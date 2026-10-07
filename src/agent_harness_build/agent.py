import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from json_repair import repair_json

APP_NAME = "harness_coding_agent"
DEFAULT_MODEL = "GLM 5.3 (HF)"
MODEL_OPTIONS = {
    "GLM 5.3 (HF)": {
        "provider": "openai",
        "model": "zai-org/GLM-5.3",
        "api_base": "https://router.huggingface.co/v1",
        "token_names": ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"],
        "label": "GLM-5.3 (Hugging Face Inference)",
    },
}
SKIP_DIRECTORIES = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
MAX_FILE_BYTES = 40_000
OFFLINE_DEMO_TASK = "Validate user name and email, normalize the email, and add tests."


def get_model_config(model_name: str = DEFAULT_MODEL) -> dict[str, Any]:
    try:
        return MODEL_OPTIONS[model_name].copy()
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc


def discover_project_dirs(root: Path | None = None) -> list[Path]:
    base = (root or Path.cwd()).resolve()
    if not base.exists() or not base.is_dir():
        return []
    candidates = []
    for child in sorted(base.iterdir(), key=lambda entry: entry.name.lower()):
        if child.is_dir() and child.name not in SKIP_DIRECTORIES:
            candidates.append(child)
    return candidates


def _safe_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path is outside the selected project.")
    return candidate


def _build_tools(
    project_root: Path, accessed_files: set[str] | None = None
) -> tuple[Any, Any, Any]:
    root = project_root.resolve()
    tracked_files = accessed_files if accessed_files is not None else set()
    supported_extensions = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
        ".html", ".css", ".json", ".toml", ".yaml", ".yml", ".md",
    }

    def list_project_files() -> list[str]:
        """List readable source and configuration files in the selected project."""
        paths = []
        for path in root.rglob("*"):
            if any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts):
                continue
            relative_path = path.relative_to(root).as_posix()
            if path.is_file() and path.suffix.lower() in supported_extensions:
                try:
                    _safe_path(root, relative_path)
                except ValueError:
                    continue
                paths.append(relative_path)
                if len(paths) >= 250:
                    break
        return sorted(paths)

    def read_project_file(path: str) -> str:
        """Read one UTF-8 text file by its relative path inside the selected project."""
        file_path = _safe_path(root, path)
        if not file_path.is_file():
            return f"File not found: {path}"
        if file_path.stat().st_size > MAX_FILE_BYTES:
            return f"File exceeds the {MAX_FILE_BYTES}-byte read limit: {path}"
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"File is not UTF-8 text: {path}"
        tracked_files.add(file_path.relative_to(root).as_posix())
        return content

    def search_project_code(query: str) -> str:
        """Search source lines for task-related terms and return file/line matches."""
        terms = [term.lower() for term in re.findall(r"[\w.-]+", query) if len(term) > 2]
        if not terms:
            return "Provide a search term with at least three characters."

        matches = []
        for path in root.rglob("*"):
            if any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts):
                continue
            if not path.is_file() or path.suffix.lower() not in supported_extensions:
                continue
            relative_path = path.relative_to(root).as_posix()
            try:
                file_path = _safe_path(root, relative_path)
            except ValueError:
                continue
            if file_path.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if any(term in line.lower() for term in terms):
                    matches.append(f"{relative_path}:{line_number}: {line[:300]}")
                    tracked_files.add(relative_path)
                    if len(matches) >= 20:
                        return "\n".join(matches)
        return "\n".join(matches) if matches else "No matching source lines found."

    return list_project_files, read_project_file, search_project_code


def _is_valid_proposal(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    # A genuine proposal must contain at least one recognized proposal key
    return any(k in obj for k in ("plan", "changes", "explanation", "validation"))


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("The model returned an empty response.")

    result: Any = None

    # Strategy 1: Look for JSON in markdown code blocks ```json ... ```
    fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for fenced in fenced_blocks:
        try:
            parsed = json.loads(fenced.strip())
            if _is_valid_proposal(parsed):
                result = parsed
                break
        except Exception:
            try:
                parsed = repair_json(fenced.strip(), return_objects=True)
                if _is_valid_proposal(parsed):
                    result = parsed
                    break
            except Exception:
                pass

    # Strategy 2: If no fenced JSON found, try parsing outermost { and }
    if not _is_valid_proposal(result):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidate_str = text[start : end + 1].strip()
            try:
                parsed = json.loads(candidate_str)
                if _is_valid_proposal(parsed):
                    result = parsed
            except Exception:
                try:
                    parsed = repair_json(candidate_str, return_objects=True)
                    if _is_valid_proposal(parsed):
                        result = parsed
                except Exception:
                    pass

    # Strategy 3: Try repair_json directly on full text
    if not _is_valid_proposal(result):
        try:
            parsed = repair_json(text, return_objects=True)
            if isinstance(parsed, list) and parsed and _is_valid_proposal(parsed[0]):
                result = parsed[0]
            elif _is_valid_proposal(parsed):
                result = parsed
        except Exception:
            pass

    # Strategy 4: Fallback heuristic parser if model responded in free-form Markdown
    if not _is_valid_proposal(result):
        plan_steps = []
        for line in text.splitlines():
            clean = line.strip()
            if re.match(r"^(\d+\.|\*|-)\s+", clean):
                plan_steps.append(re.sub(r"^(\d+\.|\*|-)\s+", "", clean))

        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        changes = []
        for block in code_blocks:
            lines = block.strip().splitlines()
            path = "service.py"
            content = block.strip()
            if lines and lines[0].startswith(("#", "//", "/*")):
                candidate_path = lines[0].lstrip("#/* ").strip()
                if "." in candidate_path and " " not in candidate_path:
                    path = candidate_path
                    content = "\n".join(lines[1:]).strip()
            changes.append({"path": path, "content": content})

        result = {
            "plan": plan_steps or ["Implement requested changes in the codebase."],
            "changes": changes,
            "explanation": text[:600].strip(),
            "validation": "",
        }

    # Normalize structure
    result.setdefault("plan", [])
    result.setdefault("changes", [])
    result.setdefault("explanation", "")
    result.setdefault("validation", "")

    if not isinstance(result["plan"], list):
        result["plan"] = [str(result["plan"])] if result["plan"] else []
    if not isinstance(result["changes"], list):
        result["changes"] = []

    return result


async def _run_agent(task: str, project_root: Path, api_key: str, model_name: str = DEFAULT_MODEL) -> dict[str, Any]:
    accessed_files: set[str] = set()
    list_files, read_file, search_code = _build_tools(project_root, accessed_files)
    model_config = get_model_config(model_name)
    lite_llm_model = f"{model_config['provider']}/{model_config['model']}"
    lite_llm_kwargs: dict[str, Any] = {"model": lite_llm_model, "api_key": api_key}
    if model_config["api_base"]:
        lite_llm_kwargs["api_base"] = model_config["api_base"]
    agent = LlmAgent(
        name="coding_agent",
        model=LiteLlm(**lite_llm_kwargs),
        description="Inspects a local project and proposes focused code changes.",
        instruction=(
            "You are an expert coding agent. First inspect the repository using list_project_files, "
            "search_project_code, and read_project_file. After understanding the code, you MUST "
            "provide your final output as a valid JSON object matching this schema:\n"
            "{\n"
            '  "plan": ["Step 1 explanation", "Step 2 explanation"],\n'
            '  "changes": [{"path": "relative/path/to/file.py", "content": "COMPLETE FILE CONTENT"}],\n'
            '  "explanation": "Summary of what was changed and why.",\n'
            '  "validation": "Suggested command to verify changes (e.g., python -m unittest ...)"\n'
            "}\n"
            "Never include unchanged files. Output only the JSON object."
        ),
        tools=[list_files, search_code, read_file],
    )
    session_service = InMemorySessionService()
    session_id = str(uuid4())
    await session_service.create_session(
        app_name=APP_NAME, user_id="local-user", session_id=session_id
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    message = types.Content(
        role="user",
        parts=[
            types.Part(
                text=(
                    f"Selected project root: {project_root.resolve()}\n"
                    f"Developer task: {task}\n\n"
                    "Inspect the project first, then prepare a safe proposal."
                )
            )
        ],
    )
    final_text = ""
    async for event in runner.run_async(
        user_id="local-user", session_id=session_id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(part.text or "" for part in event.content.parts)
    if not final_text:
        raise ValueError("The agent returned no final response.")
    result = _extract_json(final_text)
    result["inspected_files"] = sorted(accessed_files)
    return result


def offline_demo_proposal() -> dict[str, Any]:
    return {
        "plan": [
            "Validate that the payload contains a non-empty name and a plausible email.",
            "Normalize whitespace and email casing before returning the user record.",
            "Add tests for normalization and invalid payloads.",
        ],
        "changes": [
            {
                "path": "user_service.py",
                "content": '''def create_user(payload):
    """Create a user record after validating and normalizing its fields."""
    if not isinstance(payload, dict):
        raise ValueError("User payload must be a dictionary.")

    email = payload.get("email")
    name = payload.get("name")
    if not isinstance(email, str) or "@" not in email.strip():
        raise ValueError("A valid email is required.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A name is required.")

    return {"email": email.strip().lower(), "name": name.strip()}
''',
            },
            {
                "path": "test_user_service.py",
                "content": '''import unittest

from user_service import create_user


class CreateUserTests(unittest.TestCase):
    def test_normalizes_user_fields(self):
        self.assertEqual(
            create_user({"email": " DEV@EXAMPLE.COM ", "name": " Dev "}),
            {"email": "dev@example.com", "name": "Dev"},
        )

    def test_rejects_invalid_payloads(self):
        for payload in (
            None,
            {"email": "not-an-email", "name": "Dev"},
            {"email": "dev@example.com", "name": " "},
            {"email": " ", "name": "Dev"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                create_user(payload)


if __name__ == "__main__":
    unittest.main()
''',
            },
        ],
        "explanation": (
            "The sample service accepted missing or malformed fields. The proposal adds "
            "basic validation, normalizes user input, and covers both valid and invalid cases."
        ),
        "validation": "python -m unittest discover -s demo_project -v",
        "inspected_files": ["user_service.py", "test_user_service.py"],
        "offline_demo": True,
    }


def validate_python_changes(changes: list[dict[str, Any]]) -> dict[str, str]:
    results = {}
    for change in changes:
        path = change.get("path")
        content = change.get("content")
        if not isinstance(path, str) or not path.endswith(".py") or not isinstance(content, str):
            continue
        try:
            compile(content, path, "exec")
        except SyntaxError as error:
            results[path] = f"Syntax error on line {error.lineno}: {error.msg}"
        else:
            results[path] = "Python syntax passed"
    return results


def generate_proposal(
    task: str,
    project_root: Path,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    if not task.strip():
        raise ValueError("Enter a coding task first.")
    if not api_key.strip():
        raise ValueError("Add a Hugging Face token (with Inference Providers permission) before generating a proposal.")

    try:
        return asyncio.run(_run_agent(task.strip(), project_root, api_key.strip(), model_name=model_name))
    except Exception as error:
        err_msg = str(error)
        if "AuthenticationError" in err_msg or "401" in err_msg:
            raise ValueError("Invalid Hugging Face token. Please check your token permissions at https://huggingface.co/settings/tokens.") from error
        raise


def apply_changes(project_root: Path, changes: list[dict[str, Any]]) -> list[str]:
    root = project_root.resolve()
    validated_changes = []
    for change in changes:
        relative_path = change.get("path")
        content = change.get("content")
        if not isinstance(relative_path, str) or not isinstance(content, str):
            raise ValueError("Each proposed change must include a path and text content.")
        destination = _safe_path(root, relative_path)
        if destination == root:
            raise ValueError("A project directory cannot be replaced by a file.")
        validated_changes.append((destination, content))

    written = []
    for destination, content in validated_changes:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written.append(destination.relative_to(root).as_posix())
    return written


def make_unified_diff(project_root: Path, changes: list[dict[str, Any]]) -> dict[str, str]:
    import difflib

    root = project_root.resolve()
    diffs = {}
    for change in changes:
        relative_path = change.get("path")
        content = change.get("content")
        if not isinstance(relative_path, str) or not isinstance(content, str):
            continue
        destination = _safe_path(root, relative_path)
        before = destination.read_text(encoding="utf-8").splitlines(keepends=True) if destination.is_file() else []
        after = content.splitlines(keepends=True)
        diffs[relative_path] = "".join(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative_path}" if before else "/dev/null",
                tofile=f"b/{relative_path}",
            )
        )
    return diffs


def environment_api_key(model_name: str = DEFAULT_MODEL) -> str:
    model_config = get_model_config(model_name)
    for env_name in model_config["token_names"]:
        value = os.getenv(env_name, "")
        if value.strip():
            return value.strip()
    return ""