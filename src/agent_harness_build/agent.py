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
MODEL = "zai-org/GLM-5.3"
SKIP_DIRECTORIES = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
MAX_FILE_BYTES = 40_000
OFFLINE_DEMO_TASK = "Validate user name and email, normalize the email, and add tests."


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


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("The model response did not contain a JSON result.")
        text = text[start : end + 1]
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = repair_json(text, return_objects=True)
    if not isinstance(result, dict):
        raise ValueError("The model response must be a JSON object.")
    result.setdefault("plan", [])
    result.setdefault("changes", [])
    result.setdefault("explanation", "")
    result.setdefault("validation", "")
    return result


async def _run_agent(task: str, project_root: Path, api_key: str) -> dict[str, Any]:
    accessed_files: set[str] = set()
    list_files, read_file, search_code = _build_tools(project_root, accessed_files)
    agent = LlmAgent(
        name="coding_agent",
        model=LiteLlm(
            model=f"openai/{MODEL}",
            api_base="https://router.huggingface.co/v1",
            api_key=api_key,
        ),
        description="Inspects a local project and proposes focused code changes.",
        instruction=(
            "You are a careful coding agent. Start by listing files, search for task-related "
            "symbols or terms, then read the relevant files before proposing edits. Never "
            "claim to have run tests. Return only a JSON object with keys: plan (array of short steps), "
            "changes (array of objects with path and complete replacement content), "
            "explanation (string), validation (string with suggested commands/checks). "
            "Do not include unchanged files. Preserve existing project conventions."
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


def generate_proposal(task: str, project_root: Path, api_key: str) -> dict[str, Any]:
    if not task.strip():
        raise ValueError("Enter a coding task first.")
    if not api_key.strip():
        raise ValueError("Add a Hugging Face token with Inference Providers permission.")
    return asyncio.run(_run_agent(task.strip(), project_root, api_key.strip()))


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


def environment_api_key() -> str:
    return os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")