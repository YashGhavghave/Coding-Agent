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


def _safe_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path is outside the selected project.")
    return candidate


def _build_tools(project_root: Path) -> tuple[Any, Any]:
    root = project_root.resolve()

    def list_project_files() -> list[str]:
        """List readable source and configuration files in the selected project."""
        paths = []
        for path in root.rglob("*"):
            if any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts):
                continue
            if path.is_file() and path.suffix.lower() in {
                ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
                ".html", ".css", ".json", ".toml", ".yaml", ".yml", ".md",
            }:
                paths.append(path.relative_to(root).as_posix())
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
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"File is not UTF-8 text: {path}"

    return list_project_files, read_project_file


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
    list_files, read_file = _build_tools(project_root)
    agent = LlmAgent(
        name="coding_agent",
        model=LiteLlm(
            model=f"openai/{MODEL}",
            api_base="https://router.huggingface.co/v1",
            api_key=api_key,
        ),
        description="Inspects a local project and proposes focused code changes.",
        instruction=(
            "You are a careful coding agent. Inspect the selected project using the tools; "
            "read files relevant to the task before proposing edits. Never claim to have run "
            "tests. Return only a JSON object with keys: plan (array of short steps), "
            "changes (array of objects with path and complete replacement content), "
            "explanation (string), validation (string with suggested commands/checks). "
            "Do not include unchanged files. Preserve existing project conventions."
        ),
        tools=[list_files, read_file],
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
    return _extract_json(final_text)


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