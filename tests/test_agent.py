import pytest

from agent_harness_build.agent import (
    _build_tools,
    _extract_json,
    _safe_path,
    apply_changes,
    discover_project_dirs,
    generate_proposal,
    get_model_config,
    make_unified_diff,
    offline_demo_proposal,
    validate_python_changes,
)


def test_project_tools_read_files_and_reject_escape(tmp_path):
    (tmp_path / "module.py").write_text("answer = 42\n", encoding="utf-8")
    list_files, read_file, search_code = _build_tools(tmp_path)

    assert list_files() == ["module.py"]
    assert read_file("module.py") == "answer = 42\n"
    with pytest.raises(ValueError, match="outside"):
        read_file("../outside.py")


def test_search_tool_returns_matching_file_and_line(tmp_path):
    (tmp_path / "service.py").write_text("def create_user(email):\n    return email\n", encoding="utf-8")
    _, _, search_code = _build_tools(tmp_path)

    assert search_code("email") == "service.py:1: def create_user(email):\nservice.py:2:     return email"


def test_apply_changes_validates_all_paths_before_writing(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        apply_changes(
            tmp_path,
            [
                {"path": "safe.py", "content": "safe = True\n"},
                {"path": "../outside.py", "content": "unsafe = True\n"},
            ],
        )

    assert not (tmp_path / "safe.py").exists()


def test_apply_and_diff_changes(tmp_path):
    (tmp_path / "module.py").write_text("answer = 41\n", encoding="utf-8")
    changes = [{"path": "module.py", "content": "answer = 42\n"}]

    diff = make_unified_diff(tmp_path, changes)["module.py"]
    assert "-answer = 41" in diff
    assert "+answer = 42" in diff
    assert apply_changes(tmp_path, changes) == ["module.py"]
    assert (tmp_path / "module.py").read_text(encoding="utf-8") == "answer = 42\n"


def test_extract_json_from_fenced_response():
    result = _extract_json('```json\n{"plan": ["inspect"]}\n```')

    assert result["plan"] == ["inspect"]
    assert result["changes"] == []


def test_extract_json_repairs_missing_comma():
    result = _extract_json(
        '{"plan": ["inspect"], "changes": '
        '[{"path": "app.py" "content": "answer = 42"}]}'
    )

    assert result["changes"] == [{"path": "app.py", "content": "answer = 42"}]


def test_extract_json_handles_conversational_text():
    result = _extract_json(
        'Sure! Here is the proposal:\n\n'
        '```json\n{"plan": ["do something"], "changes": []}\n```\n\n'
        'Let me know if you need more changes.'
    )

    assert result["plan"] == ["do something"]


def test_extract_json_fallback_markdown():
    markdown_text = """
    Here is what needs to be done:
    1. Update the service function
    2. Write unit tests

    ```python
    # user_service.py
    def create_user(payload):
        return {}
    ```
    """
    result = _extract_json(markdown_text)

    assert len(result["plan"]) == 2
    assert result["changes"][0]["path"] == "user_service.py"
    assert "def create_user" in result["changes"][0]["content"]


def test_safe_path_rejects_absolute_escape(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        _safe_path(tmp_path, "C:/Windows/win.ini")


def test_offline_proposal_python_files_pass_syntax_validation():
    proposal = offline_demo_proposal()

    results = validate_python_changes(proposal["changes"])

    assert results == {
        "user_service.py": "Python syntax passed",
        "test_user_service.py": "Python syntax passed",
    }


def test_python_syntax_validation_reports_errors():
    results = validate_python_changes([{"path": "broken.py", "content": "def broken(:\n"}])

    assert results["broken.py"].startswith("Syntax error on line 1")


def test_model_config_supports_glm53():
    hf_config = get_model_config("GLM 5.3 (HF)")

    assert hf_config["provider"] == "openai"
    assert hf_config["model"] == "zai-org/GLM-5.3"
    assert "https://router.huggingface.co/v1" in hf_config["api_base"]


def test_discover_project_dirs_skips_generated_folders(tmp_path):
    keep_dir = tmp_path / "demo_project"
    keep_dir.mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "__pycache__").mkdir()

    assert discover_project_dirs(tmp_path) == [keep_dir, tmp_path / "notes"]


def test_generate_proposal_with_glm53(monkeypatch, tmp_path):
    calls = []

    async def fake_run_agent(task, project_root, api_key, model_name="GLM 5.3 (HF)"):
        calls.append((model_name, api_key))
        return {"plan": ["step 1"], "changes": [], "explanation": "all good", "validation": "pytest"}

    monkeypatch.setattr("agent_harness_build.agent._run_agent", fake_run_agent)

    proposal = generate_proposal("Fix it", tmp_path, "hf-token", model_name="GLM 5.3 (HF)")

    assert proposal["plan"] == ["step 1"]
    assert calls == [("GLM 5.3 (HF)", "hf-token")]


def test_cli_offline_flow(tmp_path, monkeypatch, capsys):
    from agent_harness_build.cli import run_cli

    (tmp_path / "user_service.py").write_text("def create_user(payload): return {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["patchwork", "--project", str(tmp_path), "--offline", "--apply"],
    )

    exit_code = run_cli()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[PLAN]" in captured.out
    assert "[PROPOSED CHANGES]" in captured.out
    assert "Successfully applied changes" in captured.out
    assert (tmp_path / "test_user_service.py").exists()
