import pytest

from agent_harness_build.agent import (
    _build_tools,
    _extract_json,
    _safe_path,
    apply_changes,
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