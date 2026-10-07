import os
from pathlib import Path

import streamlit as st

from agent_harness_build.agent import (
    DEFAULT_MODEL,
    apply_changes,
    environment_api_key,
    generate_proposal,
    get_model_config,
    make_unified_diff,
    OFFLINE_DEMO_TASK,
    offline_demo_proposal,
    validate_python_changes,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = WORKSPACE_ROOT / "demo_project"

st.set_page_config(page_title="Patchwork Coding Agent", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #e5eee9; --paper: #101715; --surface: #18221f; --green: #53c6a2; --coral: #f08062; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }
    .stApp { background: radial-gradient(ellipse at 100% 0%, #20372e 0%, transparent 37%), var(--paper); }
    .block-container { max-width: 1180px; padding-top: 2.6rem; }
    h1 { font-size: 2.4rem !important; letter-spacing: -0.02em !important; line-height: 1.1 !important; }
    code, pre { font-family: 'DM Mono', monospace !important; }
    [data-testid="stSidebar"] { background: #141e1b; }
    .eyebrow { color: var(--green); font: 500 0.76rem 'DM Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stStatusWidget"] { border: 1px solid #34453e; border-radius: 6px; }
    .stTextInput input, .stTextArea textarea { background: var(--surface); color: var(--ink); border-color: #40534a; }
    div[data-testid="stCode"] { background: var(--surface); }
    .stButton button[kind="primary"] { background: var(--green); border-color: var(--green); color: #10201a; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">PATCHWORK &bull; CODING AGENT</div>', unsafe_allow_html=True)
st.title("Turn coding tasks into reviewable patches.")
st.caption("Powered by GLM-5.3 and Google Agent Development Kit (ADK)")

st.markdown(
    """
    <div style="margin: 0 0 1rem 0; padding: 0.55rem 0.75rem; border: 1px solid #34453e; border-radius: 10px; background: rgba(24, 34, 31, 0.72);">
      <strong>Flow:</strong>
      1) Understand task &nbsp;→&nbsp; 2) Select files &nbsp;→&nbsp; 3) Plan &nbsp;→&nbsp; 4) Review patch &nbsp;→&nbsp; 5) Validate &nbsp;→&nbsp; 6) Apply or reject
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Connection")
    model_config = get_model_config(DEFAULT_MODEL)
    server_token = environment_api_key(DEFAULT_MODEL)
    if server_token:
        st.caption(f"Using the server's {model_config['label']} token.")
    user_token = st.text_input(
        "Use your own Hugging Face token (optional)" if server_token else "Hugging Face token",
        type="password",
        help="Token with Inference Providers permission. Used only for this session. It is not written to project files.",
    )
    token = user_token.strip() or server_token
    st.caption(f"Model: **{model_config['label']}** (`zai-org/GLM-5.3`)")
    st.divider()
    st.subheader("Project")
    hosted_space = bool(os.getenv("SPACE_ID"))
    hosted_project = Path(os.getenv("PATCHWORK_PROJECT_ROOT", str(DEFAULT_PROJECT))).expanduser()
    default_project_value = str(hosted_project if hosted_space else DEFAULT_PROJECT)

    if "selected_project_path" not in st.session_state:
        st.session_state.selected_project_path = default_project_value

    folder_button = st.button("Browse for project folder", use_container_width=True, disabled=hosted_space)
    if folder_button and not hosted_space:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(title="Select project folder")
        root.update()
        root.destroy()
        if selected:
            st.session_state.selected_project_path = selected

    project_input = st.text_input(
        "Project folder path",
        value=st.session_state.selected_project_path,
        disabled=hosted_space,
        help="Use the browse button to choose a folder in your file explorer, or type the path manually.",
    )
    project_root = Path(project_input).expanduser().resolve()

if not project_root.is_dir():
    st.error("That project folder does not exist or is not a directory.")
    st.stop()

left, right = st.columns([1.45, 1], gap="large")
with left:
    st.subheader("Step 1: Describe the task")
    st.caption("A clear task produces a clearer file selection and patch.")
    st.session_state.setdefault("task_request", "")
    task_presets = [
        ("Validate user fields", "Validate user name and email, normalize the email, and add tests."),
        ("Reject bad email", "Reject malformed email addresses and blank names; add tests."),
        ("Cover bad payloads", "Add tests for non-dictionary payloads and missing user fields."),
    ]
    preset_columns = st.columns(len(task_presets))
    for index, (label, preset) in enumerate(task_presets):
        with preset_columns[index]:
            if st.button(label, key=f"task_preset_{index}", use_container_width=True):
                st.session_state["task_request"] = preset
                st.rerun()
    task = st.text_area(
        "Describe the change",
        placeholder="Add input validation to this API and write a test for it.",
        height=130,
        label_visibility="collapsed",
        key="task_request",
    )
    generate_column, offline_column = st.columns([1.35, 1])
    with generate_column:
        generate = st.button("Analyze and propose", type="primary", use_container_width=True)
    with offline_column:
        offline_demo = st.button("Load offline example", use_container_width=True)
with right:
    st.subheader("Step 2: Selected codebase")
    st.caption("The agent inspects this project and decides which files are relevant.")
    project_files = sorted(
        path.relative_to(project_root).as_posix()
        for path in project_root.rglob("*")
        if path.is_file()
        and not any(part in {".git", ".venv", "venv", "__pycache__", "node_modules"} for part in path.relative_to(project_root).parts)
    )[:12]
    st.code("\n".join(project_files) if project_files else "No files found", language="text")

if generate or offline_demo:
    st.session_state.pop("proposal", None)
    st.session_state.pop("proposal_root", None)
    try:
        with st.status("Inspecting files and preparing a proposal...", expanded=True) as status:
            if offline_demo:
                if project_root != DEFAULT_PROJECT.resolve():
                    raise ValueError("The offline example is available for the bundled demo project only.")
                st.write("Loading the bundled example proposal. No model request is made.")
                proposal_result = offline_demo_proposal()
            else:
                st.write("Understand task → search files → inspect relevant code → draft changes.")
                proposal_result = generate_proposal(
                    task,
                    project_root,
                    token,
                    model_name=DEFAULT_MODEL,
                )
            st.session_state["proposal"] = proposal_result
            st.session_state["proposal_root"] = str(project_root)
            status.update(label="Proposal ready for review", state="complete", expanded=False)
    except Exception as error:
        st.error(f"Could not generate a proposal: {error}")

proposal = st.session_state.get("proposal")
if proposal:
    if st.session_state.get("proposal_root") != str(project_root):
        st.warning("The selected project changed. Generate a new proposal before reviewing or applying it.")
        st.stop()
    st.divider()
    st.subheader("Step 3: Plan")
    if proposal.get("offline_demo"):
        st.caption(f"Offline example: {OFFLINE_DEMO_TASK}")
    plan = proposal.get("plan", [])
    if isinstance(plan, list) and plan:
        for index, step in enumerate(plan, start=1):
            st.markdown(f"**{index}.** {step}")
    else:
        st.write("No plan was returned.")
    if proposal.get("explanation"):
        st.markdown(proposal["explanation"])

    inspected_files = proposal.get("inspected_files", [])
    if inspected_files:
        with st.expander(f"Inspected files ({len(inspected_files)})"):
            st.code("\n".join(inspected_files), language="text")

    changes = proposal.get("changes", [])
    if not isinstance(changes, list):
        changes = []
    st.subheader(f"Step 4: Proposed patch ({len(changes)} files)")
    if changes:
        diffs = make_unified_diff(project_root, changes)
        for path, diff in diffs.items():
            with st.expander(path, expanded=True):
                st.code(diff or "No changes", language="diff")
                change = next(item for item in changes if item.get("path") == path)
                with st.popover("View proposed file"):
                    st.code(change.get("content", ""), language=Path(path).suffix.lstrip(".") or "text")
        python_checks = validate_python_changes(changes)
        if python_checks:
            st.subheader("Step 5: Validation")
            for path, result in python_checks.items():
                if result.startswith("Syntax error"):
                    st.error(f"{path}: {result}")
                else:
                    st.success(f"{path}: {result}")
        apply_column, reject_column = st.columns([1, 1])
        with apply_column:
            apply_clicked = st.button("Apply reviewed changes", type="primary", use_container_width=True)
        with reject_column:
            reject_clicked = st.button("Reject proposal", use_container_width=True)
        if reject_clicked:
            st.session_state.pop("proposal", None)
            st.session_state.pop("proposal_root", None)
            st.rerun()
        if apply_clicked:
            try:
                syntax_errors = [
                    f"{path}: {result}" for path, result in python_checks.items()
                    if result.startswith("Syntax error")
                ]
                if syntax_errors:
                    raise ValueError("Fix Python syntax errors before applying: " + "; ".join(syntax_errors))
                written = apply_changes(project_root, changes)
                st.success(f"Updated: {', '.join(written)}")
                st.session_state.pop("proposal", None)
                st.session_state.pop("proposal_root", None)
            except Exception as error:
                st.error(f"Could not apply changes: {error}")
    else:
        st.info("The agent proposed no file edits for this task.")
        if st.button("Dismiss proposal"):
            st.session_state.pop("proposal", None)
            st.session_state.pop("proposal_root", None)
            st.rerun()
    if proposal.get("validation"):
        st.subheader("Step 6: Suggested validation")
        st.code(proposal["validation"], language="text")