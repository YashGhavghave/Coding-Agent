import os
from pathlib import Path

import streamlit as st

from agent_harness_build.agent import (
    apply_changes,
    environment_api_key,
    generate_proposal,
    make_unified_diff,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = WORKSPACE_ROOT / "demo_project"

st.set_page_config(page_title="Patchwork | Coding Agent", page_icon="P", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #e5eee9; --paper: #101715; --surface: #18221f; --green: #53c6a2; --coral: #f08062; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }
    .stApp { background: radial-gradient(ellipse at 100% 0%, #20372e 0%, transparent 37%), var(--paper); }
    .block-container { max-width: 1180px; padding-top: 2.6rem; }
    h1 { font-size: 2.6rem !important; letter-spacing: 0 !important; line-height: 1.08 !important; }
    code, pre { font-family: 'DM Mono', monospace !important; }
    [data-testid="stSidebar"] { background: #141e1b; }
    .eyebrow { color: var(--green); font: 500 0.76rem 'DM Mono', monospace; text-transform: uppercase; }
    div[data-testid="stStatusWidget"] { border: 1px solid #34453e; border-radius: 6px; }
    .stTextInput input, .stTextArea textarea { background: var(--surface); color: var(--ink); border-color: #40534a; }
    div[data-testid="stCode"] { background: var(--surface); }
    .stButton button[kind="primary"] { background: var(--green); border-color: var(--green); color: #10201a; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">PATCHWORK / LOCAL CODING AGENT</div>', unsafe_allow_html=True)
st.title("Turn a task into a reviewable patch.")
st.caption("Google ADK orchestrates repository inspection and proposal generation with GLM 5.3 on Hugging Face.")

with st.sidebar:
    st.subheader("Connection")
    token = st.text_input(
        "Hugging Face token",
        value=environment_api_key(),
        type="password",
        help="Used only for this session. It is not written to project files.",
    )
    st.caption("Model: zai-org/GLM-5.3 via Hugging Face Inference Providers")
    st.divider()
    st.subheader("Project")
    hosted_space = bool(os.getenv("SPACE_ID"))
    hosted_project = Path(os.getenv("PATCHWORK_PROJECT_ROOT", str(DEFAULT_PROJECT))).expanduser()
    project_input = st.text_input(
        "Project folder",
        value=str(hosted_project if hosted_space else DEFAULT_PROJECT),
        disabled=hosted_space,
        help="Hosted instances are pinned to a server-configured project folder.",
    )
    project_root = Path(project_input).expanduser().resolve()

if not project_root.is_dir():
    st.error("That project folder does not exist or is not a directory.")
    st.stop()

left, right = st.columns([1.45, 1], gap="large")
with left:
    st.subheader("Developer task")
    task = st.text_area(
        "Describe the change",
        placeholder="Add input validation to this API and write a test for it.",
        height=130,
        label_visibility="collapsed",
    )
    generate = st.button("Analyze and propose", type="primary", use_container_width=True)
with right:
    st.subheader("Selected codebase")
    project_files = sorted(
        path.relative_to(project_root).as_posix()
        for path in project_root.rglob("*")
        if path.is_file()
        and not any(part in {".git", ".venv", "venv", "__pycache__", "node_modules"} for part in path.relative_to(project_root).parts)
    )[:12]
    st.code("\n".join(project_files) if project_files else "No files found", language="text")

if generate:
    st.session_state.pop("proposal", None)
    st.session_state.pop("proposal_root", None)
    try:
        with st.status("Inspecting files and preparing a proposal...", expanded=True) as status:
            st.write("The agent is selecting relevant files, reading their contents, and drafting changes.")
            st.session_state["proposal"] = generate_proposal(task, project_root, token)
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
    st.subheader("Plan")
    plan = proposal.get("plan", [])
    if isinstance(plan, list) and plan:
        for index, step in enumerate(plan, start=1):
            st.markdown(f"**{index}.** {step}")
    else:
        st.write("No plan was returned.")
    if proposal.get("explanation"):
        st.markdown(proposal["explanation"])

    changes = proposal.get("changes", [])
    if not isinstance(changes, list):
        changes = []
    st.subheader(f"Proposed files ({len(changes)})")
    if changes:
        diffs = make_unified_diff(project_root, changes)
        for path, diff in diffs.items():
            with st.expander(path, expanded=True):
                st.code(diff or "No changes", language="diff")
                change = next(item for item in changes if item.get("path") == path)
                with st.popover("View proposed file"):
                    st.code(change.get("content", ""), language=Path(path).suffix.lstrip(".") or "text")
        if st.button("Apply reviewed changes", type="primary"):
            try:
                written = apply_changes(project_root, changes)
                st.success(f"Updated: {', '.join(written)}")
                st.session_state.pop("proposal", None)
            except Exception as error:
                st.error(f"Could not apply changes: {error}")
    else:
        st.info("The agent proposed no file edits for this task.")
    if proposal.get("validation"):
        st.subheader("Suggested validation")
        st.code(proposal["validation"], language="text")