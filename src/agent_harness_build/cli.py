import argparse
import sys
from pathlib import Path

from agent_harness_build.agent import (
    DEFAULT_MODEL,
    apply_changes,
    environment_api_key,
    generate_proposal,
    make_unified_diff,
    offline_demo_proposal,
    validate_python_changes,
)


def run_cli() -> int:
    parser = argparse.ArgumentParser(
        prog="patchwork",
        description="Patchwork: AI Coding Agent that turns tasks into reviewable patches.",
    )
    parser.add_argument(
        "--project",
        "-p",
        type=str,
        default="demo_project",
        help="Path to the target project directory (default: demo_project).",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="",
        help="Natural language coding task to execute.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run using the bundled offline demo proposal (no API key needed).",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--apply",
        "-y",
        action="store_true",
        help="Automatically apply changes without interactive confirmation.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Streamlit web interface.",
    )

    args = parser.parse_args()

    # If --ui is passed or no task/offline flag is provided, launch web UI
    if args.ui or (not args.task and not args.offline):
        import subprocess

        app_path = Path(__file__).resolve().parents[2] / "app.py"
        if not app_path.exists():
            app_path = Path.cwd() / "app.py"
        return subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)]).returncode

    project_root = Path(args.project).resolve()
    if not project_root.is_dir():
        print(f"Error: Project directory '{project_root}' does not exist.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("PATCHWORK CODING AGENT - CLI MODE")
    print("=" * 60)
    print(f"Target Project: {project_root}")
    print(f"Task: {args.task or 'Offline Example'}")
    print("-" * 60)

    if args.offline:
        print("Loading offline proposal...")
        proposal = offline_demo_proposal()
    else:
        api_key = environment_api_key(args.model)
        if not api_key:
            print(f"Error: Missing API key for {args.model}. Set HF_TOKEN or pass --offline.", file=sys.stderr)
            return 1
        print(f"Analyzing codebase and generating proposal using {args.model}...")
        try:
            proposal = generate_proposal(args.task, project_root, api_key, model_name=args.model)
        except Exception as err:
            print(f"Agent failed to generate proposal: {err}", file=sys.stderr)
            return 1

    print("\n[PLAN]")
    for i, step in enumerate(proposal.get("plan", []), start=1):
        print(f"  {i}. {step}")

    if proposal.get("inspected_files"):
        print(f"\n[INSPECTED FILES] ({len(proposal['inspected_files'])} files)")
        for f in proposal["inspected_files"]:
            print(f"  - {f}")

    if proposal.get("explanation"):
        print(f"\n[EXPLANATION]\n  {proposal['explanation']}")

    changes = proposal.get("changes", [])
    if not changes:
        print("\nNo code changes proposed.")
        return 0

    print(f"\n[PROPOSED CHANGES] ({len(changes)} files)")
    diffs = make_unified_diff(project_root, changes)
    for path, diff in diffs.items():
        print(f"\n--- Diff for {path} ---")
        print(diff if diff.strip() else "(No diff)")

    # Syntax Validation
    python_checks = validate_python_changes(changes)
    has_syntax_errors = False
    if python_checks:
        print("\n[VALIDATION]")
        for path, status in python_checks.items():
            print(f"  {path}: {status}")
            if status.startswith("Syntax error"):
                has_syntax_errors = True

    if has_syntax_errors:
        print("\nBlocked: Python syntax errors detected in proposal. Changes cannot be applied.", file=sys.stderr)
        return 1

    if proposal.get("validation"):
        print(f"\n[SUGGESTED VALIDATION COMMAND]\n  {proposal['validation']}")

    # Apply changes confirmation
    if not args.apply:
        confirm = input("\nApply these changes to the project? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Changes rejected by user. Aborting.")
            return 0

    written = apply_changes(project_root, changes)
    print(f"\nSuccessfully applied changes to: {', '.join(written)}")
    return 0
