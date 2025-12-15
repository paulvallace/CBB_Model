import os
import sys
import time
import subprocess
from datetime import date, timedelta, datetime

import streamlit as st


st.set_page_config(page_title="CBB Pipeline Runner", layout="wide")

st.title("🏀 College Basketball Pipeline (One-Click Run)")

st.caption(
    "Runs your scripts in order with live logs. "
    "Past-games step supports the interactive `input()` prompts by piping stdin."
)

# --- Helpers ---
from typing import Optional

def run_script_live(
    script_path: str,
    args=None,
    stdin_text: Optional[str] = None,
    workdir: Optional[str] = None
):
    """
    Run a python script as a subprocess and stream stdout/stderr live into Streamlit.
    Returns (returncode, full_log_text).
    """
    args = args or []
    cmd = [sys.executable, script_path] + args

    log_lines = []
    log_placeholder = st.empty()
    status_placeholder = st.empty()

    status_placeholder.info(f"Starting: `{os.path.basename(script_path)}`")

    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # feed stdin if needed (for scripts using input())
    if stdin_text is not None:
        try:
            proc.stdin.write(stdin_text)
            proc.stdin.flush()
            proc.stdin.close()
        except Exception:
            pass

    # stream output
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            log_lines.append(line.rstrip("\n"))
            # show only last N lines to keep UI fast
            tail = "\n".join(log_lines[-400:])
            log_placeholder.code(tail, language="text")
        else:
            if proc.poll() is not None:
                break
            time.sleep(0.05)

    rc = proc.returncode if proc.returncode is not None else -1
    full_log = "\n".join(log_lines)

    if rc == 0:
        status_placeholder.success(f"✅ Finished: `{os.path.basename(script_path)}`")
    else:
        status_placeholder.error(f"❌ Failed (exit {rc}): `{os.path.basename(script_path)}`")

    return rc, full_log


def script_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


# --- Locate scripts (assumes app.py is in same folder as your .py files) ---
WORKDIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    "KenPom scrape (yesterday)": os.path.join(WORKDIR, "kenpom_automate.py"),
    "Past games (OddsShark) update": os.path.join(WORKDIR, "past_games_automate.py"),
    "GameDay matchups file": os.path.join(WORKDIR, "gameday.py"),
    "Train + Predict today": os.path.join(WORKDIR, "train_model.py"),
}

with st.sidebar:
    st.header("Run options")

    # Step toggles
    run_kenpom = st.checkbox("1) KenPom scrape", value=True)
    run_past_games = st.checkbox("2) Past games update", value=False)  # default off (because it can be slow)
    run_gameday = st.checkbox("3) GameDay matchups", value=True)
    run_train = st.checkbox("4) Train + Predict", value=True)

    st.divider()
    st.subheader("Past games settings (interactive script)")
    action = st.selectbox(
        "Action (matches the script prompt)",
        options=[
            ("1", "Fetch new data only"),
            ("2", "Update master file only"),
            ("3", "Both (fetch + update master)"),
        ],
        index=2,
        format_func=lambda x: f"{x[0]}: {x[1]}",
    )[0]

    # sensible default range: last 3 days
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=3)

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)

    st.caption(
        "Note: If you choose action 2, the original script asks for a file path. "
        "This app will still prompt it via stdin unless you edit the script to accept args."
    )

    st.divider()
    stop_on_fail = st.checkbox("Stop pipeline if a step fails", value=True)

# --- Validate script files are present ---
missing = [name for name, path in SCRIPTS.items() if not script_exists(path)]
if missing:
    st.error(
        "Missing script files in the same folder as app.py:\n\n- " + "\n- ".join(missing) +
        "\n\nPut `app.py` in the same directory as your scripts (kenpom_automate.py, etc.)."
    )
    st.stop()

# --- Run buttons ---
col1, col2 = st.columns([1, 1])
with col1:
    run_all = st.button("▶️ Run selected steps", type="primary", use_container_width=True)
with col2:
    st.write("")
    st.write("")
    st.caption("You can scroll the log panels below while it runs.")

# Output area
st.subheader("Live run logs")
st.caption("Each step prints its own output here.")

# --- Pipeline runner ---
if run_all:
    selected_steps = []
    if run_kenpom:
        selected_steps.append(("KenPom scrape (yesterday)", SCRIPTS["KenPom scrape (yesterday)"], None))
    if run_past_games:
        # past_games_automate.py expects:
        # action (1/2/3), then maybe start_date/end_date (for 1 or 3), OR a file path (for 2)
        # We'll feed it lines via stdin.
        stdin_lines = []
        stdin_lines.append(action)

        if action in ("1", "3"):
            stdin_lines.append(str(start_date))
            stdin_lines.append(str(end_date))
        elif action == "2":
            # You'll likely want to customize this to a real path, OR modify script to accept args.
            stdin_lines.append("/path/to/new_data_file.csv")

        stdin_text = "\n".join(stdin_lines) + "\n"
        selected_steps.append(("Past games update", SCRIPTS["Past games (OddsShark) update"], stdin_text))

    if run_gameday:
        selected_steps.append(("GameDay matchups", SCRIPTS["GameDay matchups file"], None))
    if run_train:
        selected_steps.append(("Train + Predict", SCRIPTS["Train + Predict today"], None))

    # Run them
    all_logs = {}
    for step_name, script_path, stdin_text in selected_steps:
        st.markdown(f"### {step_name}")
        rc, log_text = run_script_live(script_path, stdin_text=stdin_text, workdir=WORKDIR)
        all_logs[step_name] = log_text

        if rc != 0 and stop_on_fail:
            st.warning("Stopped pipeline because a step failed (toggle this off in the sidebar to continue anyway).")
            break

    # Download logs
    st.divider()
    st.subheader("Download logs")
    merged = []
    for k, v in all_logs.items():
        merged.append("=" * 90)
        merged.append(k)
        merged.append("=" * 90)
        merged.append(v or "")
        merged.append("")
    merged_text = "\n".join(merged)

    st.download_button(
        "⬇️ Download run log (.txt)",
        data=merged_text.encode("utf-8"),
        file_name=f"cbb_run_log_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )
