"""Streamlit presentation layer for the QA explorer."""

from __future__ import annotations

import queue
import sys
import threading
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
BACKEND_APP = ROOT / "backend" / "app"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_APP) not in sys.path:
    sys.path.insert(0, str(BACKEND_APP))

from runner import run_exploration  # noqa: E402
from asset_manager import AssetManager  # noqa: E402


st.set_page_config(page_title="QA Explorer", page_icon="🔎", layout="wide")
st.title("QA Explorer")
st.caption("Explore an application with the configured QA planning agent.")


def is_valid_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def start_run(inputs: dict[str, object]) -> tuple[threading.Thread, queue.Queue]:
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            run_exploration(on_event=events.put, **inputs)
        except Exception:
            # The runner already emits a user-safe error event.
            pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, events


asset_manager = AssetManager()
with st.expander("Test Assets", expanded=True):
    st.caption("Upload one resume for applications that include a file-upload field.")
    uploaded_resume = st.file_uploader(
        "Resume",
        type=["pdf", "doc", "docx"],
        key="resume_upload",
    )
    if uploaded_resume is not None:
        uploaded_content = uploaded_resume.getvalue()
        upload_signature = (uploaded_resume.name, sha256(uploaded_content).hexdigest())
        if st.session_state.get("stored_resume_signature") != upload_signature:
            try:
                stored_path = asset_manager.save_resume(
                    uploaded_resume.name,
                    uploaded_content,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["stored_resume_signature"] = upload_signature
                st.success(f"Resume saved: {stored_path.name}")
    else:
        stored_resume = asset_manager.resolve_resume_path()
        if stored_resume:
            st.caption(f"Stored resume: {stored_resume.name}")


with st.form("exploration_form"):
    website_url = st.text_input("Website URL *", placeholder="https://example.com")
    username = st.text_input("Username (optional)")
    password = st.text_input("Password (optional)", type="password")
    application_context = st.text_area(
        "Application Context (optional)",
        placeholder=("E-commerce website\nHospital Management System\nBanking Portal\n\n"
                     "Add roles, business rules, important modules, pages to prioritize or avoid."),
        height=130,
    )
    exploration_goal = st.text_area(
        "Exploration Goal (optional)",
        placeholder=(
            "Go to Candidates from the sidebar, search for Nicole, edit the candidate, "
            "change the name to Max, save, then logout"
        ),
        height=100,
    )
    st.caption("Choose a recognized command when describing the page or module to open.")
    st.info(
        "Recognized commands: **Go to**, **Select**, **Choose**, **Open**, **Navigate to**."
    )
    st.success(
        "Good example: `Go to Candidates from the sidebar, search for Nicole, open Edit "
        "Candidate, change the name to Max, save, then logout.`"
    )
    st.warning(
        "Avoid: `Click on Candidates from the left sidebar...` — the current goal parser may "
        "not identify `Candidates` as the required destination."
    )
    max_steps = st.number_input("Maximum Exploration Steps", min_value=1, max_value=500, value=30, step=1)
    submitted = st.form_submit_button("Start Exploration", type="primary", use_container_width=True)

if submitted:
    if not is_valid_url(website_url):
        st.error("Enter a valid http:// or https:// website URL.")
    else:
        run_inputs = {
            "start_url": website_url.strip(), "username": username, "password": password,
            "application_context": application_context, "exploration_goal": exploration_goal,
            "max_steps": int(max_steps),
        }
        worker, events = start_run(run_inputs)
        mode = "Goal Driven Exploration" if exploration_goal.strip() else "Autonomous Discovery"
        status = "Starting exploration"
        current_url = website_url.strip()
        logs: list[str] = []
        left, right = st.columns([1, 1])
        with left:
            mode_box = st.info(f"Mode: {mode}")
            status_box = st.empty()
            url_box = st.empty()
            log_box = st.empty()
        with right:
            screenshot_box = st.empty()

        while worker.is_alive() or not events.empty():
            try:
                event = events.get(timeout=0.25)
            except queue.Empty:
                event = None
            if event:
                status = event.get("status", status)
                current_url = event.get("url", current_url)
                if event.get("type") == "log":
                    logs.append(event["message"])
                    logs = logs[-100:]
                screenshot_path = event.get("screenshot_path")
                if screenshot_path and Path(screenshot_path).exists():
                    screenshot_box.image(screenshot_path, caption="Latest screenshot", use_container_width=True)
            status_box.metric("Exploration Status", status)
            url_box.code(current_url, language=None)
            log_box.code("\n".join(logs) or "Waiting for backend logs…", language=None)

        if status.startswith("Exploration failed"):
            st.error(status)
        else:
            st.success(status)
