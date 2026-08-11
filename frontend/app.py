"""Streamlit workspace for running QA Explorer."""

from __future__ import annotations

import queue
import sys
import threading
import time
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
BACKEND_APP = BACKEND_ROOT / "app"
for path in (BACKEND_ROOT, BACKEND_APP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from asset_manager import AssetManager  # noqa: E402
from pipeline_runner import (  # noqa: E402
    add_custom_requirement,
    add_custom_test_case,
    load_custom_requirements,
    load_custom_test_cases,
    load_feedback,
    run_pipeline,
    save_feedback,
)
from runner import run_exploration  # noqa: E402


st.set_page_config(page_title="Sentinel-QA", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")


def render_landing_page() -> None:
    """Render the Sentinel-QA product landing page."""
    st.markdown(
        """
        <style>
          .stApp { background: #080d1a; color: #eaf0ff; }
          header, [data-testid="stHeader"], [data-testid="stSidebar"] { display: none; }
          .block-container { max-width: 1180px; padding-top: 1.35rem; }
          .nav { display:flex; justify-content:space-between; align-items:center; padding:.8rem 0 3.7rem; }
          .brand { color:#f7f9ff; font-size:1.18rem; font-weight:800; letter-spacing:-.03em; }
          .brand b { display:inline-flex; width:28px; height:28px; align-items:center; justify-content:center; margin-right:8px; border-radius:9px; background:linear-gradient(135deg,#7c6cff,#33d5c8); }
          .nav-note { color:#98a7c8; font-size:.85rem; }
          .hero-landing { position:relative; overflow:hidden; min-height:500px; text-align:center; padding:4.7rem 1.25rem 2rem; border:1px solid rgba(150,170,255,.14); border-radius:28px; background:radial-gradient(circle at 50% 0%,rgba(98,87,255,.22),transparent 43%),#0d1427; }
          .hero-landing:before,.hero-landing:after { content:''; position:absolute; border-radius:50%; z-index:0; }
          .hero-landing:before { width:250px; height:250px; left:8%; top:29%; background:rgba(35,215,193,.13); animation:float 8s ease-in-out infinite; }
          .hero-landing:after { width:300px; height:300px; right:5%; top:8%; background:rgba(113,91,255,.18); animation:float 10s ease-in-out infinite reverse; }
          .hero-content { position:relative; z-index:1; }
          .eyebrow-landing { display:inline-block; padding:.42rem .8rem; border:1px solid rgba(110,223,211,.36); border-radius:99px; background:rgba(46,212,194,.08); color:#79e7dc; font-size:.76rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
          .hero-landing h1 { max-width:810px; margin:1.15rem auto .9rem; color:#f5f7ff; font-size:clamp(2.7rem,6vw,5.1rem); line-height:1.02; letter-spacing:-.065em; }
          .hero-landing h1 span { background:linear-gradient(95deg,#9d91ff,#57e2d4); -webkit-background-clip:text; background-clip:text; color:transparent; }
          .hero-landing p { max-width:630px; margin:0 auto; color:#aebad2; font-size:1.08rem; line-height:1.65; }
          .orbit { position:relative; width:250px; height:112px; margin:2.8rem auto 0; }
          .ring { position:absolute; inset:0; border:1px solid rgba(145,160,255,.32); border-radius:50%; transform:rotate(-12deg); }
          .ring.two { inset:20px 26px; transform:rotate(16deg); border-color:rgba(74,224,208,.35); }
          .dot { position:absolute; width:12px; height:12px; border-radius:50%; background:#5de3d6; box-shadow:0 0 22px #5de3d6; animation:orbit 4s linear infinite; }
          .dot.two { background:#a596ff; box-shadow:0 0 20px #a596ff; animation-duration:5.5s; animation-direction:reverse; }
          @keyframes float { 0%,100% { transform:translate(0,0); } 50% { transform:translate(18px,-24px); } }
          @keyframes orbit { 0% { transform:translate(20px,48px); } 25% { transform:translate(118px,0); } 50% { transform:translate(220px,48px); } 75% { transform:translate(118px,98px); } 100% { transform:translate(20px,48px); } }
          .signals { display:flex; flex-wrap:wrap; gap:.8rem; justify-content:center; margin:1.4rem 0 3.1rem; color:#b9c4dc; font-size:.86rem; }
          .signals span { padding:.45rem .7rem; border-radius:8px; background:#111a30; border:1px solid #23304c; }
          .section { padding:5.7rem 0 1rem; text-align:center; }
          .section h2 { color:#f3f6ff; font-size:2.15rem; letter-spacing:-.045em; margin:0 0 .8rem; }
          .section p { max-width:590px; color:#9daccc; line-height:1.65; margin:0 auto 2.4rem; }
          .feature { min-height:190px; text-align:left; padding:1.5rem; border:1px solid #202c47; border-radius:17px; background:linear-gradient(145deg,rgba(22,31,55,.86),rgba(12,19,35,.86)); transition:transform .25s ease,border-color .25s ease; }
          .feature:hover { transform:translateY(-6px); border-color:#6a61d6; }
          .feature h3 { color:#f2f5ff; font-size:1.04rem; margin:.9rem 0 .45rem; }
          .feature p { color:#99a8c5; font-size:.9rem; line-height:1.58; margin:0; }
          .footer-landing { color:#687799; text-align:center; font-size:.8rem; margin:4.4rem 0 1rem; }
          .stLinkButton > a { display:inline-flex; justify-content:center; align-items:center; min-height:3.1rem; border:0; border-radius:10px; padding:0 1.4rem; background:linear-gradient(100deg,#7468f6,#42d5c6); color:white!important; font-weight:750; text-decoration:none; box-shadow:0 10px 26px rgba(82,91,229,.24); transition:transform .2s ease,box-shadow .2s ease; }
          .stLinkButton > a:hover { transform:translateY(-2px); box-shadow:0 14px 32px rgba(82,91,229,.36); }
        </style>
        <div class="nav"><div class="brand"><b>S</b>Sentinel-QA</div><div class="nav-note">Autonomous quality intelligence</div></div>
        <div class="hero-landing"><div class="hero-content"><div class="eyebrow-landing">AI-powered test exploration</div><h1>Quality assurance that <span>never looks away.</span></h1><p>Sentinel-QA explores your application like a thoughtful tester—navigating real workflows, adapting to the interface, and creating clear evidence as it works.</p></div><div class="orbit"><div class="ring"></div><div class="ring two"></div><div class="dot"></div><div class="dot two"></div></div></div>
        """,
        unsafe_allow_html=True,
    )
    action_column = st.columns([1, 1.15, 1])[1]
    with action_column:
        st.link_button("Launch automation workspace  →", "?view=automation", use_container_width=True)
    st.markdown('<div class="signals"><span>● Live browser evidence</span><span>✦ Goal-directed exploration</span><span>↗ Workflow generation</span></div><div class="section"><h2>See more. Test smarter.</h2><p>Built for teams that need practical testing momentum without losing visibility into what the automation is doing.</p></div>', unsafe_allow_html=True)
    cards = [("◉", "Explore real workflows", "Set a goal and Sentinel-QA plans, navigates, and validates the journey through your live application."), ("◌", "Stay in the loop", "Follow browser state, current activity, and detailed operational logs as exploration progresses."), ("✦", "Turn discovery into action", "Capture structured workflow output that gives your quality process a reliable starting point.")]
    for column, (icon, title, description) in zip(st.columns(3), cards):
        with column:
            st.markdown(f'<div class="feature"><div style="font-size:1.5rem">{icon}</div><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)
    st.markdown('<p class="footer-landing">Sentinel-QA · Autonomous browser exploration for modern quality teams</p>', unsafe_allow_html=True)


if st.query_params.get("view") != "automation":
    render_landing_page()
    st.stop()

st.markdown(
    """
    <style>
      .stApp { background: #f7f8fc; }
      [data-testid="stSidebar"] { background: #101828; }
      [data-testid="stSidebar"] * { color: #f8fafc; }
      .hero { padding: 1.35rem 0 .75rem; }
      .eyebrow { color: #635bff; font-weight: 700; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; }
      .hero h1 { color: #172033; font-size: 2.45rem; margin: .15rem 0 .35rem; letter-spacing: -.045em; }
      .hero p { color: #667085; font-size: 1.02rem; margin: 0; }
      .panel { background: #fff; border: 1px solid #e7eaf1; border-radius: 14px; padding: 1.15rem 1.25rem; box-shadow: 0 2px 8px rgba(16, 24, 40, .035); }
      .panel-title { color: #182230; font-weight: 700; font-size: 1rem; margin: 0 0 .2rem; }
      .panel-subtitle { color: #667085; font-size: .86rem; margin: 0 0 .95rem; }
      .status-label { color: #667085; font-size: .76rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
      .status-value { color: #172033; font-size: 1.05rem; font-weight: 700; margin-top: .15rem; }
      .stButton > button, .stFormSubmitButton > button { border-radius: 9px; font-weight: 650; min-height: 2.65rem; }
      [data-testid="stMetric"] { background: #fff; border: 1px solid #e7eaf1; border-radius: 12px; padding: .75rem; }
      [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
      [data-testid="stMain"] [data-testid="stWidgetLabel"] label,
      [data-testid="stMain"] .stTextInput label,
      [data-testid="stMain"] .stTextArea label,
      [data-testid="stMain"] .stNumberInput label { color: #172033 !important; font-weight: 650 !important; }
      [data-testid="stMain"] .stTextInput input,
      [data-testid="stMain"] .stTextArea textarea,
      [data-testid="stMain"] .stNumberInput input {
        color: #172033 !important;
        background: linear-gradient(#ffffff, #ffffff) padding-box, linear-gradient(100deg, #ec4899, #8b5cf6) border-box !important;
        border: 2px solid transparent !important;
        border-radius: 9px !important;
        box-shadow: none !important;
      }
      [data-testid="stMain"] .stTextInput input::placeholder,
      [data-testid="stMain"] .stTextArea textarea::placeholder { color: #7a8498 !important; opacity: 1; }
      [data-testid="stMain"] .stTextInput input:focus,
      [data-testid="stMain"] .stTextArea textarea:focus,
      [data-testid="stMain"] .stNumberInput input:focus { box-shadow: 0 0 0 3px rgba(168, 85, 247, .14) !important; }
      .footer-note { color: #98a2b3; font-size: .78rem; text-align: center; padding: 1.2rem 0 .3rem; }
      div[class*="st-key-add_req_btn"] button, div[class*="st-key-add_case_btn"] button,
      div[class*="st-key-add_req_btn"] button p, div[class*="st-key-add_case_btn"] button p {
        background: #111827 !important; color: #ffffff !important;
        border-color: #111827 !important;
      }
      div[class*="st-key-add_req_btn"] button:hover, div[class*="st-key-add_case_btn"] button:hover {
        background: #1f2937 !important; border-color: #1f2937 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_valid_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def start_run(inputs: dict[str, object]) -> tuple[threading.Thread, queue.Queue]:
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            run_exploration(on_event=events.put, **inputs)
        except Exception:
            # The runner emits a user-safe failure event before raising.
            pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, events


def start_pipeline_run(inputs: dict[str, object]) -> tuple[threading.Thread, queue.Queue, dict]:
    """Run the multi-agent pipeline off the UI thread; the holder carries its result."""
    events: queue.Queue = queue.Queue()
    holder: dict = {}

    def worker() -> None:
        try:
            holder["state"] = run_pipeline(on_event=events.put, **inputs)
        except Exception:
            # The pipeline emits a user-safe failure event before raising.
            pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, events, holder


def save_uploaded_docs(files) -> list[str]:
    """Persist uploaded PRD/product documents where the backend can read them."""
    docs_dir = BACKEND_ROOT / "uploads" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for uploaded in files or []:
        path = docs_dir / uploaded.name
        path.write_bytes(uploaded.getvalue())
        paths.append(str(path))
    return paths


PIPELINE_STATUS_MARKERS = (
    ("Doc Analyst", "Analyzing product documents"),
    ("App Map built", "Exploration complete — App Map ready"),
    ("Fanning out", None),  # keep the concrete fan-out message itself
    ("Test Designer produced", "Designing test cases"),
    ("Coverage Critic verdict", "Reviewing test coverage"),
    ("Healer decision", "Healing a failed test step"),
    ("Report written", "Generating the QA report"),
)


def pipeline_status_from_log(message: str) -> str | None:
    for marker, status in PIPELINE_STATUS_MARKERS:
        if marker in message:
            return status or message.split(": ", 1)[-1]
    return None


def save_uploaded_resume(asset_manager: AssetManager) -> None:
    uploaded_resume = st.session_state.get("resume_upload")
    if uploaded_resume is None:
        return
    content = uploaded_resume.getvalue()
    signature = (uploaded_resume.name, sha256(content).hexdigest())
    if st.session_state.get("stored_resume_signature") == signature:
        return
    try:
        stored_path = asset_manager.save_resume(uploaded_resume.name, content)
    except ValueError as exc:
        st.session_state["resume_error"] = str(exc)
    else:
        st.session_state["stored_resume_signature"] = signature
        st.session_state["stored_resume_name"] = stored_path.name
        st.session_state.pop("resume_error", None)


def save_uploaded_assets(asset_manager: AssetManager) -> None:
    """Store arbitrary test assets that upload test steps can reference by name."""
    import re

    for uploaded in st.session_state.get("asset_uploads") or []:
        content = uploaded.getvalue()
        signature = (uploaded.name, sha256(content).hexdigest())
        if signature in st.session_state.setdefault("stored_asset_signatures", set()):
            continue
        asset_type = re.sub(r"[^a-z0-9_]+", "_", Path(uploaded.name).stem.lower()).strip("_") or "asset"
        if not asset_type[0].isalpha():
            asset_type = f"a_{asset_type}"
        try:
            asset_manager.save_asset(asset_type, uploaded.name, content)
        except ValueError as exc:
            st.session_state["asset_error"] = str(exc)
        else:
            st.session_state["stored_asset_signatures"].add(signature)
            st.session_state.pop("asset_error", None)


def render_run_console(worker: threading.Thread, events: queue.Queue, initial_url: str, mode: str,
                       run_label: str = "Exploration") -> None:
    status = "Preparing secure browser session"
    current_url = initial_url
    logs: list[str] = []
    latest_screenshot: str | None = None
    event_count = 0

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<p class="panel-title">Live {run_label.lower()}</p>', unsafe_allow_html=True)
    st.markdown('<p class="panel-subtitle">Keep this page open while the agents work on the application.</p>', unsafe_allow_html=True)
    metric_a, metric_b, metric_c = st.columns(3)
    mode_metric = metric_a.empty()
    activity_metric = metric_b.empty()
    connection_metric = metric_c.empty()
    status_box = st.empty()
    url_box = st.empty()
    progress_box = st.empty()
    visual_col, activity_col = st.columns([1.25, 1])
    with visual_col:
        screenshot_box = st.empty()
    with activity_col:
        log_box = st.empty()

    while worker.is_alive() or not events.empty():
        try:
            event = events.get(timeout=0.25)
        except queue.Empty:
            event = None

        if event:
            event_count += 1
            status = event.get("status", status)
            current_url = event.get("url", current_url)
            if event.get("type") == "log":
                logs.append(event["message"])
                logs = logs[-80:]
                phase_status = pipeline_status_from_log(event["message"])
                if phase_status:
                    status = phase_status
            screenshot_path = event.get("screenshot_path")
            if screenshot_path and Path(screenshot_path).exists():
                latest_screenshot = screenshot_path

        mode_metric.metric("Mode", "Goal driven" if mode.startswith("Goal") else "Discovery")
        activity_metric.metric("Activity", f"{event_count} events")
        connection_metric.metric("Connection", "Running" if worker.is_alive() else "Finishing")
        status_box.markdown(f'<p class="status-label">Current activity</p><p class="status-value">{status}</p>', unsafe_allow_html=True)
        url_box.caption(f"Current page: {current_url}")
        progress_box.progress(min(0.94, 0.08 + event_count * 0.025), text=f"{run_label} is in progress")
        with visual_col:
            if latest_screenshot:
                screenshot_box.image(latest_screenshot, caption="Latest browser state", use_container_width=True)
            else:
                screenshot_box.info("A browser preview will appear after the first page observation.")
        with activity_col:
            log_box.code("\n".join(logs[-18:]) or "Waiting for activity…", language=None)

    progress_box.progress(1.0, text=f"{run_label} finished")
    if "failed" in status.casefold():
        st.error(status)
    else:
        st.success(status)
    st.markdown("</div>", unsafe_allow_html=True)


def render_pipeline_report(report: dict) -> None:
    """Render the multi-agent run's QA report with downloadable artifacts."""
    summary = report.get("summary", {})
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">QA report</p>', unsafe_allow_html=True)
    health = report.get("model_health")
    if health and health.get("failures"):
        st.warning(
            f"⚠️ The AI model was unavailable for {health['failures']} of "
            f"{health['calls']} call(s) during this run (e.g. daily rate limit reached). "
            "These results were produced by simple fallback logic, not AI test design — "
            "rerun later for a real report."
        )
    st.markdown(f'<p class="panel-subtitle">Run {report.get("run_id", "")} · generated {report.get("generated_at", "")}</p>', unsafe_allow_html=True)

    metric_columns = st.columns(6)
    metric_columns[0].metric("Tests", summary.get("total_tests", 0))
    metric_columns[1].metric("Passed", summary.get("passed", 0))
    metric_columns[2].metric("Failed", summary.get("failed", 0) + summary.get("errors", 0))
    metric_columns[3].metric("Pass rate", f'{summary.get("pass_rate", 0)}%')
    metric_columns[4].metric(
        "Requirements",
        f'{summary.get("requirements_tested", 0)}/{summary.get("requirements_total", 0)}',
    )
    metric_columns[5].metric("Blocked", summary.get("requirements_blocked", 0),
                             help="Requirements the supplied account's role cannot verify.")

    results = report.get("results", [])
    if results:
        st.markdown("**Test results**")
        st.dataframe(
            [
                {
                    "Test": result.get("title", ""),
                    "What it tests": result.get("description", ""),
                    "Requirement": result.get("requirement_id") or "—",
                    "Priority": result.get("priority", ""),
                    "Status": result.get("status", ""),
                    "Healed": "yes" if result.get("healed") else "",
                    "Duration (ms)": result.get("duration_ms", 0),
                }
                for result in results
            ],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("🧾 Test details — what each test does and why it passed or failed"):
            for result in results:
                icon = "✅" if result.get("status") == "passed" else "❌"
                st.markdown(f"**{icon} {result.get('title', '')}** · `{result.get('test_id', '')}`")
                if result.get("description"):
                    st.markdown(f"*What it tests:* {result['description']}")
                if result.get("explanation"):
                    st.markdown(f"*Outcome:* {result['explanation']}")
                st.divider()

    coverage = report.get("coverage", [])
    if coverage:
        with st.expander("Requirement coverage", expanded=False):
            st.dataframe(
                [
                    {
                        "Requirement": item.get("title", ""),
                        "Feature": item.get("feature", ""),
                        "Priority": item.get("priority", ""),
                        "Status": item.get("status", ""),
                        "Tests": item.get("test_count", 0),
                    }
                    for item in coverage
                ],
                use_container_width=True,
                hide_index=True,
            )

    if report.get("blocked"):
        with st.expander(f'🔒 Blocked requirements ({len(report["blocked"])}) — need a different role or account'):
            for entry in report["blocked"]:
                st.write(
                    f'**{entry.get("title") or entry.get("requirement_id")}** '
                    f'(`{entry.get("requirement_id")}`) — {entry.get("reason")}'
                )

    for bug in report.get("bugs", []):
        with st.expander(f'❌ {bug.get("title", "Failure")} ({bug.get("test_id", "")})'):
            if bug.get("description"):
                st.write(f'**What it tests:** {bug["description"]}')
            if bug.get("explanation"):
                st.write(f'**Why it failed:** {bug["explanation"]}')
            st.write(f'**Reason:** {bug.get("reason", "unknown")}')
            if bug.get("failed_step"):
                st.code(str(bug["failed_step"]), language=None)
            screenshot = (bug.get("evidence") or {}).get("screenshot")
            if screenshot and Path(screenshot).exists():
                st.image(screenshot, caption="Final browser state", use_container_width=True)

    artifacts = report.get("artifacts", {})
    download_columns = st.columns(3)
    for column, (label, key, mime) in zip(
        download_columns,
        (("Markdown report", "markdown", "text/markdown"),
         ("JUnit XML", "junit", "application/xml"),
         ("Raw JSON", "json", "application/json")),
    ):
        artifact_path = artifacts.get(key)
        if artifact_path and Path(artifact_path).exists():
            column.download_button(
                f"Download {label}",
                data=Path(artifact_path).read_bytes(),
                file_name=Path(artifact_path).name,
                mime=mime,
                use_container_width=True,
                key=f"download-{key}",
            )
    st.markdown("</div>", unsafe_allow_html=True)


asset_manager = AssetManager()

with st.sidebar:
    st.markdown("### Sentinel-QA")
    st.caption("Autonomous browser testing workspace")
    st.link_button("← Back to landing page", "?", use_container_width=True)
    st.divider()
    st.markdown("**How it works**")
    st.caption("1. Set a target\n\n2. Add context or a specific goal\n\n3. Review the live browser activity")
    st.divider()
    st.caption("Credentials and uploaded assets are used only for the current local exploration.")

st.markdown(
    """<div class="hero"><div class="eyebrow">Sentinel-QA · Quality assurance workspace</div>
    <h1>Explore with confidence.</h1>
    <p>Give the agent a target and intent. It will navigate the product, execute the workflow, and capture evidence as it goes.</p></div>""",
    unsafe_allow_html=True,
)

form_col, guide_col = st.columns([1.55, 1], gap="large")
with form_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">New run</p><p class="panel-subtitle">Start with a URL, then add only the context the agents need.</p>', unsafe_allow_html=True)
    run_mode = st.radio(
        "Run mode",
        ["Full QA pipeline", "Exploration only"],
        horizontal=True,
        help="The full pipeline analyses your docs, explores the app, designs and executes "
             "test cases in parallel, and produces a QA report. Exploration only runs the "
             "original discovery agent.",
    )
    pipeline_mode = run_mode == "Full QA pipeline"
    with st.form("exploration_form", clear_on_submit=False):
        website_url = st.text_input("Target URL", placeholder="https://app.example.com", help="The application entry point to explore.")
        credentials_col, steps_col = st.columns([1.5, 1])
        with credentials_col:
            with st.expander("Authentication (optional)"):
                username = st.text_input("Username", placeholder="name@example.com")
                password = st.text_input("Password", type="password")
        with steps_col:
            max_steps = st.number_input("Exploration depth", min_value=1, max_value=500, value=30, step=5, help="Maximum actions the agent may take.")
        exploration_goal = st.text_area("What should the agent accomplish?", placeholder="Example: Open Candidates, find Nicole, update the candidate name to Max, save, then log out.", height=110)
        application_context = st.text_area("Application context (optional)", placeholder="Important roles, rules, modules, or areas to prioritise and avoid.", height=80)
        if pipeline_mode:
            uploaded_docs = st.file_uploader(
                "PRD and product documents (optional)",
                type=["md", "txt", "pdf"],
                accept_multiple_files=True,
                help="Requirements are extracted from these documents and every test case "
                     "traces back to one of them.",
            )
            pipeline_col_a, pipeline_col_b = st.columns(2)
            with pipeline_col_a:
                workers = st.number_input("Parallel workers", min_value=1, max_value=8, value=3,
                                          help="Parallel budget for document analysis, test design, and test execution.")
            with pipeline_col_b:
                skip_exploration = st.checkbox(
                    "Skip exploration (design from documents only)",
                    value=False,
                    help="Faster, but test cases lose grounding in the observed UI.",
                )
                preserve_session = st.checkbox(
                    "Reuse login session across tests",
                    value=False,
                    help="Logs in once, saves the session, and skips the login steps in every "
                         "test that uses the real credentials (~15–20s faster per test). "
                         "Invalid-login tests still run fresh and logged out.",
                )
        submitted = st.form_submit_button(
            "Start QA pipeline" if pipeline_mode else "Start exploration",
            type="primary",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with guide_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Helpful context</p><p class="panel-subtitle">Specific goals produce more focused explorations.</p>', unsafe_allow_html=True)
    st.markdown("**Use action-oriented language**\n\n“Open Candidates, search for Nicole, edit the record, save.”\n\n**Mention constraints**\n\n“Use the admin role; avoid deleting data.”")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Your requirements & test cases</p><p class="panel-subtitle">Add your own alongside what the agent derives — included in every run.</p>', unsafe_allow_html=True)
    # Injected adjacent to the buttons: covers Streamlit builds without
    # st-key-* wrapper classes by targeting the two adjacent columns directly.
    st.markdown(
        """
        <style>
          div[class*="st-key-add_req_btn"] button, div[class*="st-key-add_case_btn"] button {
            background: #111827 !important; color: #ffffff !important;
            border-color: #111827 !important;
          }
          div[class*="st-key-add_req_btn"] button *, div[class*="st-key-add_case_btn"] button * {
            color: #ffffff !important;
          }
          [data-testid="stHorizontalBlock"]:has(div[class*="add_req_btn"]) button {
            background: #111827 !important; color: #ffffff !important;
            border-color: #111827 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    add_req_col, add_case_col = st.columns(2)
    if add_req_col.button("＋ Add requirement", key="add_req_btn", use_container_width=True):
        st.session_state["show_add_req"] = not st.session_state.get("show_add_req", False)
    if add_case_col.button("＋ Add test case", key="add_case_btn", use_container_width=True):
        st.session_state["show_add_case"] = not st.session_state.get("show_add_case", False)

    if st.session_state.get("show_add_req"):
        with st.form("add_requirement_form", clear_on_submit=True):
            req_title = st.text_input("Requirement", placeholder="Candidates can be created from a resume upload")
            req_feature = st.text_input("Feature/module", placeholder="candidates")
            req_description = st.text_area("Details (optional)", height=70)
            req_priority = st.selectbox("Priority", ["critical", "high", "medium", "low"], index=1)
            if st.form_submit_button("Save requirement", use_container_width=True):
                if req_title.strip():
                    saved = add_custom_requirement(req_feature, req_title, req_description, req_priority)
                    st.success(f"Saved as {saved['id']} — tests will be designed for it on every run.")
                else:
                    st.error("The requirement needs a title.")

    if st.session_state.get("show_add_case"):
        with st.form("add_test_case_form", clear_on_submit=True):
            case_title = st.text_input("Test case title", placeholder="Create a job from the Upload JD card")
            case_description = st.text_area("What it tests (optional)", height=60)
            case_steps = st.text_area(
                "Steps — one per line",
                placeholder="navigate: /login\nfill: email = {username}\nfill: password = {password}\nclick: Log In\nclick: Jobs\nupload: Choose Files = resume",
                height=140,
            )
            case_expected = st.text_area(
                "Checks — one per line",
                placeholder="url: /jobs\ntext: Job created",
                height=80,
            )
            if st.form_submit_button("Save test case", use_container_width=True):
                result = add_custom_test_case(case_title, case_description, case_steps, case_expected)
                if isinstance(result, str):
                    st.error(result)
                else:
                    st.success(f"Saved as {result['id']} — it will execute in every run.")

    custom_requirements = load_custom_requirements()
    custom_cases = load_custom_test_cases()
    if custom_requirements or custom_cases:
        st.caption(f"Stored: {len(custom_requirements)} requirement(s), {len(custom_cases)} test case(s)")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Agent memory</p><p class="panel-subtitle">Corrections the agent applies to every future run — e.g. “the Offline status is called Unavailable in the UI”, “credentials are a Provider account”.</p>', unsafe_allow_html=True)
    feedback_text = st.text_area(
        "Remembered feedback",
        value=load_feedback(),
        height=110,
        label_visibility="collapsed",
        placeholder="One correction per line. The agents read these before every run.",
    )
    if st.button("Save memory", use_container_width=True):
        save_feedback(feedback_text)
        st.success("Saved — applied to every run from now on.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Test assets</p><p class="panel-subtitle">Files the agent may upload during tests (resumes, sample PDFs, images). Test steps reference them by name.</p>', unsafe_allow_html=True)
    st.file_uploader("Assets", type=["pdf", "doc", "docx", "png", "jpg", "csv", "txt"],
                     key="asset_uploads", accept_multiple_files=True,
                     label_visibility="collapsed", on_change=save_uploaded_assets,
                     args=(asset_manager,))
    if st.session_state.get("asset_error"):
        st.error(st.session_state["asset_error"])
    stored_assets = asset_manager.list_assets()
    if stored_assets:
        st.caption("Stored: " + ", ".join(f"`{name}`" for name in stored_assets))
    else:
        st.caption("No assets stored yet — upload flows will be skipped until you add one.")
    st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    if not is_valid_url(website_url):
        st.error("Enter a complete http:// or https:// URL to begin.")
    elif pipeline_mode:
        doc_paths = save_uploaded_docs(uploaded_docs)
        pipeline_inputs = {
            "start_url": website_url.strip(),
            "docs": ",".join(doc_paths) if doc_paths else None,
            "username": username,
            "password": password,
            "goal": exploration_goal,
            "application_context": application_context,
            "max_steps": int(max_steps),
            "skip_exploration": bool(skip_exploration),
            "preserve_session": bool(preserve_session),
            "max_concurrency": int(workers),
        }
        st.write("")
        worker, events, holder = start_pipeline_run(pipeline_inputs)
        render_run_console(worker, events, website_url.strip(), "Goal Driven Exploration",
                           run_label="QA pipeline")
        report = (holder.get("state") or {}).get("report")
        if report:
            st.session_state["pipeline_report"] = report
        else:
            st.session_state.pop("pipeline_report", None)
            st.warning("The pipeline did not produce a report. Check the activity log above for the failure.")
    else:
        run_inputs = {
            "start_url": website_url.strip(),
            "username": username,
            "password": password,
            "application_context": application_context,
            "exploration_goal": exploration_goal,
            "max_steps": int(max_steps),
        }
        mode = "Goal Driven Exploration" if exploration_goal.strip() else "Autonomous Discovery"
        st.write("")
        worker, events = start_run(run_inputs)
        render_run_console(worker, events, website_url.strip(), mode)

if st.session_state.get("pipeline_report"):
    st.write("")
    render_pipeline_report(st.session_state["pipeline_report"])

st.markdown('<p class="footer-note">Sentinel-QA · Local browser exploration</p>', unsafe_allow_html=True)
