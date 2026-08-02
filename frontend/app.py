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


def render_run_console(worker: threading.Thread, events: queue.Queue, initial_url: str, mode: str) -> None:
    status = "Preparing secure browser session"
    current_url = initial_url
    logs: list[str] = []
    latest_screenshot: str | None = None
    event_count = 0

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Live exploration</p>', unsafe_allow_html=True)
    st.markdown('<p class="panel-subtitle">Keep this page open while the agent explores the application.</p>', unsafe_allow_html=True)
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
            screenshot_path = event.get("screenshot_path")
            if screenshot_path and Path(screenshot_path).exists():
                latest_screenshot = screenshot_path

        mode_metric.metric("Mode", "Goal driven" if mode.startswith("Goal") else "Discovery")
        activity_metric.metric("Activity", f"{event_count} events")
        connection_metric.metric("Connection", "Running" if worker.is_alive() else "Finishing")
        status_box.markdown(f'<p class="status-label">Current activity</p><p class="status-value">{status}</p>', unsafe_allow_html=True)
        url_box.caption(f"Current page: {current_url}")
        progress_box.progress(min(0.94, 0.08 + event_count * 0.025), text="Exploration is in progress")
        with visual_col:
            if latest_screenshot:
                screenshot_box.image(latest_screenshot, caption="Latest browser state", use_container_width=True)
            else:
                screenshot_box.info("A browser preview will appear after the first page observation.")
        with activity_col:
            log_box.code("\n".join(logs[-18:]) or "Waiting for activity…", language=None)

    progress_box.progress(1.0, text="Exploration finished")
    if status.startswith("Exploration failed"):
        st.error(status)
    else:
        st.success(status)
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
    st.markdown('<p class="panel-title">New exploration</p><p class="panel-subtitle">Start with a URL, then add only the context the agent needs.</p>', unsafe_allow_html=True)
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
        submitted = st.form_submit_button("Start exploration", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with guide_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Helpful context</p><p class="panel-subtitle">Specific goals produce more focused explorations.</p>', unsafe_allow_html=True)
    st.markdown("**Use action-oriented language**\n\n“Open Candidates, search for Nicole, edit the record, save.”\n\n**Mention constraints**\n\n“Use the admin role; avoid deleting data.”")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Test asset</p><p class="panel-subtitle">Upload a resume for file-upload workflows.</p>', unsafe_allow_html=True)
    st.file_uploader("Resume", type=["pdf", "doc", "docx"], key="resume_upload", label_visibility="collapsed", on_change=save_uploaded_resume, args=(asset_manager,))
    if st.session_state.get("resume_error"):
        st.error(st.session_state["resume_error"])
    elif st.session_state.get("stored_resume_name"):
        st.success(f"Ready: {st.session_state['stored_resume_name']}")
    else:
        stored_resume = asset_manager.resolve_resume_path()
        st.caption(f"Stored asset: {stored_resume.name}" if stored_resume else "No asset selected")
    st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    if not is_valid_url(website_url):
        st.error("Enter a complete http:// or https:// URL to begin.")
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

st.markdown('<p class="footer-note">Sentinel-QA · Local browser exploration</p>', unsafe_allow_html=True)
