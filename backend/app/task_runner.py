"""UI-facing entrypoint for a single browser-automation task run.

Mirrors ``runner.run_exploration``'s contract (events, guidance channel, log
forwarding) but drives the TaskAgent instead of the QA pipeline.  The browser
is a persistent, headed, real-Chrome profile: logins and OTP verifications
survive between runs, and consumer sites' bot detection sees an ordinary
browser instead of a fresh headless Chromium.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from browser import BrowserController
from logging_config import configure_logging
from observability.langfuse_client import create_observability
from runner import EventLogHandler
from task_agent import TaskAgent
from task_planner import GeminiTaskPlanner


EventCallback = Callable[[dict[str, Any]], None]
DEFAULT_PROFILE_DIR = Path(__file__).resolve().parent.parent / "runtime" / "task_profile"


def run_task(
    *,
    start_url: str,
    task: str,
    max_steps: int = 40,
    on_event: EventCallback | None = None,
    guidance=None,
    hitl_wait_seconds: int = 180,
    headless: bool = False,
    profile_dir: str | None = None,
) -> dict[str, Any]:
    """Run one browser task and report progress through ``on_event``."""
    callback = on_event or (lambda event: None)

    configure_logging()
    handler = EventLogHandler(callback)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)

    browser = BrowserController()
    observability = create_observability()

    callback({"type": "started", "status": "Starting browser", "mode": "Browser task", "url": start_url})
    try:
        planner = GeminiTaskPlanner(observability=observability)
        if not planner.available:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured — the task agent needs Gemini for planning"
            )
        browser.start_persistent(
            profile_dir or DEFAULT_PROFILE_DIR,
            observability=observability,
            headless=headless,
        )
        # Consumer flows legitimately cross domains (auth, maps, CDNs); the
        # payment gate and the human channel are the safety rails here, not
        # domain pinning.  Popups still close so focus stays on one page.
        browser.set_new_tab_policy(False)
        browser.set_navigation_policy(start_url, follow_external=True)
        browser.open(start_url)

        agent = TaskAgent(
            browser,
            planner,
            goal=task,
            max_steps=max_steps,
            guidance=guidance,
            on_event=callback,
            ask_timeout_s=max(30, hitl_wait_seconds),
        )
        result = agent.run()
        callback({
            "type": "completed",
            "status": f"Task {result['status']}: {result['summary']}",
            "url": browser.current_url() if browser.page else start_url,
            "mode": "Browser task",
        })
        return result
    except Exception as exc:
        logging.getLogger(__name__).exception("Task run stopped because of an unrecoverable error")
        callback({
            "type": "error",
            "status": f"Task failed: {exc}",
            "url": browser.current_url() if browser.page else start_url,
        })
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        try:
            browser.close()
        finally:
            observability.flush()
