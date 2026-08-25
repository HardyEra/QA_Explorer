"""UI-facing entrypoint for a single browser-automation task run.

Drives the LangGraph TaskAgent inside one run-level Langfuse trace: every
observation span, planner generation, Vision Analyst call, tool span, and
event of the run nests under it.  The UI still receives progress through
``on_event``; diagnostics live in Langfuse, not in code logs.

The browser is a persistent, headed, real-Chrome profile: logins and OTP
verifications survive between runs, and consumer sites' bot detection sees an
ordinary browser instead of a fresh headless Chromium.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import config  # noqa: F401  — loads backend/.env (Langfuse + Gemini keys)
from browser import BrowserController
from observability.langfuse_client import create_observability
from observability.tracing import TraceMetadata
from task_agent import TaskAgent
from task_planner import create_task_planner


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
    browser = BrowserController()
    observability = create_observability()
    run_id = f"task-{uuid4().hex[:12]}"

    callback({"type": "started", "status": "Starting browser", "mode": "Browser task", "url": start_url})
    try:
        with observability.trace(
            "Browser Task Run",
            metadata=TraceMetadata(
                project_name="qa-explorer",
                session_id=run_id,
                exploration_id=run_id,
                current_page=start_url,
                current_goal=task,
            ),
            input={"start_url": start_url, "task": task, "max_steps": max_steps},
        ) as trace:
            try:
                planner = create_task_planner(observability=observability)
                if not planner.available:
                    raise RuntimeError(
                        "Neither AZURE_OPENAI_API_KEY nor GEMINI_API_KEY is configured — "
                        "the task agent needs a planning model"
                    )
                browser.start_persistent(
                    profile_dir or DEFAULT_PROFILE_DIR,
                    observability=observability,
                    headless=headless,
                )
                # Consumer flows legitimately cross domains (auth, maps, CDNs);
                # the payment gate and the human channel are the safety rails
                # here, not domain pinning.  Popups still close so focus stays
                # on one page.
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
                    observability=observability,
                )
                result = agent.run()
                trace.update(output={
                    "status": result["status"],
                    "summary": result["summary"],
                    "steps": result["steps"],
                    "run_id": run_id,
                })
            except Exception as exc:
                observability.record_exception(
                    exc, context={"active_action": "task_runner",
                                  "current_url": browser.current_url() if browser.page else start_url}
                )
                raise
        callback({
            "type": "completed",
            "status": f"Task {result['status']}: {result['summary']}",
            "url": browser.current_url() if browser.page else start_url,
            "mode": "Browser task",
        })
        return result
    except Exception as exc:
        callback({
            "type": "error",
            "status": f"Task failed: {exc}",
            "url": browser.current_url() if browser.page else start_url,
        })
        raise
    finally:
        try:
            browser.close()
        finally:
            observability.flush()
