"""UI-facing entrypoint for a single QA exploration run.

This module owns orchestration only.  Clients pass a callback to receive small,
serialisable progress events while the existing explorer continues to own the
planning and browser work.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from browser import BrowserController
from config import ExplorationConfig
from explorer import Explorer
from graphs.generation_graph import GenerationGraph
from graphs.state import QAState
from logging_config import configure_logging
from observability.langfuse_client import create_observability


EventCallback = Callable[[dict[str, Any]], None]
SCREENSHOT_PATH = Path(__file__).resolve().parent.parent / "runtime" / "latest.png"


class EventLogHandler(logging.Handler):
    """Forward backend log records without coupling the backend to Streamlit."""

    def __init__(self, callback: EventCallback):
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.callback({"type": "log", "message": self.format(record)})
        except Exception:
            # Logging must never interrupt an exploration run.
            pass


def _discovery_history(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Read Discovery history without coupling orchestration to its graph.

    The Discovery graph owns how it records history.  Supporting both names
    allows the Generation Graph to consume current and future Discovery output
    without altering Discovery Agent internals.
    """
    for key in ("exploration_history", "execution_history", "history"):
        history = result.get(key)
        if isinstance(history, list):
            return [event for event in history if isinstance(event, dict)]
    return []


def run_exploration(
    *,
    start_url: str,
    username: str = "",
    password: str = "",
    application_context: str = "",
    exploration_goal: str = "",
    max_steps: int = 30,
    on_event: EventCallback | None = None,
) -> dict[str, Any]:
    """Run the existing explorer and report progress through ``on_event``."""
    callback = on_event or (lambda event: None)
    mode = "Goal Driven Exploration" if exploration_goal.strip() else "Autonomous Discovery"
    goal = exploration_goal.strip() or "Autonomously discover and execute the application's primary business-critical workflow."

    configure_logging()
    handler = EventLogHandler(callback)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)

    browser = BrowserController()
    observability = create_observability()

    def explorer_event(event: dict[str, Any]) -> None:
        if event.get("type") == "observation":
            try:
                SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
                browser.screenshot(str(SCREENSHOT_PATH))
                event = {**event, "screenshot_path": str(SCREENSHOT_PATH)}
            except Exception as exc:
                logging.getLogger(__name__).debug("Could not capture screenshot: %s", exc)
        callback(event)

    callback({"type": "started", "status": "Starting browser", "mode": mode, "url": start_url})
    try:
        browser.start(observability=observability)
        browser.open(start_url)
        config = ExplorationConfig(
            start_url=start_url,
            follow_external=False,
            max_steps=max_steps,
            current_goal=goal,
            application_context=application_context.strip(),
            username=username,
            password=password,
        )
        explorer = Explorer(browser, config, observability, on_event=explorer_event)
        result = explorer.explore()
        generation_state: QAState = {
            "task": goal,
            "start_url": start_url,
            "discovery_output": result,
            "exploration_history": _discovery_history(result),
            "flow": None,
            "generated_script": None,
            "readme": None,
        }
        generation_result = GenerationGraph().invoke(generation_state)
        result = {**result, "flow": generation_result["flow"]}
        callback({"type": "completed", "status": "Exploration complete", "url": browser.current_url(), "mode": mode})
        return result
    except Exception as exc:
        logging.getLogger(__name__).exception("Explorer stopped because of an unrecoverable error")
        callback({"type": "error", "status": f"Exploration failed: {exc}", "url": browser.current_url() if browser.page else start_url})
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        browser.close()
