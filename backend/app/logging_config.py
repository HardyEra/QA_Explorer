"""Shared logging setup for the QA explorer."""

import logging
from pathlib import Path


def configure_logging() -> None:
    """Write application events to a dedicated file and the console.

    The dedicated ``backend/logs/agent.log`` avoids colliding with a caller that
    redirects console output to ``backend/agent.log`` on Windows.
    """
    log_path = Path(__file__).resolve().parent.parent / "logs" / "agent.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
