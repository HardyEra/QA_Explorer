"""Live human-in-the-loop channel between the frontend chat and the agents.

The run executes in a background thread of the same process as the UI, so the
channel is a plain thread-safe object: the tester pushes guidance from the
chat at any moment; the explorer drains it before every planning step; and
when the agent is stuck it can ask a question and block (bounded) for a reply.
"""

from __future__ import annotations

import logging
import queue
import threading


logger = logging.getLogger(__name__)

DEFAULT_ASK_TIMEOUT_S = 120


class GuidanceChannel:
    """Two-way, thread-safe guidance stream for one run."""

    def __init__(self) -> None:
        self._messages: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        # [(who, text)] — "tester" or "agent"; rendered by the frontend chat.
        self.transcript: list[tuple[str, str]] = []
        self.pending_question: str | None = None

    def send(self, text: str) -> None:
        """Called by the UI when the tester submits a chat message."""
        text = str(text or "").strip()
        if not text:
            return
        with self._lock:
            self.transcript.append(("tester", text))
        self._messages.put(text)
        logger.info("Tester guidance received: %s", text)

    def drain(self) -> list[str]:
        """Non-blocking: everything the tester sent since the last check."""
        messages = []
        while True:
            try:
                messages.append(self._messages.get_nowait())
            except queue.Empty:
                return messages

    def ask(self, question: str, timeout_s: int = DEFAULT_ASK_TIMEOUT_S) -> str | None:
        """Ask the tester and wait (bounded) for an answer; ``None`` on timeout."""
        with self._lock:
            self.transcript.append(("agent", question))
            self.pending_question = question
        logger.info("Agent question for the tester: %s", question)
        try:
            return self._messages.get(timeout=max(1, timeout_s))
        except queue.Empty:
            logger.info("No tester reply within %ss; continuing autonomously", timeout_s)
            return None
        finally:
            with self._lock:
                self.pending_question = None
