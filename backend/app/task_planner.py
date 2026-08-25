"""Gemini-backed next-action planner for the browser task agent.

One call per step: the agent supplies a text description of the page (and,
only when it decides the DOM is not enough, a viewport screenshot), and the
planner returns exactly one next action as validated JSON.  A failed or
unparseable response returns ``None``; the agent owns what happens then.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from time import perf_counter

from pydantic import BaseModel, ConfigDict, ValidationError

from observability.tracing import NoopObservability


logger = logging.getLogger(__name__)

ACTION_TYPES = frozenset({
    "click", "click_text", "fill", "enter_otp", "click_at", "press_key",
    "scroll", "navigate", "ask_human", "wait", "look", "done", "fail",
})

REPAIR_PROMPT = """Your previous response could not be parsed or did not match the
required schema. Return a replacement decision now: one JSON object conforming
exactly to the response schema, with action_type set to one of: click, click_text,
fill, enter_otp, click_at, press_key, scroll, navigate, ask_human, wait, look,
done, fail. No markdown or commentary."""


class _DecisionPayload(BaseModel):
    """Strict boundary model used to validate Gemini's JSON decision.

    Every field is required and scalar so the schema stays inside Gemini's
    structured-output support: unused fields carry -1 or the empty string.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    reasoning: str
    action_type: str
    action_id: int
    target: str
    value: str
    x: int
    y: int


@dataclass(frozen=True)
class TaskDecision:
    """One planned browser action."""

    reasoning: str
    action_type: str
    action_id: int = -1
    target: str = ""
    value: str = ""
    x: int = -1
    y: int = -1

    def describe(self) -> str:
        if self.action_type == "click":
            return f"click action {self.action_id}"
        if self.action_type == "click_text":
            return f"click text '{self.value or self.target}'"
        if self.action_type == "enter_otp":
            return "enter OTP code"
        if self.action_type == "fill":
            return f"fill '{self.target}'"
        if self.action_type == "click_at":
            return f"click at ({self.x}, {self.y})"
        if self.action_type in ("press_key", "scroll", "navigate"):
            return f"{self.action_type} {self.value}"
        if self.action_type == "ask_human":
            return f"ask human: {self.value[:80]}"
        return self.action_type


class GeminiTaskPlanner:
    """Decide the single next browser action with Gemini."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 observability=None):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        # Follows the vision observer's migration default; override with
        # GEMINI_TASK_MODEL when a stronger planning model is warranted.
        self.model = model or os.getenv("GEMINI_TASK_MODEL", "gemini-3.6-flash")
        self.observability = observability or NoopObservability()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def decide(self, prompt: str, screenshot_jpeg: bytes | None = None) -> TaskDecision | None:
        """Return the validated next action, or ``None`` after retries fail."""
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not configured; task planner unavailable")
            return None
        try:
            client, types = self._create_client()
        except RuntimeError as exc:
            logger.error("Task planner unavailable: %s", exc)
            return None

        with self.observability.generation(
            "TaskPlanner",
            model=self.model,
            temperature=None,
            input={"prompt": prompt, "screenshot_attached": screenshot_jpeg is not None},
            metadata={"component": "task_planner"},
        ) as generation:
            raw, usage = self._generate(client, types, prompt, screenshot_jpeg)
            if raw is None:
                return None
            decision = self._parse(raw)
            if decision is None:
                logger.warning("Task planner returned an invalid decision; retrying once")
                raw, retry_usage = self._generate(
                    client, types, prompt + "\n\n" + REPAIR_PROMPT, screenshot_jpeg
                )
                usage = self._add_usage(usage, retry_usage)
                decision = self._parse(raw) if raw is not None else None
            generation.update(
                output={"decision": decision.__dict__ if decision else None},
                usage_details=usage or None,
            )
        if decision is None:
            logger.error("Task planner could not produce a valid decision")
        return decision

    def _create_client(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for the task planner; install backend requirements"
            ) from exc
        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=60_000),
        )
        return client, types

    def _generate(self, client, types, prompt: str,
                  screenshot_jpeg: bytes | None) -> tuple[str | None, dict[str, int]]:
        contents: list = [prompt]
        if screenshot_jpeg:
            contents.append(types.Part.from_bytes(data=screenshot_jpeg, mime_type="image/jpeg"))
        started_at = perf_counter()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    # Same rationale as the vision observer: the dedicated
                    # JSON-schema field keeps ``additionalProperties`` in the
                    # spelling Gemini's REST surface accepts.
                    response_json_schema=_DecisionPayload.model_json_schema(),
                ),
            )
        except Exception as exc:
            logger.error("Task planner request failed: %s", exc)
            self.observability.record_exception(exc, context={"active_action": "TaskPlanner"})
            return None, {}
        finally:
            logger.info("Task planner latency: %.0f ms", (perf_counter() - started_at) * 1000)
        usage = self._token_usage(getattr(response, "usage_metadata", None))
        return response.text or "", usage

    @staticmethod
    def _parse(raw: str) -> TaskDecision | None:
        try:
            payload = _DecisionPayload.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning("Task planner JSON validation failed: %s", exc)
            return None
        if payload.action_type not in ACTION_TYPES:
            logger.warning("Task planner chose unknown action_type: %s", payload.action_type)
            return None
        return TaskDecision(**payload.model_dump())

    @staticmethod
    def _token_usage(usage) -> dict[str, int]:
        if usage is None:
            return {}
        mapping = {
            "prompt_token_count": "input_tokens",
            "candidates_token_count": "output_tokens",
            "total_token_count": "total_tokens",
        }
        return {
            target: value
            for source, target in mapping.items()
            if isinstance((value := getattr(usage, source, None)), int)
        }

    @staticmethod
    def _add_usage(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
        return {key: first.get(key, 0) + second.get(key, 0) for key in first.keys() | second.keys()}
