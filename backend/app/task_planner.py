"""Next-action planners for the browser task agent.

One call per step: the agent supplies a text description of the page (and,
only when the Vision Analyst decided the DOM is not enough, a viewport
screenshot), and the planner returns exactly one next action as validated
JSON.  A failed or unparseable response returns ``None``; the agent owns what
happens then.

Two interchangeable backends share the decision contract: Azure OpenAI
(preferred when ``AZURE_OPENAI_API_KEY`` is configured) and Gemini as the
fallback.  ``create_task_planner`` picks one; the Vision Analyst stays on
Gemini either way.

Diagnostics go to Langfuse only: each call is a "TaskPlanner" generation, and
anomalies (missing key, invalid decisions, request failures) are events.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from observability.tracing import NoopObservability


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


def _parse_decision(raw: str, observability) -> TaskDecision | None:
    """Validate one raw JSON decision against the shared contract."""
    try:
        payload = _DecisionPayload.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, ValueError):
        return None
    if payload.action_type not in ACTION_TYPES:
        observability.event(
            name="planner_unknown_action",
            metadata={"action_type": payload.action_type}, level="WARNING",
        )
        return None
    return TaskDecision(**payload.model_dump())


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
            self.observability.event(
                name="planner_unavailable",
                metadata={"reason": "GEMINI_API_KEY is not configured"}, level="ERROR",
            )
            return None
        try:
            client, types = self._create_client()
        except RuntimeError as exc:
            self.observability.event(
                name="planner_unavailable",
                metadata={"reason": str(exc)}, level="ERROR",
            )
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
                self.observability.event(
                    name="planner_invalid_decision",
                    metadata={"raw": raw[:500], "action": "retrying once with a repair prompt"},
                    level="WARNING",
                )
                raw, retry_usage = self._generate(
                    client, types, prompt + "\n\n" + REPAIR_PROMPT, screenshot_jpeg
                )
                usage = self._add_usage(usage, retry_usage)
                decision = self._parse(raw) if raw is not None else None
            generation.update(
                output={"decision": decision.__dict__ if decision else None},
                usage_details=usage or None,
                status_message=None if decision else "no valid decision after repair retry",
            )
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
            self.observability.record_exception(exc, context={"active_action": "TaskPlanner"})
            return None, {}
        usage = self._token_usage(getattr(response, "usage_metadata", None))
        return response.text or "", usage

    def _parse(self, raw: str) -> TaskDecision | None:
        return _parse_decision(raw, self.observability)

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


class AzureTaskPlanner:
    """Decide the single next browser action with an Azure OpenAI deployment.

    Talks to the Responses API of one deployed model.  The endpoint and
    deployment default to the project's known deployment so only
    ``AZURE_OPENAI_API_KEY`` has to live in ``.env``.
    """

    def __init__(self, api_key: str | None = None, observability=None):
        self.api_key = api_key if api_key is not None else os.getenv("AZURE_OPENAI_API_KEY", "")
        self.endpoint = os.getenv(
            "AZURE_OPENAI_ENDPOINT", "https://abhishek-bahukhandi.openai.azure.com"
        ).rstrip("/")
        self.deployment = os.getenv("AZURE_OPENAI_TASK_DEPLOYMENT", "gpt-5.6-sol")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        self.observability = observability or NoopObservability()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def decide(self, prompt: str, screenshot_jpeg: bytes | None = None) -> TaskDecision | None:
        """Return the validated next action, or ``None`` after retries fail."""
        if not self.api_key:
            self.observability.event(
                name="planner_unavailable",
                metadata={"reason": "AZURE_OPENAI_API_KEY is not configured"}, level="ERROR",
            )
            return None
        try:
            client = self._create_client()
        except RuntimeError as exc:
            self.observability.event(
                name="planner_unavailable", metadata={"reason": str(exc)}, level="ERROR",
            )
            return None

        with self.observability.generation(
            "TaskPlanner",
            model=self.deployment,
            temperature=None,
            input={"prompt": prompt, "screenshot_attached": screenshot_jpeg is not None},
            metadata={"component": "task_planner", "provider": "azure_openai"},
        ) as generation:
            raw, usage = self._generate(client, prompt, screenshot_jpeg)
            if raw is None:
                return None
            decision = _parse_decision(raw, self.observability)
            if decision is None:
                self.observability.event(
                    name="planner_invalid_decision",
                    metadata={"raw": raw[:500], "action": "retrying once with a repair prompt"},
                    level="WARNING",
                )
                raw, retry_usage = self._generate(
                    client, prompt + "\n\n" + REPAIR_PROMPT, screenshot_jpeg
                )
                usage = GeminiTaskPlanner._add_usage(usage, retry_usage)
                decision = _parse_decision(raw, self.observability) if raw is not None else None
            generation.update(
                output={"decision": decision.__dict__ if decision else None},
                usage_details=usage or None,
                status_message=None if decision else "no valid decision after repair retry",
            )
        return decision

    def _create_client(self):
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "the openai package is required for the Azure task planner"
            ) from exc
        return AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            timeout=60.0,
        )

    def _generate(self, client, prompt: str,
                  screenshot_jpeg: bytes | None) -> tuple[str | None, dict[str, int]]:
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        if screenshot_jpeg:
            encoded = base64.b64encode(screenshot_jpeg).decode("ascii")
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encoded}",
            })
        try:
            response = client.responses.create(
                model=self.deployment,
                input=[{"role": "user", "content": content}],
                text={"format": {
                    "type": "json_schema",
                    "name": "task_decision",
                    "schema": _DecisionPayload.model_json_schema(),
                    "strict": True,
                }},
            )
        except Exception as exc:
            self.observability.record_exception(exc, context={"active_action": "TaskPlanner"})
            return None, {}
        return response.output_text or "", self._token_usage(getattr(response, "usage", None))

    @staticmethod
    def _token_usage(usage) -> dict[str, int]:
        if usage is None:
            return {}
        return {
            key: value
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance((value := getattr(usage, key, None)), int)
        }


def create_task_planner(observability=None):
    """Azure OpenAI when its key is configured; Gemini otherwise."""
    azure = AzureTaskPlanner(observability=observability)
    if azure.available:
        return azure
    return GeminiTaskPlanner(observability=observability)
