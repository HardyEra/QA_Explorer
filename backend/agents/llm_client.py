"""Shared JSON-mode model client for the orchestrated QA agents.

Every LLM-backed agent (Doc Analyst, Test Designer, Coverage Critic, Healer)
funnels its calls through this client so provider handling, JSON parsing,
timeouts, and observability stay in one place.  A missing key or a failed
request returns ``None``; each agent owns its deterministic fallback.
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

from observability.tracing import NoopObservability


load_dotenv()
logger = logging.getLogger(__name__)

MODEL_TIMEOUT_MS = int(os.getenv("MODEL_TIMEOUT_MS", "20000"))
AGENT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
AGENT_TEMPERATURE = 0
RETRY_DELAYS_S = (2, 5)

# Run-level model health so the pipeline can tell the user when results were
# produced by deterministic fallbacks rather than the AI.
_HEALTH = {"calls": 0, "failures": 0, "last_error": ""}


def reset_model_health() -> None:
    _HEALTH.update(calls=0, failures=0, last_error="")


def model_health() -> dict:
    return dict(_HEALTH)


def _record_failure(exc_text: str) -> None:
    _HEALTH["failures"] += 1
    _HEALTH["last_error"] = exc_text[:300]


def _is_retryable(error_text: str) -> bool:
    """Retry short rate-limit bursts; a spent daily quota will not recover."""
    lowered = error_text.casefold()
    if "per day" in lowered or "tpd" in lowered:
        return False
    return "429" in lowered or "rate_limit" in lowered or "503" in lowered or "502" in lowered


def _usage_details(response):
    """Normalize Groq usage metadata to provider-neutral fields."""
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    details = {}
    if getattr(usage, "prompt_tokens", None) is not None:
        details["input_tokens"] = usage.prompt_tokens
    if getattr(usage, "completion_tokens", None) is not None:
        details["output_tokens"] = usage.completion_tokens
    return details or None


class JsonModelClient:
    """Call the shared planning model and return one parsed JSON object."""

    def __init__(self, observability=None, model: str = AGENT_MODEL):
        self.observability = observability or NoopObservability()
        self.model = model
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY is not configured; agents will use deterministic fallbacks")
            self.client = None
        else:
            from groq import Groq

            self.client = Groq(api_key=api_key, timeout=MODEL_TIMEOUT_MS / 1000)

    @property
    def available(self) -> bool:
        return self.client is not None

    def complete_json(self, name: str, prompt: str) -> dict | None:
        """Return the model's JSON object, or ``None`` on any failure."""
        if self.client is None:
            return None

        _HEALTH["calls"] += 1
        with self.observability.generation(
            name,
            model=self.model,
            temperature=AGENT_TEMPERATURE,
            input=prompt,
            metadata={"provider": "groq"},
        ) as generation:
            response = None
            for attempt in range(len(RETRY_DELAYS_S) + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=AGENT_TEMPERATURE,
                        response_format={"type": "json_object"},
                    )
                    break
                except Exception as exc:
                    if attempt < len(RETRY_DELAYS_S) and _is_retryable(str(exc)):
                        delay = RETRY_DELAYS_S[attempt]
                        logger.info("%s: transient model error, retrying in %ss", name, delay)
                        import time

                        time.sleep(delay)
                        continue
                    self.observability.record_exception(
                        exc, input=prompt, context={"active_action": name}
                    )
                    _record_failure(str(exc))
                    logger.warning("%s: model request failed (%s); using fallback", name, exc)
                    return None

            content = (response.choices[0].message.content or "").strip()
            generation.update(output=content, usage_details=_usage_details(response))

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            self.observability.record_exception(
                exc, input=prompt, output=content, context={"active_action": name}
            )
            _record_failure(f"invalid JSON from model in {name}")
            logger.warning("%s: model returned invalid JSON; using fallback", name)
            return None

        if not isinstance(payload, dict):
            _record_failure(f"non-object JSON from model in {name}")
            logger.warning("%s: model returned a non-object JSON value; using fallback", name)
            return None
        return payload
