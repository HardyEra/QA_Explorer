"""Shared JSON-mode model client for the orchestrated QA agents.

Every LLM-backed agent (Doc Analyst, Test Designer, Coverage Critic, Healer)
funnels its calls through this client so provider handling, JSON parsing,
timeouts, and observability stay in one place.

Providers form a fallback chain: Azure OpenAI (when ``AZURE_OPENAI_API_KEY``
is configured) is tried first, Groq second.  A call moves down the chain on
any failure — request error or unusable JSON — and only when every provider
fails does it return ``None``; each agent owns its deterministic fallback
after that.
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
AGENT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
AGENT_TEMPERATURE = 0
RETRY_DELAYS_S = (2, 5)

AZURE_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT", "https://abhishek-bahukhandi.openai.azure.com"
)
AZURE_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_QA_DEPLOYMENT",
    os.getenv("AZURE_OPENAI_TASK_DEPLOYMENT", "gpt-5.6-sol"),
)
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

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
    """Normalize chat-completion usage metadata to provider-neutral fields."""
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
    """Call the shared planning model chain and return one parsed JSON object."""

    def __init__(self, observability=None, model: str = AGENT_MODEL):
        self.observability = observability or NoopObservability()
        self.model = model
        # Ordered (provider, model_label, client) fallback chain.
        self._providers: list[tuple[str, str, object]] = []

        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_key:
            try:
                from openai import AzureOpenAI

                self._providers.append((
                    "azure_openai",
                    AZURE_DEPLOYMENT,
                    AzureOpenAI(
                        api_key=azure_key,
                        azure_endpoint=AZURE_ENDPOINT,
                        api_version=AZURE_API_VERSION,
                        # Reasoning models routinely outlive the Groq budget.
                        timeout=max(60.0, MODEL_TIMEOUT_MS / 1000),
                    ),
                ))
            except ImportError:
                logger.warning("openai package missing; skipping the Azure provider")

        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            from groq import Groq

            self._providers.append((
                "groq", model, Groq(api_key=groq_key, timeout=MODEL_TIMEOUT_MS / 1000)
            ))

        if not self._providers:
            logger.warning(
                "Neither AZURE_OPENAI_API_KEY nor GROQ_API_KEY is configured; "
                "agents will use deterministic fallbacks"
            )

    @property
    def available(self) -> bool:
        return bool(self._providers)

    def complete_json(self, name: str, prompt: str) -> dict | None:
        """Return the first provider's valid JSON object, or ``None``."""
        if not self._providers:
            return None

        _HEALTH["calls"] += 1
        last_error = ""
        for index, (provider, model_label, client) in enumerate(self._providers):
            content = self._request(name, prompt, provider, model_label, client)
            if content is None:
                last_error = f"{provider} request failed in {name}"
            else:
                payload = self._parse(name, prompt, content)
                if payload is not None:
                    return payload
                last_error = f"invalid JSON from {provider} in {name}"
            if index + 1 < len(self._providers):
                next_provider = self._providers[index + 1][0]
                logger.warning("%s: %s; falling back to %s", name, last_error, next_provider)
        _record_failure(last_error)
        logger.warning("%s: every model provider failed; using deterministic fallback", name)
        return None

    def _request(self, name: str, prompt: str, provider: str, model_label: str,
                 client) -> str | None:
        """One provider's request with transient-error retries; None on failure."""
        with self.observability.generation(
            name,
            model=model_label,
            temperature=AGENT_TEMPERATURE if provider == "groq" else None,
            input=prompt,
            metadata={"provider": provider},
        ) as generation:
            response = None
            for attempt in range(len(RETRY_DELAYS_S) + 1):
                try:
                    kwargs = dict(
                        model=model_label,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                    )
                    if provider == "groq":
                        # Reasoning deployments reject explicit temperatures;
                        # Groq's Llama needs 0 for determinism.
                        kwargs["temperature"] = AGENT_TEMPERATURE
                    response = client.chat.completions.create(**kwargs)
                    break
                except Exception as exc:
                    if attempt < len(RETRY_DELAYS_S) and _is_retryable(str(exc)):
                        delay = RETRY_DELAYS_S[attempt]
                        logger.info("%s (%s): transient model error, retrying in %ss",
                                    name, provider, delay)
                        import time

                        time.sleep(delay)
                        continue
                    self.observability.record_exception(
                        exc, input=prompt, context={"active_action": name}
                    )
                    return None

            content = (response.choices[0].message.content or "").strip()
            generation.update(output=content, usage_details=_usage_details(response))
        return content

    def _parse(self, name: str, prompt: str, content: str) -> dict | None:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            self.observability.record_exception(
                exc, input=prompt, output=content, context={"active_action": name}
            )
            return None
        return payload if isinstance(payload, dict) else None
