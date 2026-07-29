"""Read-only, Gemini-backed visual understanding for a Playwright page.

This module intentionally has no dependency on the explorer, planner, or
executor.  Taking a viewport screenshot is its only interaction with a page.
"""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from observability.tracing import NoopObservability, ObservabilityBackend


logger = logging.getLogger(__name__)

TARGET_SCREENSHOT_WIDTH = 1280


class PlaywrightPage(Protocol):
    """The small, read-only subset of Playwright's synchronous ``Page`` API."""

    def screenshot(self, *, type: str) -> bytes: ...


@dataclass(frozen=True)
class VisionObservation:
    """A concise description of UI that is visible in the current viewport."""

    page_type: str
    summary: str
    dialogs: list[str] = field(default_factory=list)
    buttons: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    navigation: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class _VisionObservationPayload(BaseModel):
    """Strict boundary model used to validate Gemini's JSON response."""

    model_config = ConfigDict(extra="forbid", strict=True)

    page_type: str
    summary: str
    dialogs: list[str]
    buttons: list[str]
    links: list[str]
    navigation: list[str]
    forms: list[str]
    inputs: list[str]
    tables: list[str]
    warnings: list[str]

    def to_observation(self) -> VisionObservation:
        return VisionObservation(**self.model_dump())


EXPECTED_KEYS = frozenset(_VisionObservationPayload.model_fields)

VISION_PROMPT = """You are a read-only UI perception component for a QA system.
Describe only what is visibly present in the supplied current viewport image.
Do not infer off-screen content, hidden DOM content, future states, or actions.
Do not propose, plan, click, type, navigate, or otherwise execute actions.

Use concise, human-readable strings. For controls, include visible text and a
short distinguishing detail only when useful. Record only visible dialogs,
buttons, links, navigation items, forms, inputs, tables, and warnings.
Use an empty list when a category is absent. ``page_type`` should be a concise
classification such as login, dashboard, listing, form, settings, modal, or
unknown. ``summary`` should explain the visible page in one or two sentences.

Return JSON only, conforming exactly to the supplied response schema."""

REPAIR_PROMPT = """Your previous response could not be parsed or did not match the
required schema. Return a replacement response now. Return only one JSON object
that conforms exactly to the response schema; no markdown or commentary."""


class VisionObservationError(RuntimeError):
    """Raised when a screenshot cannot be converted into a valid observation."""


class VisionObserver:
    """Describe the current Playwright viewport with Gemini 2.5 Flash.

    ``observe`` is deliberately synchronous because this application uses the
    synchronous Playwright API. It captures a viewport image only; it never
    performs a browser action or waits for page state to change.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        observability: ObservabilityBackend | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
        self.observability = observability or NoopObservability()

    def observe(self, page: PlaywrightPage) -> VisionObservation:
        """Capture and describe the current viewport of ``page``."""
        if not self.api_key:
            raise VisionObservationError("GEMINI_API_KEY is not configured")

        with self.observability.generation(
            "Vision",
            model=self.model,
            temperature=None,
            input={"prompt": VISION_PROMPT, "current_url": getattr(page, "url", None)},
            metadata={"component": "vision_observer"},
        ) as generation:
            try:
                screenshot = page.screenshot(type="png")
                image_bytes, original_size, resized_size = self._resize_screenshot(screenshot)
                logger.info(
                    "Vision screenshot size: original=%sx%s, sent=%sx%s, bytes=%s",
                    *original_size,
                    *resized_size,
                    len(image_bytes),
                )

                client, types = self._create_client()
                raw_response, usage_details = self._generate(client, types, image_bytes, VISION_PROMPT)
                try:
                    observation = self._parse_observation(raw_response)
                except (json.JSONDecodeError, ValidationError, ValueError) as first_error:
                    logger.warning("Vision JSON validation failed; retrying once: %s", first_error)
                    retry_response, retry_usage = self._generate(client, types, image_bytes, REPAIR_PROMPT)
                    usage_details = self._add_usage(usage_details, retry_usage)
                    try:
                        observation = self._parse_observation(retry_response)
                    except (json.JSONDecodeError, ValidationError, ValueError) as retry_error:
                        raise VisionObservationError(
                            "Gemini returned invalid VisionObservation JSON after one retry"
                        ) from retry_error

                generation.update(
                    output={"observation": observation.__dict__, "screenshot_size": resized_size},
                    usage_details=usage_details or None,
                )
                logger.info("Parsed vision observation: %s", observation)
                return observation
            except Exception as exc:
                self.observability.record_exception(
                    exc,
                    context={"current_url": getattr(page, "url", None), "active_action": "Vision"},
                )
                raise

    def _create_client(self) -> tuple[Any, Any]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise VisionObservationError(
                "google-genai is required for VisionObserver; install backend requirements"
            ) from exc
        return genai.Client(api_key=self.api_key), types

    def _generate(
        self, client: Any, types: Any, image_bytes: bytes, prompt: str
    ) -> tuple[str, dict[str, int]]:
        started_at = perf_counter()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_VisionObservationPayload,
                ),
            )
        except Exception as exc:
            raise VisionObservationError("Gemini vision request failed") from exc
        finally:
            logger.info("Vision API latency: %.2f ms", (perf_counter() - started_at) * 1000)

        usage_details = self._token_usage(getattr(response, "usage_metadata", None))
        self._log_token_usage(usage_details)
        return response.text or "", usage_details

    @staticmethod
    def _parse_observation(raw_response: str) -> VisionObservation:
        payload = json.loads(raw_response)
        if not isinstance(payload, dict) or set(payload) != EXPECTED_KEYS:
            raise ValueError("Vision response must contain exactly the required keys")
        return _VisionObservationPayload.model_validate(payload).to_observation()

    @staticmethod
    def _resize_screenshot(screenshot: bytes) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise VisionObservationError("Pillow is required to resize vision screenshots") from exc

        with Image.open(io.BytesIO(screenshot)) as image:
            original_size = image.size
            target_height = max(1, round(image.height * TARGET_SCREENSHOT_WIDTH / image.width))
            resized = image.resize((TARGET_SCREENSHOT_WIDTH, target_height), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            resized.convert("RGB").save(output, format="JPEG", quality=85, optimize=True)
            return output.getvalue(), original_size, resized.size

    @staticmethod
    def _token_usage(usage: Any) -> dict[str, int]:
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

    @staticmethod
    def _log_token_usage(usage: dict[str, int]) -> None:
        if not usage:
            logger.info("Vision token usage: unavailable")
            return
        logger.info("Vision token usage: %s", usage)
