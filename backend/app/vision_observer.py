"""Gemini-backed, read-only interpretation of a browser screenshot.

This module deliberately has no Playwright, planner, or executor dependency.
Callers provide an already-captured screenshot and receive a typed description
of only the UI visible in that image.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


class VisionObservation(BaseModel):
    """Structured, non-actionable description of the current viewport."""

    model_config = ConfigDict(extra="forbid")

    page_name: str = ""
    page_type: str = ""
    primary_goal: str = ""
    primary_cta: str = ""
    secondary_actions: list[str] = Field(default_factory=list)
    navigation_items: list[str] = Field(default_factory=list)
    visible_forms: list[str] = Field(default_factory=list)
    visible_tables: list[str] = Field(default_factory=list)
    dialogs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


VISION_PROMPT = """You are the vision perception module for an autonomous QA agent.

Analyze ONLY the current screenshot.

Return ONLY valid JSON.

Do not include markdown.
Do not wrap the response in ```json.
Do not include explanations.

Return exactly this structure:
{
  "page_name": "",
  "page_type": "",
  "primary_goal": "",
  "primary_cta": "",
  "secondary_actions": [],
  "navigation_items": [],
  "visible_forms": [],
  "visible_tables": [],
  "dialogs": [],
  "warnings": [],
  "confidence": 0.0
}

Use a concise page_type such as login, dashboard, listing, form, wizard,
modal, or settings. Use strings in every array. confidence must be a number
from 0 to 1."""

EXPECTED_KEYS = frozenset(VisionObservation.model_fields)


class VisionObserver:
    """Send a screenshot to Gemini and return a :class:`VisionObservation`."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")

    def observe(self, screenshot_path: str | Path | None) -> VisionObservation:
        """Describe the supplied viewport image without affecting browser state.

        A failed or unavailable Gemini request deliberately yields an empty,
        zero-confidence observation so discovery retains its existing behavior.
        """
        if not screenshot_path:
            logger.warning("Vision observation skipped: no screenshot path was provided")
            return self._log_observation(VisionObservation())

        image_path = Path(screenshot_path)
        if not image_path.is_file():
            logger.warning("Vision observation skipped: screenshot does not exist: %s", image_path)
            return self._log_observation(VisionObservation())
        if not self.api_key:
            logger.warning("Vision observation skipped: GEMINI_API_KEY is not configured")
            return self._log_observation(VisionObservation())

        try:
            # Imported lazily so a missing optional SDK does not prevent the
            # existing DOM-only explorer from starting.
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=[
                    VISION_PROMPT,
                    types.Part.from_bytes(
                        data=image_path.read_bytes(),
                        mime_type=self._mime_type(image_path),
                    ),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            raw_response = response.text or ""
            logger.info(
                "======== GEMINI RAW RESPONSE ========\n%s\n=====================================",
                raw_response,
            )
            payload = self._parse_response(raw_response)
            observation = VisionObservation.model_validate(payload)
        except Exception as exc:
            logger.warning("Vision observation failed for %s: %s", image_path, exc)
            return self._log_observation(VisionObservation())

        return self._log_observation(observation)

    @staticmethod
    def _log_observation(observation: VisionObservation) -> VisionObservation:
        logger.info("Vision Observation\n%s", observation.model_dump_json(indent=2))
        return observation

    @staticmethod
    def _parse_response(raw_response: str) -> dict[str, object]:
        """Normalize Gemini text and ensure it matches the perception contract."""
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        payload = json.loads(cleaned)
        if not isinstance(payload, dict) or set(payload) != EXPECTED_KEYS:
            raise ValueError(
                "Gemini response does not contain exactly the expected VisionObservation keys"
            )
        return payload

    @staticmethod
    def _mime_type(image_path: Path) -> str:
        return "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
