"""Read-only, Gemini-backed visual understanding for a Playwright page.

This module intentionally has no dependency on the explorer, planner, or
executor.  Taking a viewport screenshot is its only interaction with a page.
"""

from __future__ import annotations

import io
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from observability.tracing import NoopObservability, ObservabilityBackend


logger = logging.getLogger(__name__)

TARGET_SCREENSHOT_WIDTH = 1280
SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


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
        # Gemini 2.5 Flash is unavailable to new API users.  Gemini 3.6 Flash
        # is Google's documented migration target and supports image input and
        # structured JSON output used by this observer.
        self.model = model or os.getenv("GEMINI_VISION_MODEL", "gemini-3.6-flash")
        self.observability = observability or NoopObservability()
        self.runtime_dir = Path(
            os.getenv("VISION_RUNTIME_DIR", str(Path(__file__).resolve().parent.parent / "runtime"))
        )

    def observe(self, page: PlaywrightPage) -> VisionObservation:
        """Capture and describe the current viewport of ``page``."""
        if not self.api_key:
            raise VisionObservationError("GEMINI_API_KEY is not configured")

        request_context: dict[str, Any] = {}
        sent_image: bytes | None = None
        with self.observability.generation(
            "Vision",
            model=self.model,
            temperature=None,
            input={"prompt": VISION_PROMPT, "current_url": getattr(page, "url", None)},
            metadata={"component": "vision_observer"},
        ) as generation:
            try:
                screenshot = page.screenshot(type="png")
                screenshot_path = self.runtime_dir / "latest.png"
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                screenshot_path.write_bytes(screenshot)
                image_bytes, original_size, resized_size = self._resize_screenshot(screenshot)
                sent_image = image_bytes
                request_context = self._request_context(
                    screenshot_path=screenshot_path,
                    original_size=original_size,
                    sent_size=resized_size,
                    image_bytes=image_bytes,
                    prompt=VISION_PROMPT,
                )
                self._validate_request(request_context, image_bytes, VISION_PROMPT)
                self._log_request(request_context)

                client, types = self._create_client()
                raw_response, usage_details = self._generate(
                    client, types, image_bytes, VISION_PROMPT, request_context
                )
                try:
                    observation = self._parse_observation(raw_response)
                except (json.JSONDecodeError, ValidationError, ValueError) as first_error:
                    logger.warning("Vision JSON validation failed; retrying once: %s", first_error)
                    retry_context = {**request_context, "prompt": REPAIR_PROMPT}
                    self._validate_request(retry_context, image_bytes, REPAIR_PROMPT)
                    self._log_request(retry_context)
                    request_context = retry_context
                    retry_response, retry_usage = self._generate(
                        client, types, image_bytes, REPAIR_PROMPT, retry_context
                    )
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
                reason = self._exception_reason(exc)
                logger.error("Vision unavailable.\nReason:\n%s", reason)
                logger.error("Falling back to DOM observation.")
                self._save_failed_request(request_context, sent_image, exc)
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
        try:
            from importlib.metadata import version
            sdk_version = version("google-genai")
        except Exception:
            sdk_version = "unknown"
        logger.info("Gemini SDK: google-genai %s", sdk_version)
        # Vision is supplementary perception: a stalled Gemini request must
        # fail fast and let DOM-first exploration continue, never hang the run.
        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=30_000),
        )
        return client, types

    def _generate(
        self, client: Any, types: Any, image_bytes: bytes, prompt: str,
        request_context: dict[str, Any],
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
                    # ``response_schema`` converts Pydantic's
                    # ``additionalProperties`` into the unsupported REST
                    # field ``additional_properties`` in recent google-genai
                    # SDKs. Send the JSON Schema through its dedicated field
                    # instead; Gemini accepts the standard spelling there.
                    response_json_schema=_VisionObservationPayload.model_json_schema(),
                ),
            )
        except Exception as exc:
            self._log_google_error(exc, request_context)
            raise VisionObservationError(
                f"Gemini vision request failed: {self._exception_reason(exc)}"
            ) from exc
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

        if not screenshot:
            raise VisionObservationError("Screenshot is empty")
        with Image.open(io.BytesIO(screenshot)) as image:
            image.verify()
        with Image.open(io.BytesIO(screenshot)) as image:
            original_size = image.size
            target_height = max(1, round(image.height * TARGET_SCREENSHOT_WIDTH / image.width))
            resized = image.resize((TARGET_SCREENSHOT_WIDTH, target_height), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            resized.convert("RGB").save(output, format="JPEG", quality=85, optimize=True)
            return output.getvalue(), original_size, resized.size

    def _request_context(
        self, *, screenshot_path: Path, original_size: tuple[int, int],
        sent_size: tuple[int, int], image_bytes: bytes, prompt: str,
    ) -> dict[str, Any]:
        try:
            from importlib.metadata import version
            sdk_version = version("google-genai")
        except Exception:
            sdk_version = "unknown"
        return {
            "model": self.model,
            "sdk_version": sdk_version,
            "screenshot_path": str(screenshot_path),
            "screenshot_exists": screenshot_path.exists(),
            "screenshot_file_size_bytes": screenshot_path.stat().st_size if screenshot_path.exists() else 0,
            "original_dimensions": f"{original_size[0]}x{original_size[1]}",
            "sent_dimensions": f"{sent_size[0]}x{sent_size[1]}",
            "image_mime_type": "image/jpeg",
            "image_size_bytes": len(image_bytes),
            "prompt_length": len(prompt),
            "prompt_preview": prompt[:300],
            "prompt": prompt,
        }

    @staticmethod
    def _validate_request(context: dict[str, Any], image_bytes: bytes, prompt: str) -> None:
        path = Path(context["screenshot_path"])
        if not path.exists():
            raise VisionObservationError(f"Screenshot does not exist: {path}")
        if path.stat().st_size <= 0 or not image_bytes:
            raise VisionObservationError("Screenshot size must be greater than zero")
        if context["image_mime_type"] not in SUPPORTED_IMAGE_MIME_TYPES:
            raise VisionObservationError(f"Unsupported image MIME type: {context['image_mime_type']}")
        if not prompt.strip():
            raise VisionObservationError("Vision prompt must not be empty")

    @staticmethod
    def _log_request(context: dict[str, Any]) -> None:
        logger.info(
            "VISION REQUEST\nModel: %s\nSDK version: %s\nScreenshot path: %s\n"
            "Screenshot exists: %s\nDimensions: %s (sent %s)\nImage MIME type: %s\n"
            "Screenshot file size: %s bytes\nSent image size: %s bytes\n"
            "Prompt: %s chars\nPrompt preview: %s",
            context["model"], context["sdk_version"], context["screenshot_path"],
            context["screenshot_exists"], context["original_dimensions"],
            context["sent_dimensions"], context["image_mime_type"],
            context["screenshot_file_size_bytes"], context["image_size_bytes"],
            context["prompt_length"],
            context["prompt_preview"].replace("\n", " "),
        )

    @classmethod
    def _exception_reason(cls, exc: BaseException) -> str:
        response = getattr(exc, "response", None)
        pieces = [str(exc) or repr(exc)]
        for name in ("code", "status", "message", "details", "body"):
            value = getattr(exc, name, None)
            if value not in (None, "") and str(value) not in pieces:
                pieces.append(f"exception.{name}={value}")
        if response is not None:
            for name in ("status_code", "status", "body", "text", "message"):
                value = getattr(response, name, None)
                if value not in (None, ""):
                    pieces.append(f"response.{name}={value}")
            pieces.append(f"response={response!r}")
        return " | ".join(pieces)

    def _log_google_error(self, exc: BaseException, context: dict[str, Any]) -> None:
        response = getattr(exc, "response", None)
        response_body = self._response_body(response)
        logger.error(
            "Gemini Vision request failed: HTTP status=%s; exception_class=%s; "
            "complete Google response body=%s; complete message=%s\n%s",
            getattr(exc, "code", None) or getattr(exc, "status_code", None)
            or getattr(response, "status_code", None),
            type(exc).__name__, response_body, self._exception_reason(exc),
            traceback.format_exc(),
        )

    @staticmethod
    def _response_body(response: Any) -> Any:
        if response is None:
            return None
        for name in ("body", "text", "content"):
            try:
                value = getattr(response, name, None)
                if value not in (None, "", b""):
                    return value
            except Exception as exc:
                return f"<unable to read response.{name}: {exc!r}>"
        return repr(response)

    def _save_failed_request(
        self, context: dict[str, Any], image_bytes: bytes | None, exc: BaseException
    ) -> None:
        if not context:
            logger.error("Vision failed before request metadata could be captured")
            return
        artifact_dir = self.runtime_dir / "debug" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "prompt.txt").write_text(context.get("prompt", ""), encoding="utf-8")
            source_screenshot = context.get("screenshot_path")
            if source_screenshot and Path(source_screenshot).exists():
                # Preserve the original Playwright PNG for visual inspection.
                (artifact_dir / "screenshot.png").write_bytes(Path(source_screenshot).read_bytes())
            if image_bytes:
                # Keep the exact JPEG bytes and MIME type used in the request.
                (artifact_dir / "sent_image.jpg").write_bytes(image_bytes)
            elif not source_screenshot:
                (artifact_dir / "screenshot.png").write_bytes(b"")
            request = {key: value for key, value in context.items() if key != "prompt"}
            request["contents"] = [
                {"text": context.get("prompt", "")},
                {"image_path": "sent_image.jpg", "mime_type": context.get("image_mime_type")},
            ]
            request["exception_class"] = type(exc).__name__
            request["error"] = self._exception_reason(exc)
            (artifact_dir / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
            metadata = {
                **request,
                "traceback": "".join(traceback.format_exception(exc)),
                "artifact_dir": str(artifact_dir),
            }
            (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            logger.error("Saved failed Vision request artifacts: %s", artifact_dir)
        except Exception:
            logger.exception("Could not save failed Vision request artifacts")

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
