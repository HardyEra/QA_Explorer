"""Langfuse implementation of the provider-neutral observability contract."""

from __future__ import annotations

import logging
import os
import traceback
from contextlib import contextmanager
from typing import Any, Iterator

from .tracing import NoopObservability, Observation, TraceMetadata

logger = logging.getLogger(__name__)


class _LangfuseObservation:
    def __init__(self, observation: Any):
        self._observation = observation

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None,
               usage_details: dict[str, int] | None = None,
               status_message: str | None = None) -> None:
        values: dict[str, Any] = {}
        if output is not None:
            values["output"] = output
        if metadata:
            values["metadata"] = metadata
        if usage_details:
            values["usage_details"] = usage_details
        if status_message:
            values.update(level="ERROR", status_message=status_message)
        if values:
            self._observation.update(**values)


class LangfuseObservability:
    """Langfuse v3/v4 adapter using context-managed nested observations."""

    def __init__(self) -> None:
        from langfuse import get_client, propagate_attributes

        self._client = get_client()
        self._propagate_attributes = propagate_attributes

    @contextmanager
    def trace(self, name: str, *, metadata: TraceMetadata,
              input: Any = None) -> Iterator[Observation]:
        attributes = self._propagate_attributes(
            trace_name=name,
            session_id=metadata.session_id,
            metadata=metadata.model_dump(mode="json"),
        )
        with attributes:
            with self._client.start_as_current_observation(
                as_type="span", name=name, input=input
            ) as observation:
                yield _LangfuseObservation(observation)

    @contextmanager
    def span(self, name: str, *, input: Any = None,
             metadata: dict[str, Any] | None = None) -> Iterator[Observation]:
        with self._client.start_as_current_observation(
            as_type="span", name=name, input=input, metadata=metadata
        ) as observation:
            yield _LangfuseObservation(observation)

    @contextmanager
    def generation(self, name: str, *, model: str, temperature: float | None,
                   input: Any = None, metadata: dict[str, Any] | None = None) -> Iterator[Observation]:
        details = {"temperature": temperature, **(metadata or {})}
        with self._client.start_as_current_observation(
            as_type="generation", name=name, model=model, input=input,
            metadata=details,
        ) as observation:
            yield _LangfuseObservation(observation)

    def record_exception(self, exception: BaseException, *, input: Any = None,
                         output: Any = None, retry_count: int = 0) -> None:
        """Annotate the active observation without changing application control flow."""
        error = {
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "stack_trace": "".join(traceback.format_exception(exception)),
            "input": input,
            "output": output,
            "retry_count": retry_count,
        }
        try:
            self._client.update_current_observation(
                level="ERROR", status_message=str(exception), metadata={"error": error}
            )
        except Exception:  # Telemetry must never interrupt QA execution.
            logger.exception("Could not send exception to Langfuse")

    def flush(self) -> None:
        self._client.flush()


def create_observability():
    """Build the configured backend, or a no-op backend for local development."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        logger.info("Langfuse credentials are absent; observability is disabled")
        return NoopObservability()
    try:
        return LangfuseObservability()
    except ImportError:
        logger.warning("langfuse package is not installed; observability is disabled")
        return NoopObservability()
