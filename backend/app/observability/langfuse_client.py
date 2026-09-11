"""Fail-safe adapter for the current OpenTelemetry-based Langfuse Python SDK.

Observability is strictly best-effort.  No SDK import, request, update, context
manager exit, or flush is allowed to alter application control flow.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator

from .tracing import NoopObservation, NoopObservability, Observation, TraceMetadata


logger = logging.getLogger(__name__)
_ACTIVE_OBSERVATIONS: ContextVar[tuple["_LangfuseObservation", ...]] = ContextVar(
    "langfuse_active_observations", default=()
)
_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}


class _LangfuseObservation:
    """A safe handle around a current v4 Langfuse span or generation."""

    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
        status_message: str | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if output is not None:
            values["output"] = output
        if metadata is not None:
            values["metadata"] = metadata
        if usage_details is not None:
            values["usage_details"] = usage_details
        if status_message is not None:
            values.update(level="ERROR", status_message=status_message)
        if not values:
            return
        try:
            # Current Langfuse SDKs update observations through their own
            # handle. Do not use removed update_current_observation APIs.
            self._observation.update(**values)
        except Exception:
            logger.exception("Failed to update Langfuse observation")


class LangfuseObservability:
    """Provider-neutral, nested Langfuse tracing that cannot interrupt QA work."""

    def __init__(self) -> None:
        # ``get_client`` and ``propagate_attributes`` are the supported v4 API.
        from langfuse import get_client, propagate_attributes

        self._client: Any = get_client()
        self._propagate_attributes: Callable[..., AbstractContextManager[Any]] = propagate_attributes

    @contextmanager
    def trace(
        self, name: str, *, metadata: TraceMetadata, input: Any = None
    ) -> Iterator[Observation]:
        attributes = self._enter_attributes(name, metadata)
        if attributes is None:
            yield NoopObservation()
            return
        try:
            with self._observation_context(
                lambda: self._client.start_as_current_observation(
                    as_type="span", name=name, input=input
                )
            ) as observation:
                yield observation
        finally:
            self._exit_context(attributes, "Langfuse trace attributes")

    @contextmanager
    def span(
        self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> Iterator[Observation]:
        with self._observation_context(
            lambda: self._client.start_as_current_observation(
                as_type="span", name=name, input=input, metadata=metadata
            )
        ) as observation:
            yield observation

    @contextmanager
    def generation(
        self,
        name: str,
        *,
        model: str,
        temperature: float | None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Observation]:
        details = {"temperature": temperature, **(metadata or {})}
        with self._observation_context(
            lambda: self._client.start_as_current_observation(
                as_type="generation", name=name, model=model, input=input, metadata=details
            )
        ) as observation:
            yield observation

    @contextmanager
    def _observation_context(
        self, factory: Callable[[], AbstractContextManager[Any]]
    ) -> Iterator[Observation]:
        """Enter and exit an SDK observation without ever swallowing app errors."""
        try:
            manager = factory()
            raw_observation = manager.__enter__()
        except Exception:
            logger.exception("Failed to start Langfuse observation")
            yield NoopObservation()
            return

        observation = _LangfuseObservation(raw_observation)
        token = _ACTIVE_OBSERVATIONS.set((*_ACTIVE_OBSERVATIONS.get(), observation))
        try:
            yield observation
        finally:
            _ACTIVE_OBSERVATIONS.reset(token)
            try:
                # Preserve the currently propagating application exception;
                # SDK exit failures are logged and discarded.
                manager.__exit__(*sys.exc_info())
            except Exception:
                logger.exception("Failed to finish Langfuse observation")

    def _enter_attributes(
        self, name: str, metadata: TraceMetadata
    ) -> AbstractContextManager[Any] | None:
        try:
            attributes = self._propagate_attributes(
                trace_name=name,
                session_id=metadata.session_id,
                metadata={
                    "project_name": metadata.project_name,
                    "exploration_id": metadata.exploration_id,
                    "current_page": metadata.current_page or "",
                    "current_goal": metadata.current_goal,
                },
            )
            attributes.__enter__()
            return attributes
        except Exception:
            logger.exception("Failed to set Langfuse trace attributes")
            return None

    @staticmethod
    def _exit_context(context: AbstractContextManager[Any], description: str) -> None:
        try:
            context.__exit__(*sys.exc_info())
        except Exception:
            logger.exception("Failed to finish %s", description)

    def record_exception(
        self,
        exception: BaseException,
        *,
        input: Any = None,
        output: Any = None,
        retry_count: int = 0,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Attach a complete error record to the innermost active observation."""
        details = {
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "traceback": "".join(traceback.format_exception(exception)),
            "current_url": None,
            "workflow_name": None,
            "active_action": None,
            "screenshot_path": None,
            "input": input,
            "output": output,
            "retry_count": retry_count,
        }
        if context:
            details.update({key: context.get(key) for key in details.keys() & context.keys()})
        active = _ACTIVE_OBSERVATIONS.get()
        if not active:
            logger.error("Application exception outside an active Langfuse observation: %s", details)
            return
        try:
            active[-1].update(metadata={"error": details}, status_message=str(exception))
        except Exception:
            # _LangfuseObservation already contains this guard, but retain the
            # outer boundary in case a future implementation changes it.
            logger.exception("Failed to send exception to Langfuse")

    def event(
        self, name: str, *, input: Any = None, output: Any = None,
        metadata: dict[str, Any] | None = None, level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """Point-in-time log line, attached to the current trace context."""
        try:
            self._client.create_event(
                name=name, input=input, output=output, metadata=metadata,
                level=level, status_message=status_message,
            )
        except Exception:
            logger.exception("Failed to send Langfuse event %s", name)

    def score_current_trace(
        self, name: str, value: float | str, *, data_type: str = "NUMERIC",
        comment: str | None = None, metadata: dict[str, Any] | None = None,
    ) -> None:
        """Send an evaluation score for the active pipeline trace."""
        try:
            self._client.score_current_trace(
                name=name, value=value, data_type=data_type, comment=comment,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Failed to send Langfuse evaluation score %s", name)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.exception("Failed to flush Langfuse")


def create_observability() -> "LangfuseObservability | NoopObservability":
    """Return a no-op backend whenever tracing is disabled or unavailable."""
    enabled = os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().casefold()
    if enabled in _DISABLED_VALUES:
        logger.info("Langfuse tracing is disabled")
        return NoopObservability()
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        logger.info("Langfuse credentials are absent; observability is disabled")
        return NoopObservability()
    try:
        return LangfuseObservability()
    except Exception:
        logger.exception("Langfuse initialization failed; observability is disabled")
        return NoopObservability()
