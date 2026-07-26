"""Prompt contract reserved for a future model-backed flow summarizer.

The current Flow Generator is deterministic.  Keeping the contract here makes
an optional LLM enhancement possible without spreading prompt text through the
normalization pipeline.
"""

FLOW_GENERATOR_SYSTEM_PROMPT = """Convert successful discovery history into a
concise business workflow. Return structured flow data only. Preserve user
entered values and meaningful page transitions. Exclude retries, failures,
telemetry, internal framework events, browser implementation details, test
cases, assertions, and all Playwright code."""
