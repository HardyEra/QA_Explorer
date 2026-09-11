"""The executable test-case contract shared by the Designer, Executor, and Reporter.

Steps and expectations are deliberately restricted to a small vocabulary the
deterministic Executor can replay without an LLM.  ``normalise_case`` is the
single validation gate: anything the Designer model emits must pass through it
before entering the pipeline state.
"""

from __future__ import annotations

import re
from typing import Any


STEP_TYPES = {"navigate", "click", "fill", "upload", "select", "check", "assert"}
EXPECTATION_TYPES = {"url_contains", "text_visible", "element_visible", "field_value", "checked"}
VALID_PRIORITIES = ("critical", "high", "medium", "low")

# Values may reference credentials symbolically; the Executor substitutes the
# real values at run time so secrets never live inside generated test cases.
USERNAME_PLACEHOLDER = "{username}"
PASSWORD_PLACEHOLDER = "{password}"


def normalise_expectation(raw):
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type", "")).strip().casefold()
    value = str(raw.get("value", "")).strip()
    if kind not in EXPECTATION_TYPES or (not value and kind != "field_value"):
        return None
    result = {"type": kind, "value": value}
    if kind in ("field_value", "checked"):
        target = str(raw.get("target", "")).strip()
        if not target or (kind == "checked" and value not in ("true", "false")):
            return None
        result["target"] = target
    return result


def normalise_case(raw: Any, fallback_id: str) -> dict[str, Any] | None:
    """Validate one model-emitted test case; return ``None`` when unusable."""
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title") or "").strip()
    if not title:
        return None

    steps = []
    for step in raw.get("steps") or []:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type") or "").strip().casefold()
        if step_type not in STEP_TYPES:
            continue
        target = str(step.get("target") or "").strip()
        # Models sometimes copy the app map's annotated descriptor verbatim —
        # "Global Search... (type=text)" — but the annotation is not UI text.
        target = re.sub(r"\s*\(type=[a-z]+\)$", "", target)
        target = re.sub(r"\s*\[(dropdown|checkbox|radio|file upload|menu item)\]$", "", target)
        if step_type in {"click", "fill", "select", "check"} and not target:
            continue
        normalised = {"type": step_type, "target": target}
        if step_type == "assert":
            expectation = normalise_expectation(step.get("expectation"))
            if expectation is None:
                return None
            normalised["expectation"] = expectation
        if step_type == "check":
            checked = str(step.get("value", "true")).casefold()
            if checked not in ("true", "false"):
                return None
            normalised["value"] = checked
        if step_type == "fill":
            normalised["value"] = str(step.get("value") or "")
        if step_type == "select":
            # ``value`` is the visible text of the option to choose.
            option = str(step.get("value") or step.get("option") or "").strip()
            if not option:
                continue
            normalised["value"] = option
        if step_type == "upload":
            # ``value`` names the test asset from the library (e.g. "resume").
            normalised["value"] = str(step.get("value") or step.get("asset") or "").strip()
        steps.append(normalised)
    if not steps:
        return None

    expected = []
    for expectation in raw.get("expected") or []:
        if not isinstance(expectation, dict):
            continue
        normalised = normalise_expectation(expectation)
        if normalised:
            expected.append(normalised)
    if not expected:
        return None

    priority = str(raw.get("priority") or "medium").casefold()
    case_id = re.sub(r"[^a-z0-9-]+", "-", str(raw.get("id") or fallback_id).casefold()).strip("-")
    return {
        "id": case_id or fallback_id,
        "requirement_id": str(raw.get("requirement_id") or "").strip(),
        "title": title,
        "description": str(raw.get("description") or "").strip(),
        "priority": priority if priority in VALID_PRIORITIES else "medium",
        "preconditions": [
            str(item).strip() for item in (raw.get("preconditions") or []) if str(item).strip()
        ],
        "steps": steps,
        "expected": expected,
    }


def describe_step(step: dict[str, Any]) -> str:
    """One plain-language phrase for a step; never exposes credential values."""
    step_type = step.get("type")
    target = str(step.get("target") or "")
    if step_type == "navigate":
        return f"open {target}" if target else "open the start page"
    if step_type == "fill":
        value = str(step.get("value") or "")
        if USERNAME_PLACEHOLDER in value:
            described_value = "the configured username"
        elif PASSWORD_PLACEHOLDER in value:
            described_value = "the configured password"
        else:
            described_value = f"'{value}'" if value else "a value"
        return f"enter {described_value} into '{target}'"
    if step_type == "click":
        return f"click '{target}'"
    if step_type == "check":
        return f"set '{target}' to {step.get('value', 'true')}"
    if step_type == "assert":
        return "verify " + describe_expectation(step.get("expectation", {}))
    if step_type == "upload":
        asset = str(step.get("value") or "a test file")
        suffix = f" via '{target}'" if target else ""
        return f"upload the stored test asset '{asset}'{suffix}"
    if step_type == "select":
        return f"choose '{step.get('value', '')}' in the '{target}' dropdown"
    return str(step_type or "perform a step")


def describe_expectation(expectation: dict[str, Any]) -> str:
    """One plain-language phrase for a verification check."""
    value = str(expectation.get("value") or "")
    kind = expectation.get("type")
    if kind == "url_contains":
        return f"the page URL contains '{value}'"
    if kind == "text_visible":
        return f"the text '{value}' is visible on the page"
    if kind == "element_visible":
        return f"a control labeled '{value}' is visible"
    if kind in ("field_value", "checked"):
        return f"'{expectation.get('target', '')}' has {kind} '{value}'"
    return f"{kind} '{value}'"


def describe_case(case: dict[str, Any]) -> str:
    """Deterministic fallback description when the model supplies none."""
    steps = "; then ".join(describe_step(step) for step in case.get("steps", []))
    checks = ", and ".join(describe_expectation(item) for item in case.get("expected", []))
    description = f"This test will {steps}."
    if checks:
        description += f" It passes only if {checks}."
    return description
