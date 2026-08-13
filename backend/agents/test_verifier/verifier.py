"""Verify designed test cases against the application's real control inventory.

Runs after design and before execution.  Deterministic (no model): every click,
select, fill and element expectation must name a control the App Map has
actually observed.  A near-miss is auto-corrected to the real label; a target
with no counterpart anywhere marks the case unverified so it is reported as
such instead of failing execution with a misleading diagnosis.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from observability.tracing import NoopObservability


logger = logging.getLogger(__name__)

# Below this similarity a "correction" would be a guess, not a fix.
AUTO_CORRECT_THRESHOLD = 0.82
# Same distinguishing words, different decoration ("View X details" vs "X 35").
TOKEN_OVERLAP_THRESHOLD = 0.6
TARGET_STEP_TYPES = {"click", "select"}
# Words that decorate a control name without identifying it.
_DECORATIVE_TOKENS = {
    "view", "details", "detail", "the", "a", "an", "of", "to", "all", "show",
    "open", "see", "page", "button", "link", "icon", "arrow", "right", "left",
}


def _normalise(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def _tokens(text: str) -> set[str]:
    """Identifying words only: decoration and counts are not identity."""
    return {
        token for token in _normalise(text).replace("-", " ").split()
        if token not in _DECORATIVE_TOKENS and not token.isdigit() and len(token) > 2
    }


class TestVerifier:
    """Check test-case targets against observed reality; fix or flag them."""

    __test__ = False

    def __init__(self, observability=None):
        self.observability = observability or NoopObservability()

    def verify(self, test_cases: list[dict], app_map: dict) -> tuple[list[dict], list[dict]]:
        """Return ``(cases, problems)``; cases keep every step, flagged if unverifiable."""
        known_controls, known_fields = self._inventory(app_map)
        if not known_controls and not known_fields:
            logger.info("Verifier: no observed inventory available; skipping verification")
            return test_cases, []

        problems: list[dict] = []
        verified: list[dict] = []
        corrections = 0
        with self.observability.span(
            "Test Verifier",
            input={"cases": len(test_cases), "known_controls": len(known_controls)},
        ) as span:
            for case in test_cases:
                case, case_problems, case_corrections = self._verify_case(
                    case, known_controls, known_fields
                )
                corrections += case_corrections
                problems.extend(case_problems)
                verified.append(case)
            span.update(output={"corrections": corrections, "problems": len(problems)})
        logger.info(
            "Test Verifier: %s auto-correction(s), %s unverifiable target(s) across %s case(s)",
            corrections, len(problems), len(test_cases),
        )
        return verified, problems

    @staticmethod
    def _inventory(app_map: dict) -> tuple[dict[str, str], dict[str, str]]:
        """Observed control labels and field names, normalised → original."""
        controls: dict[str, str] = {}
        fields: dict[str, str] = {}
        for page in app_map.get("pages", []):
            for action in page.get("actions", []):
                label = action.get("label", "")
                if label:
                    controls.setdefault(_normalise(label), label)
            for control in page.get("controls", []):
                if control:
                    controls.setdefault(_normalise(control), control)
            for field in page.get("fields", []):
                for key in ("placeholder", "name"):
                    value = field.get(key)
                    if value:
                        fields.setdefault(_normalise(value), value)
            for fill in page.get("fills", []):
                value = fill.get("field")
                if value:
                    fields.setdefault(_normalise(value), value)
        return controls, fields

    def _verify_case(self, case: dict, controls: dict[str, str],
                     fields: dict[str, str]) -> tuple[dict, list[dict], int]:
        problems: list[dict] = []
        corrections = 0
        steps = []
        for index, step in enumerate(case.get("steps", []), start=1):
            step_type = step.get("type")
            target = step.get("target", "")
            if step_type in TARGET_STEP_TYPES:
                known = controls
            elif step_type == "fill":
                # A fill may legitimately target a field or a labelled control.
                known = {**fields, **controls}
            else:
                steps.append(step)
                continue

            resolved, corrected = self._resolve(target, known)
            if resolved is None:
                problems.append({
                    "test_id": case.get("id"),
                    "step_index": index,
                    "target": target,
                    "reason": f"no observed control or field is named {target!r}",
                })
            else:
                if corrected:
                    logger.info(
                        "Verifier corrected %s step %s: %r -> %r",
                        case.get("id"), index, target, resolved,
                    )
                    corrections += 1
                step = {**step, "target": resolved}
            steps.append(step)

        expected, expectation_problems = self._verify_expectations(case, controls, fields)
        problems.extend(expectation_problems)
        verified_case = {**case, "steps": steps, "expected": expected}
        if problems:
            verified_case["unverified"] = [problem["reason"] for problem in problems]
        return verified_case, problems, corrections

    def _verify_expectations(self, case: dict, controls: dict[str, str],
                             fields: dict[str, str]) -> tuple[list[dict], list[dict]]:
        """Drop element expectations naming controls nobody has ever observed."""
        known = {**controls, **fields}
        kept, problems = [], []
        for expectation in case.get("expected", []):
            if expectation["type"] != "element_visible":
                kept.append(expectation)
                continue
            resolved, _ = self._resolve(expectation["value"], known)
            if resolved is None:
                problems.append({
                    "test_id": case.get("id"),
                    "step_index": None,
                    "target": expectation["value"],
                    "reason": f"expected control {expectation['value']!r} was never observed",
                })
                continue
            kept.append({**expectation, "value": resolved})
        if not kept:
            # Never leave a case with nothing to verify.
            kept = [item for item in case.get("expected", []) if item["type"] == "url_contains"]
        return kept, problems

    @staticmethod
    def _resolve(target: str, known: dict[str, str]) -> tuple[str | None, bool]:
        """Exact → substring → fuzzy. Returns (resolved_label, was_corrected)."""
        normalised = _normalise(target)
        if not normalised:
            return None, False
        if normalised in known:
            return known[normalised], False
        # Observed labels often carry extra text ("Onboarded Vendors 35").
        for key, original in known.items():
            if normalised in key or key in normalised:
                return original, original.casefold() != normalised
        # Same identifying words, different decoration: "View X details" ↔ "X 35".
        target_tokens = _tokens(target)
        if target_tokens:
            best_token_key, best_token_score = None, 0.0
            for key, original in known.items():
                key_tokens = _tokens(original)
                if not key_tokens:
                    continue
                overlap = len(target_tokens & key_tokens) / len(target_tokens | key_tokens)
                if overlap > best_token_score:
                    best_token_key, best_token_score = key, overlap
            if best_token_key and best_token_score >= TOKEN_OVERLAP_THRESHOLD:
                return known[best_token_key], True
        best_key, best_score = None, 0.0
        for key in known:
            score = SequenceMatcher(None, normalised, key).ratio()
            if score > best_score:
                best_key, best_score = key, score
        if best_key and best_score >= AUTO_CORRECT_THRESHOLD:
            return known[best_key], True
        return None, False
