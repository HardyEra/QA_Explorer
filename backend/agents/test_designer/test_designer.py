"""Test Designer agent: one feature's requirements × the App Map → test cases.

The orchestrator fans out one Designer call per feature so each prompt stays
small and grounded.  Emitted cases are validated by ``normalise_case`` and
their expectations are grounded against text actually observed in the
application (or promised by the requirements) before they can enter pipeline
state.  Requirements the supplied account's role cannot verify are reported as
``blocked`` instead of being turned into tests that can only fail.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import urlparse

from agents.llm_client import JsonModelClient
from app_map_builder import AppMapBuilder
from observability.tracing import NoopObservability

from .models import (
    PASSWORD_PLACEHOLDER,
    USERNAME_PLACEHOLDER,
    describe_case,
    normalise_case,
)


logger = logging.getLogger(__name__)

DESIGNER_PROMPT = """You are a professional QA engineer designing end-to-end UI test cases.

Feature under test: {feature}

Requirements for this feature (each has a stable id — every test case MUST
reference one via requirement_id):
{requirements}

Observed application map (real pages and controls discovered by exploring the
live application; prefer these exact labels for click targets):
{app_map}

Credentials: {credentials_note}

Application context supplied by the user (authoritative for account roles and
access — e.g. which role the credentials belong to):
{application_context}

Where these requirements live in the application, according to map analysis
(detected account role: {detected_role}):
{feature_locations}
Design each case against the pages listed as "found" for its requirement.
A requirement marked "not found" because it needs a different role belongs in
"blocked"; one marked "not found — unexplored" deserves at most a cautious
navigation-level case.
{critique_note}
ROLE RULE: design ONLY test cases the supplied account can actually execute.
If a requirement can only be verified by a role, account, or screen that is
not accessible according to the context and the application map, do NOT design
steps for it — list it under "blocked" with its requirement_id and a short
reason instead. A blocked requirement is reported honestly; a doomed test case
is noise.

Design focused test cases for THIS feature only. Cover the happy path and, when
the requirements imply them, negative and edge cases. Keep each case short and
independently runnable from a fresh browser session.

CRITICAL: every case starts in a NEW browser that is LOGGED OUT at the start
URL. A case that needs an authenticated or prepared state must include the
login (and any setup clicks) as its own first steps — never assume state left
over from another case.

Allowed step types (the runner is deterministic, no other types exist):
- {{"type": "navigate", "target": "<absolute or relative URL>"}}
- {{"type": "click", "target": "<visible label of the control>"}}
- {{"type": "fill", "target": "<field placeholder/name/label>", "value": "<text>"}}
- {{"type": "upload", "target": "<visible label of the upload control, or empty>",
   "value": "<asset name from the list below>"}}
Stored test assets available for upload steps: {assets_note}
Design an upload step ONLY when the application map shows a file input or an
upload control (e.g. "Choose Files", "Upload"), and only with a listed asset.
Use "{username_placeholder}" and "{password_placeholder}" as fill values when
credentials are needed; never invent real-looking credentials.
Fill targets are matched against the field's placeholder, name, id, label, and
input type. Use the exact form-field names from the application map when
present. For unlabeled login fields, use the target "email" (or "username")
and "password" — the runner resolves those by input type.

Allowed expectation types (checked after the final step):
- {{"type": "url_contains", "value": "<substring of the expected URL>"}}
- {{"type": "text_visible", "value": "<text expected on the page>"}}
- {{"type": "element_visible", "value": "<visible label of an expected control>"}}
GROUNDING RULE: text_visible and element_visible values must be copied verbatim
from the application map or from the requirements (e.g. a documented error
message). If you do not know the exact UI text, assert url_contains instead —
never guess interface copy.

Return exactly one JSON object:
{{
  "test_cases": [
    {{
      "id": "auth-login-valid",
      "requirement_id": "<one of the ids above>",
      "title": "...",
      "description": "One or two plain sentences a non-technical reader understands: what this test does in the product and what outcome proves it works.",
      "priority": "critical",
      "preconditions": ["..."],
      "steps": [ ... ],
      "expected": [ ... ]
    }}
  ],
  "blocked": [
    {{"requirement_id": "<id>", "reason": "<state the specific role or access this account lacks — leave this list EMPTY unless the context above explicitly says the account cannot reach the screen>"}}
  ]
}}"""


class TestDesigner:
    """Design executable test cases for a single feature."""

    # The Test* name describes the agent, not a pytest suite.
    __test__ = False

    def __init__(self, observability=None, model_client: JsonModelClient | None = None):
        self.observability = observability or NoopObservability()
        self.model_client = model_client or JsonModelClient(self.observability)

    def design(
        self,
        feature: str,
        requirements: list[dict],
        app_map: dict,
        start_url: str,
        has_credentials: bool = False,
        critique: str = "",
        application_context: str = "",
        feature_locations: list[dict] | None = None,
        detected_role: str = "",
        assets: list[str] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Return ``(test_cases, blocked_requirements)`` for one feature."""
        with self.observability.span(
            "Test Designer",
            input={"feature": feature, "requirement_count": len(requirements)},
        ) as span:
            payload = self.model_client.complete_json(
                "Test Designer.design_cases",
                self._prompt(feature, requirements, app_map, has_credentials, critique,
                             application_context, feature_locations, detected_role, assets),
            )
            cases = self._from_payload(payload, feature)
            blocked = self._blocked_from_payload(payload, feature, requirements)
            if not cases and not blocked:
                cases = self._fallback(feature, requirements, app_map, start_url)
            corpus = self._grounding_corpus(requirements, app_map)
            cases = [self._ground_case(case, corpus, start_url) for case in cases]
            # A reviewer must be able to understand every case without reading
            # its JSON. Grounding may alter expectations, so describe last.
            cases = [
                case if case.get("description") else {**case, "description": describe_case(case)}
                for case in cases
            ]
            span.update(output={"case_count": len(cases), "blocked_count": len(blocked)})
        logger.info(
            "Test Designer produced %s case(s), %s blocked requirement(s) for feature %r",
            len(cases), len(blocked), feature,
        )
        return cases, blocked

    def _prompt(
        self,
        feature: str,
        requirements: list[dict],
        app_map: dict,
        has_credentials: bool,
        critique: str,
        application_context: str,
        feature_locations: list[dict] | None = None,
        detected_role: str = "",
        assets: list[str] | None = None,
    ) -> str:
        requirement_lines = json.dumps(
            [
                {
                    "id": requirement.get("id"),
                    "title": requirement.get("title"),
                    "description": requirement.get("description", ""),
                    "acceptance_criteria": requirement.get("acceptance_criteria", []),
                    "priority": requirement.get("priority", "medium"),
                }
                for requirement in requirements
            ],
            indent=2,
            ensure_ascii=False,
        ) if requirements else "None supplied — design smoke tests from the application map."
        critique_note = (
            f"\nA coverage review found gaps in the previous design round. Address them:\n{critique}\n"
            if critique
            else ""
        )
        location_lines = []
        for item in feature_locations or []:
            status = item.get("status", "uncertain")
            if status == "found":
                line = f"- [{item.get('requirement_id')}] FOUND on: {', '.join(item.get('pages', []))}"
                if item.get("evidence"):
                    line += f" (evidence: {item['evidence']})"
            else:
                line = f"- [{item.get('requirement_id')}] {status.upper()}"
                if item.get("note"):
                    line += f": {item['note']}"
            location_lines.append(line)
        return DESIGNER_PROMPT.format(
            detected_role=detected_role or "unknown",
            feature_locations="\n".join(location_lines) or "No map analysis available.",
            assets_note=", ".join(assets) if assets else
            "none stored — do NOT design upload steps",
            feature=feature or "general application smoke",
            requirements=requirement_lines,
            app_map=AppMapBuilder.compact_text(app_map) or "No exploration data available.",
            credentials_note=(
                "a username and password ARE available via the placeholders below"
                if has_credentials
                else "no credentials are available; do not design login-dependent cases unless required"
            ),
            application_context=application_context.strip() or "None supplied.",
            critique_note=critique_note,
            username_placeholder=USERNAME_PLACEHOLDER,
            password_placeholder=PASSWORD_PLACEHOLDER,
        )

    @staticmethod
    def _from_payload(payload: dict | None, feature: str) -> list[dict]:
        if not payload or not isinstance(payload.get("test_cases"), list):
            return []
        cases = []
        for index, raw in enumerate(payload["test_cases"], start=1):
            case = normalise_case(raw, fallback_id=f"{feature or 'case'}-{index}")
            if case:
                cases.append(case)
            else:
                logger.warning("Test Designer dropped an invalid case for feature %r", feature)
        return cases

    @staticmethod
    def _blocked_from_payload(
        payload: dict | None, feature: str, requirements: list[dict]
    ) -> list[dict]:
        """Keep only blocked entries that reference a real requirement."""
        if not payload or not isinstance(payload.get("blocked"), list):
            return []
        by_id = {requirement.get("id"): requirement for requirement in requirements}
        blocked = []
        for entry in payload["blocked"]:
            if not isinstance(entry, dict):
                continue
            requirement_id = str(entry.get("requirement_id") or "").strip()
            requirement = by_id.get(requirement_id)
            if not requirement:
                continue
            blocked.append(
                {
                    "requirement_id": requirement_id,
                    "feature": feature,
                    "title": requirement.get("title", ""),
                    "reason": str(entry.get("reason") or "").strip()
                    or "not executable with the supplied account",
                }
            )
        return blocked

    # ------------------------------------------------------------------
    # Expectation grounding
    # ------------------------------------------------------------------

    @staticmethod
    def _grounding_corpus(requirements: list[dict], app_map: dict) -> str:
        """All text the application or the requirements are known to contain."""
        parts: list[str] = []
        for requirement in requirements:
            parts.append(str(requirement.get("title", "")))
            parts.append(str(requirement.get("description", "")))
            parts.extend(str(criterion) for criterion in requirement.get("acceptance_criteria", []))
        for page in app_map.get("pages", []):
            parts.append(str(page.get("title", "")))
            parts.append(str(page.get("url", "")))
            parts.extend(str(action.get("label", "")) for action in page.get("actions", []))
            parts.extend(str(control) for control in page.get("controls", []))
            for field in page.get("fields", []):
                parts.append(str(field.get("name", "")))
                parts.append(str(field.get("placeholder", "")))
            parts.extend(str(fill.get("field", "")) for fill in page.get("fills", []))
        return "\n".join(part for part in parts if part).casefold()

    @staticmethod
    def _ground_case(case: dict, corpus: str, start_url: str) -> dict:
        """Drop expectations nothing observed or documented supports.

        This includes URLs: models invent URL schemes ("slide-1") that no
        observed page ever had, producing false failures.
        """
        domain = (urlparse(start_url).netloc or start_url).casefold()
        grounded = []
        for expectation in case.get("expected", []):
            value = expectation["value"].casefold()
            if expectation["type"] == "url_contains":
                if value in corpus or domain in value or value in start_url.casefold():
                    grounded.append(expectation)
                else:
                    logger.info(
                        "Dropped ungrounded URL expectation %r from case %s",
                        expectation["value"], case.get("id"),
                    )
                continue
            if value in corpus:
                grounded.append(expectation)
            else:
                logger.info(
                    "Dropped ungrounded expectation %r from case %s",
                    expectation["value"], case.get("id"),
                )
        if not grounded:
            # A case must still verify something: at minimum the run stayed on
            # the application rather than an error page.
            grounded = [{"type": "url_contains", "value": urlparse(start_url).netloc or start_url}]
        return {**case, "expected": grounded}

    @staticmethod
    def _fallback(feature: str, requirements: list[dict], app_map: dict, start_url: str) -> list[dict]:
        """One deterministic smoke case per requirement (or one for the app).

        Assertions stay URL-based on purpose: a page's tab title is often not
        rendered as visible text, so asserting it produces false failures.
        """
        domain = urlparse(start_url).netloc or start_url

        def smoke(case_id: str, requirement: dict | None) -> dict:
            # Assert only that the application answered: a deeper page may
            # redirect to login, which is not a failure of reachability.
            target_url = (requirement or {}).get("source_url") or start_url
            expected = [{"type": "url_contains", "value": domain}]
            case = {
                "id": case_id,
                "requirement_id": (requirement or {}).get("id", ""),
                "title": f"Smoke: {(requirement or {}).get('title', 'application loads')}",
                "priority": (requirement or {}).get("priority", "medium"),
                "preconditions": [],
                "steps": [{"type": "navigate", "target": target_url}],
                "expected": expected,
            }
            return {**case, "description": describe_case(case)}

        if requirements:
            return [
                smoke(f"{requirement['id']}-smoke", requirement) for requirement in requirements
            ]
        return [smoke(f"{feature or 'app'}-smoke", None)]
