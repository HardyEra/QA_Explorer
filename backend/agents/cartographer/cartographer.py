"""Cartographer agent: make sense of the explored map against the PRD.

Sits between exploration and test design.  It reads the App Map (URLs, page
titles, button labels) alongside the extracted requirements and produces a
semantic map: which screen implements which requirement, which requirements
were never found (unexplored, or hidden behind another role), and what role
the supplied account appears to hold.  A deterministic keyword matcher backs
the model so the pipeline always gets an answer.
"""

from __future__ import annotations

import json
import logging
import re

from agents.llm_client import JsonModelClient
from app_map_builder import AppMapBuilder
from observability.tracing import NoopObservability


logger = logging.getLogger(__name__)

FEATURE_STATUSES = {"found", "not_found", "uncertain"}
_TOKEN_STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
    "are", "can", "must", "shall", "be", "user", "users", "system", "platform",
}

CARTOGRAPHER_PROMPT = """You are a QA cartographer. Your job is to connect a product's
requirements to the real screens discovered by exploring the live application.

Requirements (from the product documents):
{requirements}

Observed application map (real URLs, page titles, and control labels):
{app_map}

Account context supplied by the user:
{application_context}

For every requirement, decide:
- status "found": one or more observed pages clearly implement it (name them).
- status "not_found": nothing observed implements it — say whether that is
  more likely because it needs a different role/account or because exploration
  has not reached it yet.
- status "uncertain": partial or ambiguous evidence.
Base every judgement ONLY on the observed URLs, titles, and control labels —
do not invent screens that are not in the map.

Also infer which role the logged-in account appears to hold, from the screens
it could reach.

Return exactly one JSON object:
{{
  "detected_role": "short role name or 'unknown'",
  "features": [
    {{
      "requirement_id": "<id>",
      "status": "found",
      "pages": ["<url from the map>"],
      "evidence": "which observed labels/URLs support this",
      "note": "for not_found: likely reason (other role vs unexplored)"
    }}
  ]
}}"""


SYNTHESIS_PROMPT = """You are a QA analyst. No product documents were supplied, so the observed
application itself is the only source of truth. From the exploration map below,
derive the testable requirements this application evidently has.

Observed application map (real URLs, page titles, control labels, and form
fields discovered in the live application):
{app_map}

Account context supplied by the user:
{application_context}

Rules:
- One requirement per distinct module or capability you can see evidence for:
  navigation entries (e.g. sidebar modules), forms, search, authentication.
- IMPORTANT: when a page exposes a set of module navigation entries (a sidebar
  or menu such as "Jobs", "Candidates", "Clients"), create ONE requirement PER
  module entry — "Candidates module opens and shows its content", "Jobs module
  opens and shows its content", and so on. Never collapse the modules into a
  single generic navigation requirement; each module must be independently
  testable so the report can state per module whether it works.
- Base every requirement ONLY on observed controls/pages — do not invent
  features that leave no trace in the map.
- Name the feature after the observed module (e.g. "candidates", "jobs").
- Acceptance criteria must be phrased so a UI test could verify them using the
  observed labels.
- priority: critical for authentication/core flows, high for main modules,
  medium/low for secondary UI.

Return exactly one JSON object:
{{
  "requirements": [
    {{
      "feature": "candidates",
      "title": "Candidates module opens and lists candidates",
      "description": "The sidebar exposes a 'Candidates' entry ...",
      "acceptance_criteria": ["Clicking 'Candidates' opens a page ..."],
      "priority": "high"
    }}
  ]
}}"""


class Cartographer:
    """Map requirements onto observed screens; detect the account's role."""

    def __init__(self, observability=None, model_client: JsonModelClient | None = None):
        self.observability = observability or NoopObservability()
        self.model_client = model_client or JsonModelClient(self.observability)

    def map_features(
        self,
        requirements: list[dict],
        app_map: dict,
        application_context: str = "",
    ) -> dict:
        if not requirements:
            return {"detected_role": "unknown", "features": []}
        with self.observability.span(
            "Cartographer",
            input={
                "requirement_count": len(requirements),
                "page_count": len(app_map.get("pages", [])),
            },
        ) as span:
            payload = self.model_client.complete_json(
                "Cartographer.map_features",
                CARTOGRAPHER_PROMPT.format(
                    requirements=json.dumps(
                        [
                            {
                                "id": requirement.get("id"),
                                "feature": requirement.get("feature"),
                                "title": requirement.get("title"),
                                "description": requirement.get("description", "")[:300],
                            }
                            for requirement in requirements
                        ],
                        indent=2,
                        ensure_ascii=False,
                    ),
                    app_map=AppMapBuilder.compact_text(app_map) or "No pages observed yet.",
                    application_context=application_context.strip() or "None supplied.",
                ),
            )
            semantic_map = self._from_payload(payload, requirements, app_map)
            if semantic_map is None:
                semantic_map = self._fallback(requirements, app_map)
            span.update(
                output={
                    "detected_role": semantic_map["detected_role"],
                    "found": sum(
                        1 for item in semantic_map["features"] if item["status"] == "found"
                    ),
                }
            )
        logger.info(
            "Cartographer: role=%s; %s/%s requirement(s) located on observed screens",
            semantic_map["detected_role"],
            sum(1 for item in semantic_map["features"] if item["status"] == "found"),
            len(requirements),
        )
        return semantic_map

    def synthesize_requirements(
        self, app_map: dict, application_context: str = ""
    ) -> list[dict]:
        """Derive requirements from the observed application when no docs exist.

        This is what lets exploration-only runs still produce a per-feature
        test plan and a coverage table instead of three shallow smoke tests.
        """
        if not app_map.get("pages"):
            return []
        with self.observability.span(
            "Cartographer.synthesize", input={"page_count": len(app_map["pages"])}
        ) as span:
            payload = self.model_client.complete_json(
                "Cartographer.synthesize_requirements",
                SYNTHESIS_PROMPT.format(
                    app_map=AppMapBuilder.compact_text(app_map),
                    application_context=application_context.strip() or "None supplied.",
                ),
            )
            requirements = self._requirements_from_payload(payload)
            if not requirements:
                requirements = self._synthesis_fallback(app_map)
            span.update(output={"requirement_count": len(requirements)})
        logger.info(
            "Cartographer synthesized %s requirement(s) from the observed application",
            len(requirements),
        )
        return requirements

    @staticmethod
    def _requirements_from_payload(payload: dict | None) -> list[dict]:
        if not payload or not isinstance(payload.get("requirements"), list):
            return []
        requirements = []
        for index, item in enumerate(payload["requirements"], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            priority = str(item.get("priority") or "medium").casefold()
            criteria = item.get("acceptance_criteria")
            requirements.append(
                {
                    "id": f"app-R{index}",
                    "feature": str(item.get("feature") or "general").strip().casefold() or "general",
                    "title": title,
                    "description": str(item.get("description") or "").strip(),
                    "acceptance_criteria": [
                        str(criterion).strip()
                        for criterion in (criteria if isinstance(criteria, list) else [])
                        if str(criterion).strip()
                    ],
                    "priority": priority if priority in {"critical", "high", "medium", "low"} else "medium",
                    "source_doc": "observed application",
                }
            )
        return requirements

    @staticmethod
    def _synthesis_fallback(app_map: dict) -> list[dict]:
        """One reachability requirement per observed page."""
        requirements = []
        for index, page in enumerate(app_map.get("pages", []), start=1):
            title = page.get("title") or page.get("url", "")
            controls = ", ".join(page.get("controls", [])[:10])
            requirements.append(
                {
                    "id": f"app-R{index}",
                    "feature": (page.get("title") or "page").casefold(),
                    "title": f"Page '{title}' is reachable and renders its controls",
                    "description": f"Observed controls: {controls}" if controls else "",
                    "acceptance_criteria": [],
                    "priority": "medium",
                    "source_doc": "observed application",
                    # Lets the Designer's fallback target the page itself
                    # instead of sending every smoke test to the start URL.
                    "source_url": page.get("url", ""),
                }
            )
        return requirements

    @staticmethod
    def _from_payload(payload: dict | None, requirements: list[dict], app_map: dict) -> dict | None:
        if not payload or not isinstance(payload.get("features"), list):
            return None
        requirement_ids = {requirement.get("id") for requirement in requirements}
        known_urls = {page.get("url") for page in app_map.get("pages", [])}
        features = []
        for item in payload["features"]:
            if not isinstance(item, dict):
                continue
            requirement_id = str(item.get("requirement_id") or "").strip()
            if requirement_id not in requirement_ids:
                continue
            status = str(item.get("status") or "uncertain").casefold()
            # The cartographer reads the map, it does not draw new pages on it.
            pages = [
                str(url) for url in (item.get("pages") or [])
                if str(url) in known_urls
            ]
            if status == "found" and not pages:
                status = "uncertain"
            features.append(
                {
                    "requirement_id": requirement_id,
                    "status": status if status in FEATURE_STATUSES else "uncertain",
                    "pages": pages,
                    "evidence": str(item.get("evidence") or "").strip(),
                    "note": str(item.get("note") or "").strip(),
                }
            )
        if not features:
            return None
        covered = {feature["requirement_id"] for feature in features}
        for requirement in requirements:
            if requirement.get("id") not in covered:
                features.append(
                    {
                        "requirement_id": requirement.get("id", ""),
                        "status": "uncertain",
                        "pages": [],
                        "evidence": "",
                        "note": "the cartographer model did not report on this requirement",
                    }
                )
        return {
            "detected_role": str(payload.get("detected_role") or "unknown").strip() or "unknown",
            "features": features,
        }

    @classmethod
    def _fallback(cls, requirements: list[dict], app_map: dict) -> dict:
        """Keyword-overlap matching when no model is available."""
        page_tokens = {
            page.get("url", ""): cls._tokens(
                " ".join(
                    [
                        page.get("title", ""),
                        page.get("url", ""),
                        " ".join(page.get("controls", [])),
                        " ".join(action.get("label", "") for action in page.get("actions", [])),
                    ]
                )
            )
            for page in app_map.get("pages", [])
        }
        features = []
        for requirement in requirements:
            wanted = cls._tokens(
                f"{requirement.get('feature', '')} {requirement.get('title', '')}"
            )
            matches = [
                url for url, tokens in page_tokens.items() if len(wanted & tokens) >= 2
            ]
            features.append(
                {
                    "requirement_id": requirement.get("id", ""),
                    "status": "found" if matches else "not_found",
                    "pages": matches,
                    "evidence": "keyword overlap with page titles/controls" if matches else "",
                    "note": "" if matches else "no observed screen matches; needs exploration or another role",
                }
            )
        return {"detected_role": "unknown", "features": features}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        tokens = set()
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            if token in _TOKEN_STOP_WORDS or len(token) < 3:
                continue
            # Singularise so "requests" matches "request".
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            tokens.add(token)
        return tokens
