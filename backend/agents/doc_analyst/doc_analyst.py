"""Doc Analyst agent: one product document in, structured requirements out.

The orchestrator fans out one Doc Analyst per document, so this module never
sees the whole corpus.  When the model is unavailable, a deterministic
heading-based fallback keeps the pipeline moving.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agents.llm_client import JsonModelClient
from observability.tracing import NoopObservability

from .models import VALID_PRIORITIES, Requirement


logger = logging.getLogger(__name__)

MAX_DOC_CHARS = 24_000

ANALYST_PROMPT = """You are a senior QA analyst. Extract the testable product requirements
from the document below.

Rules:
- Extract only behaviour a QA engineer could verify through the product's UI or API.
- Group related requirements under a short feature name (e.g. "authentication", "checkout").
- Each requirement needs a concise title, a one-to-three sentence description, and
  concrete acceptance criteria phrased as verifiable statements.
- priority must be one of: critical, high, medium, low.
- Do not invent behaviour that is not stated or clearly implied by the document.

Document name: {doc_name}

Document content:
---
{doc_text}
---

Return exactly one JSON object:
{{
  "requirements": [
    {{
      "feature": "authentication",
      "title": "User can log in with valid credentials",
      "description": "...",
      "acceptance_criteria": ["..."],
      "priority": "critical"
    }}
  ]
}}"""


class DocAnalyst:
    """Extract requirements from a single document."""

    def __init__(self, observability=None, model_client: JsonModelClient | None = None):
        self.observability = observability or NoopObservability()
        self.model_client = model_client or JsonModelClient(self.observability)

    def analyze(self, doc_path: str) -> list[dict]:
        path = Path(doc_path)
        text = self._read_document(path)
        if not text.strip():
            logger.warning("Doc Analyst: %s is empty or unreadable; skipping", doc_path)
            return []

        with self.observability.span(
            "Doc Analyst", input={"doc": path.name, "chars": len(text)}
        ) as span:
            payload = self.model_client.complete_json(
                "Doc Analyst.extract_requirements",
                ANALYST_PROMPT.format(doc_name=path.name, doc_text=text[:MAX_DOC_CHARS]),
            )
            requirements = self._from_payload(payload, path) or self._fallback(text, path)
            span.update(output={"requirement_count": len(requirements)})
        logger.info("Doc Analyst extracted %s requirement(s) from %s", len(requirements), path.name)
        return [requirement.to_dict() for requirement in requirements]

    @staticmethod
    def _read_document(path: Path) -> str:
        if not path.is_file():
            logger.warning("Doc Analyst: document %s does not exist", path)
            return ""
        if path.suffix.casefold() == ".pdf":
            try:
                from pypdf import PdfReader

                return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
            except Exception as exc:
                logger.warning("Doc Analyst: could not read PDF %s (%s)", path.name, exc)
                return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Doc Analyst: could not read %s (%s)", path.name, exc)
            return ""

    def _from_payload(self, payload: dict | None, path: Path) -> list[Requirement]:
        if not payload or not isinstance(payload.get("requirements"), list):
            return []
        slug = self._slug(path)
        requirements = []
        for index, item in enumerate(payload["requirements"], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            criteria = item.get("acceptance_criteria")
            priority = str(item.get("priority") or "medium").casefold()
            requirements.append(
                Requirement(
                    id=f"{slug}-R{index}",
                    feature=str(item.get("feature") or "general").strip().casefold() or "general",
                    title=title,
                    description=str(item.get("description") or "").strip(),
                    acceptance_criteria=[
                        str(criterion).strip()
                        for criterion in (criteria if isinstance(criteria, list) else [])
                        if str(criterion).strip()
                    ],
                    priority=priority if priority in VALID_PRIORITIES else "medium",
                    source_doc=path.name,
                )
            )
        return requirements

    def _fallback(self, text: str, path: Path) -> list[Requirement]:
        """Turn markdown headings (or leading paragraph lines) into requirements.

        Deliberately coarse: it preserves traceability to the document even
        when no model is available, at the cost of granularity.
        """
        slug = self._slug(path)
        headings = [
            (match.group("title").strip(), match.end())
            for match in re.finditer(r"^#{1,3}\s+(?P<title>.+)$", text, re.MULTILINE)
        ]
        requirements: list[Requirement] = []
        if headings:
            for index, (title, start) in enumerate(headings, start=1):
                end = headings[index][1] - len(title) - 4 if index < len(headings) else len(text)
                body = " ".join(text[start:end].split())[:400]
                requirements.append(
                    Requirement(
                        id=f"{slug}-R{index}",
                        feature=title.casefold(),
                        title=title,
                        description=body,
                        source_doc=path.name,
                    )
                )
        else:
            paragraphs = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
            for index, paragraph in enumerate(paragraphs[:10], start=1):
                first_line = paragraph.splitlines()[0][:120]
                requirements.append(
                    Requirement(
                        id=f"{slug}-R{index}",
                        feature="general",
                        title=first_line,
                        description=" ".join(paragraph.split())[:400],
                        source_doc=path.name,
                    )
                )
        logger.info("Doc Analyst used the heading fallback for %s", path.name)
        return requirements

    @staticmethod
    def _slug(path: Path) -> str:
        return re.sub(r"[^a-z0-9]+", "-", path.stem.casefold()).strip("-") or "doc"
