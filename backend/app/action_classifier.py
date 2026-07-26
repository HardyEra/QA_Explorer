"""Deterministic action classification and ranking for QA exploration."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import logging
from typing import Iterable

from models import Action


logger = logging.getLogger(__name__)

# Categories are checked in this order. Specific business intents must win over
# generic navigation words such as "cart" or "account".
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "logout": ("logout", "log out", "sign out"),
    "checkout": ("checkout", "place order", "complete order", "payment"),
    "purchase": (
        "add to cart", "add to bag", "buy now", "shopping cart", "view cart",
        "order now", "purchase",
    ),
    "authentication": (
        "log in", "login", "sign in", "signin", "sign up", "signup",
        "register", "create account", "forgot password", "reset password",
    ),
    "create": (
        "add candidate", "create candidate", "create job", "new candidate",
        "new job", "add", "create", "new",
    ),
    "delete": ("delete", "remove", "archive", "discard"),
    "edit": ("edit", "update", "modify", "change"),
    "upload": ("upload", "attach", "import"),
    "download": ("download", "export", "get file"),
    "submit": ("submit", "apply", "confirm"),
    "save": ("save", "continue", "next", "proceed", "generate", "invite"),
    "search": ("search", "find products", "find items"),
    "filter": ("filter", "filters", "dropdown", "sort", "sorting", "tab"),
    "menu": ("menu", "open menu", "hamburger", "more options", "more"),
    "profile": ("user avatar", "avatar", "profile", "my account"),
    "settings": ("settings", "preferences", "account settings"),
    "notifications": ("notifications", "notification", "alerts"),
    "candidate_card": ("candidate card", "candidate profile", "candidate details"),
    "view": ("view", "open", "details", "preview", "show"),
    "social": (
        "twitter", "facebook", "linkedin", "instagram", "youtube", "tiktok",
        "pinterest", "discord",
    ),
    "footer": (
        "privacy", "privacy policy", "terms", "terms of service", "careers",
        "copyright", "accessibility", "cookie policy", "legal",
    ),
    "external": ("external", "visit website", "company website", "partner site"),
    "navigation": (
        "home", "products", "catalog", "shop", "about", "contact",
        "back", "next", "previous", "view details", "learn more",
    ),
    "help": ("help", "support"),
}

CATEGORY_PRIORITY: dict[str, int] = {
    "create": 100,
    "purchase": 99,
    "checkout": 98,
    "submit": 96,
    "save": 96,
    "upload": 96,
    "download": 70,
    "authentication": 95,
    "logout": 5,
    "delete": 90,
    "edit": 90,
    "view": 60,
    "search": 70,
    "navigation": 60,
    "unknown": 50,
    "filter": 40,
    "menu": 25,
    "candidate_card": 20,
    "notifications": 15,
    "profile": 10,
    "settings": 5,
    "help": 5,
    "navigation_module": 60,
    "dialog_action": 50,
    "form_action": 50,
    "toolbar_action": 25,
    "header_action": 25,
    "footer": 5,
    "social": 1,
    "external": 1,
}


@dataclass(frozen=True)
class ClassifiedAction:
    """An extracted action enriched solely with deterministic decisions."""

    id: int
    text: str
    type: str
    category: str
    priority: int

    @classmethod
    def from_action(cls, action: Action, category: str) -> "ClassifiedAction":
        return cls(
            id=action.id,
            text=action.text,
            type=action.type,
            category=category,
            priority=CATEGORY_PRIORITY[category],
        )


class ActionClassifier:
    """Classify visible action labels using an extensible keyword mapping."""

    def classify(self, action: Action) -> ClassifiedAction:
        normalized_text = self._semantic_text(action)
        text_category = next(
            (
                category
                for category, keywords in CATEGORY_KEYWORDS.items()
                if any(keyword in normalized_text for keyword in keywords)
            ),
            "unknown",
        )
        category = self._context_category(action, text_category)
        return ClassifiedAction.from_action(action, category)

    def classify_all(self, actions: Iterable[Action]) -> list[ClassifiedAction]:
        return [self.classify(action) for action in actions]

    @staticmethod
    def _semantic_text(action: Action) -> str:
        """Normalize every user-perceivable label exposed by an action.

        The current extractors place the accessible name, visible text,
        ``aria-label``, title, and fallback placeholder into ``Action.text``.
        The optional attributes keep classification correct when a richer
        extractor supplies those fields separately in the future.
        """
        values = [action.text]
        for field in ("aria_label", "title", "placeholder", "accessible_name", "name"):
            value = getattr(action, field, None)
            if value:
                values.append(str(value))
        return " ".join(" ".join(values).casefold().split())

    @staticmethod
    def _context_category(action: Action, text_category: str) -> str:
        """Use extracted layout semantics when text alone is ambiguous."""
        context = (action.container_context or "").casefold()
        # A header avatar/menu is account navigation even if its label is only
        # an icon name. Preserve explicit authentication and logout intent.
        if context == "profile" and text_category not in {"authentication", "logout"}:
            return "profile"
        if context in {"navigation", "sidebar", "menu"} and text_category in {"unknown", "view", "navigation"}:
            return "navigation_module"
        if context == "dialog" and text_category == "unknown":
            return "dialog_action"
        if context == "form" and text_category == "unknown":
            return "form_action"
        if context == "toolbar" and text_category == "unknown":
            return "toolbar_action"
        if context == "header" and text_category == "unknown":
            return "header_action"
        return text_category


class ActionRanker:
    """Sort actions by priority while preserving page order for score ties."""

    NEW_ACTION_BONUS = 35
    PERSISTENT_UI_PENALTY = 30
    PERSISTENT_UI_CATEGORIES = {
        "navigation", "menu", "profile", "settings", "notifications",
        "filter", "candidate_card", "social", "footer", "help", "external",
        "navigation_module", "toolbar_action", "header_action",
    }

    def __init__(self) -> None:
        self._previous_action_keys: set[tuple[str, str, str]] | None = None

    def rank(self, actions: Iterable[ClassifiedAction]) -> list[ClassifiedAction]:
        actions = list(actions)
        current_keys = {self._action_key(action) for action in actions}
        new_keys = (
            current_keys - self._previous_action_keys
            if self._previous_action_keys is not None
            else set()
        )
        content_fully_explored = not any(
            action.category not in self.PERSISTENT_UI_CATEGORIES for action in actions
        )

        ranked = []
        for action in actions:
            score = self._score(action)
            key = self._action_key(action)
            if key in new_keys and action.category not in self.PERSISTENT_UI_CATEGORIES:
                score += self.NEW_ACTION_BONUS
                logger.info("Newly introduced action: %s (+%s)", action.text, self.NEW_ACTION_BONUS)
            elif action.category in self.PERSISTENT_UI_CATEGORIES and not content_fully_explored:
                # A navigation module can be the required entry point to an
                # explicit workflow (for example, Candidates). Keep its base
                # score while still withholding the new-content bonus.
                penalty = 0 if action.category == "navigation_module" else self.PERSISTENT_UI_PENALTY
                score -= penalty
                if penalty:
                    logger.info(
                        "Deferring persistent UI action: %s (-%s)",
                        action.text,
                        penalty,
                    )
            ranked.append(replace(action, priority=score))

        ranked.sort(key=lambda action: action.priority, reverse=True)
        for action in ranked:
            logger.info("%s (%s,%s)", action.text, action.category, action.priority)
        self._previous_action_keys = current_keys
        return ranked

    @staticmethod
    def _action_key(action: ClassifiedAction) -> tuple[str, str, str]:
        return (
            action.category,
            " ".join(action.text.casefold().split()),
            action.type,
        )

    @staticmethod
    def _score(action: ClassifiedAction) -> int:
        """Apply small intent-specific differences inside each category."""
        text = " ".join(action.text.casefold().split())

        if action.category == "create":
            if "candidate" in text:
                return 100
            if text.startswith(("add ", "create ")):
                return 98
            return 97
        if action.category == "workflow":
            return 98 if "checkout" in text else 96
        if action.category == "authentication":
            return 97 if any(term in text for term in ("login", "log in", "sign in")) else 95
        return CATEGORY_PRIORITY[action.category]
