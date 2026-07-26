"""Deterministic action classification and ranking for QA exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from models import Action


# Categories are checked in this order. Specific business intents must win over
# generic navigation words such as "cart" or "account".
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "purchase": (
        "add to cart", "add to bag", "buy now", "checkout", "shopping cart",
        "view cart", "place order", "payment", "order now",
    ),
    "authentication": (
        "log in", "login", "sign in", "sign up", "register", "create account",
        "forgot password", "reset password", "log out", "logout",
    ),
    "search": ("search", "find products", "find items"),
    "form": ("submit", "continue", "save", "confirm", "apply", "subscribe"),
    "settings": ("settings", "preferences", "profile", "account settings"),
    "menu": ("menu", "open menu", "hamburger", "more options", "more"),
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
        "home", "products", "catalog", "shop", "about", "contact", "help",
        "support", "back", "next", "previous", "view details", "learn more",
    ),
}

CATEGORY_PRIORITY: dict[str, int] = {
    "purchase": 100,
    "authentication": 95,
    "form": 90,
    "search": 80,
    "navigation": 60,
    "unknown": 50,
    "menu": 30,
    "settings": 20,
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
        normalized_text = " ".join(action.text.lower().split())
        category = next(
            (
                category
                for category, keywords in CATEGORY_KEYWORDS.items()
                if any(keyword in normalized_text for keyword in keywords)
            ),
            "unknown",
        )
        return ClassifiedAction.from_action(action, category)

    def classify_all(self, actions: Iterable[Action]) -> list[ClassifiedAction]:
        return [self.classify(action) for action in actions]


class ActionRanker:
    """Sort actions by priority while preserving page order for score ties."""

    def rank(self, actions: Iterable[ClassifiedAction]) -> list[ClassifiedAction]:
        return sorted(actions, key=lambda action: action.priority, reverse=True)
