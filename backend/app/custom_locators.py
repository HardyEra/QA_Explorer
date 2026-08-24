"""Turn user-pasted element HTML into narrow, stable custom locator fallbacks."""

from __future__ import annotations

import re


_TAG_PATTERN = re.compile(r"^\s*<(?P<tag>[a-z][\w:-]*)\b", re.IGNORECASE)
_ATTRIBUTE_PATTERN = re.compile(
    r"(?P<name>[\w:-]+)\s*=\s*(?P<quote>['\"])(?P<value>.*?)\2", re.DOTALL
)
_SELECTOR_ATTRIBUTES = ("data-test", "data-testid", "id", "aria-label", "name")


def build_custom_locator(name: str, element_html: str) -> dict[str, str]:
    """Return a safe CSS locator derived only from stable semantic attributes."""
    label = " ".join(name.split())
    if not label:
        raise ValueError("Give the custom control a short name, such as 'cart'.")
    if not _TAG_PATTERN.search(element_html):
        raise ValueError("Paste one HTML element, starting with an opening tag such as <a ...>.")

    attributes = {
        match.group("name").casefold(): match.group("value").strip()
        for match in _ATTRIBUTE_PATTERN.finditer(element_html)
        if match.group("value").strip()
    }
    for attribute in _SELECTOR_ATTRIBUTES:
        value = attributes.get(attribute)
        if value:
            return {
                "name": label,
                "selector": f'[{attribute}="{_css_escape(value)}"]',
                "attribute": attribute,
            }
    raise ValueError(
        "The pasted element needs one stable attribute: data-test, data-testid, id, aria-label, or name."
    )


def _css_escape(value: str) -> str:
    """Escape a string for the quoted CSS attribute selector we generate."""
    return value.replace("\\", "\\\\").replace('"', '\\"')
