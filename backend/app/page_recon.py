"""Live inventory of every mapped page: what controls really exist, by name.

Exploration discovers which pages exist; recon walks those pages once and
records their real, verifiable control names (visible text AND accessible
names), input fields, and control roles.  The Test Verifier then checks every
designed step against this inventory, which is what stops the Designer from
shipping steps like ``click "Log In"`` at an app whose button says "Sign in".
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MAX_PAGES = 20
MAX_CONTROLS_PER_PAGE = 60
PAGE_TIMEOUT_MS = 20_000


class PageRecon:
    """Visit mapped URLs with one logged-in browser and inventory their controls."""

    def __init__(self, observability=None, headless: bool | None = None):
        import os

        from observability.tracing import NoopObservability

        self.observability = observability or NoopObservability()
        if headless is None:
            headless = os.getenv("QA_RECON_HEADLESS", "true").strip().casefold() != "false"
        self.headless = headless

    def inventory(self, start_url: str, urls: list[str], username: str = "",
                  password: str = "") -> dict[str, dict]:
        """Return ``{url: {controls, fields, roles, title}}`` for reachable pages."""
        from browser import BrowserController

        pages: dict[str, dict] = {}
        browser = BrowserController()
        with self.observability.span("Page Recon", input={"page_count": len(urls)}) as span:
            try:
                browser.start(observability=self.observability, headless=self.headless)
                # Logged-out pages first: once authenticated, the login page
                # redirects to the dashboard and its controls become invisible
                # to recon — which is exactly where "Sign in" lives.
                logged_out_urls = [url for url in urls[:MAX_PAGES] if self._is_public(url)]
                for url in logged_out_urls or [start_url]:
                    entry = self._inventory_page(browser, url)
                    if entry:
                        pages[url] = entry

                if username and password and not self._login(browser, username, password):
                    logger.warning("Recon could not log in; inventorying public pages only")
                for url in urls[:MAX_PAGES]:
                    if url in pages:
                        continue
                    entry = self._inventory_page(browser, url)
                    if entry:
                        pages[url] = entry
            except Exception:
                logger.warning("Page recon failed; continuing without it", exc_info=True)
            finally:
                try:
                    browser.close()
                except Exception:
                    logger.debug("Could not close the recon browser", exc_info=True)
            span.update(output={"pages_inventoried": len(pages)})
        logger.info("Page recon inventoried %s page(s)", len(pages))
        return pages

    @staticmethod
    def _is_public(url: str) -> bool:
        """Pages a logged-out visitor sees — inventory these before logging in."""
        lowered = url.casefold()
        return any(
            marker in lowered
            for marker in ("/login", "/signin", "/sign-in", "/register", "/forgot")
        ) or lowered.rstrip("/").count("/") <= 2

    @staticmethod
    def _login(browser, username: str, password: str) -> bool:
        filled = browser.fill_input("email", username) or browser.fill_input("username", username)
        if not (filled and browser.fill_input("password", password)):
            return False
        before = browser.current_url()
        for label in ("Sign in", "Sign In", "Log In", "Login", "Submit"):
            try:
                browser.page.get_by_role("button", name=label).first.click(timeout=2_000)
                break
            except Exception:
                continue
        else:
            return False
        browser.wait_for_url_change(before, timeout_ms=15_000)
        return browser.current_url() != before

    def _inventory_page(self, browser, url: str) -> dict | None:
        try:
            browser.open(url)
            browser.wait_for_page_ready(timeout_ms=PAGE_TIMEOUT_MS)
            page = browser.page
            controls: list[str] = []
            roles: dict[str, str] = {}

            for selector, role in (
                ("button:visible", "button"),
                ("a[href]:visible", "link"),
                ("[role='button']:visible", "button"),
                ("[role='tab']:visible", "tab"),
                ("select:visible", "combobox"),
                ("[role='combobox']:visible", "combobox"),
                ("input[type='checkbox']:visible", "checkbox"),
                ("input[type='file']", "file_upload"),
            ):
                elements = page.locator(selector)
                for index in range(min(elements.count(), MAX_CONTROLS_PER_PAGE)):
                    element = elements.nth(index)
                    for name in self._names(element):
                        if name and name not in controls and len(controls) < MAX_CONTROLS_PER_PAGE:
                            controls.append(name)
                            if role != "button" or name not in roles:
                                roles[name] = role

            fields = []
            inputs = page.locator("input:visible, textarea:visible")
            for index in range(min(inputs.count(), 30)):
                element = inputs.nth(index)
                field = {
                    "type": element.get_attribute("type") or "text",
                    "name": element.get_attribute("name") or "",
                    "placeholder": element.get_attribute("placeholder")
                    or element.get_attribute("aria-label") or "",
                }
                if (field["name"] or field["placeholder"]) and field not in fields:
                    fields.append(field)

            return {
                "url": browser.current_url(),
                "title": page.title(),
                "controls": controls,
                "fields": fields,
                "roles": roles,
            }
        except Exception as exc:
            logger.info("Recon skipped %s (%s)", url, type(exc).__name__)
            return None

    @staticmethod
    def _names(element) -> list[str]:
        """Every name a test could legitimately use to target this control."""
        names = []
        for getter in (
            lambda: element.inner_text(timeout=500),
            lambda: element.get_attribute("aria-label"),
            lambda: element.get_attribute("title"),
        ):
            try:
                value = " ".join(str(getter() or "").split())
            except Exception:
                value = ""
            if value and len(value) <= 80 and value not in names:
                names.append(value)
        return names
