from playwright.sync_api import sync_playwright
from extractor import PageExtractor
from action_registry import ActionRegistry
from playwright.sync_api import Error, TimeoutError
import logging
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

class BrowserController:

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.extractor = None
        self.action_registry = None
        self.explore_new_tabs = False
        self.follow_external = False
        self.base_domain = None

    def start(self):
        self.playwright = sync_playwright().start()

        self.browser = self.playwright.chromium.launch(
            headless=False
        )

        self.page = self.browser.new_page()
        # Catch popups as soon as the browser context creates them. Checking only
        # after a click misses delayed window.open() calls from social links.
        self.page.context.on("page", self._handle_new_page)

        self.action_registry = ActionRegistry()

        self.extractor = PageExtractor(
            self.page,
            self.action_registry
        )

    def open(self, url):
        self.page.goto(url)

    def set_new_tab_policy(self, explore_new_tabs=False):
        """Choose whether click-triggered popup tabs should remain open."""
        self.explore_new_tabs = explore_new_tabs

    def set_navigation_policy(self, start_url, follow_external=False):
        """Keep default exploration on the start domain and primary page."""
        self.base_domain = urlparse(start_url).netloc
        self.follow_external = follow_external

    def _handle_new_page(self, popup):
        if self.explore_new_tabs or popup == self.page:
            return
        try:
            logger.info("Closing child tab immediately: %s", popup.url)
            popup.close()
        except Error as exc:
            logger.warning("Could not close child tab: %s", exc)

    def _close_child_tabs(self):
        """Defence-in-depth for popups created before Playwright raises its event."""
        if self.explore_new_tabs:
            return
        for popup in self.page.context.pages:
            if popup != self.page:
                self._handle_new_page(popup)

    def _return_to_main_domain(self, previous_url):
        if self.follow_external or not self.base_domain:
            return
        if urlparse(self.page.url).netloc == self.base_domain:
            return
        logger.info("Returning from external URL %s to %s", self.page.url, previous_url)
        try:
            self.page.go_back(wait_until="domcontentloaded", timeout=5000)
        except TimeoutError:
            logger.warning("Timed out returning from external URL; opening previous URL directly")
            self.page.goto(previous_url, wait_until="domcontentloaded")

    def title(self):
        return self.page.title()

    def current_url(self):
        return self.page.url

    def screenshot(self, name="page.png"):
        self.page.screenshot(path=f"screenshots/{name}")

    def close(self):
        self.browser.close()
        self.playwright.stop()

    def click(self, selector):
        self.page.locator(selector).click()

    def get_buttons(self):
        return self.extractor.get_buttons()
    
    def get_inputs(self):
        return self.extractor.get_inputs()
    
    def get_forms(self):
        return self.extractor.get_forms()
    
    def observe(self):
        return self.extractor.observe()
    
    def get_actions(self):
        return self.extractor.get_actions()

    def get_action(self, action_id):
        return self.action_registry.get(action_id)
    
    def click_action(self, action_id):

        action = self.action_registry.get(action_id)

        if action is None:
            logger.warning("Action %s was not found in the current action registry", action_id)
            return False

        try:
            before_url = self.page.url
            action["locator"].click(timeout=5000)
            # The context-level page event normally closes popups synchronously.
            # Keep this short check for delayed popup creation.
            self.page.wait_for_timeout(250)
            self._close_child_tabs()
            self._return_to_main_domain(before_url)
            # A click can trigger a client-side route without a formal navigation.
            # Waiting for a quiet network gives the next graph observation stable DOM data.
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                self.page.wait_for_load_state("networkidle", timeout=2000)
            except TimeoutError:
                logger.info("Page did not become network-idle after action %s; continuing", action_id)
            logger.info(
                "Clicked action %s (%s); URL changed from %s to %s",
                action_id,
                action["text"],
                before_url,
                self.page.url,
            )
            return True

        except (Error, TimeoutError) as exc:
            logger.exception("Failed to click action %s: %s", action_id, exc)
            print(f"Failed to click action {action_id}")
            return False

    def fill_input(self, target, value):

        strategies = [
            lambda: self.page.get_by_placeholder(target),
            lambda: self.page.locator(f'[name="{target}"]'),
            lambda: self.page.locator(f'#{target}'),
            lambda: self.page.get_by_label(target),
            lambda: self.page.locator(f'[aria-label="{target}"]'),
        ]

        for strategy in strategies:
            try:
                locator = strategy()
                locator.first.wait_for(state="visible", timeout=500)
                locator.first.fill(value)
                return True
            except Exception:
                pass

        print(f"Couldn't find input: {target}")
        return False
