from playwright.sync_api import sync_playwright
from extractor import PageExtractor
from action_registry import ActionRegistry
from playwright.sync_api import TimeoutError

class BrowserController:

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.extractor = None
        self.action_registry = None

    def start(self):
        self.playwright = sync_playwright().start()

        self.browser = self.playwright.chromium.launch(
            headless=False
        )

        self.page = self.browser.new_page()

        self.action_registry = ActionRegistry()

        self.extractor = PageExtractor(
            self.page,
            self.action_registry
        )

    def open(self, url):
        self.page.goto(url)

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
            return False

        try:
            action["locator"].click(timeout=5000)
            return True

        except TimeoutError:
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