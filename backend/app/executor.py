import logging

from asset_manager import AssetManager
from playwright.sync_api import Error, TimeoutError


logger = logging.getLogger(__name__)

AUTHENTICATION_TERMS = (
    "login", "log in", "sign in", "signin", "authenticate", "authentication",
)


class Executor:

    def __init__(self, browser, observability):
        self.browser = browser
        self.observability = observability
        self.asset_manager = AssetManager()

    def execute(self, plan):
        for step in plan["steps"]:
            print(step)
            before_state = self.browser.capture_action_state(step)

            if step["type"] == "fill":
                success = self.browser.fill_input(
                    step["target"],
                    step["value"]
                )
                if not success:
                    return False
                self.browser.wait_after_action(step, before_state)

            elif step["type"] == "click":
                action_id = step.get("action_id")
                if action_id is None:
                    print("Click step is missing action_id.")
                    return False
                source_url = self.browser.current_url()
                action_details = self.browser.get_action(action_id) or {}
                if action_details.get("type") == "file_upload":
                    resume_path = self.asset_manager.resolve_resume_path()
                    if not resume_path:
                        logger.error("File upload action selected, but no resume asset is available")
                        return False
                    success = self.browser.upload_file_action(action_id, resume_path)
                    if success:
                        logger.info("Uploaded stored resume through selected file-upload action")
                else:
                    success = self.browser.click_action(action_id)

                if not success and action_details.get("type") != "file_upload":
                    success = self._recover_intercepted_click(action_id)
                if not success:
                    logger.error(
                        "Click action %s failed after interception recovery; recording failure and continuing exploration",
                        action_id,
                    )
                    return False
                if self._is_authentication_action(step, action_details):
                    if self.browser.wait_for_url_change(source_url):
                        logger.info("Waiting for destination page...")
                        destination_ready = self.browser.wait_for_page_ready(
                            previous_fingerprint=before_state.get("fingerprint"),
                            wait_for_content_update=True,
                        )
                        if destination_ready:
                            logger.info("Destination page ready.")
                        else:
                            logger.warning("Destination page did not become ready before the timeout.")
                    else:
                        logger.info("URL did not change; falling back to DOM-based synchronization.")
                        self.browser.wait_after_action(step, before_state)
                else:
                    self.browser.wait_after_action(step, before_state)

                # Action locators belong to the page that was observed and
                # planned. A navigation invalidates every remaining locator in
                # this plan, so let the graph observe the destination page.
                if self.browser.current_url() != source_url:
                    return True

        return True

    @staticmethod
    def _is_authentication_action(step, action_details) -> bool:
        if str(step.get("category", "")).casefold() == "authentication":
            return True
        labels = (
            step.get("target"),
            step.get("label"),
            step.get("text"),
            action_details.get("text"),
        )
        label_text = " ".join(str(label or "").casefold() for label in labels)
        return any(term in label_text for term in AUTHENTICATION_TERMS)

    def _recover_intercepted_click(self, action_id) -> bool:
        """Clear common overlays and retry one normal Playwright click.

        ``BrowserController.click_action`` intentionally catches Playwright
        click failures to keep Discovery moving.  Recovery therefore starts
        after a failed click and never uses ``force=True``.
        """
        logger.warning("Click action %s failed; attempting pointer-interception recovery", action_id)
        try:
            self.browser.page.keyboard.press("Escape")
            logger.info("Pressed Escape while recovering click action %s", action_id)
        except (Error, TimeoutError) as exc:
            logger.debug("Escape recovery failed for action %s: %s", action_id, exc)

        backdrop = self.browser.page.locator(
            ".modal-backdrop, .backdrop, [data-testid='backdrop'], [data-backdrop='true']"
        )
        try:
            if backdrop.count():
                backdrop.first.click(timeout=1_000)
                logger.info("Clicked backdrop while recovering click action %s", action_id)
        except (Error, TimeoutError) as exc:
            logger.debug("Backdrop recovery failed for action %s: %s", action_id, exc)

        logger.info("Retrying click action %s after pointer-interception recovery", action_id)
        return self.browser.click_action(action_id)
