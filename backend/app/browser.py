from playwright.sync_api import sync_playwright
from extractor import PageExtractor
from action_registry import ActionRegistry
from playwright.sync_api import Error, TimeoutError
import logging
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
from contextlib import contextmanager
from time import monotonic, perf_counter

from observability.tracing import NoopObservability


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
        self.observability = NoopObservability()
        self.latest_screenshot_path = None
        self.workflow_name = None
        # Set only by start_persistent(); owns browser shutdown when present.
        self._persistent_context = None

    def start(self, observability=None, headless=False, storage_state=None):
        self.observability = observability or NoopObservability()
        self.playwright = sync_playwright().start()

        self.browser = self.playwright.chromium.launch(
            headless=headless
        )

        if storage_state:
            # A saved session's cookies are injected into an otherwise fresh
            # context: the test starts already logged in, isolation intact.
            self.page = self.browser.new_context(storage_state=storage_state).new_page()
        else:
            self.page = self.browser.new_page()
        # Catch popups as soon as the browser context creates them. Checking only
        # after a click misses delayed window.open() calls from social links.
        self.page.context.on("page", self._handle_new_page)

        self.action_registry = ActionRegistry()

        self.extractor = PageExtractor(
            self.page,
            self.action_registry
        )

    def start_persistent(self, user_data_dir, observability=None, headless=False,
                         channel="chrome", viewport=(1280, 800)):
        """Launch a persistent real-Chrome profile instead of a fresh Chromium.

        Cookies, logins, and OTP verifications survive between runs, and
        consumer sites' bot detection sees an ordinary installed browser.
        Falls back to the bundled Chromium when Chrome is not installed.
        """
        self.observability = observability or NoopObservability()
        self.playwright = sync_playwright().start()
        Path(user_data_dir).mkdir(parents=True, exist_ok=True)
        launch_kwargs = dict(
            user_data_dir=str(user_data_dir),
            headless=headless,
            viewport={"width": viewport[0], "height": viewport[1]},
        )
        try:
            context = self.playwright.chromium.launch_persistent_context(
                channel=channel, **launch_kwargs
            )
        except Error as exc:
            logger.warning("Chrome channel '%s' unavailable (%s); using bundled Chromium",
                           channel, exc)
            context = self.playwright.chromium.launch_persistent_context(**launch_kwargs)
        self._persistent_context = context
        self.browser = context.browser
        self.page = context.pages[0] if context.pages else context.new_page()
        self.page.context.on("page", self._handle_new_page)
        self.action_registry = ActionRegistry()
        self.extractor = PageExtractor(self.page, self.action_registry)

    @contextmanager
    def _browser_action(self, action, selector=None):
        """Record browser operations through the provider-neutral interface."""
        started_at = perf_counter()
        initial_url = self.page.url if self.page else None
        success = False
        with self.observability.span(
            "Playwright." + action,
            input={"action": action, "selector": selector, "page_url": initial_url},
            metadata={"observation_type": "browser_action"},
        ) as span:
            try:
                yield
                success = True
            except Exception as exc:
                self.observability.record_exception(
                    exc,
                    input={"action": action, "selector": selector, "page_url": initial_url},
                    context={
                        "current_url": initial_url,
                        "workflow_name": self.workflow_name,
                        "active_action": action,
                        "screenshot_path": self.latest_screenshot_path,
                    },
                )
                raise
            finally:
                span.update(output={
                    "action": action,
                    "selector": selector,
                    "page_url": self.page.url if self.page else initial_url,
                    "success": success,
                    "execution_time_ms": round((perf_counter() - started_at) * 1000, 2),
                })

    def open(self, url):
        with self._browser_action("navigate", selector=url):
            self.page.goto(url)
        self.wait_for_page_ready()

    def go_back(self):
        with self._browser_action("navigate", selector="browser.go_back"):
            response = self.page.go_back()
        self.wait_for_page_ready()
        return response

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
            with self._browser_action("navigate", selector="browser.go_back"):
                self.page.go_back(wait_until="domcontentloaded", timeout=5000)
            self.wait_for_page_ready()
        except TimeoutError:
            logger.warning("Timed out returning from external URL; opening previous URL directly")
            with self._browser_action("navigate", selector=previous_url):
                self.page.goto(previous_url, wait_until="domcontentloaded")
            self.wait_for_page_ready()

    def title(self):
        return self.page.title()

    def current_url(self):
        return self.page.url

    def screenshot(self, path):
        """Save a screenshot at a caller-provided absolute or relative path."""
        with self._browser_action("screenshot", selector=str(path)):
            self.page.screenshot(path=path)
        self.latest_screenshot_path = str(path)

    def save_session(self, path):
        """Persist the current context's cookies/session for later reuse."""
        self.page.context.storage_state(path=str(path))

    def close(self):
        if self._persistent_context is not None:
            # Closing the persistent context shuts down its browser process.
            try:
                self._persistent_context.close()
            except Error as exc:
                logger.warning("Could not close the persistent context: %s", exc)
            self._persistent_context = None
        elif self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def click(self, selector):
        with self._browser_action("click", selector):
            self.page.locator(selector).click()
        self.wait_for_page_ready()

    def select_option(self, selector, value):
        with self._browser_action("select", selector):
            self.page.locator(selector).select_option(value)
        self.wait_for_page_ready()

    def click_at(self, x_px, y_px):
        """Click a raw viewport coordinate — for canvas/visually-rendered UI."""
        with self._browser_action("click_at", f"{x_px},{y_px}"):
            self.page.mouse.click(x_px, y_px)
        self.wait_for_page_ready()

    # ------------------------------------------------------------------
    # Frame-aware interaction.  Consumer sites mount login and payment UI
    # in child iframes (e.g. Zomato's auth-login-ui on accounts.zomato.com);
    # main-frame-only extraction sees an empty page there and the agent
    # starves.  Everything below searches the main frame first, then every
    # attached child frame.
    # ------------------------------------------------------------------

    def _candidate_frames(self, limit=6):
        frames = []
        for frame in self.page.frames:
            if frame == self.page.main_frame:
                continue
            try:
                if frame.is_detached() or not frame.url:
                    continue
            except Error:
                continue
            frames.append(frame)
            if len(frames) >= limit:
                break
        return frames

    def _all_frames(self):
        return [self.page.main_frame, *self._candidate_frames()]

    def frame_elements(self, max_per_frame=40):
        """Visible clickable texts and inputs inside child iframes.

        Returns [{name, url, clickables: [str], inputs: [{type, placeholder,
        name, maxlength}]}] for frames that contain anything interactive.
        """
        result = []
        for frame in self._candidate_frames():
            info = {"name": frame.name or "", "url": frame.url[:120],
                    "clickables": [], "inputs": []}
            try:
                clickable = frame.locator(
                    "button, [role='button'], a, input[type='submit']"
                )
                seen = set()
                for index in range(min(clickable.count(), max_per_frame)):
                    element = clickable.nth(index)
                    try:
                        if not element.is_visible():
                            continue
                        text = " ".join((element.inner_text(timeout=200) or "").split())[:60]
                        if text and text.casefold() not in seen:
                            seen.add(text.casefold())
                            info["clickables"].append(text)
                    except Error:
                        continue
                inputs = frame.locator("input, textarea")
                for index in range(min(inputs.count(), 15)):
                    element = inputs.nth(index)
                    try:
                        if not element.is_visible():
                            continue
                        info["inputs"].append({
                            "type": element.get_attribute("type") or "text",
                            "placeholder": element.get_attribute("placeholder") or "",
                            "name": element.get_attribute("name") or "",
                            "maxlength": element.get_attribute("maxlength") or "",
                        })
                    except Error:
                        continue
            except Error as exc:
                logger.debug("Could not inspect frame %s: %s", frame.name, exc)
            if info["clickables"] or info["inputs"]:
                result.append(info)
        return result

    def click_text(self, text, exact=False):
        """Click an element by its visible text, searching every frame.

        This is the reliable fallback when action extraction misses a control
        (iframes, portals, custom widgets): exactly how a human describes the
        click. Tries text match first, then button/link roles by name.
        """
        text = str(text or "").strip()
        if not text:
            return False
        for frame in self._all_frames():
            try:
                locator = frame.get_by_text(text, exact=exact).first
                locator.wait_for(state="visible", timeout=1_500)
                with self._browser_action("click_text", text):
                    locator.click(timeout=3_000)
                self.wait_for_page_ready()
                return True
            except (Error, TimeoutError):
                continue
        for frame in self._all_frames():
            for role in ("button", "link"):
                try:
                    locator = frame.get_by_role(role, name=text).first
                    locator.wait_for(state="visible", timeout=1_000)
                    with self._browser_action("click_text", text):
                        locator.click(timeout=3_000)
                    self.wait_for_page_ready()
                    return True
                except (Error, TimeoutError):
                    continue
        logger.warning("click_text found no visible element with text: %s", text)
        return False

    def find_otp_boxes(self):
        """Locate a row of 4–8 single-character inputs (an OTP widget)."""
        for frame in self._all_frames():
            try:
                boxes = frame.locator("input[maxlength='1']")
                if 4 <= boxes.count() <= 8:
                    return boxes
            except Error:
                continue
        return None

    def fill_otp(self, code):
        """Type a one-time code into split OTP boxes.

        Split-box widgets auto-advance focus per keystroke, so the reliable
        input method is: click the first box, then send real key events —
        a locator ``fill`` puts the whole code in one box and fails.
        """
        code = str(code or "").strip().replace(" ", "")
        if not code:
            return False
        boxes = self.find_otp_boxes()
        if boxes is not None:
            with self._browser_action("fill_otp", f"{len(code)} chars"):
                boxes.first.click()
                self.page.keyboard.type(code, delay=120)
            return True
        # No maxlength markers: fall back to the first visible input of a
        # child frame (auth modals are usually the only frame present).
        for frame in self._candidate_frames():
            try:
                first = frame.locator("input").first
                first.wait_for(state="visible", timeout=1_000)
                with self._browser_action("fill_otp", f"{len(code)} chars (frame fallback)"):
                    first.click()
                    self.page.keyboard.type(code, delay=120)
                return True
            except (Error, TimeoutError):
                continue
        logger.warning("fill_otp could not locate OTP inputs")
        return False

    def press_key(self, key):
        with self._browser_action("press_key", key):
            self.page.keyboard.press(key)

    def scroll(self, delta_y):
        with self._browser_action("scroll", str(delta_y)):
            self.page.mouse.wheel(0, delta_y)

    def wait_for_timeout(self, timeout_ms):
        with self._browser_action("wait", selector=f"timeout:{timeout_ms}ms"):
            self.page.wait_for_timeout(timeout_ms)

    def get_buttons(self):
        return self.extractor.get_buttons()
    
    def get_inputs(self):
        return self.extractor.get_inputs()
    
    def get_forms(self):
        return self.extractor.get_forms()
    
    def observe(self):
        self.wait_for_page_ready()
        return self.extractor.observe()

    def wait_for_page_ready(
        self,
        timeout_ms=20_000,
        previous_fingerprint=None,
        wait_for_content_update=False,
        content_update_timeout_ms=8_000,
    ) -> bool:
        """Wait for usable, optionally post-navigation, page content.

        When a pre-action fingerprint is supplied, the page must expose both
        interactive controls and meaningfully different content. This prevents
        a client-side URL update from being mistaken for a completed route
        while the old page DOM is still visible.
        """
        logger.info("Waiting for page to become interactive...")
        deadline = monotonic() + timeout_ms / 1000

        def remaining_timeout_ms(limit_ms):
            return max(1, min(limit_ms, int((deadline - monotonic()) * 1000)))

        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=remaining_timeout_ms(5_000))
        except (TimeoutError, Error):
            logger.warning("Timed out waiting for domcontentloaded; checking interactive elements")

        if monotonic() < deadline:
            try:
                self.page.wait_for_load_state("networkidle", timeout=remaining_timeout_ms(3_000))
            except (TimeoutError, Error):
                logger.debug("Page did not become network-idle; continuing with interactive-element check")

        interactive = self.page.locator(
            "button, input, select, textarea, a, [role='button']"
        )
        while monotonic() < deadline:
            try:
                count = interactive.count()
                content_changed = (
                    previous_fingerprint is None
                    or self._dom_changed_significantly(
                        previous_fingerprint,
                        self._dom_fingerprint(),
                    )
                )
                if count and content_changed:
                    logger.info("Found %s interactive elements.", count)
                    if wait_for_content_update:
                        return self._wait_for_destination_content_update(
                            count,
                            self._dom_fingerprint(),
                            min(content_update_timeout_ms, remaining_timeout_ms(content_update_timeout_ms)),
                        )
                    logger.info("Page ready.")
                    return True
            except Error as exc:
                logger.debug("Could not inspect interactive elements yet: %s", exc)
            try:
                self.page.wait_for_timeout(500)
            except Error as exc:
                logger.debug("Could not wait for the next interactive-element poll: %s", exc)

        if previous_fingerprint is not None:
            logger.warning("Timed out waiting for destination content to replace the previous page.")
        else:
            logger.warning("Timed out waiting for interactive elements.")
        return False

    def _wait_for_destination_content_update(
        self,
        initial_interactive_count,
        initial_fingerprint,
        timeout_ms,
    ) -> bool:
        """Wait for deferred application content after a successful route change.

        Modern dashboards often expose a shell of controls before their module
        navigation and business content are rendered. A URL change plus an
        initial interactive count therefore is not sufficient after login.
        This waits for a real DOM or interactive-count update, polling rather
        than sleeping for a fixed duration.
        """
        logger.info("Waiting for destination content to finish rendering...")
        deadline = monotonic() + timeout_ms / 1000
        interactive = self.page.locator(
            "button, input, select, textarea, a, [role='button']"
        )

        while monotonic() < deadline:
            try:
                count = interactive.count()
                fingerprint = self._dom_fingerprint()
                if (
                    count > initial_interactive_count
                    or self._dom_changed_significantly(initial_fingerprint, fingerprint)
                ):
                    logger.info(
                        "Destination content updated (%s -> %s interactive elements).",
                        initial_interactive_count,
                        count,
                    )
                    logger.info("Page ready.")
                    return True
            except Error as exc:
                logger.debug("Could not inspect destination content yet: %s", exc)
            try:
                self.page.wait_for_timeout(500)
            except Error as exc:
                logger.debug("Could not wait for destination-content poll: %s", exc)

        # The initial page shell was already usable. Do not fail an otherwise
        # valid login merely because the destination has no deferred update.
        logger.info("Destination content did not change further; continuing with the ready page.")
        logger.info("Page ready.")
        return True
    
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
            with self._browser_action("click", str(action_id)):
                action["locator"].click(timeout=5000)
            self._close_child_tabs()
            self._return_to_main_domain(before_url)
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
            self.observability.record_exception(
                exc,
                input={"action": "click", "selector": str(action_id), "page_url": self.page.url},
                context={
                    "current_url": self.page.url,
                    "workflow_name": self.workflow_name,
                    "active_action": action.get("text") if action else str(action_id),
                    "screenshot_path": self.latest_screenshot_path,
                },
            )
            print(f"Failed to click action {action_id}")
            return False

    def fill_input(self, target, value):
        """Fill a field by label/placeholder/name/id, searching every frame.

        The main frame is tried first with the full strategy list; child
        frames follow (login modals live there).  Inside a child frame a
        final blind fallback fills the first visible text-like input — auth
        iframes routinely expose a single input with no attributes at all.
        """
        for frame in self._all_frames():
            in_child_frame = frame != self.page.main_frame
            if self._fill_input_in_frame(frame, target, value, allow_blind=in_child_frame):
                return True

        self.observability.record_exception(
            RuntimeError(f"No fillable input matched: {target}"),
            input={"action": "fill", "selector": target, "page_url": self.page.url},
            context={
                "current_url": self.page.url,
                "workflow_name": self.workflow_name,
                "active_action": target,
                "screenshot_path": self.latest_screenshot_path,
            },
        )
        print(f"Couldn't find input: {target}")
        return False

    def _fill_input_in_frame(self, frame, target, value, allow_blind=False):
        strategies = [
            lambda: frame.get_by_placeholder(target),
            lambda: frame.locator(f'[name="{target}"]'),
            lambda: frame.locator(f'#{target}'),
            lambda: frame.get_by_label(target),
            lambda: frame.locator(f'[aria-label="{target}"]'),
            # Case-insensitive attribute matches: planners and designers often
            # capitalise a field name ("Email") that the DOM stores lowercase.
            lambda: frame.locator(f'[name="{target}" i]'),
            lambda: frame.locator(f'[id="{target}" i]'),
            lambda: frame.locator(f'[aria-label="{target}" i]'),
        ]
        # Semantic fallbacks by intent: custom login forms frequently expose no
        # label, name, or placeholder at all, but their input types are still
        # unambiguous. "username"/"email"/"password" targets must reach them.
        intent = str(target or "").casefold()
        if "password" in intent:
            strategies.append(lambda: frame.locator('input[type="password"]'))
        if any(term in intent for term in ("email", "e-mail", "username", "user name", "login id", "userid")):
            strategies.append(lambda: frame.locator('input[type="email"]'))
            strategies.append(lambda: frame.locator('input[autocomplete="username"]'))
            strategies.append(lambda: frame.locator('form input[type="text"]'))
        if allow_blind:
            strategies.append(lambda: frame.locator(
                "input:not([type=hidden]):not([type=checkbox]):not([type=radio])"
                ":not([type=submit]):not([type=button]), textarea"
            ))

        for strategy in strategies:
            # Resolve quietly first: a strategy miss is normal, only a failed
            # fill on a found element is worth recording.
            try:
                locator = strategy().first
                locator.wait_for(state="visible", timeout=500)
            except Exception:
                continue
            try:
                with self._browser_action("fill", target):
                    locator.fill(value)
                return True
            except Exception:
                continue
        return False

    def upload_file_inputs(self, file_path):
        """Set a stored test asset on every HTML file input on the page."""
        file_inputs = self.page.locator('input[type="file"]')
        try:
            count = file_inputs.count()
        except Error as exc:
            logger.warning("Could not inspect file inputs: %s", exc)
            return False

        if not count:
            return False

        uploaded = False
        for index in range(count):
            input_locator = file_inputs.nth(index)
            try:
                with self._browser_action("upload_file", str(file_path)):
                    input_locator.set_input_files(str(file_path))
                logger.info("Uploaded test asset %s to file input %s", file_path, index)
                uploaded = True
            except (Error, TimeoutError) as exc:
                logger.warning("Failed to upload test asset to file input %s: %s", index, exc)
        return uploaded

    def upload_file_action(self, action_id, file_path):
        """Attach a file to the exact file input selected by the planner."""
        action = self.action_registry.get(action_id)
        if action is None or action.get("type") != "file_upload":
            logger.warning("Action %s is not a registered file-upload control", action_id)
            return False
        try:
            with self._browser_action("upload_file", str(file_path)):
                action["locator"].set_input_files(str(file_path))
            logger.info("Uploaded test asset %s using action %s", file_path, action_id)
            return True
        except (Error, TimeoutError) as exc:
            logger.warning("Failed to upload test asset using action %s: %s", action_id, exc)
            return False

    def has_file_inputs(self):
        """Return whether the current page exposes an HTML file input."""
        try:
            return self.page.locator('input[type="file"]').count() > 0
        except Error as exc:
            logger.warning("Could not inspect file inputs: %s", exc)
            return False

    def capture_action_state(self, action):
        """Capture the small amount of state needed for post-action waiting."""
        clicked_locator = None
        if action.get("type") == "click":
            registered_action = self.action_registry.get(action.get("action_id"))
            if registered_action:
                clicked_locator = registered_action.get("locator")
        return {
            "url": self.page.url,
            "fingerprint": self._dom_fingerprint(),
            "clicked_locator": clicked_locator,
        }

    def wait_after_action(self, action, before_state, timeout_ms=20_000) -> bool:
        """Wait for an action's observable outcome instead of stale page controls.

        Clicks wait for a navigation, a meaningful DOM update, removal of the
        clicked control, or a complete spinner cycle. Text fills do not require
        a transition, but still wait for the page's DOM-ready milestone.
        """
        action_type = action.get("type", "unknown")
        if action_type != "click":
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=min(timeout_ms, 2_000))
            except (TimeoutError, Error):
                logger.debug("DOM was not ready after %s; continuing", action_type)
            logger.info("Post-action wait finished: %s does not require a transition.", action_type)
            return True

        logger.info("Waiting for click outcome...")
        deadline = monotonic() + timeout_ms / 1000
        spinner_seen = False
        before_url = before_state.get("url")
        before_fingerprint = before_state.get("fingerprint", {})
        clicked_locator = before_state.get("clicked_locator")

        while monotonic() < deadline:
            if self.page.url != before_url:
                logger.info("Post-action wait finished: URL changed from %s to %s.", before_url, self.page.url)
                return True

            if self._dom_changed_significantly(before_fingerprint, self._dom_fingerprint()):
                logger.info("Post-action wait finished: DOM changed significantly.")
                return True

            if clicked_locator is not None and self._locator_disappeared(clicked_locator):
                logger.info("Post-action wait finished: clicked element disappeared or detached.")
                return True

            spinner_visible = self._spinner_visible()
            spinner_seen = spinner_seen or spinner_visible
            if spinner_seen and not spinner_visible:
                logger.info("Post-action wait finished: loading spinner disappeared.")
                return True

            try:
                self.page.wait_for_timeout(500)
            except Error as exc:
                logger.debug("Could not wait for the next post-action poll: %s", exc)

        logger.warning("Post-action wait timed out without a visible page transition.")
        return False

    def wait_for_url_change(self, previous_url, timeout_ms=20_000) -> bool:
        """Wait specifically for a navigation away from ``previous_url``."""
        logger.info("Current URL: %s", previous_url)
        logger.info("Waiting for URL change...")
        deadline = monotonic() + timeout_ms / 1000
        while monotonic() < deadline:
            current_url = self.page.url
            if current_url != previous_url:
                logger.info("URL changed: %s", current_url)
                return True
            try:
                self.page.wait_for_timeout(500)
            except Error as exc:
                logger.debug("Could not wait for the next URL-change poll: %s", exc)

        logger.warning("Timed out waiting for URL change from %s.", previous_url)
        return False

    def _dom_fingerprint(self):
        try:
            text = self.page.locator("body").inner_text(timeout=1_000)
        except Error:
            text = ""
        normalized = " ".join(text.split())[:12_000]
        return {
            "text": normalized,
            "digest": sha256(normalized.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _dom_changed_significantly(before_fingerprint, after_fingerprint) -> bool:
        before_text = before_fingerprint.get("text", "")
        after_text = after_fingerprint.get("text", "")
        if before_fingerprint.get("digest") == after_fingerprint.get("digest"):
            return False
        if not before_text or not after_text:
            return True
        return SequenceMatcher(None, before_text, after_text, autojunk=False).ratio() < 0.92

    @staticmethod
    def _locator_disappeared(locator) -> bool:
        try:
            return not locator.is_visible(timeout=100)
        except Error:
            return True

    def _spinner_visible(self) -> bool:
        spinner = self.page.locator(
            "[role='progressbar'], [aria-busy='true'], .loading, .loader, .spinner"
        )
        try:
            for index in range(spinner.count()):
                if spinner.nth(index).is_visible(timeout=100):
                    return True
        except Error:
            return False
        return False
