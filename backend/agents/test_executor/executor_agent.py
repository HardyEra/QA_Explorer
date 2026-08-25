"""Executor worker: deterministically replay one test case, then validate it.

This is intentionally NOT an LLM agent.  Each worker owns a private browser
(safe to run in parallel graph branches — every LangGraph branch runs in its
own thread, and each thread starts its own Playwright instance).  The embedded
Validator asserts the case's expectations; the Healer is consulted at most
once per step, and only on failure.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from agents.healer import Healer
from agents.test_designer.models import (
    PASSWORD_PLACEHOLDER,
    USERNAME_PLACEHOLDER,
    describe_step,
)
from observability.tracing import NoopObservability


logger = logging.getLogger(__name__)

RUNS_ROOT = Path(__file__).resolve().parent.parent.parent / "runtime" / "runs"
STEP_TIMEOUT_MS = 5_000
POST_LOGIN_REDIRECT_TIMEOUT_MS = 8_000
VALIDATION_WINDOW_MS = 10_000

# Parallel workers share one test account. Concurrent logins can trip
# throttling or session invalidation, so real-credential login windows (the
# credential fills plus the submitting click) run one at a time; everything
# after login stays parallel.
_LOGIN_LOCK = threading.Lock()


def create_login_session(start_url: str, username: str, password: str, path,
                         observability=None, headless: bool | None = None) -> str | None:
    """Log in once and save the session so eligible tests can skip their logins.

    Best-effort: any failure returns ``None`` and tests fall back to their own
    login steps, exactly as before.
    """
    from pathlib import Path as _Path

    from browser import BrowserController

    if headless is None:
        headless = os.getenv("QA_EXECUTOR_HEADLESS", "true").strip().casefold() != "false"
    browser = BrowserController()
    try:
        browser.start(observability=observability, headless=headless)
        browser.open(start_url)
        if not browser.fill_input("email", username) and not browser.fill_input("username", username):
            logger.warning("Session bootstrap: could not find a username/email field")
            return None
        if not browser.fill_input("password", password):
            logger.warning("Session bootstrap: could not find a password field")
            return None
        before_url = browser.current_url()
        for label in ("Log In", "Login", "Sign In", "Sign in", "Submit"):
            try:
                browser.page.get_by_role("button", name=label).first.click(timeout=2_000)
                break
            except Exception:
                continue
        else:
            logger.warning("Session bootstrap: could not find a login button")
            return None
        browser.wait_for_url_change(before_url, timeout_ms=15_000)
        if browser.current_url() == before_url:
            logger.warning("Session bootstrap: login did not leave the login page")
            return None
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        browser.save_session(path)
        logger.info("Login session saved; eligible tests will skip their login steps")
        return str(path)
    except Exception:
        logger.warning("Session bootstrap failed; tests will log in individually", exc_info=True)
        return None
    finally:
        try:
            browser.close()
        except Exception:
            logger.debug("Could not close the bootstrap browser", exc_info=True)


class ExecutorAgent:
    """Replay one test case in an isolated browser and return a TestResult dict."""

    def __init__(self, observability=None, healer: Healer | None = None,
                 headless: bool | None = None, on_event=None):
        self.observability = observability or NoopObservability()
        self.healer = healer or Healer(self.observability)
        if headless is None:
            headless = os.getenv("QA_EXECUTOR_HEADLESS", "true").strip().casefold() != "false"
        self.headless = headless
        # Streams live execution progress (current test, current step, verdict)
        # to the UI so a run is watchable, not a black box.
        self.on_event = on_event or (lambda event: None)

    def run(self, test_case: dict, start_url: str, username: str = "", password: str = "",
            run_id: str = "run", session_state_path: str | None = None,
            headless: bool | None = None) -> dict:
        # Imported lazily so this module stays importable without Playwright
        # and each worker thread builds its own browser stack.
        from browser import BrowserController

        # A saved session only replaces REAL-credential logins: BOTH the
        # username and password placeholders must be present. A test that
        # fills a literal invalid username with the real password (or only
        # toggles the password field) is testing the logged-OUT experience —
        # a pre-injected session would falsify it.
        storage_state = None
        steps = test_case.get("steps", [])
        boundary = self._login_boundary(steps)
        uses_real_credentials = boundary is not None and any(
            USERNAME_PLACEHOLDER in str(step.get("value", "")) for step in steps
        )
        if session_state_path and uses_real_credentials:
            storage_state = session_state_path
            skipped = test_case["steps"][: boundary + 1]
            test_case = {**test_case, "steps": test_case["steps"][boundary + 1:]}
            logger.info(
                "Reusing saved login session for %s; skipped %s login step(s)",
                test_case.get("id"), len(skipped),
            )

        started_at = perf_counter()
        result: dict = {
            "test_id": test_case.get("id", "unknown"),
            "requirement_id": test_case.get("requirement_id", ""),
            "title": test_case.get("title", ""),
            "description": test_case.get("description", ""),
            "unverified": test_case.get("unverified", []),
            "priority": test_case.get("priority", "medium"),
            "status": "error",
            "failed_step": None,
            "failure_reason": None,
            "healed": False,
            "expectations": [],
            "evidence": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.on_event({
            "type": "test_start",
            "test_id": result["test_id"],
            "title": result["title"],
            "total_steps": len(test_case.get("steps", [])),
            "status": f"Running test: {result['title']}",
        })
        browser = BrowserController()
        with self.observability.span(
            "Test Executor", input={"test_id": result["test_id"], "title": result["title"]}
        ) as span:
            try:
                browser.start(observability=self.observability,
                              headless=self.headless if headless is None else headless,
                              storage_state=storage_state)
                browser.open(self._absolute_url(start_url, ""))
                self._run_steps(browser, test_case, start_url, username, password, result)
                if result["failure_reason"] is None:
                    self._validate(browser, test_case, result)
            except Exception as exc:
                logger.exception("Executor crashed on test %s", result["test_id"])
                result["status"] = "error"
                result["failure_reason"] = f"executor error: {exc}"
            finally:
                self._capture_evidence(browser, run_id, result)
                try:
                    browser.close()
                except Exception:
                    logger.warning("Executor could not close the browser cleanly", exc_info=True)
            result["duration_ms"] = round((perf_counter() - started_at) * 1000, 1)
            span.update(output={"status": result["status"], "duration_ms": result["duration_ms"]})
        self.on_event({
            "type": "test_result",
            "test_id": result["test_id"],
            "title": result["title"],
            "result_status": result["status"],
            "reason": result.get("failure_reason"),
            "duration_ms": result["duration_ms"],
            "screenshot_path": (result.get("evidence") or {}).get("screenshot"),
            "status": f"{result['status'].upper()}: {result['title']}",
        })
        return result

    # ------------------------------------------------------------------
    # Step replay
    # ------------------------------------------------------------------

    def _run_steps(self, browser, test_case: dict, start_url: str, username: str,
                   password: str, result: dict) -> None:
        steps = test_case.get("steps", [])
        login_boundary = self._login_boundary(steps)
        lock_held = False
        if login_boundary is not None:
            _LOGIN_LOCK.acquire()
            lock_held = True
        try:
            for position, raw_step in enumerate(steps):
                index = position + 1
                step = self._substitute(raw_step, username, password)
                # Describe the RAW step: the substituted one holds credentials.
                self.on_event({
                    "type": "test_step",
                    "test_id": result["test_id"],
                    "index": index,
                    "total": len(steps),
                    "description": describe_step(raw_step),
                    "status": f"Step {index}/{len(steps)}: {describe_step(raw_step)}",
                })
                url_before_step = browser.current_url() if browser.page else None
                error = self._execute_step(browser, step, start_url)
                self._emit_live_frame(browser, result["test_id"])
                if error is not None:
                    verdict = self.healer.heal(
                        test_case.get("title", ""), step, error, self._page_snapshot(browser)
                    )
                    if verdict["decision"] == "retry":
                        revised = self._substitute(verdict["revised_step"], username, password)
                        if self._execute_step(browser, revised, start_url) is None:
                            result["healed"] = True
                            logger.info(
                                "Healed step %s of %s: %r -> %r",
                                index, result["test_id"], step.get("target"), revised.get("target"),
                            )
                            error = None
                    if error is not None:
                        result["status"] = "failed"
                        # Record the raw step, not the substituted one: the
                        # substituted step contains the real credentials and
                        # this record ends up in reports.
                        result["failed_step"] = {"index": index, **raw_step}
                        result["failure_reason"] = verdict.get("reason") or error
                        return

                if position == login_boundary:
                    # SPA logins often update the URL a beat after the click;
                    # validating against the stale login page causes flakes.
                    if url_before_step and browser.current_url() == url_before_step:
                        browser.wait_for_url_change(
                            url_before_step, timeout_ms=POST_LOGIN_REDIRECT_TIMEOUT_MS
                        )
                    if lock_held:
                        _LOGIN_LOCK.release()
                        lock_held = False
        finally:
            if lock_held:
                _LOGIN_LOCK.release()

    @staticmethod
    def _login_boundary(steps: list[dict]) -> int | None:
        """Index of the click that submits real credentials, if any.

        Detection keys on the password *placeholder* so negative-credential
        tests (which fill literal wrong values) are neither serialized nor
        made to wait for a redirect that will never come.
        """
        password_fill = None
        for position, step in enumerate(steps):
            if step.get("type") == "fill" and PASSWORD_PLACEHOLDER in str(step.get("value", "")):
                password_fill = position
        if password_fill is None:
            return None
        for position in range(password_fill + 1, len(steps)):
            if steps[position].get("type") == "click":
                return position
        return password_fill

    def _execute_step(self, browser, step: dict, start_url: str) -> str | None:
        """Run one step; return an error description or ``None`` on success."""
        step_type = step.get("type")
        target = step.get("target", "")
        try:
            if step_type == "navigate":
                browser.open(self._absolute_url(start_url, target))
                return None
            if step_type == "fill":
                if browser.fill_input(target, step.get("value", "")):
                    return None
                return f"no fillable input matched {target!r}"
            if step_type == "click":
                return self._click_by_label(browser, target)
            if step_type == "select":
                return self._select_option(browser, target, step.get("value", ""))
            if step_type == "upload":
                return self._upload_asset(browser, target, step.get("value", ""))
            return f"unsupported step type {step_type!r}"
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _select_option(browser, target: str, option: str) -> str | None:
        """Choose an option in a native <select> or a custom dropdown."""
        page = browser.page
        # Native selects first: select_option works without opening the menu.
        for locate in (
            lambda: page.get_by_label(target),
            lambda: page.locator(f'select[name="{target}" i]'),
            lambda: page.locator(f'select[aria-label="{target}" i]'),
        ):
            try:
                locate().first.select_option(label=option, timeout=1_500)
                browser.wait_for_page_ready()
                return None
            except Exception:
                continue
        # Custom dropdowns (React selects, comboboxes): open, then click the option.
        open_error = ExecutorAgent._click_by_label(browser, target)
        if open_error is not None:
            return f"could not open the {target!r} dropdown: {open_error}"
        for locate in (
            lambda: page.get_by_role("option", name=option),
            lambda: page.get_by_text(option, exact=True),
            lambda: page.get_by_text(option),
        ):
            try:
                locator = locate().first
                locator.wait_for(state="visible", timeout=2_000)
                locator.click(timeout=STEP_TIMEOUT_MS)
                browser.wait_for_page_ready()
                return None
            except Exception:
                continue
        return f"opened {target!r} but no option matched {option!r}"

    @staticmethod
    def _upload_asset(browser, target: str, asset_name: str) -> str | None:
        """Attach a stored test asset; bypasses the OS file dialog entirely."""
        from asset_manager import AssetManager

        manager = AssetManager()
        asset_path = None
        if asset_name:
            asset_path = manager.get_asset_path(asset_name)
        if asset_path is None:
            assets = manager.list_assets()
            asset_path = next(iter(assets.values()), None)
        if asset_path is None:
            return (
                f"no stored test asset available for {asset_name or 'upload'!r} — "
                "add one in the Test assets panel"
            )

        # Native file inputs first: set_input_files works even when the input
        # is visually hidden behind a styled button.
        if browser.has_file_inputs() and browser.upload_file_inputs(asset_path):
            browser.wait_for_page_ready()
            return None
        # Custom upload buttons open a file chooser; intercept it.
        if target:
            try:
                with browser.page.expect_file_chooser(timeout=STEP_TIMEOUT_MS) as chooser_info:
                    browser.page.get_by_text(target).first.click(timeout=STEP_TIMEOUT_MS)
                chooser_info.value.set_files(str(asset_path))
                browser.wait_for_page_ready()
                return None
            except Exception as exc:
                return f"could not upload via {target!r}: {type(exc).__name__}"
        return "no file input or upload control found on the page"

    @staticmethod
    def _click_by_label(browser, label: str) -> str | None:
        """Click a control by its visible label using resilient strategies."""
        page = browser.page
        strategies = (
            lambda: page.get_by_role("button", name=label, exact=True),
            lambda: page.get_by_role("link", name=label, exact=True),
            lambda: page.get_by_role("button", name=label),
            lambda: page.get_by_role("link", name=label),
            lambda: page.get_by_text(label, exact=True),
            lambda: page.get_by_text(label),
            # Accessible names: icon controls often carry their label only in
            # aria-label (e.g. "View Onboarded Vendors details"), which
            # visible-text matching can never find.
            lambda: page.get_by_label(label, exact=True),
            lambda: page.get_by_label(label),
            lambda: page.locator(f'[aria-label="{label}" i]'),
            lambda: page.locator(f'[title="{label}" i]'),
        )
        last_error = None
        for strategy in strategies:
            try:
                locator = strategy().first
                locator.wait_for(state="visible", timeout=STEP_TIMEOUT_MS // len(strategies))
                locator.click(timeout=STEP_TIMEOUT_MS)
                browser.wait_for_page_ready()
                return None
            except Exception as exc:
                last_error = exc
        return f"no clickable control matched {label!r} ({type(last_error).__name__})"

    @staticmethod
    def _substitute(step: dict, username: str, password: str) -> dict:
        """Resolve credential placeholders at run time; secrets never persist."""
        step = dict(step)
        value = step.get("value")
        if isinstance(value, str):
            step["value"] = value.replace(USERNAME_PLACEHOLDER, username).replace(
                PASSWORD_PLACEHOLDER, password
            )
        return step

    @staticmethod
    def _absolute_url(start_url: str, target: str) -> str:
        if not target:
            return start_url
        if target.startswith(("http://", "https://")):
            return target
        return start_url.rstrip("/") + "/" + target.lstrip("/")

    # ------------------------------------------------------------------
    # Validator
    # ------------------------------------------------------------------

    def _validate(self, browser, test_case: dict, result: dict) -> None:
        """Assert every expectation; a test passes only when all of them hold.

        Checks poll until they hold or the validation window closes: uploads
        process asynchronously and realtime UIs render a beat after the last
        step, so instant checks would fail flows that actually work.
        """
        from time import monotonic

        checks = [
            {"type": expectation["type"], "value": expectation["value"], "passed": False}
            for expectation in test_case.get("expected", [])
        ]
        deadline = monotonic() + VALIDATION_WINDOW_MS / 1000
        while True:
            for check in checks:
                if not check["passed"]:
                    check["passed"] = self._check(browser, check)
            if all(check["passed"] for check in checks) or monotonic() >= deadline:
                break
            try:
                browser.page.wait_for_timeout(500)
            except Exception:
                break
        result["expectations"] = checks
        failed = [check for check in checks if not check["passed"]]
        if failed:
            result["status"] = "failed"
            result["failure_reason"] = "; ".join(
                f"expected {check['type']} {check['value']!r}" for check in failed
            )
        else:
            result["status"] = "passed"

    @staticmethod
    def _check(browser, expectation: dict) -> bool:
        # Each probe is quick; the polling loop in _validate provides patience.
        value = expectation["value"]
        try:
            if expectation["type"] == "url_contains":
                return value.casefold() in browser.current_url().casefold()
            if expectation["type"] == "text_visible":
                body = browser.page.locator("body").inner_text(timeout=1_000)
                if value.casefold() in body.casefold():
                    return True
                # Input placeholders (e.g. "Global Search...") are real,
                # user-visible text but never part of the body's inner text.
                return browser.page.get_by_placeholder(value).first.is_visible(timeout=500)
            if expectation["type"] == "element_visible":
                if browser.page.get_by_text(value).first.is_visible(timeout=1_000):
                    return True
                if browser.page.get_by_placeholder(value).first.is_visible(timeout=500):
                    return True
                # aria-labelled controls have no visible text to match.
                return browser.page.get_by_label(value).first.is_visible(timeout=500)
        except Exception:
            return False
        return False

    # ------------------------------------------------------------------
    # Evidence and healing context
    # ------------------------------------------------------------------

    def _emit_live_frame(self, browser, test_id: str) -> None:
        """Publish the current browser view so the UI can show execution live."""
        try:
            from runner import SCREENSHOT_PATH

            SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Per-worker temp name keeps parallel executors from clobbering
            # each other mid-write; the rename itself is atomic.
            tmp_path = SCREENSHOT_PATH.with_name(f"exec-{test_id}.tmp.png")
            browser.screenshot(str(tmp_path))
            os.replace(tmp_path, SCREENSHOT_PATH)
            self.on_event({"type": "test_frame", "screenshot_path": str(SCREENSHOT_PATH)})
        except Exception:
            logger.debug("Could not publish a live execution frame", exc_info=True)

    @staticmethod
    def _page_snapshot(browser) -> dict:
        """Small textual snapshot of the live page for the Healer prompt."""
        snapshot = {"url": "", "controls": [], "fields": []}
        try:
            snapshot["url"] = browser.current_url()
            controls = browser.page.locator("button, a, [role='button'], [role='link']")
            seen = set()
            for index in range(min(controls.count(), 60)):
                text = " ".join((controls.nth(index).inner_text(timeout=300) or "").split())
                if text and text.casefold() not in seen:
                    seen.add(text.casefold())
                    snapshot["controls"].append(text[:80])
            fields = browser.page.locator("input, textarea, select")
            for index in range(min(fields.count(), 30)):
                field = fields.nth(index)
                label = (
                    field.get_attribute("placeholder")
                    or field.get_attribute("name")
                    or field.get_attribute("aria-label")
                    or ""
                )
                if label:
                    snapshot["fields"].append(label[:80])
        except Exception:
            logger.debug("Could not snapshot the page for healing", exc_info=True)
        return snapshot

    def _capture_evidence(self, browser, run_id: str, result: dict) -> None:
        try:
            if browser.page is None:
                return
            evidence_dir = RUNS_ROOT / run_id
            evidence_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = evidence_dir / f"{result['test_id']}.png"
            browser.screenshot(str(screenshot_path))
            result["evidence"]["screenshot"] = str(screenshot_path)
            result["evidence"]["final_url"] = browser.current_url()
        except Exception:
            logger.debug("Could not capture evidence for %s", result["test_id"], exc_info=True)
