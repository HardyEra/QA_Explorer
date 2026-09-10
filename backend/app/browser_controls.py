"""Observed, unambiguous browser interactions shared by task and QA execution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import monotonic


class ControlError(RuntimeError):
    """A target or its resulting state could not be established."""


def attribute(name, value):
    return f"[{name}={json.dumps(str(value))}]"


class BrowserControls:
    def __init__(self, page, timeout_ms=5000):
        self.page = page
        self.timeout_ms = timeout_ms

    def resolve(self, target, kind="control"):
        """Prefer exact identity; refuse duplicate fields rather than guessing first."""
        if not target:
            raise ControlError("A specific target is required")
        deadline = monotonic() + self.timeout_ms / 1000
        while True:
            for strategy in ("id", "name", "label", "placeholder", "role", "text", "intent"):
                candidates = []
                for frame in self.page.frames:
                    if strategy == "id":
                        locator = frame.locator(attribute("id", target.lstrip("#")))
                    elif strategy == "name":
                        locator = frame.locator(attribute("name", target))
                    elif strategy == "label":
                        locator = frame.get_by_label(target, exact=True)
                    elif strategy == "placeholder":
                        locator = frame.get_by_placeholder(target, exact=True)
                    elif strategy == "role":
                        role = {"check": "checkbox", "select": "combobox"}.get(kind, "button")
                        locator = frame.get_by_role(role, name=target, exact=True)
                        if kind == "control":
                            for role in ("link", "option", "tab", "menuitem"):
                                locator = locator.or_(
                                    frame.get_by_role(role, name=target, exact=True)
                                )
                    elif strategy == "intent":
                        intent = str(target).casefold()
                        if kind != "field" or intent not in ("email", "username", "password"):
                            continue
                        locator = frame.locator(
                            'input[type="password"]'
                            if intent == "password"
                            else 'input[type="email"], input[autocomplete="username"]'
                        )
                    else:
                        if kind in ("field", "value", "file", "check"):
                            continue
                        locator = frame.get_by_text(target, exact=True)
                    for i in range(locator.count()):
                        item = locator.nth(i)
                        info = item.evaluate("e => ({tag:e.tagName, type:e.type})")
                        if kind in ("field", "value") and (
                            info["tag"]
                            not in (
                                ("INPUT", "TEXTAREA", "SELECT")
                                if kind == "value"
                                else ("INPUT", "TEXTAREA")
                            )
                            or info.get("type")
                            in ("hidden", "checkbox", "radio", "file", "submit", "button")
                        ):
                            continue
                        if kind == "file" and info.get("type") != "file":
                            continue
                        if kind == "check" and info.get("type") != "checkbox":
                            continue
                        if kind not in ("file", "check") and not item.is_visible():
                            continue
                        candidates.append(item)
                if len(candidates) > 1:
                    raise ControlError(
                        f"Ambiguous {kind} target {target!r}: use its unique field id"
                    )
                if candidates:
                    return candidates[0]
            if monotonic() >= deadline:
                raise ControlError(f"No observed {kind} matches {target!r}")
            self.page.wait_for_timeout(150)

    def _poll(self, predicate, message, timeout_ms=None):
        deadline = monotonic() + (timeout_ms or self.timeout_ms) / 1000
        while True:
            if predicate():
                return
            if monotonic() >= deadline:
                raise ControlError(message)
            self.page.wait_for_timeout(100)

    def fill(self, target, value):
        field = self.resolve(target, "field")
        value = str(value)
        phone = field.get_attribute("type") == "tel"

        def matches():
            actual = field.input_value(timeout=500)
            if phone:
                return re.sub(r"\D", "", actual) == re.sub(r"\D", "", value)
            return actual == value

        field.fill(value, timeout=self.timeout_ms)
        field.blur()
        # Give controlled React inputs time to render their actual stored value.
        self.page.wait_for_timeout(200)
        try:
            self._poll(matches, "Field did not retain the requested value", 700)
        except ControlError:
            # One bounded repair, on the SAME field. Never guess a country code.
            flag = field.locator("xpath=..").locator("[title*='+']")
            prefix = None
            if phone and flag.count() == 1:
                match = re.search(r"\+\s*(\d+)", flag.get_attribute("title") or "")
                if match and value.startswith("+" + match[1]):
                    prefix = "+" + match[1]
            field.fill(prefix or "", timeout=self.timeout_ms)
            self.page.wait_for_timeout(250)
            field.fill(value, timeout=self.timeout_ms)
            field.blur()
            self.page.wait_for_timeout(200)
            self._poll(matches, f"Value mismatch after filling {target!r}")

    def click(self, target):
        item = self.resolve(target)
        # Click only once. A later missing outcome must not resubmit a record.
        item.click(timeout=self.timeout_ms)

    def check(self, target, checked=True):
        field = self.resolve(target, "check")
        if field.is_checked() == checked:
            return
        if field.is_visible():
            field.set_checked(checked, timeout=self.timeout_ms)
        else:
            labels = field.evaluate("e => Array.from(e.labels || []).map(l => l.htmlFor)")
            field_id = field.get_attribute("id")
            if not field_id or field_id not in labels:
                raise ControlError("Hidden checkbox has no associated label")
            frame = field.element_handle().owner_frame()
            label = frame.locator("label" + attribute("for", field_id))
            if label.count() != 1:
                raise ControlError("Checkbox label is ambiguous")
            label.click(timeout=self.timeout_ms)
        self._poll(lambda: field.is_checked() == checked, "Checkbox state did not update")

    def select(self, target, option):
        control = self.resolve(target, "select")
        if control.evaluate("e => e.tagName") == "SELECT":
            control.select_option(label=option, timeout=self.timeout_ms)
            self._poll(
                lambda: control.locator("option:checked").inner_text().strip() == option,
                "Selected option did not update",
            )
            return
        control.click(timeout=self.timeout_ms)
        # Options can be rendered in a portal, so resolve against the current page.
        self.resolve(option).click(timeout=self.timeout_ms)

        stable_id = control.get_attribute("id")

        def selected():
            node = self.resolve(stable_id or target, "select")
            return node.evaluate(
                "(e, value) => (e.value || e.innerText || '').trim() === value", option
            )

        self._poll(selected, "Dropdown did not retain the selected option")

    def upload(self, target, path):
        path = Path(path)
        if not path.is_file():
            raise ControlError("The requested upload asset does not exist")
        if target:
            field = self.resolve(target, "file")
        else:
            fields = [f.locator('input[type="file"]') for f in self.page.frames]
            if sum(f.count() for f in fields) != 1:
                raise ControlError("Multiple or missing file inputs: specify the upload field id")
            field = next(f for f in fields if f.count() == 1)
        field.set_input_files(str(path), timeout=self.timeout_ms)
        self._poll(
            lambda: (
                field.evaluate("e => Array.from(e.files || []).map(f => f.name)") == [path.name]
            ),
            "File attachment was not retained",
        )


def page_evidence(page):
    """Capture current form state without password/OTP values or hidden tokens."""
    frames = []
    for frame in page.frames:
        try:
            frames.append(
                frame.evaluate("""() => {
                const visible = e => !!(e.getClientRects().length) && getComputedStyle(e).visibility !== 'hidden';
                return {
                    url: location.href,
                    text: (document.body?.innerText || '').slice(0, 18000),
                    fields: Array.from(document.querySelectorAll('input,textarea,select')).filter(e =>
                        e.type !== 'hidden' && (visible(e) || ['file','checkbox'].includes(e.type))
                    ).slice(0, 100).map(e => ({
                        id:e.id, name:e.name, label:Array.from(e.labels || []).map(l=>l.innerText).join(' '),
                        placeholder:e.placeholder || '', type:e.type, required:e.required,
                        disabled:e.disabled, checked:e.checked,
                        value: /password|otp|secret|token|verification.?code/i.test([e.type,e.name,e.id,e.autocomplete].join(' '))
                            ? '[REDACTED]' : e.type === 'file' ? Array.from(e.files||[]).map(f=>f.name).join(', ') : e.value,
                        valid:e.validity?.valid, validation:e.validationMessage || ''
                    })),
                    alerts:Array.from(document.querySelectorAll('[role="alert"],[aria-live="assertive"]')).filter(visible).map(e=>e.innerText)
                };
            }""")
            )
        except Exception:
            continue
    return frames


def check_expectation(page, expectation):
    """Check one explicit UI assertion; field assertions require unique targets."""
    kind, value = expectation.get("type"), str(expectation.get("value", ""))
    try:
        if kind == "url_contains":
            return bool(value) and value.casefold() in page.url.casefold()
        if kind in ("field_value", "checked"):
            controls = BrowserControls(page, timeout_ms=250)
            field = controls.resolve(
                expectation.get("target", ""), "check" if kind == "checked" else "value"
            )
            if kind == "checked":
                return value in ("true", "false") and field.is_checked() == (value == "true")
            return field.input_value() == value
        if not value or kind not in ("text_visible", "element_visible"):
            return False
        for frame in page.frames:
            locators = [frame.get_by_text(value, exact=False)]
            if kind == "element_visible":
                locators.extend(
                    [
                        frame.get_by_label(value, exact=True),
                        frame.get_by_placeholder(value, exact=True),
                    ]
                )
            for locator in locators:
                if any(locator.nth(i).is_visible() for i in range(locator.count())):
                    return True
    except Exception:
        return False
    return False
