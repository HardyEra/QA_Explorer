from models import Action, Observation, Page
from action_deduplicator import ActionDeduplicator
from accessibility_extractor import AccessibilityExtractor
from page_summary import PageSummaryGenerator
import logging
import re


_UTILITY_LABEL = re.compile(
    r"^(?:[a-z-]+:)*(?:p|px|py|pt|pr|pb|pl|m|mx|my|mt|mr|mb|ml|w|h|min-w|max-w|min-h|max-h|text|bg|border|rounded|shadow|z)(?:-|\[)",
    re.IGNORECASE,
)
_UTILITY_WORDS = {
    "block", "inline", "inline-block", "flex", "grid", "hidden", "visible",
    "fixed", "absolute", "relative", "sticky", "bottom", "top", "left", "right",
}
_GENERATED_LABEL = re.compile(r"^[a-z0-9_-]{20,}$", re.IGNORECASE)


logger = logging.getLogger(__name__)

class PageExtractor:

    def __init__(self, page, action_registry):
        self.page = page
        self.action_registry = action_registry
        self.deduplicator = ActionDeduplicator()
        self.accessibility_extractor = AccessibilityExtractor(page, action_registry)
        self.page_summary_generator = PageSummaryGenerator()
        self._overlay = {"ui_context": "NORMAL_PAGE", "action_context": "page", "container": None, "selector": None}

    def _detect_active_overlay(self):
        """Identify the top-most visible interaction surface without UI-library coupling.

        Framework selectors are only signals: ARIA roles, visibility, viewport
        geometry and stacking order decide which surface is active.  The
        returned selector scopes later locator queries, including React portals
        mounted outside the application's root node.
        """
        try:
            overlay = self.page.evaluate("""() => {
                const visible = node => {
                    const style = getComputedStyle(node), rect = node.getBoundingClientRect();
                    return style.display !== 'none' && style.visibility !== 'hidden'
                        && Number(style.opacity || 1) > 0 && rect.width > 0 && rect.height > 0;
                };
                const selectorFor = node => {
                    const parts = [];
                    for (let current = node; current && current.nodeType === 1 && current !== document.body; current = current.parentElement) {
                        let part = current.tagName.toLowerCase();
                        if (current.id) { part += '#' + CSS.escape(current.id); parts.unshift(part); break; }
                        let index = 1, sibling = current;
                        while ((sibling = sibling.previousElementSibling)) if (sibling.tagName === current.tagName) index++;
                        part += `:nth-of-type(${index})`;
                        parts.unshift(part);
                    }
                    return 'body > ' + parts.join(' > ');
                };
                const titleFor = node => {
                    const labelled = (node.getAttribute('aria-labelledby') || '').split(/\\s+/)
                        .map(id => document.getElementById(id)?.innerText || '').filter(Boolean).join(' ');
                    const heading = node.querySelector('[role="heading"], h1, h2, h3, h4, h5, h6');
                    return (labelled || node.getAttribute('aria-label') || heading?.innerText || '').replace(/\\s+/g, ' ').trim() || null;
                };
                const candidates = [];
                const add = (node, kind, signal) => {
                    if (!node || !visible(node) || node.classList.contains('MuiBackdrop-root')) return;
                    const rect = node.getBoundingClientRect(), style = getComputedStyle(node);
                    const z = Number.parseFloat(style.zIndex) || 0;
                    // A presentation element is often a wrapper. It is a
                    // surface only when it has dialog-like content or is a
                    // full-screen overlay layer.
                    if (signal === 'presentation' && !node.querySelector('[role="dialog"], dialog, [aria-modal="true"]')
                        && !(rect.width >= innerWidth * .8 && rect.height >= innerHeight * .8 && z > 0)) return;
                    candidates.push({node, kind, signal, z, area: rect.width * rect.height, order: candidates.length});
                };
                document.querySelectorAll('[role="dialog"], dialog, [aria-modal="true"]').forEach(n => add(n, 'MODAL', 'dialog'));
                document.querySelectorAll('.MuiModal-root, .MuiDrawer-root, .MuiPopover-root, [data-radix-popper-content-wrapper], [data-headlessui-state]').forEach(n => {
                    const classes = n.className?.toString() || '';
                    add(n, /Drawer/i.test(classes) ? 'DRAWER' : /Popover|popper/i.test(classes) ? 'POPOVER' : 'MODAL', 'library');
                });
                document.querySelectorAll('[role="presentation"]').forEach(n => add(n, 'MODAL', 'presentation'));
                // Generic portals: a hidden application root plus locked body
                // is strong evidence that an outside-root child is the active overlay.
                const rootHidden = document.querySelector('#root[aria-hidden="true"]');
                const locked = /overflow\\s*:\\s*hidden/i.test(document.body.getAttribute('style') || '')
                    || getComputedStyle(document.body).overflow === 'hidden';
                if (rootHidden && locked) {
                    Array.from(document.body.children).filter(n => n !== rootHidden).forEach(n => add(n, 'MODAL', 'portal'));
                }
                if (!candidates.length) return {ui_context: 'NORMAL_PAGE', action_context: 'page', container: null, selector: null};
                candidates.sort((a, b) => b.z - a.z || b.order - a.order || a.area - b.area);
                const best = candidates[0];
                return {ui_context: best.kind, action_context: best.kind.toLowerCase(), container: titleFor(best.node), selector: selectorFor(best.node), signal: best.signal};
            }""")
            if overlay["ui_context"] != "NORMAL_PAGE":
                logger.info("Overlay detected: %s (%s)", overlay.get("signal", "generic overlay"), overlay["ui_context"])
                logger.info("Context: %s", overlay["ui_context"])
                logger.info("Extracting actions only from active %s.", overlay["action_context"])
            return overlay
        except Exception as exc:
            logger.debug("Overlay detection failed; using page surface: %s", exc)
            return {"ui_context": "NORMAL_PAGE", "action_context": "page", "container": None, "selector": None}

    def _surface(self):
        selector = self._overlay.get("selector")
        return self.page.locator(selector) if selector else self.page.locator("body")
    # --------------------------
    # Safe helpers
    # --------------------------

    def _safe_text(self, element):
        """Return a human-readable label using semantic sources only."""
        candidates = []
        try:
            candidates.append(element.inner_text(timeout=500))
        except Exception:
            pass

        for attribute in ("aria-label",):
            try:
                candidates.append(element.get_attribute(attribute))
            except Exception:
                pass

        candidates.append(self._labelled_by_text(element))

        for attribute in ("title", "placeholder", "value"):
            try:
                candidates.append(element.get_attribute(attribute))
            except Exception:
                pass

        candidates.append(self._accessible_name(element))
        for candidate in candidates:
            label = self._normalise_label(candidate)
            if self._is_meaningful_label(label):
                return label
        return ""


    def _infer_name(self, element):
        # Kept for backward compatibility with callers; class/id inference is
        # intentionally forbidden because it yields implementation noise.
        label = self._accessible_name(element)
        return label if self._is_meaningful_label(label) else None

    @staticmethod
    def _normalise_label(value):
        # Icon controls sometimes expose only a zero-width space as text.
        # Treat that as empty so semantic fallbacks can name a real control;
        # it must never become a human-facing action label.
        text = str(value or "").replace("\u200b", "").replace("\ufeff", "")
        return " ".join(text.split())

    @staticmethod
    def _is_meaningful_label(label):
        if not label:
            return False
        normalized = label.casefold()
        if normalized in _UTILITY_WORDS or _UTILITY_LABEL.match(normalized):
            return False
        if normalized.startswith(("hover:", "focus:", "active:", "fixed bottom", "block ")):
            return False
        return not _GENERATED_LABEL.fullmatch(normalized)

    @staticmethod
    def _labelled_by_text(element):
        try:
            return element.evaluate(
                """node => (node.getAttribute('aria-labelledby') || '').split(/\\s+/)
                    .filter(Boolean)
                    .map(id => document.getElementById(id))
                    .filter(Boolean)
                    .map(label => label.innerText || label.textContent || '')
                    .join(' ')"""
            )
        except Exception:
            return ""

    @staticmethod
    def _accessible_name(element):
        try:
            return element.evaluate(
                """node => {
                    const labels = node.labels ? Array.from(node.labels) : [];
                    return labels.map(label => label.innerText || label.textContent || '')
                        .join(' ') || node.getAttribute('alt') || '';
                }"""
            )
        except Exception:
            return ""

    @staticmethod
    def _action_context(element):
        """Extract small semantic context without retaining DOM references."""
        try:
            return element.evaluate(
                """node => {
                    const ancestors = [];
                    let current = node.parentElement;
                    while (current && ancestors.length < 5) {
                        ancestors.push({
                            tag: current.tagName.toLowerCase(),
                            role: current.getAttribute('role') || '',
                            label: [
                                current.getAttribute('aria-label'), current.getAttribute('title'),
                                current.id, current.getAttribute('data-testid')
                            ].filter(Boolean).join(' ').toLowerCase()
                        });
                        current = current.parentElement;
                    }
                    const semantic = ancestors.map(item => `${item.tag} ${item.role} ${item.label}`).join(' ');
                    let container = null;
                    if (/avatar|profile|account|user/.test(semantic)) container = 'profile';
                    else if (ancestors.some(item => item.role === 'dialog' || item.tag === 'dialog')) container = 'dialog';
                    else if (ancestors.some(item => item.tag === 'form' || item.role === 'form')) container = 'form';
                    else if (ancestors.some(item => item.tag === 'nav' || item.role === 'navigation')) container = 'navigation';
                    else if (ancestors.some(item => item.tag === 'aside' || item.role === 'complementary')) container = 'sidebar';
                    else if (ancestors.some(item => item.role === 'toolbar')) container = 'toolbar';
                    else if (ancestors.some(item => item.role === 'menu' || item.role === 'menubar')) container = 'menu';
                    else if (ancestors.some(item => item.tag === 'header' || item.role === 'banner')) container = 'header';
                    return {
                        role: node.getAttribute('role') || node.tagName.toLowerCase(),
                        parent_role: node.parentElement?.getAttribute('role') || null,
                        container_context: container,
                        ancestor_tags: ancestors.map(item => item.tag),
                        aria_role: node.getAttribute('role'),
                        tag_name: node.tagName.toLowerCase(),
                    };
                }"""
            )
        except Exception:
            return {}
    # --------------------------
    # Inputs
    # --------------------------

    def get_inputs(self):

        inputs = []

        for element in self._surface().locator("input").all():

            try:
                inputs.append({
                    "type": element.get_attribute("type"),
                    "name": element.get_attribute("name"),
                    "placeholder": element.get_attribute("placeholder"),
                    "id": element.get_attribute("id")
                })
            except:
                continue

        return inputs

    # --------------------------
    # Buttons
    # --------------------------

    def get_buttons(self):

        buttons = []

        locator = self._surface().locator(
            "button, input[type='submit'], input[type='button']"
        )

        for element in locator.all():

            text = self._safe_text(element)

            if text:
                buttons.append(text)

        return buttons

    # --------------------------
    # Forms
    # --------------------------

    def get_forms(self):

        forms = []

        for form in self._surface().locator("form").all():

            try:

                form_data = {
                    "inputs": [],
                    "buttons": []
                }

                for input_element in form.locator("input").all():

                    try:
                        form_data["inputs"].append({
                            "type": input_element.get_attribute("type"),
                            "name": input_element.get_attribute("name"),
                            "placeholder": input_element.get_attribute("placeholder"),
                            "id": input_element.get_attribute("id")
                        })
                    except:
                        continue

                for button in form.locator(
                    "button, input[type='submit'], input[type='button']"
                ).all():

                    text = self._safe_text(button)

                    if text:
                        form_data["buttons"].append(text)

                forms.append(form_data)

            except:
                continue

        return forms

    # --------------------------
    # Actions
    # --------------------------

    def get_actions(self):

        actions = []

        selectors = {
            "link": "a",
            "button": "button, input[type='submit'], input[type='button']",
            # Native file inputs are often visually styled by a companion
            # label/button. They are still the reliable Playwright target for
            # a user-supplied asset and must remain within the active overlay.
            "file_upload": "input[type='file']",
            "clickable": (
                "[role='button'], [role='link'], [role='menuitem'], [role='treeitem'], "
                "[role='option'], [onclick], [tabindex], [class~='cursor-pointer'], "
                "[class*='cursor-pointer'], [style*='cursor: pointer'], div, span"
            ),
            # Some applications attach React handlers directly to an SVG menu
            # icon. ``_semantic_icon_label`` filters this broad selector down
            # to the one verified hamburger glyph below.
            "icon": "svg",
        }

        for action_type, selector in selectors.items():

            for element in self._surface().locator(selector).all():

                try:
                    # Do not plan clicks on hidden menu entries, modals, or
                    # duplicate controls. They cannot be interacted with yet.
                    if not self._is_actionable_element(element, action_type):
                        continue

                    text = (
                        self._safe_text(element)
                        or ("Choose Files" if action_type == "file_upload" else "")
                        or self._semantic_test_label(element)
                        or self._semantic_icon_label(element)
                    )

                    if not text:
                        continue

                    action_id = self.action_registry.register(
                        locator=element,
                        text=text,
                        action_type=action_type
                    )

                    actions.append(
                        Action(
                            id=action_id,
                            text=text,
                            type=action_type,
                            **self._action_context(element),
                            context=self._overlay["action_context"],
                            container=self._overlay.get("container"),
                        )
                    )

                except Exception as e:

                    print(f"[Extractor] Skipping {action_type}: {e}")

                    continue

        # Design-system option cards are frequently implemented as a ``div``
        # with a framework event handler. They therefore expose only a
        # paragraph/StaticText in the accessibility tree (for example,
        # "Client"), not a button or link. Register the visible label as the
        # click target: a normal Playwright click on that text bubbles to the
        # card handler without needing a force click or implementation-specific
        # CSS selector.
        actions.extend(self._dialog_option_actions())

        if self._overlay["ui_context"] != "NORMAL_PAGE":
            logger.info(
                "%s actions found: %s",
                self._overlay["action_context"].capitalize(),
                ", ".join(action.text for action in actions) or "None",
            )
            logger.info("Background actions suppressed: active overlay is the only extraction surface")

        return actions

    def _dialog_option_actions(self):
        """Find visible, card-like choices in a dialog with missing semantics.

        This is intentionally scoped to dialogs and requires a compact leaf
        paragraph inside a card-sized ancestor. It avoids turning explanatory
        modal copy or the dialog title into planner actions while supporting
        custom React/Vue option cards with no ARIA click role.
        """
        actions = []
        seen_labels = set()
        option_labels = self._surface().locator("p, [role='paragraph']")

        for element in option_labels.all():
            try:
                if not self._is_actionable_element(element):
                    continue
                text = self._safe_text(element)
                normalized = self._normalise_label(text).casefold()
                if not self._is_meaningful_label(text) or normalized in seen_labels:
                    continue
                if not self._is_dialog_card_label(element):
                    continue

                action_id = self.action_registry.register(
                    locator=element,
                    text=text,
                    action_type="dialog_option",
                )
                actions.append(
                    Action(
                        id=action_id,
                        text=text,
                        type="dialog_option",
                        **self._action_context(element),
                        context=self._overlay["action_context"],
                        container=self._overlay.get("container"),
                    )
                )
                seen_labels.add(normalized)
                logger.info("Discovered custom dialog option: %s", text)
            except Exception as exc:
                logger.debug("Skipping custom dialog option: %s", exc)

        return actions

    @staticmethod
    def _is_dialog_card_label(element):
        """Return whether a paragraph is the label of a visual dialog option."""
        try:
            return bool(element.evaluate("""node => {
                let dialog = node.closest('[role="dialog"], dialog');
                // Some custom modals omit role=dialog. Recognise only a
                // centred, overlay-like panel as a fallback, never an
                // arbitrary application container.
                if (!dialog) {
                    for (let current = node.parentElement; current; current = current.parentElement) {
                        const rect = current.getBoundingClientRect();
                        const style = window.getComputedStyle(current);
                        const centered = Math.abs((rect.left + rect.width / 2) - window.innerWidth / 2)
                            <= window.innerWidth * 0.2;
                        const panelSized = rect.width >= 240 && rect.width <= window.innerWidth * 0.85
                            && rect.height >= 120 && rect.height <= window.innerHeight * 0.9;
                        const overlayLayer = style.position === 'fixed' || style.position === 'absolute'
                            || Number.parseInt(style.zIndex || '0', 10) > 0;
                        if (centered && panelSized && overlayLayer) {
                            dialog = current;
                            break;
                        }
                    }
                }
                if (!dialog || node.children.length || !node.textContent.trim()) return false;
                const dialogRect = dialog.getBoundingClientRect();
                const text = node.textContent.trim();
                for (let current = node.parentElement;
                     current && current !== dialog;
                     current = current.parentElement) {
                    const rect = current.getBoundingClientRect();
                    const style = window.getComputedStyle(current);
                    const actionHint = current.matches(
                        'button, a, [role="button"], [role="link"], [onclick], [tabindex]'
                    ) || style.cursor === 'pointer';
                    const cardSized = rect.width >= 72 && rect.height >= 72
                        && rect.width <= dialogRect.width * 0.7
                        && rect.height <= dialogRect.height * 0.9;
                    const hasVisualChoice = Boolean(current.querySelector('img, svg'))
                        && current.innerText.trim() === text;
                    if (actionHint || (cardSized && hasVisualChoice)) return true;
                }
                return false;
            }"""))
        except Exception:
            return False

    @staticmethod
    def _is_actionable_element(element, action_type=None):
        """Reject DOM nodes that are visible but intentionally non-interactive.

        Component libraries often keep hidden native inputs in the layout for
        a combobox. Those inputs can look visible to Playwright while being
        aria-hidden or tabindex=-1, and they must never become planner actions.
        """
        try:
            if action_type != "file_upload" and not element.is_visible():
                return False
            if action_type == "icon":
                return True
            if action_type != "file_upload" and (element.get_attribute("aria-hidden") or "").casefold() == "true":
                return False
            if element.get_attribute("disabled") is not None:
                return False
            tabindex = element.get_attribute("tabindex")
            if tabindex and int(tabindex) < 0:
                return False
            if (element.evaluate("node => node.tagName.toLowerCase()") or "") == "input":
                input_type = (element.get_attribute("type") or "text").casefold()
                if input_type == "hidden" or (input_type == "file" and action_type != "file_upload"):
                    return False
            if action_type == "clickable":
                # React commonly delegates handlers, so a listener is not
                # reliably introspectable. These are the framework-neutral DOM
                # interaction signals that remain observable.
                interactive = element.evaluate("""node => {
                    const style = getComputedStyle(node);
                    return node.hasAttribute('onclick') || typeof node.onclick === 'function'
                        || node.matches('[role="button"], [role="link"], [role="menuitem"], [role="treeitem"], [role="option"]')
                        || node.tabIndex >= 0 || style.cursor === 'pointer'
                        || /(^|\\s)cursor-pointer(\\s|$)/.test(node.className?.toString() || '');
                }""")
                if not interactive:
                    return False
                # A pointer-styled layout wrapper can contain several real
                # controls. Registering it produces a concatenated label such
                # as "Log In ... Forgot Password ... Log In" and competes with
                # the controls a user can actually identify. Keep leaf divs
                # (including ``div.cursor-pointer`` upload targets), but let a
                # semantic descendant own the interaction when one exists.
                if element.evaluate("""node => {
                    const tag = node.tagName.toLowerCase();
                    return (tag === 'div' || tag === 'span') && Boolean(node.querySelector(
                        'button, a, input, select, textarea, [role="button"], [role="link"], '
                        + '[role="menuitem"], [role="treeitem"], [role="option"], [onclick], [tabindex]'
                    ));
                }"""):
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def _semantic_icon_label(element):
        """Name the one common unlabeled control needed for navigation.

        Some applications render a hamburger as an icon-only clickable node
        without an accessible name. We do not infer labels from CSS or class
        names. This intentionally narrow fallback recognises only a compact,
        SVG-only, top-left header control and names it from its user-visible
        function: opening the navigation menu.
        """
        try:
            return element.evaluate(
                """node => {
                    const rect = node.getBoundingClientRect();
                    const role = (node.getAttribute('role') || '').toLowerCase();
                    const tag = node.tagName.toLowerCase();
                    const isClickable = tag === 'button' || role === 'button'
                        || node.hasAttribute('onclick') || node.tabIndex >= 0;
                    const inDialog = Boolean(node.closest('[role="dialog"], dialog'));
                    const isCompactHeaderControl = rect.top >= 0 && rect.top <= 96
                        && rect.left >= 0 && rect.left <= Math.max(160, window.innerWidth * 0.2)
                        && rect.width > 0 && rect.width <= 80 && rect.height > 0 && rect.height <= 80;
                    const hamburgerGlyph = tag === 'svg' && node.getAttribute('viewBox') === '0 0 24 24'
                        && Array.from(node.querySelectorAll('path')).some(path => {
                            const data = path.getAttribute('d') || '';
                            return (data.match(/[hH]20/g) || []).length >= 3
                                && (data.match(/[vV]2/g) || []).length >= 3;
                        });
                    if (!inDialog && isCompactHeaderControl && (
                        (isClickable && node.querySelector('svg')) || hamburgerGlyph
                    )) {
                        return 'Open navigation menu';
                    }
                    return '';
                }"""
            ) or ""
        except Exception:
            return ""

    @staticmethod
    def _semantic_test_label(element):
        """Name the one explicitly identified, otherwise unlabeled cart link.

        Some test applications expose key navigation only through stable
        semantic hooks. A cart link can be entirely unlabeled, while a sidebar
        logout entry can lose its text during a transient menu render. Keep
        these exceptions exact rather than inferring actions from CSS names or
        arbitrary implementation ids.
        """
        try:
            return element.evaluate(
                """node => {
                    const hook = (node.getAttribute('data-test') || '').trim().toLowerCase();
                    const labels = {
                        'shopping-cart-link': 'Shopping cart',
                        'logout-sidebar-link': 'Logout',
                    };
                    return labels[hook] || '';
                }"""
            ) or ""
        except Exception:
            return ""

    # --------------------------
    # Observation
    # --------------------------

    def observe(self):

        self.action_registry.clear()
        self._overlay = self._detect_active_overlay()

        # The accessibility tree is the primary source of page semantics, but
        # it may omit custom sidebar/menu controls exposed by component
        # libraries. Supplement it with meaningful DOM actions instead of
        # losing a workflow entry point such as a navigation module.
        # AX snapshots describe the whole document and cannot reliably map a
        # portal action back to its owning DOM subtree.  For an active overlay,
        # DOM extraction is both more complete (custom div click targets) and
        # safely scoped to the interaction surface.
        accessibility = None if self._overlay["ui_context"] != "NORMAL_PAGE" else self.accessibility_extractor.extract()
        if accessibility is not None:
            page = Page(title=self.page.title(), url=self.page.url)
            accessibility_labels = {
                self._normalise_label(action.text).casefold()
                for action in accessibility.actions
            }
            dom_actions = self.get_actions()
            supplemental_dom_actions = [
                action
                for action in dom_actions
                if self._normalise_label(action.text).casefold() not in accessibility_labels
            ]
            if supplemental_dom_actions:
                logger.info(
                    "Supplemented accessibility actions with %s DOM action(s)",
                    len(supplemental_dom_actions),
                )
            return Observation(
                page=page,
                actions=self.deduplicator.deduplicate(
                    [*accessibility.actions, *supplemental_dom_actions]
                ),
                inputs=accessibility.inputs,
                forms=accessibility.forms,
                accessibility_tree=accessibility.tree,
                page_summary=self.page_summary_generator.generate(page.title, accessibility.tree),
                page_title=page.title,
                ui_context=self._overlay["ui_context"],
                active_container=self._overlay.get("container"),
            )

        return Observation(
            page=Page(
                title=self.page.title(),
                url=self.page.url
            ),
            actions=self.deduplicator.deduplicate(
                self.get_actions()
            ),
            inputs=self.get_inputs(),
            forms=self.get_forms(),
            accessibility_tree=[],
            page_summary="",
            page_title=self.page.title(),
            ui_context=self._overlay["ui_context"],
            active_container=self._overlay.get("container"),
        )
