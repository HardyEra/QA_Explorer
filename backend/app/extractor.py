from models import Action, Observation, Page
from action_deduplicator import ActionDeduplicator

class PageExtractor:

    def __init__(self, page, action_registry):
        self.page = page
        self.action_registry = action_registry
        self.deduplicator = ActionDeduplicator()
    # --------------------------
    # Safe helpers
    # --------------------------

    def _safe_text(self, element):
        """
        Safely extract visible text from an element.
        Never throws.
        """

        try:
            text = element.text_content(timeout=500)
            if text and text.strip():
                return text.strip()
        except:
            pass

        for attr in [
            "value",
            "aria-label",
            "title",
            "placeholder",
            "alt",
        ]:
            try:
                value = element.get_attribute(attr)
                if value and value.strip():
                    return value.strip()
            except:
                pass

        return ""


    def _infer_name(self, element):

            attributes = []

            for attr in [
                "aria-label",
                "title",
                "placeholder",
                "data-testid",
                "id",
                "class",
                "name",
            ]:
                try:
                    value = element.get_attribute(attr)
                    if value:
                        attributes.append(value.lower())
                except:
                    pass

            combined = " ".join(attributes)

            mappings = {
                "shopping_cart": "Shopping Cart",
                "cart": "Shopping Cart",
                "basket": "Shopping Cart",
                "checkout": "Checkout",
                "burger": "Open Menu",
                "menu": "Open Menu",
                "hamburger": "Open Menu",
                "profile": "Profile",
                "account": "Profile",
                "user": "Profile",
                "notification": "Notifications",
                "bell": "Notifications",
                "search": "Search",
                "filter": "Filter",
                "sort": "Sort",
                "home": "Home",
                "settings": "Settings",
                "logout": "Logout",
                "login": "Login",
                "close": "Close",
                "back": "Back",
                "next": "Next",
            }

            for key, value in mappings.items():
                if key in combined:
                    return value

            if attributes:
                return attributes[0].replace("-", " ").replace("_", " ").title()

            return None
    # --------------------------
    # Inputs
    # --------------------------

    def get_inputs(self):

        inputs = []

        for element in self.page.locator("input").all():

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

        locator = self.page.locator(
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

        for form in self.page.locator("form").all():

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
            "clickable": "[role='button'], [onclick], [tabindex], div[role='button']"
        }

        for action_type, selector in selectors.items():

            for element in self.page.locator(selector).all():

                try:

                    text = self._safe_text(element)

                    if not text:
                        text = self._infer_name(element)

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
                            type=action_type
                        )
                    )

                except Exception as e:

                    print(f"[Extractor] Skipping {action_type}: {e}")

                    continue

        return actions

    # --------------------------
    # Observation
    # --------------------------

    def observe(self):

        self.action_registry.clear()

        return Observation(
            page=Page(
                title=self.page.title(),
                url=self.page.url
            ),
            actions=self.deduplicator.deduplicate(
                self.get_actions()
            ),
            inputs=self.get_inputs(),
            forms=self.get_forms()
        )