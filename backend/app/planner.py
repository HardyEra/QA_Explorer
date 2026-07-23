from google import genai
from config import GOOGLE_API_KEY
import json
import logging
import os


logger = logging.getLogger(__name__)
MODEL_TIMEOUT_MS = int(os.getenv("MODEL_TIMEOUT_MS", "10000"))

class Planner:

    def __init__(self):
        # self.client = Groq(api_key=GROQ_API_KEY)
        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY is not configured; using local fallback planning")
            self.client = None
            self.remote_planning_available = False
        else:
            self.client = genai.Client(
                api_key=GOOGLE_API_KEY,
                http_options=genai.types.HttpOptions(timeout=MODEL_TIMEOUT_MS),
            )
            self.remote_planning_available = True

    def _fallback_plan(self, observation):
        """Make conservative progress when the model service is unavailable.

        The fallback is intentionally deterministic: it completes a familiar
        username/password form, then explores one remaining action at a time.
        This keeps the graph usable during transient model API failures.
        """
        inputs = observation.inputs
        username = next(
            (
                item
                for item in inputs
                if any(
                    term in " ".join(
                        str(item.get(key) or "").lower()
                        for key in ("name", "placeholder", "id")
                    )
                    for term in ("username", "user-name", "email")
                )
            ),
            None,
        )
        password = next(
            (item for item in inputs if item.get("type") == "password"),
            None,
        )

        if username and password:
            login_action = next(
                (
                    action
                    for action in observation.actions
                    if "login" in action.text.lower() or "sign in" in action.text.lower()
                ),
                observation.actions[0] if observation.actions else None,
            )
            if login_action:
                def target(item):
                    return item.get("placeholder") or item.get("name") or item.get("id")

                logger.warning("Using local fallback plan to submit the login form")
                return {
                    "steps": [
                        {
                            "type": "fill",
                            "target": target(username),
                            "value": os.getenv("QA_USERNAME", "standard_user"),
                        },
                        {
                            "type": "fill",
                            "target": target(password),
                            "value": os.getenv("QA_PASSWORD", "secret_sauce"),
                        },
                        {"type": "click", "action_id": login_action.id},
                    ]
                }

        if observation.actions:
            action = observation.actions[0]
            logger.warning(
                "Using local fallback plan to explore action %s (%s)",
                action.id,
                action.text,
            )
            return {"steps": [{"type": "click", "action_id": action.id}]}

        return {"steps": []}

    def plan(self, observation, visited_pages, visited_actions):

        visited_pages_text = "\n".join(visited_pages) or "None"

        visited_actions_text = "\n".join(visited_actions) or "None"

        actions = "\n".join(
            f"{a.id} - {a.text} ({a.type})"
            for a in observation.actions
        )

        inputs = "\n".join(
            f"- Placeholder: {i.get('placeholder', '')}, Name: {i.get('name', '')}, Type: {i.get('type', '')}"
            for i in observation.inputs
        )

        prompt = f"""
You are an autonomous QA explorer.

Current Page

Title:
{observation.page.title}

URL:
{observation.page.url}

Available Inputs:
{inputs}

Available Actions:
{actions}

Already Visited Pages:
{visited_pages_text}

Already Executed Actions:
{visited_actions_text}

Authentication Rules:

- If the current URL contains "inventory", "cart", "checkout", "inventory-item", or any page other than the login page, you are already logged in.
- Never attempt to log in again unless the current URL is the login page.
Rules:
- Login ONLY if the current page is the login page.
- Otherwise never generate login steps.
- Prefer unexplored pages.
- If an action cannot be clicked because another UI element blocks it (menu, modal, dropdown), first open that UI element.
- Return exactly ONE JSON object.
- Do not explain your reasoning.
- Do not use markdown.
- Do not output anything except JSON.

Never repeat an action already executed on the same page.
Prefer actions that lead to unexplored pages.

Your goal is to explore the application autonomously.

Avoid redundant interactions.
If multiple actions have the same purpose, explore only one representative example unless there is evidence they behave differently.
Prioritize discovering new pages and unique functionality over repeating identical operations.

JSON Format:

{{
    "steps":[
        {{
            "type":"fill",
            "target":"Username",
            "value":"standard_user"
        }},
        {{
            "type":"fill",
            "target":"Password",
            "value":"secret_sauce"
        }},
        {{
            "type":"click",
            "action_id":0
        }}
    ]
}}
"""

        if not self.remote_planning_available:
            return self._fallback_plan(observation)

        logger.info("Requesting plan from Gemini (timeout=%sms)", MODEL_TIMEOUT_MS)
        try:
            response = self.client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
        except Exception:
            # A 5xx/timeout is transient and should not terminate browser QA.
            # Avoid making every subsequent graph step wait for another timeout.
            logger.exception("Gemini plan request failed; switching to local fallback planning")
            self.remote_planning_available = False
            return self._fallback_plan(observation)

        content = response.text.strip()

        try:
            plan = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Gemini returned invalid JSON; using local fallback plan")
            return self._fallback_plan(observation)

        if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
            logger.warning("Gemini returned an invalid plan shape; using local fallback plan")
            return self._fallback_plan(observation)

        print(content)
        logger.info("Received and parsed Gemini plan")


        return plan
