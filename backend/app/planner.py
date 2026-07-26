from groq import Groq
from config import GROQ_API_KEY
import json
import logging
import os


logger = logging.getLogger(__name__)
MODEL_TIMEOUT_MS = int(os.getenv("MODEL_TIMEOUT_MS", "10000"))
PLANNER_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PLANNER_TEMPERATURE = 0


def _usage_details(response):
    """Normalize Groq usage metadata to Langfuse's provider-neutral fields."""
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    details = {}
    if input_tokens is not None:
        details["input_tokens"] = input_tokens
    if output_tokens is not None:
        details["output_tokens"] = output_tokens
    return details or None

class Planner:

    def __init__(self, observability):
        self.observability = observability
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY is not configured; using local fallback planning")
            self.client = None
            self.remote_planning_available = False
        else:
            self.client = Groq(
                api_key=GROQ_API_KEY,
                timeout=MODEL_TIMEOUT_MS / 1000,
            )
            self.remote_planning_available = True

    def _fallback_plan(self, observation, provided_username="", provided_password=""):
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
                            "value": provided_username or os.getenv("QA_USERNAME", "standard_user"),
                        },
                        {
                            "type": "fill",
                            "target": target(password),
                            "value": provided_password or os.getenv("QA_PASSWORD", "secret_sauce"),
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

    def plan(self, observation, visited_pages, visited_actions, config=None):

        visited_pages_text = "\n".join(visited_pages) or "None"

        visited_actions_text = "\n".join(visited_actions) or "None"

        actions = json.dumps(
            [
                {
                    "id": action.id,
                    "text": action.text,
                    "category": action.category,
                    "priority": action.priority,
                }
                for action in observation.actions
            ],
            indent=2,
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

Application context supplied by the user:
{getattr(config, "application_context", "") or "None"}

Exploration objective:
{getattr(config, "current_goal", "Autonomously discover application pages and actions")}

Credentials supplied by the user (use only when a matching login form is present):
Username: {getattr(config, "username", "") or "Not supplied"}
Password: {getattr(config, "password", "") or "Not supplied"}

Your objective is to discover meaningful business workflows.

Each available action contains a deterministic category and priority. Higher priority
means the action is more likely to advance primary application functionality. The
actions are already ranked in descending priority.

Always prioritize actions that advance the application's primary functionality.
Avoid low-value actions such as social media, footer links, and legal pages unless
there are no meaningful alternatives. Prefer unexplored pages and unique workflows.
Do not repeat an action already executed on the same page.

If an action cannot be clicked because another UI element blocks it (menu, modal,
dropdown), first open that UI element. Return exactly ONE JSON object, with no
reasoning, markdown, or text outside the JSON.

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
            return self._fallback_plan(observation, getattr(config, "username", ""), getattr(config, "password", ""))

        logger.info("Requesting plan from Groq model %s (timeout=%sms)", PLANNER_MODEL, MODEL_TIMEOUT_MS)
        with self.observability.generation(
            "Planner.generate_plan", model=PLANNER_MODEL,
            temperature=PLANNER_TEMPERATURE, input=prompt,
            metadata={"provider": "groq", "retry_count": 0},
        ) as generation:
            try:
                response = self.client.chat.completions.create(
                    model=PLANNER_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=PLANNER_TEMPERATURE,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                self.observability.record_exception(exc, input=prompt, retry_count=0)
                logger.exception("Groq plan request failed; switching to local fallback planning")
                self.remote_planning_available = False
                return self._fallback_plan(observation, getattr(config, "username", ""), getattr(config, "password", ""))

            content = (response.choices[0].message.content or "").strip()
            generation.update(output=content, usage_details=_usage_details(response))

            try:
                plan = json.loads(content)
            except json.JSONDecodeError as exc:
                self.observability.record_exception(exc, input=prompt, output=content, retry_count=0)
                logger.warning("Groq returned invalid JSON; using local fallback plan")
                return self._fallback_plan(observation, getattr(config, "username", ""), getattr(config, "password", ""))

            if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
                exc = ValueError("Groq returned an invalid plan shape")
                self.observability.record_exception(exc, input=prompt, output=content, retry_count=0)
                logger.warning("Groq returned an invalid plan shape; using local fallback plan")
                return self._fallback_plan(observation, getattr(config, "username", ""), getattr(config, "password", ""))

        print(content)
        logger.info("Received and parsed Groq plan")


        return plan
