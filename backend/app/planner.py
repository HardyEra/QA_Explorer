from groq import Groq
from config import GROQ_API_KEY
import json
import logging
import os
import re
from dataclasses import dataclass


logger = logging.getLogger(__name__)
MODEL_TIMEOUT_MS = int(os.getenv("MODEL_TIMEOUT_MS", "10000"))
PLANNER_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PLANNER_TEMPERATURE = 0
BUSINESS_WORKFLOW_CATEGORIES = {
    "authentication", "create", "submit", "save", "upload", "download",
    "purchase", "checkout", "edit", "delete",
}
WORKFLOW_CONTINUATION_CATEGORIES = {
    "submit", "save", "upload",
}
GENERIC_GOALS = {
    "explore important business workflows",
    "autonomously discover and execute the application's primary business-critical workflow",
}
WORKFLOW_STOP_WORDS = {
    "add", "and", "an", "a", "click", "create", "explore", "go", "important",
    "in", "menu", "navigate", "open", "the", "to", "upload", "workflow", "workflows",
}
NEGATED_ACTION_PATTERN = re.compile(
    r"\b(?:do\s+not|don't|never|avoid)\s+"
    r"(?:click|select|open|visit|use|go\s+to|navigate\s+to)?\s*"
    r"(?P<label>.+?)(?=(?:[.;),]|\s+(?:instead|but|then)\b|$))",
    re.IGNORECASE,
)
GOAL_TARGET_PATTERN = re.compile(
    r"\b(?:go\s+to|select|choose|open|navigate\s+to)\s+"
    r"(?:the\s+)?(?P<label>[a-z0-9][a-z0-9\s-]*?)"
    r"(?=\s+(?:in|from|via)\s+(?:the\s+)?(?:menu|sidebar|navigation)\b"
    r"|\s*(?:,|\.|\)|\band\b|\bthen\b|$))",
    re.IGNORECASE,
)
NON_FILLABLE_INPUT_TYPES = {
    "button", "checkbox", "file", "hidden", "image", "radio", "reset", "submit",
}


@dataclass
class WorkflowMemory:
    """Short-term planning focus for one explicitly requested workflow."""

    goal: str
    started: bool = False
    last_action: str | None = None
    target_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    target_reached: bool = False
    creation_started: bool = False


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
        self.workflow_memory: WorkflowMemory | None = None

    def _fallback_plan(self, observation, provided_username="", provided_password=""):
        """Make conservative progress when the model service is unavailable.

        The fallback is intentionally deterministic: it completes a familiar
        username/password form, then explores one remaining action at a time.
        This keeps the graph usable during transient model API failures.
        """
        inputs = observation.inputs
        decision_actions = self._decision_candidates(observation.actions)
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
            (
                item for item in inputs
                if item.get("type") == "password"
                or "password" in " ".join(
                    str(item.get(key) or "").casefold()
                    for key in ("name", "placeholder", "id")
                )
            ),
            None,
        )

        if username and password:
            login_action = next(
                (
                    action
                    for action in observation.actions
                    if "login" in action.text.lower() or "sign in" in action.text.lower()
                ),
                decision_actions[0] if decision_actions else None,
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

        form_plan = self._workflow_form_plan(inputs, decision_actions)
        if form_plan:
            return form_plan

        if decision_actions:
            action = decision_actions[0]
            logger.warning(
                "Using local fallback plan to explore action %s (%s)",
                action.id,
                action.text,
            )
            return {"steps": [{"type": "click", "action_id": action.id}]}

        return {"steps": []}

    def _decision_candidates(self, actions):
        """Prefer business-workflow actions from the ranked candidate list.

        ``Explorer`` has already classified and ranked these actions.  The
        planner does not re-rank them; it only excludes secondary UI from the
        immediate decision when at least one workflow-advancing action exists.
        """
        actions = list(actions)
        if self.workflow_memory:
            prohibited_actions = [
                action for action in actions
                if self._is_prohibited_action(action) or self._is_automatic_logout(action)
            ]
            for action in prohibited_actions:
                if self._is_automatic_logout(action):
                    logger.info("Excluding automatic logout action: %s", action.text)
                else:
                    logger.info(
                        "Excluding action %s because the workflow explicitly forbids it",
                        action.text,
                    )
            allowed_actions = [
                action for action in actions
                if action not in prohibited_actions
            ]
            # Goal relevance must be evaluated before the broad business filter:
            # a target module such as "Candidates" is often navigation, but it
            # is the correct first step for a Candidate workflow.
            workflow_actions = self._workflow_actions(allowed_actions)
            if workflow_actions:
                selected = workflow_actions
                logger.info("Continuing active workflow: %s", self.workflow_memory.goal)
            else:
                authentication_actions = [
                    action for action in allowed_actions
                    if getattr(action, "category", "") == "authentication"
                ]
                if authentication_actions:
                    selected = authentication_actions
                    logger.info("Authentication is required before continuing workflow: %s", self.workflow_memory.goal)
                else:
                    selected = self._workflow_navigation_actions(allowed_actions)
                if selected and not authentication_actions:
                    logger.info("Workflow target is not visible; selecting a workflow navigation control")
                elif not selected:
                    selected = []
                    logger.warning("No action advances the active workflow: %s", self.workflow_memory.goal)
        else:
            actions = [
                action for action in actions
                if not self._is_automatic_logout(action)
            ]
            business_actions = [
                action for action in actions
                if getattr(action, "category", "") in BUSINESS_WORKFLOW_CATEGORIES
            ]
            selected = business_actions or actions
        for action in selected:
            logger.info(
                "Planner decision candidate: %s (%s,%s)",
                action.text,
                getattr(action, "category", "unknown"),
                getattr(action, "priority", "unknown"),
            )
        return selected

    def _activate_workflow(self, config) -> None:
        goal = " ".join(str(getattr(config, "current_goal", "") or "").split())
        normalized_goal = goal.casefold().rstrip(".!")
        if not goal or normalized_goal in GENERIC_GOALS:
            return
        existing_goal = (
            self.workflow_memory.goal.casefold().rstrip(".!")
            if self.workflow_memory
            else None
        )
        if existing_goal != normalized_goal:
            positive_goal = self._goal_without_negative_instructions(goal)
            forbidden_phrases = self._forbidden_goal_phrases(goal)
            target_phrases = self._workflow_target_phrases(positive_goal)
            self.workflow_memory = WorkflowMemory(
                goal=goal,
                target_phrases=target_phrases,
                forbidden_phrases=forbidden_phrases,
            )
            logger.info("Workflow memory activated: %s", goal)
            if target_phrases:
                logger.info("Workflow target(s): %s", ", ".join(target_phrases))
            if forbidden_phrases:
                logger.info("Workflow exclusion(s): %s", ", ".join(forbidden_phrases))

    def _workflow_actions(self, actions):
        """Keep decision making on the active workflow and its completion actions."""
        assert self.workflow_memory is not None

        # A named menu/module target is a hard gate. For example, "Clients"
        # must be selected before an unrelated Add or Submit control can begin
        # the requested workflow. Exact matching also prevents "All Client"
        # from standing in for the distinct "Clients" module.
        if self.workflow_memory.target_phrases and not self.workflow_memory.target_reached:
            target_actions = [
                action for action in actions
                if self._is_target_action(action)
            ]
            if target_actions:
                return self._sort_workflow_actions(target_actions, self._workflow_goal_terms())
            return []

        goal_terms = self._workflow_goal_terms()
        matching = [
            action for action in actions
            if goal_terms.intersection(self._workflow_tokens(action.text))
        ]
        if matching:
            return self._sort_workflow_actions(matching, goal_terms)

        # Once the requested module has been reached, an unlabeled generic
        # creation control (for example, "Add New") is the expected next step.
        # It is deliberately unavailable before the target module is reached.
        if self.workflow_memory.target_reached and not self.workflow_memory.creation_started:
            create_actions = [
                action for action in actions
                if getattr(action, "category", "") == "create"
            ]
            if create_actions:
                return self._sort_workflow_actions(create_actions, goal_terms)

        # Generic completion controls are relevant only after a goal-matching
        # creation action was selected. This prevents unrelated actions such
        # as Change Password from being treated as workflow continuation.
        if self.workflow_memory.creation_started:
            return [
                action for action in actions
                if getattr(action, "category", "") in WORKFLOW_CONTINUATION_CATEGORIES
            ]
        return []

    def _workflow_navigation_actions(self, actions):
        """Use a menu only when the goal explicitly requires menu navigation."""
        goal_words = set(re.findall(r"[a-z0-9]+", self.workflow_memory.goal.casefold()))
        if "menu" not in goal_words:
            return []
        menu_actions = [
            action for action in actions
            if getattr(action, "category", "") == "menu"
        ]
        if menu_actions:
            return menu_actions
        return [
            action for action in actions
            if getattr(action, "category", "") == "navigation_module"
        ]

    def _workflow_goal_terms(self) -> set[str]:
        assert self.workflow_memory is not None
        return {
            term
            for term in self._workflow_tokens(
                self._goal_without_negative_instructions(self.workflow_memory.goal)
            )
            if term not in WORKFLOW_STOP_WORDS
        }

    @staticmethod
    def _goal_without_negative_instructions(goal: str) -> str:
        """Remove prohibitions before deriving positive workflow intent."""
        return NEGATED_ACTION_PATTERN.sub(" ", goal)

    @classmethod
    def _forbidden_goal_phrases(cls, goal: str) -> tuple[str, ...]:
        """Extract the action labels a user explicitly told us not to use."""
        phrases: list[str] = []
        for match in NEGATED_ACTION_PATTERN.finditer(goal):
            label = match.group("label")
            # Context such as "in the dashboard home" describes location, not
            # the forbidden action label. Keep just the leading label phrase.
            label = re.split(
                r"\s+(?:in|on|from|at)\s+(?:the\s+)?|\s+(?:dashboard|home|page)\b",
                label,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            normalized = cls._normalized_phrase(label)
            if normalized and normalized not in phrases:
                phrases.append(normalized)
        return tuple(phrases)

    @classmethod
    def _workflow_target_phrases(cls, goal: str) -> tuple[str, ...]:
        """Extract explicit navigation/module targets from a workflow goal."""
        phrases: list[str] = []
        # Quoted labels are intentional names, e.g. select "Clients".
        raw_phrases = re.findall(r'["\']([^"\']+)["\']', goal)
        raw_phrases.extend(match.group("label") for match in GOAL_TARGET_PATTERN.finditer(goal))
        for phrase in raw_phrases:
            normalized = cls._normalized_phrase(phrase)
            # "go to menu and select Clients" describes how to reach the
            # destination. The menu is a control, not an additional target
            # that must be present alongside the Clients module.
            if normalized in {"menu", "sidebar", "navigation"}:
                continue
            if normalized and normalized not in phrases:
                phrases.append(normalized)
        return tuple(phrases)

    @staticmethod
    def _normalized_phrase(text: str) -> str:
        return " ".join(
            Planner._singularize(token)
            for token in re.findall(r"[a-z0-9]+", text.casefold())
        )

    def _is_prohibited_action(self, action) -> bool:
        if not self.workflow_memory:
            return False
        action_label = self._normalized_phrase(action.text)
        return any(
            action_label == phrase or action_label.startswith(f"{phrase} ")
            for phrase in self.workflow_memory.forbidden_phrases
        )

    def _is_target_action(self, action) -> bool:
        if not self.workflow_memory:
            return False
        action_label = self._normalized_phrase(action.text)
        return action_label in self.workflow_memory.target_phrases

    def _is_automatic_logout(self, action) -> bool:
        """Keep a discovery run authenticated unless logout is the stated goal."""
        if getattr(action, "category", "") != "logout":
            return False
        goal = self.workflow_memory.goal if self.workflow_memory else ""
        return not any(term in goal.casefold() for term in ("logout", "log out", "sign out"))

    @staticmethod
    def _workflow_tokens(text: str) -> set[str]:
        """Normalize workflow nouns so Candidate matches the Candidates module."""
        return {
            Planner._singularize(term)
            for term in re.findall(r"[a-z0-9]+", text.casefold())
        }

    @staticmethod
    def _singularize(term: str) -> str:
        if len(term) > 4 and term.endswith("ies"):
            return f"{term[:-3]}y"
        if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
            return term[:-1]
        return term

    @staticmethod
    def _sort_workflow_actions(actions, goal_terms):
        def score(action):
            action_terms = Planner._workflow_tokens(action.text)
            overlap = len(goal_terms.intersection(action_terms))
            category_bonus = {
                "create": 30,
                "upload": 25,
                "submit": 20,
                "save": 15,
                "navigation_module": 10,
            }.get(getattr(action, "category", ""), 0)
            return overlap * 100 + category_bonus + getattr(action, "priority", 0)

        return sorted(actions, key=score, reverse=True)

    def _workflow_form_plan(self, inputs, decision_actions):
        """Fill a creation form with safe dummy data before its submit action.

        This is intentionally limited to an explicit workflow after its target
        module and creation control have both been selected. It avoids filling
        dashboard filters, login inputs, or an unrelated form.
        """
        if not self.workflow_memory or not self.workflow_memory.creation_started:
            return None

        submit_action = next(
            (
                action for action in decision_actions
                if getattr(action, "category", "") in {"submit", "save"}
            ),
            None,
        )
        fill_steps = []
        for item in inputs:
            input_type = str(item.get("type") or "text").casefold()
            if input_type in NON_FILLABLE_INPUT_TYPES:
                continue
            target = item.get("placeholder") or item.get("name") or item.get("id")
            if not target:
                continue
            fill_steps.append(
                {
                    "type": "fill",
                    "target": target,
                    "value": self._dummy_value_for_input(item),
                }
            )

        if not fill_steps:
            return None
        if submit_action:
            fill_steps.append({"type": "click", "action_id": submit_action.id})
        logger.info(
            "Preparing %s form value(s) for active workflow%s",
            len(fill_steps) - int(submit_action is not None),
            " before submit" if submit_action else "",
        )
        return {"steps": fill_steps}

    @staticmethod
    def _dummy_value_for_input(item) -> str:
        label = " ".join(
            str(item.get(key) or "").casefold()
            for key in ("name", "placeholder", "id")
        )
        input_type = str(item.get("type") or "").casefold()
        if input_type == "email" or "email" in label:
            return "qa.test@example.com"
        if input_type in {"tel", "phone"} or any(term in label for term in ("phone", "mobile")):
            return "5551234567"
        if input_type == "number" or any(term in label for term in ("count", "number", "quantity")):
            return "1"
        if input_type == "date" or "date" in label:
            return "2026-07-27"
        if input_type == "url" or "website" in label or "url" in label:
            return "https://example.com"
        if "contact" in label:
            return "QA Test Contact"
        if any(term in label for term in ("name", "client", "company", "organization", "contact")):
            return "QA Test Client"
        if any(term in label for term in ("address", "location")):
            return "123 Test Street"
        if any(term in label for term in ("description", "note", "comment", "detail")):
            return "Created by automated QA exploration"
        return "QA Test"

    def _remember_workflow_action(self, plan, actions) -> None:
        if not self.workflow_memory:
            return

    def record_execution(self, plan, actions) -> None:
        """Advance workflow memory only after the executor reports success."""
        self._remember_workflow_action(plan, actions)
        by_id = {action.id: action for action in actions}
        for step in plan.get("steps", []):
            if step.get("type") != "click":
                continue
            action = by_id.get(step.get("action_id"))
            if action:
                category = getattr(action, "category", "")
                if category != "authentication":
                    self.workflow_memory.started = True
                if self._is_target_action(action):
                    self.workflow_memory.target_reached = True
                    logger.info("Workflow target reached: %s", action.text)
                elif category == "create" and (
                    self.workflow_memory.target_reached
                    or not self.workflow_memory.target_phrases
                ):
                    self.workflow_memory.creation_started = True
                    logger.info("Workflow creation step started: %s", action.text)
                self.workflow_memory.last_action = action.text
                logger.info(
                    "Workflow memory: %s -> %s",
                    self.workflow_memory.goal,
                    action.text,
                )
            return

    def plan(self, observation, visited_pages, visited_actions, config=None):

        visited_pages_text = "\n".join(visited_pages) or "None"

        visited_actions_text = "\n".join(visited_actions) or "None"

        self._activate_workflow(config)
        exception_context = {
            "current_url": getattr(getattr(observation, "page", None), "url", None),
            "workflow_name": getattr(config, "current_goal", None),
            "active_action": "Plan",
        }
        decision_actions = self._decision_candidates(observation.actions)
        vision_context = self._vision_context(observation)
        actions = json.dumps(
            [
                {
                    "id": action.id,
                    "text": action.text,
                    "category": action.category,
                    "priority": action.priority,
                }
                for action in decision_actions
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

Visual UI Observation (Gemini screenshot analysis; current viewport only):
{vision_context}

Available Actions:
{actions}

These are the preferred decision candidates selected from the ranked action
list. If any are business-workflow actions, choose one of them before any
secondary UI such as filters, cards, avatars, notifications, menus, settings,
tabs, sorting, social links, or footer links.

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
            return self._fallback_plan(
                observation,
                getattr(config, "username", ""),
                getattr(config, "password", ""),
            )

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
                self.observability.record_exception(
                    exc, input=prompt, retry_count=0, context=exception_context
                )
                logger.exception("Groq plan request failed; switching to local fallback planning")
                self.remote_planning_available = False
                return self._fallback_plan(observation, getattr(config, "username", ""), getattr(config, "password", ""))

            content = (response.choices[0].message.content or "").strip()
            generation.update(output=content, usage_details=_usage_details(response))

            try:
                plan = json.loads(content)
            except json.JSONDecodeError as exc:
                self.observability.record_exception(
                    exc, input=prompt, output=content, retry_count=0, context=exception_context
                )
                logger.warning("Groq returned invalid JSON; using local fallback plan")
                return self._fallback_plan(observation, getattr(config, "username", ""), getattr(config, "password", ""))

            if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
                exc = ValueError("Groq returned an invalid plan shape")
                self.observability.record_exception(
                    exc, input=prompt, output=content, retry_count=0, context=exception_context
                )
                logger.warning("Groq returned an invalid plan shape; using local fallback plan")
                return self._fallback_plan(observation, getattr(config, "username", ""), getattr(config, "password", ""))

            allowed_action_ids = {action.id for action in decision_actions}
            selected_click_ids = {
                step.get("action_id") for step in plan["steps"]
                if step.get("type") == "click"
            }
            if selected_click_ids - allowed_action_ids:
                logger.warning("Planner selected an action outside the preferred business candidates; using fallback")
                return self._fallback_plan(observation, getattr(config, "username", ""), getattr(config, "password", ""))

        print(content)
        logger.info("Received and parsed Groq plan")
        return plan

    @staticmethod
    def _vision_context(observation) -> str:
        """Serialize optional Gemini perception for Groq without exposing an image.

        Gemini receives the screenshot. Groq receives only its validated,
        structured description, which keeps the planner prompt compact and
        makes the hand-off visible in application logs and Langfuse.
        """
        vision = getattr(observation, "vision_observation", None)
        if vision is None:
            logger.info("Planner visual context: unavailable")
            return "Unavailable for this observation. Use the DOM/accessibility actions."
        payload = {
            key: getattr(vision, key, None)
            for key in (
                "page_type", "summary", "dialogs", "buttons", "links",
                "navigation", "forms", "inputs", "tables", "warnings",
            )
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        logger.info("Planner visual context: %s", serialized)
        return serialized
