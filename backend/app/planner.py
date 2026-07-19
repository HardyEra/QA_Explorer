from google import genai
from config import GOOGLE_API_KEY
import json

class Planner:

    def __init__(self):
        # self.client = Groq(api_key=GROQ_API_KEY)
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

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

        response = self.client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        content = response.text.strip()

        while True:
            try:
                plan = json.loads(content)
                break
            except json.JSONDecodeError:
                if content.endswith("}"):
                    content = content[:-1].rstrip()
                else:
                    raise

        print(content)


        return plan