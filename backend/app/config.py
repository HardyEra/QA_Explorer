from dotenv import load_dotenv
import os
from uuid import uuid4
from urllib.parse import urlparse
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class ExplorationConfig:

    def __init__(self,
                 start_url,
                 follow_external=False,
                 explore_new_tabs=False,
                 max_steps=100,
                 max_planner_actions=10,
                 project_name="qa-explorer",
                 session_id=None,
                 exploration_id=None,
                 current_goal="Autonomously discover application pages and actions",
                 application_context="",
                 username="",
                 password="",
                 evidence_dir="",
                 hitl_wait_seconds=0):

        self.start_url = start_url
        self.follow_external = follow_external
        self.explore_new_tabs = explore_new_tabs
        self.max_steps = max_steps
        self.max_planner_actions = max_planner_actions
        self.project_name = project_name
        self.session_id = session_id or str(uuid4())
        self.exploration_id = exploration_id or str(uuid4())
        self.current_goal = current_goal
        self.application_context = application_context
        self.username = username
        self.password = password
        # A PRD-driven run keeps immutable screenshots for every observed
        # screen.  Downstream design and verification use these as evidence.
        self.evidence_dir = evidence_dir
        # >0 enables "ask the tester when stuck" during exploration.
        self.hitl_wait_seconds = hitl_wait_seconds

        self.base_domain = urlparse(start_url).netloc

    def in_scope(self, url):

        if self.follow_external:
            return True

        return urlparse(url).netloc == self.base_domain
