from planner import Planner
from executor import Executor
from exploration_memory import ExplorationMemory
class Explorer:

    def __init__(self, browser, config):
        self.browser = browser
        self.config = config
        self.planner = Planner()
        self.executor = Executor(browser)
        self.memory = ExplorationMemory()

    def observe(self):
        observation = self.browser.observe()

        print(observation.page.title)
        print(observation.page.url)

        print("\nAvailable Actions:")

        for action in observation.actions:
            print(action)

        return observation
    
    def explore(self):

        while True:

            observation = self.observe()
            current_url = observation.page.url

            if not self.config.in_scope(current_url):

                print("\nOutside exploration scope.")
                print(current_url)

                self.browser.page.go_back()

                continue

            self.memory.mark_page(observation.page.url)

            print("\n==========================")
            print(observation.page.title)
            print(observation.page.url)
            print("==========================")

            # Keep only unexplored actions
            available_actions = [
                action
                for action in observation.actions
                if not self.memory.has_executed_action(
                    observation.page.url,
                    action.id
                )
            ]

            # Stop if nothing left on this page
            if not available_actions:
                print("\n [OK] Exploration complete.")
                break

            # Give planner only unexplored actions
            observation.actions = available_actions

            plan = self.planner.plan(
                observation,
                self.memory.visited_pages,
                self.memory.executed_actions
            )

            print(plan)

            success = self.executor.execute(plan)

            if not success:
                print("Execution failed.")

                for step in plan["steps"]:
                    if step["type"] == "click":
                        self.memory.mark_action(
                            observation.page.url,
                            step["action_id"]
                        )

                continue

            # Mark executed actions
            for step in plan["steps"]:
                if step["type"] == "click":
                    self.memory.mark_action(
                        observation.page.url,
                        step["action_id"]
                    )