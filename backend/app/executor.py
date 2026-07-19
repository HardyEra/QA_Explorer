class Executor:

    def __init__(self, browser):
        self.browser = browser

    def execute(self, plan):

        for step in plan["steps"]:
            print(step)

            if step["type"] == "fill":
                self.browser.fill_input(
                    step["target"],
                    step["value"]
                )

            elif step["type"] == "click":

                success = self.browser.click_action(step["action_id"])

                if not success:
                    return False

        return True