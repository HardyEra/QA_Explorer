class Executor:

    def __init__(self, browser, observability):
        self.browser = browser
        self.observability = observability

    def execute(self, plan):

        for step in plan["steps"]:
            print(step)

            if step["type"] == "fill":
                success = self.browser.fill_input(
                    step["target"],
                    step["value"]
                )
                if not success:
                    return False

            elif step["type"] == "click":
                action_id = step.get("action_id")
                if action_id is None:
                    print("Click step is missing action_id.")
                    return False
                source_url = self.browser.current_url()
                success = self.browser.click_action(action_id)

                if not success:
                    return False

                # Action locators belong to the page that was observed and
                # planned. A navigation invalidates every remaining locator in
                # this plan, so let the graph observe the destination page.
                if self.browser.current_url() != source_url:
                    return True

        return True
