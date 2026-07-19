class ActionRegistry:

    def __init__(self):
        self.actions = {}
        self.next_id = 0

    def register(self, locator, text, action_type):
        action_id = self.next_id

        self.actions[action_id] = {
            "locator": locator,
            "text": text,
            "type": action_type
        }

        self.next_id += 1

        return action_id
    
    def get(self, action_id):
        return self.actions.get(action_id)
    
    def clear(self):
        self.actions.clear()
        self.next_id = 0