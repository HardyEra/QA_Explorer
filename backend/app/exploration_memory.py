class ExplorationMemory:
    def __init__(self):
        self.visited_pages = set()
        self.executed_actions = set()

    def page_key(self, url):
        return url.split("#")[0]

    def action_key(self, page_url, action_id):
        return f"{self.page_key(page_url)}::{action_id}"

    def mark_page(self, url):
        self.visited_pages.add(self.page_key(url))

    def has_seen_page(self, url):
        return self.page_key(url) in self.visited_pages

    def mark_action(self, page_url, action_id):
        self.executed_actions.add(self.action_key(page_url, action_id))

    def has_executed_action(self, page_url, action_id):
        return self.action_key(page_url, action_id) in self.executed_actions