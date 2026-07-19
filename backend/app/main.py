from browser import BrowserController
from explorer import Explorer
from config import ExplorationConfig

browser = BrowserController()
browser.start()

start_url = "https://www.saucedemo.com"

browser.open(start_url)

config = ExplorationConfig(
    start_url=start_url,
    follow_external=False
)

explorer = Explorer(browser, config)
explorer.explore()

input("Press Enter to exit...")

browser.close()