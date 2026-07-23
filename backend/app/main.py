from browser import BrowserController
from explorer import Explorer
from config import ExplorationConfig
from logging_config import configure_logging
import logging

configure_logging()

logger = logging.getLogger(__name__)
browser = BrowserController()

try:
    browser.start()

    start_url = "https://www.saucedemo.com"

    browser.open(start_url)

    config = ExplorationConfig(
        start_url=start_url,
        follow_external=False
    )

    explorer = Explorer(browser, config)
    explorer.explore()
except Exception:
    logger.exception("Explorer stopped because of an unrecoverable error")
    raise
finally:
    browser.close()
