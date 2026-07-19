from dotenv import load_dotenv
import os
from urllib.parse import urlparse
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class ExplorationConfig:

    def __init__(self,
                 start_url,
                 follow_external=False):

        self.start_url = start_url
        self.follow_external = follow_external

        self.base_domain = urlparse(start_url).netloc

    def in_scope(self, url):

        if self.follow_external:
            return True

        return urlparse(url).netloc == self.base_domain