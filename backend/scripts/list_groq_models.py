"""Print the Groq models available to the configured account."""

from pathlib import Path
import sys

from groq import Groq


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from config import GROQ_API_KEY  # noqa: E402

client = Groq(api_key=GROQ_API_KEY)

for model in client.models.list().data:
    print(model.name)
