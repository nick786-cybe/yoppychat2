import os
from dotenv import load_dotenv

load_dotenv()

def load_config():
    """Loads configuration from environment variables."""
    return {
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN"),
        "telegram_bot_username": os.environ.get("TELEGRAM_BOT_USERNAME"),
        "app_base_url": os.environ.get("APP_BASE_URL"),
    }