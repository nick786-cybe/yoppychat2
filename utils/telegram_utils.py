import requests
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Configure logging
log = logging.getLogger(__name__)

def get_bot_token_and_url():
    """Loads Telegram bot token and constructs the base API URL."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("TELEGRAM_BOT_TOKEN not found in environment variables")
        return None, None
    return token, f"https://api.telegram.org/bot{token}"

def send_message(chat_id: int, text: str, **kwargs):
    """
    Sends a message to a given Telegram chat ID.
    This corrected version accepts any optional keyword argument (like reply_to_message_id)
    and passes it directly to the Telegram API.
    """
    token, base_url = get_bot_token_and_url()
    if not token:
        log.error("TELEGRAM_BOT_TOKEN not set. Cannot send message.")
        return None

    url = f"{base_url}/sendMessage"
    
    # --- THIS IS THE FIX ---
    # Create a base payload and then add any extra arguments passed in.
    # This makes the function flexible for things like replying to messages.
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown',  # Default parse mode
        'disable_web_page_preview': True
    }
    payload.update(kwargs) # Merge any other arguments like reply_markup or reply_to_message_id
    # --- END FIX ---

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        log.error(f"Failed to send message to chat_id {chat_id}: {e}")
        return None

def set_webhook():
    """Sets the application's webhook URL with Telegram."""
    token, base_url = get_bot_token_and_url()
    app_url = os.environ.get("APP_BASE_URL")

    if not all([token, app_url]):
        log.error("Cannot set webhook. Missing TELEGRAM_BOT_TOKEN or APP_BASE_URL in environment variables.")
        return False

    webhook_secret = token.split(':')[-1][:10] # Use part of the token as a secret
    webhook_url = f"{app_url}/telegram/webhook/{webhook_secret}"

    url = f"{base_url}/setWebhook"
    payload = {
        'url': webhook_url,
        'secret_token': webhook_secret
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("ok"):
            log.info(f"Webhook successfully set to: {webhook_url}")
            return True
        else:
            log.error(f"Failed to set webhook: {result.get('description')}")
            return False
    except requests.exceptions.RequestException as e:
        log.error(f"Exception while setting webhook: {e}")
        return False

def create_channel_keyboard(channels: list):
    """Creates a keyboard with a list of user's channels."""
    if not channels:
        return None

    keyboard = [[{'text': f"Ask: {ch}"}] for ch in channels]
    keyboard.insert(0, [{'text': "Ask: General Q&A"}]) # Add a general option

    return {
        'keyboard': keyboard,
        'resize_keyboard': True,
        'one_time_keyboard': True,
        'input_field_placeholder': 'Select a channel to query'
    }