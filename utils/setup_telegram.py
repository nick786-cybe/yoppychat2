from flask import Flask
from dotenv import load_dotenv
from telegram_utils import set_webhook

# Load environment variables from your .env file
load_dotenv()

# Create a temporary Flask app instance
app = Flask(__name__)

# The with app.app_context() is crucial to make sure
# the function can access application configuration if needed.
with app.app_context():
    print("Attempting to set the Telegram webhook...")
    if set_webhook():
        print("Webhook set successfully!")
    else:
        print("Failed to set webhook. Please check your .env file for TELEGRAM_BOT_TOKEN and APP_BASE_URL.")