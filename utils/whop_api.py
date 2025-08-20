import os
import requests
import logging

log = logging.getLogger(__name__)

# --- Configuration (from your .env file) ---
WHOP_CLIENT_ID = os.environ.get("WHOP_CLIENT_ID")
WHOP_CLIENT_SECRET = os.environ.get("WHOP_CLIENT_SECRET")

# --- Official Whop URLs from Documentation ---
WHOP_DATA_API_BASE_URL = "https://data.whop.com"
WHOP_TOKEN_URL = "https://api.whop.com/oauth/token"

def exchange_code_for_token(code: str, redirect_uri: str) -> dict:
    """Exchanges an authorization code for a Whop access token."""
    payload = {
        'grant_type': 'authorization_code',
        'code': code,
        'client_id': WHOP_CLIENT_ID,
        'client_secret': WHOP_CLIENT_SECRET,
        'redirect_uri': redirect_uri
    }
    
    # --- THIS IS THE FIX ---
    # Send the payload as form data, not JSON
    response = requests.post(WHOP_TOKEN_URL, data=payload)
    # --- END FIX ---

    response.raise_for_status()
    return response.json()

def get_whop_user_data(access_token: str) -> dict:
    """Fetches the authenticated user's data from Whop."""
    url = f"{WHOP_DATA_API_BASE_URL}/v2/me"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def get_user_memberships(access_token: str) -> list:
    """Fetches the user's community memberships from Whop."""
    url = f"{WHOP_DATA_API_BASE_URL}/v2/me/memberships"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get('data', [])

def validate_license(access_token: str, whop_user_id: str, product_id: str) -> bool:
    """Checks if a user has a valid license for a specific product."""
    url = f"{WHOP_DATA_API_BASE_URL}/v2/licenses/validate"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'user_id': whop_user_id, 'product_id': product_id}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get('valid', False)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return False
        log.error(f"Error validating license: {e}")
        return False