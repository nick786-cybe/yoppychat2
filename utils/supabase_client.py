from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from typing import Optional
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env if present
load_dotenv()

log = logging.getLogger(__name__)

# --- START OF OPTIMIZATION ---

# Initialize the Admin Client ONCE at the module level when the app starts.
# This creates a single, shared instance that can be reused.
_supabase_admin_client: Optional[Client] = None
try:
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set for the admin client.")

    print(f"DEBUG ADMIN: Initializing SHARED Supabase admin client...")
    print(f"DEBUG ADMIN: SUPABASE_URL loaded: '{url}'")
    print(f"DEBUG ADMIN: SUPABASE_SERVICE_KEY loaded (first 10 chars): '{key[:10]}...' Length: {len(key)}")
    _supabase_admin_client = create_client(url, key)

except Exception as e:
    log.error(f"CRITICAL: Failed to initialize shared Supabase admin client on startup: {e}")

def get_supabase_admin_client() -> Client:
    """
    Returns the single, shared Supabase admin client instance.
    This function no longer creates a new client on every call, improving performance.
    """
    if not _supabase_admin_client:
        raise ConnectionError("Supabase admin client is not available. Check initial configuration and logs.")
    return _supabase_admin_client

# --- END OF OPTIMIZATION ---


def get_supabase_client(access_token: Optional[str] = None) -> Optional[Client]:
    """
    Initializes and returns a user-specific Supabase client.
    This function correctly creates a new client for each user request,
    ensuring their specific permissions are used via the access_token.
    """
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        log.error("SUPABASE_URL or SUPABASE_ANON_KEY not set in environment variables!")
        return None

    headers = {}
    if access_token:
        # If we have a user's token, set the Authorization header.
        headers["Authorization"] = f"Bearer {access_token}"

    # Pass the headers dictionary to ClientOptions.
    options = ClientOptions(headers=headers)

    # Initialize the client. The `key` sets the `apikey` header,
    # and `options` adds the `Authorization` header if present.
    supabase: Client = create_client(url, key, options=options)

    return supabase


def refresh_supabase_session(refresh_token: str) -> Optional[dict]:
    """
    Attempts to refresh a Supabase session using a refresh token.
    Returns the new session dictionary if successful, None otherwise.
    """
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_ANON_KEY")
    
    if not url or not key:
        log.error("SUPABASE_URL or SUPABASE_ANON_KEY not set for refreshing session.")
        return None
    
    supabase_anon_client = create_client(url, key)
    try:
        # Use the refresh_session method from the Supabase client's auth object
        response = supabase_anon_client.auth.refresh_session(refresh_token)
        log.info(f"Supabase session refreshed successfully for user: {response.user.id}")
        return response.session.dict() # Return the new session data as a dictionary
    except Exception as e:
        log.error(f"Error refreshing Supabase session: {e}", exc_info=True)
        return None