# yoppychat2/utils/whop_api.py

import os
import requests
from typing import Optional, Dict, Any, Tuple
import jwt

WHOP_API_BASE = "https://api.whop.com"
APP_API_KEY = os.getenv("WHOP_APP_API_KEY", "").strip()

def get_embedded_user_token(req) -> Optional[str]:
    token = req.headers.get("x-whop-user-token")
    if token:
        return token
    return req.cookies.get("whop_user_token") or req.cookies.get("x-whop-user-token")

def decode_jwt_no_verify(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
    except Exception:
        return {}

def _http_get(url: str, headers: Dict[str, str]) -> Tuple[int, Any]:
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "application/json" in content_type:
            return r.status_code, r.json()
        return r.status_code, r.text
    except Exception as e:
        return 0, {"error": str(e)}

def get_user_from_token(user_token: str) -> Optional[dict]:
    headers = {"Authorization": f"Bearer {user_token}"}
    code, data = _http_get(f"{WHOP_API_BASE}/v5/me", headers)
    return data if code == 200 and isinstance(data, dict) else None

def get_current_company(user_token: str = None) -> Optional[dict]:
    """
    Fetches the company based on the user's token if provided,
    otherwise falls back to the static company associated with the app key.
    """
    if user_token:
        try:
            decoded_token = decode_jwt_no_verify(user_token)
            # The company_id is expected in the 'org_id' claim of the Whop JWT.
            company_id = decoded_token.get('org_id')

            if company_id:
                # Use the user's token to make an authenticated request on their behalf.
                headers = {"Authorization": f"Bearer {user_token}"}
                code, data = _http_get(f"{WHOP_API_BASE}/v5/me/companies/{company_id}", headers)
                if code == 200 and isinstance(data, dict):
                    return data
        except Exception as e:
            # If any error occurs, we'll just fall through to the default method.
            print(f"Could not fetch company via user token: {e}. Falling back to default.")

    # Fallback to the original, static method
    if not APP_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {APP_API_KEY}"}
    code, data = _http_get(f"{WHOP_API_BASE}/v5/company", headers)
    return data if code == 200 and isinstance(data, dict) else None

def get_user_role_in_company(user_id: str, company_data: dict) -> Optional[str]:
    """
    Determine user's role in the provided company.
    """
    if not APP_API_KEY:
        # If there's no app key, we can't verify roles, so default to admin.
        return "admin"

    try:
        company_id = company_data.get("id")
        if not company_id:
            # Cannot determine role without a company ID.
            print("Warning: company_data missing 'id'. Cannot determine user role.")
            return "admin"

        # Check if the user is the owner of the current company via the authorized_user field.
        # This is populated when fetching a company via /v5/me/companies/{id}
        authorized_user = company_data.get("authorized_user")
        if authorized_user and authorized_user.get('user_id') == user_id and authorized_user.get('role') == 'owner':
            return "admin"

        # Whop App-level owner check
        if company_data.get("owner_id") == user_id:
            return "admin"

        # If not the owner, check their membership status within this specific company.
        app_headers = {"Authorization": f"Bearer {APP_API_KEY}"}

        # Use the specific company endpoint and filter by user_id for efficiency.
        # We also filter by 'valid=true' to only get active, completed memberships.
        url = f"{WHOP_API_BASE}/v5/companies/{company_id}/memberships?filter[user_id][eq]={user_id}&filter[valid][eq]=true"
        mem_code, mem_data = _http_get(url, app_headers)

        if mem_code != 200 or not isinstance(mem_data, dict):
            print(f"Failed to fetch memberships for company {company_id}. Defaulting role to admin.")
            return "admin"

        # If the 'data' array is not empty, it means the user has at least one valid membership.
        memberships = mem_data.get("data", [])
        if not memberships:
            # No valid membership found for this user in this company.
            return "member" # Or handle as "none" if you have a state for that.

        # Check if any of the user's valid memberships are for an admin-level plan.
        ADMIN_PLAN_IDS = os.getenv("WHOP_ADMIN_PLAN_IDS", "").split(',')
        if any(m.get("plan_id") in ADMIN_PLAN_IDS for m in memberships):
            return "admin"

        return "member"

    except Exception as e:
        print(f"Error determining user role for {user_id}: {e}. Defaulting to admin.")
        return "admin"
