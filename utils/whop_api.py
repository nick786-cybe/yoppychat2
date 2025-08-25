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

def get_current_company(user_token: str = None, company_id: str = None) -> Optional[dict]:
    """
    Fetches the company based on the provided company_id,
    otherwise falls back to the static company associated with the app key.
    """
    if company_id and user_token:
        print(f"[DEBUG] Attempting to fetch company {company_id} via user token.")
        try:
            # Use the user's token to make an authenticated request on their behalf.
            headers = {"Authorization": f"Bearer {user_token}"}
            print(f"[DEBUG] Fetching company from API: /v5/me/companies/{company_id}")
            code, data = _http_get(f"{WHOP_API_BASE}/v5/me/companies/{company_id}", headers)
            print(f"[DEBUG] API response code: {code}")
            if code == 200 and isinstance(data, dict):
                print(f"[DEBUG] Successfully fetched company {company_id} via user token.")
                return data
            else:
                print(f"[DEBUG] API response data: {data}")
        except Exception as e:
            print(f"Could not fetch company via user token: {e}. Falling back to default.")

    # Fallback to the original, static method
    print("[DEBUG] Falling back to static company fetch method.")
    if not APP_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {APP_API_KEY}"}
    code, data = _http_get(f"{WHOP_API_BASE}/v5/company", headers)
    print(f"[DEBUG] Static fetch response code: {code}")
    return data if code == 200 and isinstance(data, dict) else None

def get_user_role_in_company(user_id: str, company_data: dict, user_token: str = None) -> Optional[str]:
    """
    Determine user's role in the provided company.
    Uses user_token for authentication if provided, otherwise falls back to APP_API_KEY.
    """
    try:
        company_id = company_data.get("id")
        if not company_id:
            print("Warning: company_data missing 'id'. Cannot determine user role.")
            return "member"

        # Check for ownership first, as it doesn't require a membership check.
        authorized_user = company_data.get("authorized_user")
        if (authorized_user and authorized_user.get('user_id') == user_id and authorized_user.get('role') == 'owner') or \
           company_data.get("owner_id") == user_id:
            return "admin"

        # Determine which authentication method to use
        if user_token:
            # Use the user's own token for a more secure, context-aware check.
            headers = {"Authorization": f"Bearer {user_token}"}
            url = f"{WHOP_API_BASE}/v5/me/memberships?filter[company_id][eq]={company_id}&filter[valid][eq]=true"
            log_prefix = "user's"
        else:
            # Fallback to the app key. This may fail if the key is not authorized for the company.
            if not APP_API_KEY:
                print("Warning: No user_token and no APP_API_KEY. Defaulting role to member.")
                return "member"
            headers = {"Authorization": f"Bearer {APP_API_KEY}"}
            url = f"{WHOP_API_BASE}/v5/companies/{company_id}/memberships?filter[user_id][eq]={user_id}&filter[valid][eq]=true"
            log_prefix = "app's"

        mem_code, mem_data = _http_get(url, headers)

        if mem_code != 200 or not isinstance(mem_data, dict):
            print(f"Failed to fetch memberships for company {company_id} using {log_prefix} token. Defaulting role to member.")
            return "member"

        memberships = mem_data.get("data", [])
        if not memberships:
            return "member"

        ADMIN_PLAN_IDS = os.getenv("WHOP_ADMIN_PLAN_IDS", "").split(',')
        if any(m.get("plan_id") in ADMIN_PLAN_IDS for m in memberships):
            return "admin"

        return "member"

    except Exception as e:
        print(f"Error determining user role for {user_id}: {e}. Defaulting to member.")
        return "member"

def get_company_by_id(company_id: str, user_token: str) -> Optional[dict]:
    """Fetches a specific company by its ID using a user token."""
    if not company_id or not user_token:
        return None
    headers = {"Authorization": f"Bearer {user_token}"}
    code, data = _http_get(f"{WHOP_API_BASE}/v5/me/companies/{company_id}", headers)
    return data if code == 200 and isinstance(data, dict) else None
