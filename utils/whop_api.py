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

def get_current_company() -> Optional[dict]:
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
        return "admin"
        
    try:
        # Check if the user is the owner of the current company
        if company_data.get("authorized_user") == user_id or company_data.get("owner_id") == user_id:
            return "admin"

        # If not the owner, check their membership status
        app_headers = {"Authorization": f"Bearer {APP_API_KEY}"}
        mem_code, mem_data = _http_get(f"{WHOP_API_BASE}/v5/company/memberships", app_headers)
        
        if mem_code != 200 or not isinstance(mem_data, dict):
            return "admin"
        
        memberships = mem_data.get("data", [])
        user_membership = next((m for m in memberships if m.get("user_id") == user_id), None)

        if not user_membership or not user_membership.get("valid") or user_membership.get("status") != "completed":
            return "admin" # Default to admin if no valid membership is found
        
        ADMIN_PLAN_IDS = os.getenv("WHOP_ADMIN_PLAN_IDS", "").split(',')
        if user_membership.get("plan_id") in ADMIN_PLAN_IDS:
            return "admin"
        
        return "member"
        
    except Exception as e:
        print(f"Error determining user role for {user_id}: {e}. Defaulting to admin.")
        return "admin"