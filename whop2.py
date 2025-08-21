# whop.py (Corrected)

import os
import json
from typing import Optional, Dict, Any, Tuple

from flask import Flask, request, jsonify, render_template_string
import requests
from dotenv import load_dotenv
import jwt  # pyjwt

load_dotenv()

WHOP_API_BASE = "https://api.whop.com"
APP_API_KEY = os.getenv("WHOP_APP_API_KEY", "").strip()
PORT = int(os.getenv("PORT", "5000"))

# Remember to set your Admin Plan ID here
ADMIN_PLAN_IDS = ["plan_FyOCkxDxw57Qx"] 

app = Flask(__name__)

# -----------------------------
# Helpers
# -----------------------------

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

def bearer(headers: Dict[str, str], token: str) -> Dict[str, str]:
    h = dict(headers or {})
    h["Authorization"] = f"Bearer {token}"
    return h

def app_headers() -> Dict[str, str]:
    if not APP_API_KEY:
        return {}
    return {"Authorization": f"Bearer {APP_API_KEY}"}

def http_get(url: str, headers: Dict[str, str]) -> Tuple[int, Any]:
    try:
        r = requests.get(url, headers=headers, timeout=10)
        return r.status_code, (r.json() if r.headers.get("content-type","").startswith("application/json") else r.text)
    except Exception as e:
        return 0, {"error": str(e)}

# -----------------------------
# Whop API calls
# -----------------------------

def me_user(user_token: str):
    code, data = http_get(f"{WHOP_API_BASE}/v5/me", bearer({}, user_token))
    return code, data

def me_companies(user_token: str):
    code, data = http_get(f"{WHOP_API_BASE}/v5/me/companies", bearer({}, user_token))
    return code, data

def me_accounts(user_token: str):
    code, data = http_get(f"{WHOP_API_BASE}/v5/me/social_accounts", bearer({}, user_token))
    return code, data

def company_current(app_key_headers: Dict[str,str]):
    code, data = http_get(f"{WHOP_API_BASE}/v5/company", app_key_headers)
    return code, data

# ---- CORRECTED FUNCTION ----
def company_memberships(app_key_headers: Dict[str,str], company_id: Optional[str] = None):
    # GET /v5/company/memberships (requires app key)
    # This endpoint is already scoped to the API key's company.
    code, data = http_get(f"{WHOP_API_BASE}/v5/company/memberships", app_key_headers)
    
    # The bug was here: The API uses 'page_id' on memberships, not 'company_id'.
    # We will filter by 'page_id' to be safe, though the endpoint is already scoped.
    if code == 200 and isinstance(data, dict) and company_id:
        items = data.get("data") or data.get("items") or data
        if isinstance(items, list):
            # CORRECTED LINE: Use 'page_id' for filtering
            items = [m for m in items if str(m.get("page_id")) == str(company_id)]
            data['data'] = items # Ensure the filtered list is put back
            
    return code, data
# ---- END CORRECTION ----

def get_user_role_in_company(user_id: str, company_id: str, app_key_headers: Dict[str, str], admin_plan_ids: list) -> Optional[str]:
    """
    Determine user's role in the company.
    Returns 'owner', 'admin', 'member', or None
    """
    try:
        # First, check if user is the company owner
        comp_code, comp_data = company_current(app_key_headers)
        if comp_code == 200 and isinstance(comp_data, dict):
            if comp_data.get("authorized_user") == user_id or comp_data.get("owner_id") == user_id:
                return "owner"
        
        # Get all memberships for the company
        mem_code, mem_data = company_memberships(app_key_headers, company_id)
        if mem_code != 200 or not isinstance(mem_data, dict):
            # If memberships can't be fetched, default to admin as requested.
            return "admin"
        
        memberships = mem_data.get("data", [])
        if not isinstance(memberships, list):
            return "admin"
        
        # Look for the user's specific membership
        user_membership = None
        for membership in memberships:
            if membership.get("user_id") == user_id:
                user_membership = membership
                break
        
        if not user_membership:
            # --- MODIFICATION AS REQUESTED ---
            # This makes any user without a membership an admin.
            return "admin"
        
        # Check if the membership is active and valid
        if not user_membership.get("valid", False) or user_membership.get("status") != "completed":
            return None # Invalid members get no role
        
        # Check if the user's plan ID matches any of the explicit admin plan IDs
        if user_membership.get("plan_id") in admin_plan_ids:
            return "admin"
        
        # If they have a valid membership and it's not an admin plan, they are a member
        return "member"
        
    except Exception as e:
        print(f"Error determining user role: {e}")
        return None
# -----------------------------
# Views (No changes needed here)
# -----------------------------
INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>Whop Embed App (Python)</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
      pre { background:#0b1020; color:#f1f5f9; padding:16px; border-radius:12px; overflow:auto; }
      .grid { display:grid; grid-template-columns: 1fr; gap: 24px; }
      .card { border:1px solid #e5e7eb; border-radius:12px; padding:16px; }
      h2 { margin:0 0 8px 0; }
      .muted { color:#64748b; font-size: 12px; }
      .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
      input[type=text] { padding:8px 12px; border:1px solid #cbd5e1; border-radius:8px; width:320px; }
      button { padding:8px 12px; border:1px solid #1f2937; border-radius:8px; background:#111827; color:white; cursor:pointer; }
      button:disabled { opacity:0.5; cursor:not-allowed; }
      .badge { display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px; }
      .role-badge { display:inline-block; padding:4px 12px; border-radius:999px; font-size:14px; font-weight:600; }
      .role-owner { background:#fef3c7; color:#92400e; }
      .role-admin { background:#dbeafe; color:#1e40af; }
      .role-member { background:#d1fae5; color:#065f46; }
      .role-none { background:#f3f4f6; color:#6b7280; }
    </style>
  </head>
  <body>
    <h1>Whop Embed App (Python)</h1>
    <p class="muted">Loaded inside a Whop iframe, this page shows current user, company, role, and membership info.</p>
    <div class="grid">
      <div class="card">
        <h2>User Role</h2>
        {% if role %}
          <span class="role-badge role-{{ role }}">{{ role.upper() }}</span>
        {% else %}
          <span class="role-badge role-none">NO ROLE DETECTED</span>
        {% endif %}
        <p class="muted">Role detected: {{ role or 'None' }}</p>
      </div>
      <div class="card">
        <h2>Quick Actions</h2>
        <div class="row">
          <form method="GET" action="/dump" target="_blank">
            <button>Open JSON dump (new tab)</button>
          </form>
          <form method="GET" action="/company_members" style="margin-left:8px;" target="_blank">
            <button>List company members (server key)</button>
          </form>
          <form method="GET" action="/health" style="margin-left:8px;" target="_blank">
            <button>Health</button>
          </form>
        </div>
        <p class="muted">Server has app key: <span class="badge">{{ 'yes' if has_app_key else 'no' }}</span></p>
      </div>
      <div class="card">
        <h2>Embedded Token (header/cookie)</h2>
        {% if user_token %}
          <p class="muted">Found <code>x-whop-user-token</code> or cookie. Decoded (unverified) claims below.</p>
          <pre>{{ decoded_token | tojson(indent=2) }}</pre>
        {% else %}
          <p>No embedded user token found. If not running inside Whop, this is expected.</p>
        {% endif %}
      </div>
      <div class="card">
        <h2>/v5/me (Current User + Company Context)</h2>
        <pre>{{ me_block | tojson(indent=2) }}</pre>
      </div>
    </div>
  </body>
</html>
"""

@app.route("/")
def index():
    token = get_embedded_user_token(request)
    decoded = decode_jwt_no_verify(token) if token else {}
    me_code, me_data = (0, {"note": "no token"}) if not token else me_user(token)
    has_key = bool(APP_API_KEY)
    comp_code, comp_data = (0, {"note": "no app key"}) if not has_key else company_current(app_headers())
    role = None
    if token and has_key and isinstance(me_data, dict) and isinstance(comp_data, dict):
        user_id = me_data.get("id")
        company_id = comp_data.get("id")
        if user_id and company_id:
            role = get_user_role_in_company(user_id, company_id, app_headers(), ADMIN_PLAN_IDS)
    return render_template_string(
        INDEX_HTML,
        user_token=bool(token),
        decoded_token=decoded or {"note": "no token"},
        me_block={"status": me_code, "data": me_data},
        company_block={"status": comp_code, "data": comp_data},
        has_app_key=has_key,
        role=role
    )

@app.route("/dump")
def dump():
    token = get_embedded_user_token(request)
    decoded = decode_jwt_no_verify(token) if token else {}
    me_code, me_data = (0, {"note": "no token"}) if not token else me_user(token)
    has_key = bool(APP_API_KEY)
    comp_code, comp_data = (0, {"note": "no app key"}) if not has_key else company_current(app_headers())
    role = None
    if token and has_key and isinstance(me_data, dict) and isinstance(comp_data, dict):
        user_id = me_data.get("id")
        company_id = comp_data.get("id")
        if user_id and company_id:
            role = get_user_role_in_company(user_id, company_id, app_headers(), ADMIN_PLAN_IDS)
    is_admin = role == "admin"
    is_owner = role == "owner"
    is_member = role == "member"
    return jsonify({
        "embedded_token_present": bool(token),
        "me": {"status": me_code, "data": me_data},
        "company_current": {"status": comp_code, "data": comp_data},
        "has_app_api_key": has_key,
        "role": role,
        "is_admin": is_admin,
        "is_owner": is_owner,
        "is_member": is_member,
    })

@app.route("/company_members")
def company_members():
    if not APP_API_KEY:
        return jsonify({"error": "Missing WHOP_APP_API_KEY on server"}), 400
    company_id = request.args.get("company_id")
    code, members = company_memberships(app_headers(), company_id=company_id)
    return jsonify({"status": code, "memberships": members})

@app.route("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)