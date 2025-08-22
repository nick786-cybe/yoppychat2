# yoppychat2/utils/subscription_utils.py

import os
from functools import wraps
from flask import session, jsonify
import redis
import json
from .supabase_client import get_supabase_admin_client
from .db_utils import get_profile
from . import whop_api

# --- Redis Caching Setup ---
try:
    redis_client = redis.from_url(os.environ.get("REDIS_URL"))
    CACHE_DURATION_SECONDS = 300 # Cache for 5 minutes
    print("Successfully connected to Redis for caching.")
except Exception as e:
    redis_client = None
    print(f"Could not connect to Redis for caching: {e}. Caching will be disabled.")

# --- Plan Definitions ---
COMMUNITY_PLANS = {
    'basic_community': {
        'name': 'Basic Community',
        'shared_channels_allowed': 1,
        'queries_per_month': 50
    },
    'pro_community': {
        'name': 'Pro Community',
        'shared_channels_allowed': 2,
        'queries_per_month': 100
    },
    'rich_community': {
        'name': 'Rich Community',
        'shared_channels_allowed': 5,
        'queries_per_month': 250
    }
}

PLANS = {
    'free': { 'name': 'Free', 'max_channels': 0, 'max_queries_per_month': 50 },
    'creator': { 'name': 'Creator', 'max_channels': 10, 'max_queries_per_month': 2500 },
    'pro': { 'name': 'Pro', 'max_channels': float('inf'), 'max_queries_per_month': 10000 },
    'whop_basic_member': { 'name': 'Basic Member', 'max_channels': 2, 'max_queries_per_month': 50 },
    'whop_pro_member': { 'name': 'Pro Member', 'max_channels': 5, 'max_queries_per_month': 100 },
    'whop_rich_member': { 'name': 'Rich Member', 'max_channels': 10, 'max_queries_per_month': 300 }
}

def get_community_status(community_id: str) -> dict:
    """
    Fetches a community's plan, limits, and current usage.
    """
    cache_key = f"community_status:{community_id}"
    if redis_client:
        try:
            cached_status = redis_client.get(cache_key)
            if cached_status:
                return json.loads(cached_status)
        except redis.RedisError as e:
            print(f"Redis GET error for community {community_id}: {e}. Fetching from DB.")

    supabase_admin = get_supabase_admin_client()
    response = supabase_admin.table('communities').select('*').eq('id', community_id).single().execute()
    if not response.data:
        return None

    community_data = response.data
    plan_id = community_data.get('plan_id', 'basic_community')
    plan_details = COMMUNITY_PLANS.get(plan_id, COMMUNITY_PLANS['basic_community'])

    status = {
        'community_id': community_id,
        'plan_id': plan_id,
        'plan_name': plan_details['name'],
        'limits': {
            'shared_channel_limit': plan_details['shared_channels_allowed'],
            'queries_per_month': plan_details.get('queries_per_month', 50),
            'query_limit': community_data.get('query_limit', 0),
            'owner_trial_limit': 10
        },
        'usage': {
            'queries_used': community_data.get('queries_used', 0),
            'trial_queries_used': community_data.get('trial_queries_used', 0)
        }
    }

    if redis_client:
        redis_client.setex(cache_key, CACHE_DURATION_SECONDS, json.dumps(status))

    return status

def get_user_status(user_id: str, active_community_id: str = None) -> dict:
    """
    Fetches a user's status with the final, corrected logic.
    """
    cache_key = f"user_status:{user_id}:community:{active_community_id or 'none'}"
    if redis_client:
        try:
            cached_status = redis_client.get(cache_key)
            if cached_status:
                cached_data = json.loads(cached_status)
                if cached_data.get('limits', {}).get('max_channels') == 'inf':
                    cached_data['limits']['max_channels'] = float('inf')
                if cached_data.get('limits', {}).get('max_queries_per_month') == 'inf':
                    cached_data['limits']['max_queries_per_month'] = float('inf')
                return cached_data
        except redis.RedisError as e:
            print(f"Redis GET error for user {user_id}: {e}. Fetching from DB.")

    supabase_admin = get_supabase_admin_client()
    profile = get_profile(user_id)
    if not profile:
        return None

    is_whop_user = bool(profile.get('whop_user_id'))
    personal_plan_id = profile.get('personal_plan_id') or profile.get('direct_subscription_plan')

    raw_plan_id = None
    plan_details = {}

    # 1. Determine the user's plan ID based on their status and subscriptions.
    if personal_plan_id:
        # User has a personal plan. Check if it's valid for their user type.
        is_valid_plan = False
        if is_whop_user and personal_plan_id.startswith('whop_'):
            is_valid_plan = True
        elif not is_whop_user and personal_plan_id in ['creator', 'pro', 'free']:
            is_valid_plan = True

        if is_valid_plan and personal_plan_id in PLANS:
            raw_plan_id = personal_plan_id
        else:
            # Plan ID in DB is invalid or doesn't match user type. Fall through to defaults.
            personal_plan_id = None # Nullify to trigger default logic.

    if not raw_plan_id:
        # User has no valid personal plan, determine default.
        if is_whop_user and active_community_id:
            raw_plan_id = 'community_base' # Virtual plan for default community members
        elif is_whop_user:
            raw_plan_id = 'whop_base' # Virtual plan for Whop users outside a community
        else:
            raw_plan_id = 'free'

    # 2. Get the plan's limits and name.
    if raw_plan_id == 'community_base':
        community_status = get_community_status(active_community_id)
        plan_details = {
            'name': community_status.get('plan_name', 'Community Member') if community_status else 'Community Member',
            'max_channels': 0,
            'max_queries_per_month': community_status['limits'].get('queries_per_month', 50) if community_status else 50
        }
    elif raw_plan_id == 'whop_base':
        plan_details = { 'name': 'Basic Member', 'max_channels': 0, 'max_queries_per_month': 50 }
    else:
        # It's a real plan from the PLANS dict.
        plan_details = PLANS.get(raw_plan_id, PLANS['free'])

    # 3. Construct the final status object. No more stacking or overwrites needed.
    status = {
        'user_id': user_id,
        'plan_id': raw_plan_id,
        'plan_name': plan_details.get('name', 'Unknown Plan'),
        'has_personal_plan': bool(personal_plan_id),
        'is_whop_user': is_whop_user,
        'active_community_id': active_community_id,
        'is_active_community_owner': False,
        'community_role': None,
        'limits': plan_details.copy(),
        'usage': {
            'queries_this_month': get_profile(user_id).get('queries_this_month', 0),
            'channels_processed': get_profile(user_id).get('channels_processed', 0)
        }
    }

    if active_community_id:
        community_res = supabase_admin.table('communities').select('owner_user_id').eq('id', active_community_id).single().execute()
        if community_res.data and str(community_res.data['owner_user_id']) == str(user_id):
            status['is_active_community_owner'] = True
        role = whop_api.get_user_role_in_company(user_id)
        status['community_role'] = role

    if redis_client:
        serializable_status = json.loads(json.dumps(status, default=lambda o: 'inf' if o == float('inf') else o))
        redis_client.setex(cache_key, CACHE_DURATION_SECONDS, json.dumps(serializable_status))

    return status

def limit_enforcer(check_type: str):
    """
    Decorator to enforce limits. Handles both direct users and community users.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                return jsonify({'status': 'error', 'message': 'Authentication required.'}), 401

            user_id = session['user']['id']
            active_community_id = session.get('active_community_id')
            user_status = get_user_status(user_id, active_community_id)

            if not user_status:
                 return jsonify({'status': 'error', 'message': 'Could not verify user.'}), 500

            # --- Query Limit Logic ---
            if check_type == 'query':
                max_queries = user_status['limits'].get('max_queries_per_month', 0)
                if max_queries != float('inf') and user_status['usage']['queries_this_month'] >= max_queries:
                    return jsonify({
                        'status': 'limit_reached',
                        'message': f"You've reached your monthly query limit of {int(max_queries)}."
                    }), 403

            # --- Personal Channel Limit Logic ---
            elif check_type == 'channel':
                max_channels = user_status['limits'].get('max_channels', 0)
                current_channels = user_status['usage'].get('channels_processed', 0)
                if max_channels != float('inf') and current_channels >= max_channels:
                    message = f"You have reached the maximum of {max_channels} personal channels for your plan."
                    if user_status.get('is_whop_user'):
                        return jsonify({
                            'status': 'limit_reached',
                            'message': message,
                            'action': 'show_upgrade_popup'
                        }), 403
                    else:
                        return jsonify({
                            'status': 'limit_reached',
                            'message': message
                        }), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator

def community_channel_limit_enforcer(f):
    """
    Decorator for community owners adding shared channels.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return jsonify({'status': 'error', 'message': 'Authentication required.'}), 401

        user_id = session['user']['id']
        active_community_id = session.get('active_community_id')

        if not active_community_id:
            return jsonify({'status': 'error', 'message': 'No active community context found.'}), 400

        user_status = get_user_status(user_id, active_community_id)

        if not user_status.get('is_active_community_owner'):
            return jsonify({'status': 'error', 'message': 'Only community owners can perform this action.'}), 403

        community_status = get_community_status(active_community_id)
        if not community_status:
            return jsonify({'status': 'error', 'message': 'Could not verify community status.'}), 500

        from . import db_utils
        current_shared_channels = db_utils.count_shared_channels(active_community_id)
        max_shared_channels = community_status['limits'].get('shared_channel_limit', 0)

        if current_shared_channels >= max_shared_channels:
            return jsonify({
                'status': 'limit_reached',
                'message': f"You have reached the maximum of {max_shared_channels} shared channels for your community's plan."
            }), 403

        return f(*args, **kwargs)

    return decorated_function
