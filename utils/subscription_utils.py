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
    },
    'pro_community': {
        'name': 'Pro Community',
        'shared_channels_allowed': 2,
    },
    'rich_community': {
        'name': 'Rich Community',
        'shared_channels_allowed': 5,
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
    Fetches a user's status, now aware of the active community context.
    The cache key now includes the active_community_id to store different contexts.
    """
    cache_key = f"user_status:{user_id}:community:{active_community_id or 'none'}"
    if redis_client:
        try:
            cached_status = redis_client.get(cache_key)
            if cached_status:
                return json.loads(cached_status)
        except redis.RedisError as e:
            print(f"Redis GET error for user {user_id}: {e}. Fetching from DB.")

    supabase_admin = get_supabase_admin_client()
    profile = get_profile(user_id)
    if not profile:
        return None

    is_whop_user = bool(profile.get('whop_user_id'))
    
    # Determine the user's plan based on their type (Whop vs. regular)
    raw_plan_id = profile.get('personal_plan_id') or profile.get('direct_subscription_plan')
    
    if not raw_plan_id:
        if is_whop_user and active_community_id:
            # Whop members without a personal plan inherit a default member plan
            # based on their community's subscription level.
            community_status = get_community_status(active_community_id)
            community_plan_id = community_status.get('plan_id') if community_status else 'basic_community'
            
            if community_plan_id == 'pro_community':
                raw_plan_id = 'whop_pro_member'
            elif community_plan_id == 'rich_community':
                raw_plan_id = 'whop_rich_member'
            else: # Default for 'basic_community' and any other case
                raw_plan_id = 'whop_basic_member'
        else:
            # Regular users default to the 'free' plan.
            raw_plan_id = 'free'
            
    plan_details = PLANS.get(raw_plan_id, PLANS['free']) # Fallback to 'free' just in case

    # Get personal usage stats
    usage_resp = supabase_admin.table('usage_stats').select('*').eq('user_id', user_id).maybe_single().execute()
    usage_data = usage_resp.data if hasattr(usage_resp, 'data') and usage_resp.data else {}

    status = {
        'user_id': user_id,
        'plan_id': raw_plan_id,
        'plan_name': plan_details['name'],
        'is_whop_user': is_whop_user,
        'active_community_id': active_community_id,
        'is_active_community_owner': False,
        'community_role': None,
        'limits': plan_details,
        'usage': {
            'queries_this_month': usage_data.get('queries_this_month', 0),
            'channels_processed': usage_data.get('channels_processed', 0)
        }
    }

    # If there's an active community, check ownership and determine the user's role
    if active_community_id:
        community_res = supabase_admin.table('communities').select('owner_user_id').eq('id', active_community_id).single().execute()
        if community_res.data and str(community_res.data['owner_user_id']) == str(user_id):
            status['is_active_community_owner'] = True

        role = whop_api.get_user_role_in_company(user_id)
        status['community_role'] = role

    if redis_client:
        serializable_status = json.loads(json.dumps(status).replace('Infinity', '"inf"'))
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
                # For Whop users, check community limits
                if active_community_id:
                    community_status = get_community_status(active_community_id)
                    if not community_status:
                        return jsonify({'status': 'error', 'message': 'Active community not found.'}), 404

                    if user_status.get('is_active_community_owner') and community_status['usage']['trial_queries_used'] < community_status['limits']['owner_trial_limit']:
                        return f(*args, **kwargs) # Owner is in trial period

                    # --- REFINED LOGIC ---
                    # Check if the community's plan has no queries (trial ended and no paid plan)
                    if community_status['limits']['query_limit'] == 0:
                        # For the owner, the trial has expired.
                        if user_status.get('is_active_community_owner'):
                             return jsonify({
                                'status': 'limit_reached',
                                'message': "Your trial has ended. Please subscribe to a community plan to continue."
                            }), 403
                        # For members of that community.
                        else:
                            return jsonify({
                                'status': 'limit_reached',
                                'message': "This community does not have an active plan. Please contact the owner."
                            }), 403

                    # Check if the community has used all its monthly queries
                    if community_status['usage']['queries_used'] >= community_status['limits']['query_limit']:
                        return jsonify({
                            'status': 'limit_reached',
                            'message': "This community's query limit has been reached for the month."
                        }), 403
                # For direct users, check personal limits
                else:
                    if user_status['usage']['queries_this_month'] >= user_status['limits']['max_queries_per_month']:
                        return jsonify({'status': 'limit_reached', 'message': f"You've reached your monthly query limit of {user_status['limits']['max_queries_per_month']}."}), 403

            # --- Personal Channel Limit Logic (for all users) ---
            elif check_type == 'channel':
                max_channels = user_status['limits'].get('max_channels', 0)
                current_channels = user_status['usage'].get('channels_processed', 0)
                if current_channels >= max_channels:
                    return jsonify({
                        'status': 'limit_reached',
                        'message': f"You have reached the maximum of {max_channels} personal channels for your plan.",
                        'action': 'upgrade_personal_plan' # More specific action
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
