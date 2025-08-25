# In utils/subscription_utils.py

import os
import logging
from datetime import date
from functools import wraps
from flask import session, jsonify
import redis
import json
from .supabase_client import get_supabase_admin_client

# --- (This section is unchanged) ---
try:
    redis_client = redis.from_url(os.environ.get("REDIS_URL"))
    CACHE_DURATION_SECONDS = 300 
    print("Successfully connected to Redis for caching.")
except Exception as e:
    redis_client = None
    print(f"Could not connect to Redis for caching: {e}. Caching will be disabled.")

PLANS = {
    'free': {
        'name': 'Free',
        'max_channels': 3,
        'max_queries_per_month': 50,
        'max_videos_per_channel': 1
    },
    'creator': {
        'name': 'Creator',
        'max_channels': 10,
        'max_queries_per_month': 2500,
        'max_videos_per_channel': 250
    },
    'pro': {
        'name': 'Pro',
        'max_channels': float('inf'),
        'max_queries_per_month': 10000,
        'max_videos_per_channel': float('inf')
    }
}


# --- START: COMPLETE AND CORRECTED FUNCTION ---
def get_user_subscription_status(user_id):
    """
    Fetches the user's profile, subscription plan, and usage stats.
    Caches results in Redis for 5 minutes to improve performance.
    """
    cache_key = f"user_status:{user_id}"
    if redis_client:
        try:
            cached_status = redis_client.get(cache_key)
            if cached_status:
                logging.info(f"Cache HIT for user {user_id}.")
                loaded_data = json.loads(cached_status)
                # Convert 'inf' strings back to float('inf')
                if loaded_data.get('limits', {}).get('max_queries_per_month') == 'inf':
                    loaded_data['limits']['max_queries_per_month'] = float('inf')
                if loaded_data.get('limits', {}).get('max_channels') == 'inf':
                    loaded_data['limits']['max_channels'] = float('inf')
                return loaded_data
        except redis.RedisError as e:
            logging.error(f"Redis GET error for user {user_id}: {e}. Fetching from DB.")
    
    logging.info(f"Cache MISS for user {user_id}. Fetching from database.")

    try:
        supabase_admin = get_supabase_admin_client()
        profile_data = None
        
        # 1. Get user profile
        profile_resp = supabase_admin.table('user_profiles').select('*').eq('id', user_id).execute()
        if not profile_resp.data:
            # This is a fallback and shouldn't happen for logged-in users, but is safe to have
            return None 
        profile_data = profile_resp.data[0]

        plan_name = profile_data.get('subscription_plan', 'free')
        plan_limits = PLANS.get(plan_name, PLANS['free'])

        # 2. Get usage stats
        usage_resp = supabase_admin.table('usage_stats').select('*').eq('user_id', user_id).execute()
        if not usage_resp.data:
            # Create usage stats if they don't exist
            usage_insert_resp = supabase_admin.table('usage_stats').insert({
                'user_id': user_id,
                'queries_this_month': 0,
                'channels_processed': 0,
                'last_reset_date': date.today().isoformat()
            }).execute()
            usage_data = usage_insert_resp.data[0]
        else:
            usage_data = usage_resp.data[0]

        # 3. Reset monthly query count if a new month has started
        if usage_data.get('last_reset_date'):
            last_reset = date.fromisoformat(usage_data['last_reset_date'])
            if last_reset.month != date.today().month:
                logging.info(f"Resetting monthly query count for user {user_id}.")
                update_resp = supabase_admin.table('usage_stats').update({
                    'queries_this_month': 0,
                    'last_reset_date': date.today().isoformat()
                }).eq('user_id', user_id).execute()
                if update_resp.data:
                    usage_data = update_resp.data[0]

        queries_used = usage_data.get('queries_this_month', 0)
        max_queries = plan_limits.get('max_queries_per_month', 0)
        
        queries_remaining_value = float('inf') if max_queries == float('inf') else int(max_queries - queries_used)

        # 4. Construct the complete status dictionary
        status_to_cache = {
            'profile': profile_data,
            'plan': plan_name,
            'usage': usage_data,
            'limits': plan_limits,
            'queries_remaining': queries_remaining_value 
        }

        # 5. Cache the result in Redis before returning
        if redis_client:
            try:
                def json_default(o):
                    if o == float('inf'):
                        return 'inf' 
                    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')
                
                redis_client.setex(
                    cache_key, 
                    CACHE_DURATION_SECONDS, 
                    json.dumps(status_to_cache, default=json_default)
                )
                logging.info(f"Cache SET for user {user_id}.")
            except redis.RedisError as e:
                 logging.error(f"Redis SETEX error for user {user_id}: {e}.")

        return status_to_cache
        
    except Exception as e:
        logging.error(f"Unhandled exception in get_user_subscription_status for user {user_id}: {e}", exc_info=True)
        return None
# --- END: COMPLETE AND CORRECTED FUNCTION ---


def subscription_required(check_type):
    """
    A decorator to verify a user's subscription status before allowing an action.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                return jsonify({'status': 'error', 'message': 'Authentication required.'}), 401
            
            user_id = session['user']['id']
            status = get_user_subscription_status(user_id)

            if not status or 'limits' not in status or 'usage' not in status:
                 return jsonify({'status': 'error', 'message': 'Could not verify user subscription.'}), 500
            
            # Use .get() for safer dictionary access
            max_channels = status['limits'].get('max_channels', 0)
            max_queries = status['limits'].get('max_queries_per_month', 0)

            if check_type == 'channel':
                current_channels = status['usage'].get('channels_processed', 0)
                if current_channels >= max_channels:
                    return jsonify({
                        'status': 'limit_reached', 
                        'message': f"You have reached the maximum of {max_channels} channels for the '{status['plan']}' plan."
                    }), 403
            
            elif check_type == 'query':
                current_queries = status['usage'].get('queries_this_month', 0)
                if current_queries >= max_queries:
                     return jsonify({
                        'status': 'limit_reached', 
                        'message': f"You have reached your monthly query limit of {max_queries} for the '{status['plan']}' plan."
                    }), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator