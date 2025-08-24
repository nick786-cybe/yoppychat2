# In utils/db_utils.py

import logging
from .supabase_client import get_supabase_admin_client

# It's good practice to use the shared admin client for these utility functions
# as they are often called from background tasks or trusted server-side routes.
supabase = get_supabase_admin_client()
log = logging.getLogger(__name__)

def get_profile(user_id: str):
    """Fetches a user's complete profile data."""
    try:
        response = supabase.table('profiles').select('*').eq('id', user_id).maybe_single().execute()
        # Add a check to ensure response is not None before accessing .data
        return response.data if response else None
    except Exception as e:
        log.error(f"Error getting profile for user {user_id}: {e}")
        return None

def link_user_to_community(user_id: str, community_id: str):
    """Creates a link in the user_communities join table."""
    try:
        supabase.table('user_communities').upsert(
            {'user_id': user_id, 'community_id': community_id},
            ignore_duplicates=True
        ).execute()
        return True
    except Exception as e:
        log.error(f"Error linking user {user_id} to community {community_id}: {e}")
        return False

def find_channel_by_url(channel_url: str):
    """Checks if a channel already exists in the master channels table."""
    try:
        response = supabase.table('channels').select('id, status').eq('channel_url', channel_url).maybe_single().execute()
        return response.data
    except Exception as e:
        log.error(f"Error finding channel by URL {channel_url}: {e}")
        return None

def link_user_to_channel(user_id: str, channel_id: int):
    """Creates a link in the user_channels join table."""
    try:
        # Upsert with ignore_duplicates=True is a safe way to ensure the link exists
        # without causing an error if it's already there.
        response = supabase.table('user_channels').upsert(
            {'user_id': user_id, 'channel_id': channel_id},
            ignore_duplicates=True
        ).execute()
        return response.data
    except Exception as e:
        log.error(f"Error linking user {user_id} to channel {channel_id}: {e}")
        return None

def create_channel(channel_url: str, user_id: str, is_shared: bool = False, community_id: str = None):
    """Adds a new channel to the master list with a 'pending' status."""
    try:
        channel_payload = {
            'channel_url': channel_url,
            'user_id': user_id, # Store who originally added it
            'status': 'pending',
            'is_shared': is_shared,
            'community_id': community_id
        }
        response = supabase.table('channels').insert(channel_payload).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        log.error(f"Error creating channel for URL {channel_url}: {e}")
        return None

def add_community(community_data: dict):
    """Adds a new community with default plan values."""
    try:
        # Set default values for a new community based on the 'basic_community' plan
        defaults = {
            'plan_id': 'basic_community',
            'query_limit': 0, # Will be set by Whop webhook based on member count
            'queries_used': 0,
            'shared_channel_limit': 1,
            'trial_queries_used': 0
        }
        # Merge provided data with defaults, letting provided data take precedence
        final_data = {**defaults, **community_data}

        response = supabase.table('communities').upsert(
            final_data, on_conflict='whop_community_id'
        ).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        log.error(f"Error adding community {community_data.get('whop_community_id')}: {e}")
        return None

def count_channels_for_user(user_id: str) -> int:
    """Counts the total number of channels (personal or shared) created by a specific user."""
    try:
        response = supabase.table('channels').select('id', count='exact').eq('user_id', user_id).execute()
        return response.count or 0
    except Exception as e:
        log.error(f"Error counting channels for user {user_id}: {e}")
        return 0

def count_shared_channels(community_id: str) -> int:
    """Counts the number of shared channels for a given community."""
    try:
        response = supabase.table('channels').select('id', count='exact').eq('community_id', community_id).eq('is_shared', True).execute()
        return response.count or 0
    except Exception as e:
        log.error(f"Error counting shared channels for community {community_id}: {e}")
        return 0

def increment_community_query_usage(community_id: str, is_trial: bool):
    """
    Increments the query counter for a specific community.
    Handles both the owner's trial and the shared community pool.
    """
    try:
        # Calls the updated RPC function that takes community_id directly.
        params = {'p_community_id': community_id, 'p_is_trial': is_trial}
        supabase.rpc('increment_query_usage', params).execute()
    except Exception as e:
        log.error(f"Error incrementing query usage for community {community_id}: {e}")

def increment_personal_query_usage(user_id: str):
    """
    Increments the query counter for a specific user.
    """
    try:
        params = {'p_user_id': user_id}
        supabase.rpc('increment_personal_query_usage', params).execute()
    except Exception as e:
        log.error(f"Error incrementing personal query usage for user {user_id}: {e}")

def increment_channels_processed(user_id: str):
    """
    Increments the channels_processed counter for a specific user.
    This should be called only when a new, unique channel is added to a user's list.
    """
    try:
        params = {'p_user_id': user_id}
        # Assumes a corresponding RPC function exists in the database.
        supabase.rpc('increment_channels_processed', params).execute()
    except Exception as e:
        log.error(f"Error incrementing channels processed for user {user_id}: {e}")

def create_initial_usage_stats(user_id: str):
    """Creates the initial usage_stats row for a new user."""
    try:
        # Using upsert is safe and prevents errors if the row somehow already exists.
        supabase.table('usage_stats').upsert({'user_id': user_id}).execute()
        return True
    except Exception as e:
        log.error(f"Error creating initial usage stats for user {user_id}: {e}")
        return False

def create_or_update_profile(profile_data: dict):
    """Creates or updates a user profile. Used for both direct and Whop users."""
    try:
        # Using upsert is efficient. It will update if 'id' exists, or insert if it doesn't.
        response = supabase.table('profiles').upsert(profile_data).execute()

        # Check if the upsert was successful
        if response.data:
            print(f"Successfully upserted profile for user ID: {profile_data.get('id')}")
            return response.data[0]

        # If upsert fails or returns no data, attempt a direct select
        user_id = profile_data.get('id')
        if user_id:
            return get_profile(user_id)

        return None
    except Exception as e:
        log.error(f"Error upserting profile: {e}")
        return None
