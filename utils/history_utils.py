import logging
import json
from datetime import datetime
from .supabase_client import get_supabase_client, get_supabase_admin_client
from supabase import Client as SupabaseClient

# --- FIX: Modified save_chat_history to accept a Supabase client ---
def save_chat_history(supabase_client: SupabaseClient, user_id, channel_name, question, answer, sources):
    """
    Save chat history to the database using the provided Supabase client.
    This allows the caller to decide whether to use a user-authenticated
    client or an admin client.
    """
    try:
        data = {
            'user_id': user_id,
            'channel_name': channel_name,
            'question': question,
            'answer': answer,
            'sources': sources,
            'created_at': datetime.utcnow().isoformat()
        }
        # Use the client passed into the function
        supabase_client.table('chat_history').insert(data).execute()
    except Exception as e:
        # Log the specific RLS error if it occurs
        logging.error(f"Error saving chat history for user {user_id}: {e}", exc_info=True)

# --- FIX: Modified get_chat_history to use an authenticated client ---
def get_chat_history(user_id, channel_name, access_token: str, limit=None):
    """Get chat history from the database using an authenticated client."""
    # Always use the user-specific client to respect RLS policies
    supabase = get_supabase_client(access_token=access_token)
    if not supabase:
        logging.error("Failed to initialize Supabase client in get_chat_history.")
        return []
        
    try:
        query = supabase.table('chat_history')\
            .select('*')\
            .eq('user_id', user_id)\
            .eq('channel_name', channel_name)\
            .order('created_at', desc=True)
        if limit is not None:
            query = query.limit(limit)
        
        response = query.execute()
        
        history = list(reversed(response.data))
        
        # Deserialize sources if needed
        for qa in history:
            if isinstance(qa.get('sources'), str):
                try:
                    qa['sources'] = json.loads(qa['sources'])
                except Exception:
                    qa['sources'] = []
            if qa.get('sources') is None:
                qa['sources'] = []
        return history
    except Exception as e:
        logging.error(f"Error getting chat history for user {user_id}: {e}", exc_info=True)
        return []
