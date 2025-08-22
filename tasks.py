# In tasks.py
import re
import os
import json
import redis
from postgrest.exceptions import APIError
from huey import SqliteHuey, RedisHuey
from huey.exceptions import TaskException
from utils.youtube_utils import (
    get_transcripts_from_channel, # <-- Use the robust function for new channels
    get_transcripts_from_urls,    # <-- Use the targeted function for syncing
    youtube_api
)
from utils.embed_utils import create_and_store_embeddings
from utils.supabase_client import get_supabase_admin_client
from utils.telegram_utils import send_message, create_channel_keyboard
from utils.config_utils import load_config
from utils.qa_utils import answer_question_stream, extract_topics_from_text,generate_channel_summary
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import logging
from utils.history_utils import save_chat_history

logger = logging.getLogger(__name__)

# Load environment variables from .env if present
load_dotenv()

redis_url = os.environ.get('REDIS_URL')

if redis_url:
    print("Connecting to Redis for Huey task queue...")
    huey = RedisHuey(url=redis_url)
else:
    print("Using SqliteHuey for task queue.")
    os.makedirs('data', exist_ok=True)
    huey = SqliteHuey(filename='data/huey_queue.db')

# --- Redis connection setup ---
try:
    redis_client = redis.from_url(os.environ.get('REDIS_URL'))
    print("Successfully connected to Redis for progress updates.")
except Exception as e:
    redis_client = None
    print(f"Could not connect to Redis for progress updates: {e}. Progress feature will be disabled.")


# --- Helper function ---
def update_task_progress(task_id, status, progress, message):
    """Updates the progress of a task in Redis."""
    if not redis_client:
        return
    progress_data = json.dumps({'status': status, 'progress': progress, 'message': message})
    redis_client.set(f"task_progress:{task_id}", progress_data, ex=3600)


# --- REFACTORED process_channel_task ---
@huey.task(context=True)
def process_channel_task(channel_id, task=None):
    """
    [REFACTORED] Background task for a NEW channel using the intelligent fetcher.
    """
    task_id = task.id if task else None
    supabase_admin = get_supabase_admin_client()

    try:
        update_task_progress(task_id, 'processing', 5, 'Fetching channel details...')
        channel_resp = supabase_admin.table('channels').select('channel_url, user_id').eq('id', channel_id).single().execute()
        
        if not channel_resp.data:
            raise ValueError(f"Channel with ID {channel_id} not found.")

        channel_url = channel_resp.data['channel_url']
        user_id_who_submitted = channel_resp.data['user_id']
        
        print(f"--- [TASK STARTED] Processing NEW channel ID: {channel_id} ({channel_url}) ---")
        supabase_admin.table('channels').update({'status': 'processing'}).eq('id', channel_id).execute()
        
        update_task_progress(task_id, 'processing', 10, 'Scanning for long-form videos...')
        # --- THIS IS THE CORRECTED LOGIC ---
        transcripts, thumbnail, subs = get_transcripts_from_channel(
            youtube_api, 
            channel_url, 
            target_video_count=250 # Target this many long-form videos
        )
        if not transcripts:
            raise ValueError("Could not find any long-form videos with transcripts.")
        # --- END OF CORRECTION ---

        update_task_progress(task_id, 'processing', 75, 'Building AI knowledge base...')
        create_and_store_embeddings(transcripts, channel_id, user_id_who_submitted)
        
        text_sample = " ".join([t['transcript'] for t in transcripts[:5]])[:10000]
        update_task_progress(task_id, 'processing', 90, 'Identifying topics...')
        topics = extract_topics_from_text(text_sample)
        
        update_task_progress(task_id, 'processing', 95, 'Finalizing...')
        summary = generate_channel_summary(text_sample)
        
        video_data = list(reversed([
            {'video_id': t['video_id'], 'title': t['title'], 'url': t['url'], 'upload_date': t['upload_date']}
            for t in transcripts
        ]))
        channel_name = transcripts[0]['uploader'].strip() if transcripts else "Unknown Channel"
        
        supabase_admin.table('channels').update({
            'channel_name': channel_name,
            'channel_thumbnail': thumbnail,
            'videos': video_data,
            'subscriber_count': subs,
            'topics': topics,
            'summary': summary,
            'status': 'ready'
        }).eq('id', channel_id).execute()

        update_task_progress(task_id, 'complete', 100, f"Success! The AI for '{channel_name}' is ready.")
        return f"Successfully processed {channel_name}"

    except Exception as e:
        logging.error(f"Task failed for channel ID {channel_id}: {e}", exc_info=True)
        supabase_admin.table('channels').update({'status': 'failed'}).eq('id', channel_id).execute()
        update_task_progress(task_id, 'failed', 0, str(e))
        raise e


# --- REFACTORED sync_channel_task ---
@huey.task(context=True)
def sync_channel_task(channel_id, task=None):
    """
    [REFACTORED] Background task to sync a channel, processing only new long-form videos.
    """
    task_id = task.id if task else None
    supabase_admin = get_supabase_admin_client()
    
    try:
        print(f"--- [SYNC TASK STARTED] Syncing channel_id: {channel_id} ---")
        update_task_progress(task_id, 'syncing', 5, 'Checking for new content...')

        channel_resp = supabase_admin.table('channels').select('channel_url, videos, user_id').eq('id', channel_id).single().execute()
        if not channel_resp.data:
            raise ValueError("Channel not found.")
        
        channel_url = channel_resp.data['channel_url']
        user_id = channel_resp.data['user_id']
        existing_videos = {v['video_id'] for v in channel_resp.data.get('videos', [])}
        print(f"Found {len(existing_videos)} existing videos for channel {channel_id}.")

        update_task_progress(task_id, 'syncing', 15, 'Scanning for new videos...')

        # To find new videos, we unfortunately can't use the intelligent scanner directly.
        # We must first find ALL recent videos and then determine which ones are new.
        # Note: 'extract_channel_videos' is no longer in youtube_utils, so we replicate its core logic here.
        
        # 1. Get the uploads playlist ID
        match = re.search(r'(?:channel/|c/|@|user/)([^/?\s]+)', channel_url)
        if not match: raise ValueError("Could not parse channel identifier.")
        search_resp = youtube_api.search().list(q=match.group(1), type='channel', part='id', maxResults=1).execute()
        if not search_resp.get('items'): raise ValueError("Channel not found via search.")
        yt_channel_id = search_resp['items'][0]['id']['channelId']
        channel_details = youtube_api.channels().list(part="contentDetails", id=yt_channel_id).execute()
        uploads_id = channel_details['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # 2. Get recent video IDs from the playlist
        latest_video_ids = []
        next_page_token = None
        # We scan up to 250 recent videos to check for new ones
        for _ in range(5): # 5 pages * 50 results = 250 videos
            playlist_resp = youtube_api.playlistItems().list(
                part="contentDetails", playlistId=uploads_id, maxResults=50, pageToken=next_page_token
            ).execute()
            latest_video_ids.extend([item['contentDetails']['videoId'] for item in playlist_resp.get('items', [])])
            next_page_token = playlist_resp.get('nextPageToken')
            if not next_page_token:
                break
        
        new_video_ids = [vid for vid in latest_video_ids if vid not in existing_videos]

        if not new_video_ids:
            print("No new videos to process.")
            update_task_progress(task_id, 'complete', 100, 'Channel is already up-to-date.')
            return "Channel is already up-to-date."

        print(f"Found {len(new_video_ids)} new videos to check.")
        update_task_progress(task_id, 'syncing', 30, f'Found {len(new_video_ids)} new videos. Analyzing for long-form content...')

        # 3. Process only the new video URLs and filter for long-form content
        new_video_urls = [f"https://www.youtube.com/watch?v={vid}" for vid in new_video_ids]
        new_transcripts = get_transcripts_from_urls(youtube_api, new_video_urls)
        
        if not new_transcripts:
            print("None of the new videos were long-form or had transcripts.")
            update_task_progress(task_id, 'complete', 100, 'No new long-form content found.')
            return "No new long-form content to add."
        
        update_task_progress(task_id, 'syncing', 70, 'Updating the AI knowledge base...')
        create_and_store_embeddings(new_transcripts, channel_id, user_id)
        
        update_task_progress(task_id, 'syncing', 95, 'Finalizing...')
        new_video_data = [
            {'video_id': t['video_id'], 'title': t['title'], 'url': t['url'], 'upload_date': t['upload_date']} 
            for t in new_transcripts
        ]
        
        updated_video_list = new_video_data + channel_resp.data.get('videos', [])
        supabase_admin.table('channels').update({'videos': updated_video_list}).eq('id', channel_id).execute()

        update_task_progress(task_id, 'complete', 100, f"Sync complete! Added {len(new_transcripts)} new videos.")
        print(f"--- [SYNC TASK SUCCESS] Channel {channel_id} updated with {len(new_transcripts)} new videos. ---")
        return f"Successfully added {len(new_transcripts)} videos."

    except Exception as e:
        print(f"--- [SYNC TASK FAILED] Critical error for channel {channel_id}: {e} ---")
        logging.error(f"Sync task failed for {channel_id}", exc_info=True)
        update_task_progress(task_id, 'failed', 0, str(e))
        raise e

# (The rest of the file: consume_answer_stream, process_private_message, etc. remains unchanged)

def consume_answer_stream(question, config, channel_data, video_ids, user_id, access_token):
    """
    This is the corrected helper function.
    """
    full_answer = ""
    sources = []
    # This now correctly passes the question argument to the underlying stream function.
    stream = answer_question_stream(
        question_for_prompt=question,
        question_for_search=question,
        channel_data=channel_data,
        video_ids=video_ids,
        user_id=user_id,
        access_token=access_token
    )

    for chunk in stream:
        if chunk.startswith('data: '):
            data_str = chunk.replace('data: ', '').strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                if data.get('answer'):
                    full_answer += data['answer']
                if data.get('sources'):
                    sources = data['sources']
            except json.JSONDecodeError:
                continue
    return full_answer, sources

def process_private_message(message: dict):
    """
    This is the complete and corrected function for handling private Telegram messages.
    It re-initializes the Supabase client to prevent stale connection errors.
    """
    chat_id = message['chat']['id']
    text = message.get('text', '').strip()
    telegram_username = message['from'].get('username', f"user_{message['from']['id']}")
    print(f"[Private Chat] Received message from chat_id {chat_id}: '{text}'")

    # --- THIS IS THE FIX ---
    # Get a fresh database connection every time the task runs.
    supabase_admin = get_supabase_admin_client()
    # --- END FIX ---

    if text.startswith('/connect'):
        code = text.split(' ')[-1]
        print(f"[Private Chat] Attempting to connect with code: {code}")

        if len(code) < 16:
            send_message(chat_id, "This doesn't look like a valid connection code.")
            return

        ten_minutes_ago = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        connection_resp = supabase_admin.table('telegram_connections').select('*').eq('connection_code', code).eq('is_active', False).gte('created_at', ten_minutes_ago).limit(1).execute()

        if connection_resp.data:
            connection = connection_resp.data[0]
            print(f"[Private Chat] Found valid connection record: {connection['id']}")
            supabase_admin.table('telegram_connections').update({
                'is_active': True,
                'telegram_chat_id': chat_id,
                'telegram_username': telegram_username
            }).eq('id', connection['id']).execute()

            send_message(chat_id, "✅ Success! Your account is now connected.")
            
            user_id = connection['app_user_id']
            channels_resp = supabase_admin.table('user_channels').select('channels(channel_name)').eq('user_id', user_id).execute()
            user_channel_names = [item['channels']['channel_name'] for item in channels_resp.data if item.get('channels')]
            
            keyboard = create_channel_keyboard(user_channel_names)
            send_message(chat_id, "Which channel do you want to ask about? Please select one from the keyboard or just type a question.", reply_markup=keyboard)
        else:
            print(f"[Private Chat] Invalid or expired connection code received: {code}")
            send_message(chat_id, "❌ This connection code is invalid or has expired.")
        return

    active_connection_resp = supabase_admin.table('telegram_connections').select('*').eq('telegram_chat_id', chat_id).eq('is_active', True).limit(1).execute()

    if not active_connection_resp.data:
        print(f"[Private Chat] No active connection found for chat_id {chat_id}.")
        config = load_config()
        app_url = config.get("app_base_url", "your website")
        connect_url = f"{app_url}/telegram/connect"
        send_message(chat_id, f"Welcome! Please connect your account first:\n{connect_url}")
        return

    connection = active_connection_resp.data[0]
    user_id = connection['app_user_id']
    print(f"[Private Chat] Active session for user_id: {user_id}")
    
    if text == '/start' or text == '/ask':
        channels_resp = supabase_admin.table('user_channels').select('channels(channel_name)').eq('user_id', user_id).execute()
        user_channel_names = [item['channels']['channel_name'] for item in channels_resp.data if item.get('channels')]
        keyboard = create_channel_keyboard(user_channel_names)
        
        message_text = "Welcome! Which channel would you like to ask about?" if text == '/start' else "Which channel do you want to ask about? Please select one from the keyboard or type its name."
        send_message(chat_id, message_text, reply_markup=keyboard)
        return

    if text.startswith('Ask: '):
        channel_context = text.replace('Ask: ', '').strip()
        supabase_admin.table('telegram_connections').update({'last_channel_context': channel_context if channel_context != "General Q&A" else None}).eq('id', connection['id']).execute()
        send_message(chat_id, f"OK. Context set to '{channel_context}'. What would you like to ask?")
        return

    try:
        send_message(chat_id, "Thinking...")

        channel_data = None
        video_ids = None

        if connection.get('last_channel_context'):
            channel_name_context = connection['last_channel_context']
            channel_details_resp = supabase_admin.table('channels').select('*').eq('channel_name', channel_name_context).limit(1).execute()
            if channel_details_resp.data:
                channel_data = channel_details_resp.data[0]
                video_ids = {v['video_id'] for v in channel_data.get('videos', [])}

        config = load_config()
        full_answer, sources = consume_answer_stream(text, config, channel_data, video_ids, user_id, access_token=None)

        if not full_answer:
            full_answer = "I couldn't find an answer to your question."

        response_text = full_answer
        if sources:
            response_text += "\n\n*Sources:*"
            for i, source in enumerate(sources[:3]):
                response_text += f"\n{i+1}. [{source['title']}]({source['url']})"

        send_message(chat_id, response_text, parse_mode='Markdown')

    except Exception as e:
        print(f"[Private Chat] Error processing question for chat_id {chat_id}: {e}")
        send_message(chat_id, "Sorry, an error occurred while processing your question.")


def process_group_message(message: dict):
    """
    This is the complete and corrected function for handling group chat messages.
    """
    chat_id = message['chat']['id']
    chat_title = message['chat'].get('title', 'Unknown Group')
    text = message.get('text', '').strip()
    print(f"[Group Chat] Received message from {chat_title} ({chat_id}): '{text}'")

    # Re-initialize the client inside the task for a fresh connection
    supabase_admin = get_supabase_admin_client()

    if text.startswith('/link_channel'):
        code = text.split(' ')[-1]
        print(f"[Group Chat] Attempting to link group with code: {code}")
        conn_resp = supabase_admin.table('group_connections').select('*').eq('connection_code', code).limit(1).execute()

        if conn_resp.data:
            supabase_admin.table('group_connections').update({
                'is_active': True,
                'group_chat_id': chat_id,
                'group_title': chat_title
            }).eq('connection_code', code).execute()
            send_message(chat_id, "✅ This group is now successfully linked! Community members can now ask questions by mentioning the bot.")
        else:
            print(f"[Group Chat] Invalid connection code received: {code}")
            send_message(chat_id, "❌ That connection code is invalid or expired.")
        return

    config = load_config()
    bot_token = config.get("telegram_bot_token", "")
    bot_username = config.get("telegram_bot_username")

    if not bot_username:
        print("telegram_bot_username not set in config.json. Cannot detect mentions.")
        return

    is_reply_to_bot = message.get('reply_to_message', {}).get('from', {}).get('is_bot', False)
    if not (bot_username in text or is_reply_to_bot):
        print(f"Ignoring group message as it does not mention '{bot_username}'.")
        return

    group_conn_resp = supabase_admin.table('group_connections').select('*, channels(*)').eq('group_chat_id', chat_id).eq('is_active', True).limit(1).execute()

    if not group_conn_resp.data:
        send_message(chat_id, "This group is not linked to a YouTube channel.")
        return

    try:
        user_who_asked = message.get('from', {})
        user_first_name = user_who_asked.get('first_name', 'User')

        connection = group_conn_resp.data[0]
        channel_data = connection.get('channels')
        if not channel_data:
            send_message(chat_id, "Error: The linked YouTube channel data could not be found.")
            return

        owner_user_id = connection['owner_user_id']
        question = text.replace(bot_username, "").strip()
        video_ids = {v['video_id'] for v in channel_data.get('videos', [])}

        # We now pass 'access_token=None' as the last argument.
        full_answer, sources = consume_answer_stream(question, config, channel_data, video_ids, owner_user_id, access_token=None)

        if not full_answer:
            full_answer = "I couldn't find an answer to your question in the video transcripts."

        response_text = f"Hey {user_first_name}!\n\n{full_answer}"
        if sources:
            response_text += "\n\n*Sources from the videos:*"
            for i, source in enumerate(sources[:2]):
                response_text += f"\n- [{source['title']}]({source['url']})"

        # Using the flexible send_message helper function
        send_message(chat_id, response_text, parse_mode='Markdown', reply_to_message_id=message['message_id'], disable_web_page_preview=True)

    except Exception as e:
        print(f"[Group Chat] Error processing question for chat_id {chat_id}: {e}")



@huey.task()
def process_telegram_update_task(update: dict):
    print(f"--- New Task Received by Huey ---")
    print(f"Update Data: {json.dumps(update, indent=2)}")

    message = update.get('message')
    if not message:
        print("Update received without a 'message' key. Ignoring.")
        return

    chat = message.get('chat')
    if not chat:
        print("Message received without a 'chat' key. Ignoring.")
        return

    is_group_chat = chat.get('type') in ['group', 'supergroup']

    if is_group_chat:
        process_group_message(message)
    else:
        process_private_message(message)

@huey.task()
def delete_channel_task(channel_id: int, user_id: str):
    """
    Background task to UNLINK a channel and PERMANENTLY DELETE all its
    associated data if it becomes orphaned.
    """
    try:
        print(f"--- [DELETE TASK STARTED] Unlinking Channel ID: {channel_id} from User ID: {user_id} ---")
        supabase_admin = get_supabase_admin_client()

        # --- FIX: Decrement user's personal channel count if it's a personal channel ---
        # We must do this BEFORE unlinking, while we can still easily check the channel's status.
        channel_details_resp = supabase_admin.table('channels').select('is_shared').eq('id', channel_id).single().execute()
        if channel_details_resp.data and not channel_details_resp.data.get('is_shared'):
            print(f"Channel {channel_id} is a personal channel. Decrementing count for user {user_id}.")
            supabase_admin.rpc('decrement_channel_count', {'p_user_id': user_id}).execute()
        # --- END FIX ---

        # Step 1: Unlink the user from the channel.
        supabase_admin.table('user_channels').delete().match({
            'user_id': user_id,
            'channel_id': channel_id
        }).execute()
        print(f"Successfully unlinked channel {channel_id} from user {user_id}.")

        # Step 2: Check if the channel is now orphaned.
        other_users_response = supabase_admin.table('user_channels') \
            .select('user_id', count='exact') \
            .eq('channel_id', channel_id) \
            .execute()

        if other_users_response.count == 0:
            print(f"Channel {channel_id} is orphaned. Deleting all associated data.")

            # Step 3: Get the channel details to find its associated videos.
            channel_details_response = supabase_admin.table('channels').select('videos').eq('id', channel_id).single().execute()
            
            if channel_details_response.data and channel_details_response.data.get('videos'):
                video_ids = [v['video_id'] for v in channel_details_response.data['videos']]
                
                # Step 4: Delete all embeddings linked to those videos.
                if video_ids:
                    print(f"Found {len(video_ids)} videos. Deleting associated embeddings...")
                    supabase_admin.table('embeddings').delete().in_('video_id', video_ids).execute()
                    print(f"Deleted embeddings for videos: {video_ids}")

            # Step 5: Finally, delete the master channel record.
            supabase_admin.table('channels').delete().eq('id', channel_id).execute()
            print(f"Deleted master record for channel {channel_id}.")
            print(f"--- [DELETE TASK SUCCESS] Permanently deleted channel {channel_id}. ---")
        
        else:
            print(f"--- [DELETE TASK SUCCESS] Channel {channel_id} is still in use by {other_users_response.count} other users. ---")

    except Exception as e:
        if isinstance(e, APIError):
            error_message = e.message
        else:
            error_message = str(e)
        
        print(f"--- [DELETE TASK FAILED] Error for channel {channel_id}: {error_message} ---")
        raise

    
@huey.task()
def post_answer_processing_task(user_id, channel_name, question, answer, sources):
    """
    Handles background database operations after an answer is streamed.
    """
    try:
        # Get a fresh admin client for background tasks
        admin_supabase = get_supabase_admin_client()

        # 1. Increment the user's query count
        print(f"Incrementing query count for user {user_id}")
        admin_supabase.rpc('increment_query_count', {'p_user_id': user_id}).execute()

        # 2. Save the chat history
        # Note: We use the admin client here for reliability in background tasks
        print(f"Saving chat history for user {user_id}")
        save_chat_history(
            supabase_client=admin_supabase,
            user_id=user_id,
            channel_name=channel_name,
            question=question,
            answer=answer,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error in post-answer processing for user {user_id}: {e}", exc_info=True)