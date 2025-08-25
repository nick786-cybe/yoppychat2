import logging
from functools import wraps
from utils.youtube_utils import is_youtube_video_url, clean_youtube_url
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, Response
import os
import json
import secrets
from datetime import datetime, timedelta, timezone
from tasks import huey, process_channel_task, sync_channel_task, process_telegram_update_task,delete_channel_task
from utils.qa_utils import answer_question_stream,search_and_rerank_chunks
from utils.supabase_client import get_supabase_client, get_supabase_admin_client,refresh_supabase_session
from utils.history_utils import get_chat_history, save_chat_history
from utils.telegram_utils import set_webhook, get_bot_token_and_url
from utils.config_utils import load_config
from utils.subscription_utils import get_user_subscription_status,subscription_required
from utils import prompts
import time
import redis
from postgrest.exceptions import APIError
from markupsafe import Markup
import markdown
from huey.exceptions import TaskException
from dotenv import load_dotenv
from flask_compress import Compress


logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
Compress(app)

@app.context_processor
def inject_subscription_data():
    if 'user' in session:
        user_id = session['user']['id']
        subscription_data = get_user_subscription_status(user_id)
        return dict(subscription=subscription_data, user=session.get('user'))
    return dict(subscription=None, user=None)

@app.template_filter('markdown')
def markdown_filter(text):
    return Markup(markdown.markdown(text))

app.secret_key = os.environ.get('SECRET_KEY', 'a_default_dev_secret_key')

try:
    redis_client = redis.from_url(os.environ.get('REDIS_URL'))
except Exception:
    redis_client = None

def get_user_channels():
    """
    Gets all channels linked to the current user, with Redis caching.
    Results are cached for 15 seconds to speed up navigation.
    """
    if 'user' not in session:
        return {}

    user_id = session['user']['id']
    cache_key = f"user_channels:{user_id}"

    if redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                #(f"CACHE HIT for user channels: {user_id}")
                return json.loads(cached_data)
        except Exception as e:
            logging.error(f"Redis GET error for user channels: {e}")

    #(f"CACHE MISS for user channels: {user_id}. Fetching from DB.")
    access_token = session.get('access_token')
    supabase = get_supabase_client(access_token)
    if not supabase:
        return {}

    user_channels = {}
    try:
        response = supabase.table('user_channels').select('channels(*)').eq('user_id', user_id).execute()
        if response.data:
            linked_channels = [item['channels'] for item in response.data if item.get('channels')]
            user_channels = {
                item['channel_name']: item
                for item in linked_channels
                if item and item.get('channel_name')
            }
    except APIError as e:
        logging.error(f"Could not fetch user channels due to APIError: {e.message}")
        if 'JWT expired' in e.message:
            session.clear()
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_user_channels: {e}")

    if redis_client and user_channels:
        try:
            redis_client.setex(cache_key, 15, json.dumps(user_channels))
        except Exception as e:
            logging.error(f"Redis SET error for user channels: {e}")
    
    return user_channels

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            logging.warning("Unauthorized access: User not in session.")
            return jsonify({'status': 'error', 'message': 'Authentication required. Please log in again.'}), 401
        try:
            return f(*args, **kwargs)
        except APIError as e:
            if 'JWT' in e.message and 'expired' in e.message:
                logging.warning("Caught expired JWT. Clearing session and sending 401.")
                session.clear()
                return jsonify({
                    'status': 'error',
                    'message': 'Your session has expired. The page will now reload.',
                    'action': 'logout'
                }), 401
            else:
                raise e
    return decorated_function

@app.before_request
def check_token_expiry():
    """
    Runs before every request to check if the user's token is about to expire.
    If it is, it attempts to refresh it in the background.
    """
    # Check if user is logged in and we have an expiry time
    if 'user' in session and session.get('expires_at'):
        # time.time() is in seconds, so we compare
        current_time = time.time()

        # Check if token expires in the next 60 seconds
        if session['expires_at'] < (current_time + 60):
            #("JWT is about to expire. Attempting to refresh...")

            refresh_token = session.get('refresh_token')
            if not refresh_token:
                logging.warning("No refresh token found. Logging user out.")
                session.clear()
                return

            new_session_data = refresh_supabase_session(refresh_token)

            if new_session_data:
                # Update the session with the new token and expiry time
                session['access_token'] = new_session_data.get('access_token')
                session['refresh_token'] = new_session_data.get('refresh_token')
                session['expires_at'] = new_session_data.get('expires_at')
                #("Successfully refreshed JWT session.")
            else:
                logging.warning("Failed to refresh JWT session. Logging user out.")
                session.clear() # Log out if refresh fails

@app.route('/auth/callback')
def auth_callback():
    return render_template(
        'callback.html',
        SUPABASE_URL=('https://glmtdjegibqaojifyxzf.supabase.co'),
        SUPABASE_ANON_KEY=('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdsbXRkamVnaWJxYW9qaWZ5eHpmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA2NjA2MDUsImV4cCI6MjA2NjIzNjYwNX0.AFqnq49ZBp-jiJ1GEHr4QDNoL0QGw3dPYFRu_2YvNVA')
    )

@app.route('/auth/set-cookie', methods=['POST'])
def set_auth_cookie():
    """
    This endpoint receives the full session from the client-side Supabase instance
    and sets the Flask session, including the refresh token and expiry time.
    """
    try:
        data = request.get_json()
        access_token = data.get('access_token')
        refresh_token = data.get('refresh_token')
        expires_at = data.get('expires_at')

        if not access_token:
            return jsonify({'status': 'error', 'message': 'Access token is missing.'}), 400

        supabase = get_supabase_client(access_token)
        user_response = supabase.auth.get_user(access_token)

        user = user_response.user
        if not user:
            return jsonify({'status': 'error', 'message': 'Invalid token.'}), 401

        # Store all necessary session info
        session['user'] = user.dict()
        session['access_token'] = access_token
        session['refresh_token'] = refresh_token
        session['expires_at'] = expires_at

        #(f"Successfully set session for user: {user.email}")
        return jsonify({'status': 'success', 'message': 'Session set successfully.'})

    except Exception as e:
        logging.error(f"Error in set-cookie endpoint: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal error occurred.'}), 500


#
# --- START: NEW SPA API ENDPOINTS ---
#


@app.route('/api/channel_details/<path:channel_name>')
@login_required
def get_channel_details(channel_name):
    """
    A fast API endpoint that returns only the channel metadata.
    It uses the cache and skips the slow chat history query.
    """
    all_user_channels = get_user_channels() # This is the fast, cached function
    current_channel = all_user_channels.get(channel_name)

    if not current_channel:
        return jsonify({'error': 'Channel not found or permission denied'}), 404

    # Return just the channel data immediately
    return jsonify({
        'current_channel': current_channel,
        'saved_channels': all_user_channels
    })

@app.route('/api/chat_history/<path:channel_name>')
@login_required
def get_chat_history_api(channel_name):
    """
    A dedicated API endpoint that returns only the chat history.
    This is the slower query that we will call second.
    """
    user_id = session['user']['id']
    access_token = session.get('access_token')

    history = get_chat_history(user_id, channel_name, access_token)

    return jsonify({'history': history})

#
# --- END: NEW SPA API ENDPOINTS ---
#

@app.route('/stream_answer', methods=['POST'])
@login_required
@subscription_required('query')
def stream_answer():
    user_id = session['user']['id']
    question = request.form.get('question', '').strip()
    channel_name = request.form.get('channel_name')
    tone = request.form.get('tone', 'Casual')
    access_token = session.get('access_token')

    MAX_CHAT_MESSAGES = 20
    current_channel_name_for_history = channel_name or 'general'
    history = get_chat_history(user_id, current_channel_name_for_history, access_token=access_token)

    if len(history) >= MAX_CHAT_MESSAGES:
        def limit_exceeded_stream():
            error_data = {
                "error": "QUERY_LIMIT_REACHED",
                "message": f"You have reached the chat limit of {MAX_CHAT_MESSAGES} messages. Please use the 'Clear Chat' button to start a new conversation."
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
        return Response(limit_exceeded_stream(), mimetype='text/event-stream')
    
    chat_history_for_prompt = ""
    for qa in history[-5:]: 
        chat_history_for_prompt += f"Human: {qa['question']}\nAI: {qa['answer']}\n\n"

    final_question_with_history = question
    if chat_history_for_prompt:
        final_question_with_history = (
            "Given the following conversation history:\n"
            f"{chat_history_for_prompt}"
            "--- End History ---\n\n"
            "Now, answer this new question, considering the history as context:\n"
            f"{question}"
        )

    search_query = question
    # if history:
    #     last_q = history[-1]['question']
    #     last_a = history[-1]['answer']
    #     search_query = f"In response to the question '{last_q}' and the answer '{last_a}', the user is now asking: {question}"
        
    channel_data = None
    video_ids = None

    if channel_name:
        all_user_channels = get_user_channels() 
        channel_data = all_user_channels.get(channel_name)
        if channel_data:
            video_ids = {v['video_id'] for v in channel_data.get('videos', [])}

    stream = answer_question_stream(
        question_for_prompt=final_question_with_history,
        question_for_search=search_query,
        channel_data=channel_data,
        video_ids=video_ids,
        user_id=user_id,
        access_token=access_token,
        tone=tone 
    )
    
    return Response(stream, mimetype='text/event-stream')

# @app.route('/')
# def index():
#     return redirect(url_for('channel'))

# @app.route('/landing')
# def landing():
#     return render_template('landing.html')

@app.route('/')
def home():
    """Renders the main landing page."""
    # If a user is already logged in, you might want to send them to their channels
    if 'user' in session:
        return redirect(url_for('channel'))
    
    # Otherwise, show the landing page
    return render_template('landing.html')

@app.route('/about')
def about():
    """Renders the about us page."""
    return render_template('about.html')

@app.route('/ask', methods=['GET'])
@login_required
def ask():
    user_id = session['user']['id']
    access_token = session.get('access_token')
    history = get_chat_history(user_id, 'general', access_token=access_token)
    return render_template('ask.html',
                           history=history,
                           saved_channels=get_user_channels())

@app.route('/ask/channel/<path:channel_name>')
@login_required
def ask_channel(channel_name):
    user = session.get('user')
    user_id = user['id']
    access_token = session.get('access_token')
    all_user_channels = get_user_channels()
    current_channel = all_user_channels.get(channel_name)
    if not current_channel:
        flash(f"You do not have access to the channel '{channel_name}' or it does not exist.", "error")
        return redirect(url_for('channel'))
    history = get_chat_history(user_id, channel_name, access_token)
    return render_template(
        'ask.html',
        user=user,
        history=history,
        channel_name=channel_name,
        current_channel=current_channel,
        saved_channels=all_user_channels,
        SUPABASE_URL=os.environ.get('SUPABASE_URL'),
        SUPABASE_ANON_KEY=os.environ.get('SUPABASE_ANON_KEY')
    )

@app.route('/delete_channel/<int:channel_id>', methods=['POST'])
@login_required
def delete_channel_route(channel_id):
    user_id = session['user']['id']
    supabase_admin = get_supabase_admin_client()
    try:
        ownership_check = supabase_admin.table('user_channels') \
            .select('channel_id') \
            .eq('user_id', user_id) \
            .eq('channel_id', channel_id) \
            .limit(1).single().execute()
        delete_channel_task(channel_id, user_id)
        return jsonify({
            'status': 'success',
            'message': 'Channel deletion has been started in the background.'
        })
    except APIError as e:
        if 'PGRST116' in e.message:
            return jsonify({'status': 'error', 'message': 'Channel not found or you do not have permission.'}), 404
        logging.error(f"API Error on channel deletion for {channel_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'A database error occurred.'}), 500
    except Exception as e:
        logging.error(f"Error initiating deletion for channel {channel_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An error occurred while starting the deletion process.'}), 500

@app.route('/channel', methods=['GET', 'POST'])
def channel():
    try:
        if request.method == 'POST':
            if 'user' not in session:
                return jsonify({'status': 'error', 'message': 'Authentication required.'}), 401
            user_id = session['user']['id']
            channel_url = request.form.get('channel_url', '').strip()
            if not channel_url:
                return jsonify({'status': 'error', 'message': 'Channel URL is required'}), 400
            cleaned_channel_url = clean_youtube_url(channel_url)
            supabase = get_supabase_client(session.get('access_token'))
            existing_channel_res = supabase.table('channels').select('id, status').eq('channel_url', cleaned_channel_url).limit(1).execute()
            if existing_channel_res.data:
                channel_id = existing_channel_res.data[0]['id']
                link_res = supabase.table('user_channels').select('channel_id').eq('user_id', user_id).eq('channel_id', channel_id).limit(1).execute()
                if not link_res.data:
                    supabase.table('user_channels').insert({'user_id': user_id, 'channel_id': channel_id}).execute()
                return jsonify({'status': 'success', 'message': 'Channel already exists and has been added.'})
            else:
                task = process_channel_task.schedule(
                    args=(cleaned_channel_url, user_id),
                    delay=1
                )
                return jsonify({'status': 'processing', 'task_id': task.id})
        return render_template(
            'channel.html',
            saved_channels=get_user_channels(),
            user=session.get('user'),
            SUPABASE_URL=os.environ.get('SUPABASE_URL'),
            SUPABASE_ANON_KEY=os.environ.get('SUPABASE_ANON_KEY')
        )
    except APIError as e:
        if 'JWT' in e.message and 'expired' in e.message:
            session.clear()
            return jsonify({'status': 'error', 'message': 'Your session has expired.'}), 401
        logging.error(f"API Error in /channel POST: {e}")
        return jsonify({'status': 'error', 'message': str(e.message)}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred in /channel: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500

@app.route('/task_result/<task_id>')
@login_required
def task_result(task_id):
    if redis_client:
        progress_data = redis_client.get(f"task_progress:{task_id}")
        if progress_data:
            return jsonify(json.loads(progress_data))
    try:
        result = huey.result(task_id, preserve=True)
        if result is None:
            return jsonify({'status': 'processing', 'progress': 5, 'message': 'Task is starting...'})
        else:
            return jsonify({'status': 'complete', 'progress': 100, 'message': result})
    except TaskException as e:
        logging.error(f"Task {task_id} failed: {e}")
        return jsonify({'status': 'failed', 'progress': 0, 'message': str(e)})

# ... (Keep all your other routes like /terms, /privacy, /logout, /clear_chat, etc. They are fine)
@app.route('/terms')
def terms():
    """Renders the terms and conditions page."""
    return render_template('terms.html',saved_channels=get_user_channels())

@app.route('/privacy')
def privacy():
    """Renders the terms and conditions page."""
    return render_template('privacy.html',saved_channels=get_user_channels())
    
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been successfully logged out.', 'success')
    return redirect(url_for('channel'))

@app.route('/clear_chat', methods=['POST'])
@login_required
def clear_chat():
    channel_name = request.form.get('channel_name') or 'general'
    user_id = session['user']['id']
    try:
        supabase = get_supabase_client(session.get('access_token'))
        supabase.table('chat_history').delete().eq('user_id', user_id).eq('channel_name', channel_name).execute()
        return jsonify({'status': 'success', 'message': f'Chat history cleared for {channel_name}'})
    except Exception as e:
        logging.error(f"Error clearing chat history: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/refresh_channel/<int:channel_id>', methods=['POST'])
@login_required
def refresh_channel_route(channel_id):
    user_id = session['user']['id']
    access_token = session.get('access_token')
    try:
        supabase = get_supabase_client(access_token)
        ownership_check = supabase.table('user_channels') \
            .select('channel_id') \
            .eq('user_id', user_id) \
            .eq('channel_id', channel_id) \
            .limit(1).single().execute()
        task = sync_channel_task(channel_id)
        return jsonify({'status': 'success', 'message': 'Channel refresh has been queued.', 'task_id': task.id})
    except APIError:
        return jsonify({'status': 'error', 'message': 'Channel not found or you do not have permission.'}), 404
    except Exception as e:
        logging.error(f"Error initiating refresh for channel {channel_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An error occurred while starting the refresh.'}), 500

@app.route('/telegram/connect', methods=['GET', 'POST'])
def connect_telegram():
    if 'user' not in session:
        return render_template('connect_telegram.html', user=None, saved_channels={})

    user_id = session['user']['id']
    supabase_admin = get_supabase_admin_client()
    existing_response = supabase_admin.table('telegram_connections').select('*').eq('app_user_id', user_id).limit(1).execute()
    if existing_response.data and existing_response.data[0]['is_active']:
        existing_data = existing_response.data[0]
        return render_template('connect_telegram.html',
                               connection_status='connected',
                               telegram_username=existing_data.get('telegram_username', 'N/A'),
                               saved_channels=get_user_channels(),
                               user=session.get('user'))

    if request.method == 'POST':
        connection_code = secrets.token_hex(8)
        data_to_store = {
            'app_user_id': user_id,
            'telegram_chat_id': 0, 
            'connection_code': connection_code,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'is_active': False
        }
        supabase_admin.table('telegram_connections').upsert(data_to_store, on_conflict='app_user_id').execute()
        token, _ = get_bot_token_and_url()
        bot_username = token.split(':')[0] if token else "YourBot"
        return render_template('connect_telegram.html',
                               connection_status='code_generated',
                               connection_code=connection_code,
                               bot_username=bot_username,
                               saved_channels=get_user_channels(),
                               user=session.get('user'))

    return render_template('connect_telegram.html',
                           connection_status='not_connected',
                           saved_channels=get_user_channels(),
                           user=session.get('user'))

@app.route('/channel/<int:channel_id>/disconnect_group', methods=['POST'])
@login_required
def disconnect_group(channel_id):
    user_id = session['user']['id']
    supabase = get_supabase_client(session.get('access_token'))
    supabase_admin = get_supabase_admin_client()
    link_check = supabase.table('user_channels').select('channels(channel_name)') \
        .eq('user_id', user_id).eq('channel_id', channel_id).single().execute()
    if not (link_check.data and link_check.data.get('channels')):
        flash("You do not have permission to modify this channel's connection.", "error")
        return redirect(url_for('channel'))
    channel_name = link_check.data['channels']['channel_name']
    try:
        supabase_admin.table('group_connections').delete().eq('linked_channel_id', channel_id).execute()
        flash("Telegram group successfully disconnected.", "success")
    except APIError as e:
        flash(f"An error occurred while disconnecting: {e.message}", "error")
    return redirect(url_for('ask_channel', channel_name=channel_name))

@app.route('/channel/<int:channel_id>/connect_group')
@login_required
def connect_group(channel_id):
    supabase_admin = get_supabase_admin_client()
    supabase = get_supabase_client(session.get('access_token'))
    user_id = session['user']['id']
    link_check = supabase.table('user_channels').select('channel_id').eq('user_id', user_id).eq('channel_id', channel_id).limit(1).execute()
    if not link_check.data:
        flash("You do not have permission to access this channel.", "error")
        return redirect(url_for('channel'))
    channel_resp = supabase_admin.table('channels').select('id, channel_name').eq('id', channel_id).single().execute()
    if not channel_resp.data:
        flash("Channel not found.", "error")
        return redirect(url_for('channel'))
    response = supabase_admin.table('group_connections').select('*').eq('linked_channel_id', channel_id).eq('is_active', True).limit(1).execute()
    if response.data:
        return render_template('connect_group.html',
                               connection_status='connected',
                               channel=channel_resp.data,
                               group_details=response.data[0],
                               saved_channels=get_user_channels())
    connection_code = secrets.token_hex(10)
    supabase_admin.table('group_connections').upsert({
        'owner_user_id': user_id,
        'linked_channel_id': channel_id,
        'connection_code': connection_code,
        'is_active': False
    }, on_conflict='linked_channel_id').execute()
    token, _ = get_bot_token_and_url()
    bot_username = token.split(':')[0] if token else "YourBot"
    return render_template('connect_group.html',
                           connection_status='code_generated',
                           channel=channel_resp.data,
                           connection_code=connection_code,
                           bot_username=bot_username,
                           saved_channels=get_user_channels())

@app.route('/telegram/webhook/<webhook_secret>', methods=['POST'])
def telegram_webhook(webhook_secret):
    config = load_config()
    token = config.get("telegram_bot_token")
    if not token:
        logging.error("Webhook received but TELEGRAM_BOT_TOKEN is not configured.")
        return "Configuration error", 500
    expected_secret = token.split(':')[-1][:10]
    header_secret = request.headers.get('X-Telegram-Bot-Api-Secret-Token')
    if not (secrets.compare_digest(webhook_secret, expected_secret) and header_secret and secrets.compare_digest(header_secret, expected_secret)):
        logging.warning("Unauthorized webhook access attempt.")
        return "Unauthorized", 403
    update = request.get_json()
    process_telegram_update_task(update)
    return jsonify({'status': 'ok'})

@app.template_filter('format_subscribers')
def format_subscribers_filter(value):
    try:
        num = int(value)
        if num < 1000:
            return str(num)
        if num < 1000000:
            k_value = f"{num / 1000:.1f}"
            return k_value.replace('.0', '') + "K"
        m_value = f"{num / 1000000:.1f}"
        return m_value.replace('.0', '') + "M"
    except (ValueError, TypeError):
        return ""
    
if __name__ == '__main__':

    app.run(debug=False, host='0.0.0.0', port=5000)