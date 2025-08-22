import logging
from functools import wraps
from utils.youtube_utils import is_youtube_video_url, clean_youtube_url
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, Response
import os
import json
import secrets
from datetime import datetime, timezone
from tasks import huey, process_channel_task, sync_channel_task, process_telegram_update_task, delete_channel_task
from utils.qa_utils import answer_question_stream
from utils.supabase_client import get_supabase_client, get_supabase_admin_client, refresh_supabase_session
from utils.history_utils import get_chat_history
from utils.telegram_utils import set_webhook, get_bot_token_and_url
from utils.config_utils import load_config
from utils.subscription_utils import get_user_status, limit_enforcer, community_channel_limit_enforcer, get_community_status, admin_channel_limit_enforcer
from utils import db_utils
import time
import redis
from postgrest.exceptions import APIError
from markupsafe import Markup
import markdown
from huey.exceptions import TaskException
from dotenv import load_dotenv
from flask_compress import Compress
import jwt
from utils import whop_api
from utils import db_utils
import hmac
import hashlib
import base64
from utils.subscription_utils import PLANS, COMMUNITY_PLANS

logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

Compress(app)
app.secret_key = os.environ.get('SECRET_KEY', 'a_default_dev_secret_key')


@app.template_filter('markdown')
def markdown_filter(text):
    return Markup(markdown.markdown(text))

try:
    redis_client = redis.from_url(os.environ.get('REDIS_URL'))
except Exception:
    redis_client = None

@app.context_processor
def inject_user_status():
    if 'user' in session:
        user_id = session['user']['id']
        active_community_id = session.get('active_community_id')
        user_status = get_user_status(user_id, active_community_id)
        is_embedded = session.get('is_embedded_whop_user', False)

        community_status = None
        if active_community_id:
            community_status = get_community_status(active_community_id)

        return dict(
            user_status=user_status,
            user=session.get('user'),
            is_embedded_whop_user=is_embedded,
            community_status=community_status
        )
    return dict(user_status=None, user=None, is_embedded_whop_user=False, community_status=None)

def get_user_channels():
    """
    Return all channels visible to the logged-in user.
    This includes their personal channels and any shared channels from their active community.
    """
    if 'user' not in session:
        return {}

    user_id = session['user']['id']
    active_community_id = session.get('active_community_id')

    # The cache key must now be context-aware
    cache_key = f"user_visible_channels:{user_id}:community:{active_community_id or 'none'}"

    if redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logging.error(f"Redis GET error: {e}")

    # Use the admin client for robust channel fetching
    supabase = get_supabase_admin_client()
    all_channels = {}

    try:
        # 1. Fetch user's personal channels
        personal_channels_resp = supabase.table('user_channels').select('channels(*)').eq('user_id', user_id).execute()
        if personal_channels_resp.data:
            for item in personal_channels_resp.data:
                channel = item.get('channels')
                if channel and channel.get('channel_name'):
                    all_channels[channel['channel_name']] = channel

        # 2. If in a community, fetch shared channels
        if active_community_id:
            shared_channels_resp = supabase.table('channels').select('*').eq('is_shared', True).eq('community_id', active_community_id).execute()
            if shared_channels_resp.data:
                for channel in shared_channels_resp.data:
                    if channel and channel.get('channel_name'):
                        # This will merge shared channels and overwrite any personal channel with the same name
                        all_channels[channel['channel_name']] = channel

    except APIError as e:
        logging.error(f"Supabase error in get_user_channels: {e.message}")
        if 'JWT expired' in e.message:
            session.clear()
    except Exception as e:
        logging.error(f"Unexpected error in get_user_channels: {e}")

    if redis_client and all_channels:
        try:
            redis_client.setex(cache_key, 15, json.dumps(all_channels))
        except Exception as e:
            logging.error(f"Redis SET error: {e}")

    return all_channels

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return jsonify({'status': 'error', 'message': 'Authentication required.'}), 401
        try:
            return f(*args, **kwargs)
        except APIError as e:
            if 'JWT' in e.message and 'expired' in e.message:
                session.clear()
                return jsonify({'status': 'error', 'message': 'Session expired.', 'action': 'logout'}), 401
            raise e
    return decorated_function





@app.route('/whop/')
def whop_landing():
    """Serves the landing page for Whop users."""
    # This route now simply renders the page. The button on the page
    # will point to the Node.js service to start the actual login.
    return render_template('landing_whop.html')

@app.route('/auth/whop/installation-callback')
def whop_installation_callback():
    token = request.args.get('token')
    jwt_secret = os.environ.get('JWT_SECRET')
    if not token or not jwt_secret:
        flash('Installation failed: Invalid token.', 'error')
        return redirect(url_for('home'))

    try:
        decoded_payload = jwt.decode(token, jwt_secret, algorithms=['HS256'])
        owner_email = decoded_payload['owner_email']
        whop_community_id = decoded_payload['whop_community_id']

        supabase_admin = get_supabase_admin_client()
        list_of_users = supabase_admin.auth.admin.list_users()
        auth_user = next((u for u in list_of_users if u.email == owner_email), None)

        if not auth_user:
            new_user_res = supabase_admin.auth.admin.create_user({'email': owner_email, 'email_confirm': True, 'password': secrets.token_urlsafe(16)})
            auth_user = new_user_res.user

        app_user_id = str(auth_user.id)

        # Create the community first to get its ID
        community_data = {
            'whop_community_id': whop_community_id,
            'owner_user_id': app_user_id
        }
        community = db_utils.add_community(community_data)
        if not community:
            flash('Failed to create community record.', 'error')
            return redirect(url_for('home'))

        # Now link the owner's profile to the community
        profile_data = {
            'id': app_user_id,
            'whop_user_id': decoded_payload.get('owner_user_id'),
            'full_name': decoded_payload.get('owner_full_name'),
            'avatar_url': decoded_payload.get('owner_avatar_url'),
            'email': owner_email,
            'is_community_owner': True,
            'community_id': community['id']
        }
        db_utils.create_or_update_profile(profile_data)
        db_utils.create_initial_usage_stats(app_user_id)

        # Set the session to log the owner in
        session['user'] = auth_user.model_dump()
        flash('Your community has been successfully installed! You have 10 free queries to test out the bot.', 'success')
        return redirect(url_for('channel'))

    except Exception as e:
        logging.error(f"An error occurred during Whop installation callback: {e}", exc_info=True)
        flash('An unexpected error occurred during installation.', 'error')
        return redirect(url_for('home'))


# --- Whop Webhooks ---

def validate_whop_webhook(f):
    """
    Decorator to validate that a webhook request is genuinely from Whop.
    This uses a HMAC-SHA256 signature.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Assumes the signature is in a header like 'X-Whop-Signature-SHA256'
        signature_header = request.headers.get('X-Whop-Signature-SHA256')
        if not signature_header:
            logging.warning("Webhook received without signature header.")
            return jsonify({'status': 'error', 'message': 'Missing signature'}), 401

        secret = os.environ.get('WHOP_WEBHOOK_SECRET')
        if not secret:
            logging.error("WHOP_WEBHOOK_SECRET is not configured. Cannot validate webhook.")
            return jsonify({'status': 'error', 'message': 'Server configuration error'}), 500

        request_body = request.get_data()

        try:
            digest = hmac.new(secret.encode('utf-8'), request_body, digestmod=hashlib.sha256).digest()
            computed_hmac = base64.b64encode(digest)
        except Exception:
            return jsonify({'status': 'error', 'message': 'Internal server error during validation'}), 500

        if not hmac.compare_digest(computed_hmac, signature_header.encode('utf-8')):
            logging.warning("Invalid webhook signature.")
            return jsonify({'status': 'error', 'message': 'Invalid signature'}), 403

        return f(*args, **kwargs)
    return decorated_function


@app.route('/whop/webhook/membership-update', methods=['POST'])
@validate_whop_webhook
def whop_membership_update_webhook():
    """
    Handles membership update webhooks from Whop to manage personal plans.
    Example payload: { "data": { "whop_user_id": "...", "new_plan_id": "..." } }
    """
    payload = request.get_json()
    logging.info(f"Received Whop membership webhook: {payload}")

    data = payload.get('data', {})
    whop_user_id = data.get('whop_user_id')
    new_plan_id = data.get('new_plan_id')

    if not whop_user_id or not new_plan_id:
        logging.warning(f"Webhook payload missing whop_user_id or new_plan_id: {data}")
        return jsonify({'status': 'error', 'message': 'Missing required fields in payload'}), 400

    if new_plan_id not in PLANS:
        logging.warning(f"Webhook received with unrecognized plan_id: {new_plan_id}")
        return jsonify({'status': 'error', 'message': 'Unrecognized plan_id'}), 400

    try:
        supabase_admin = get_supabase_admin_client()
        profile_res = supabase_admin.table('profiles').select('id').eq('whop_user_id', whop_user_id).maybe_single().execute()

        if not profile_res.data:
            logging.warning(f"Webhook for non-existent user with whop_user_id: {whop_user_id}")
            return jsonify({'status': 'not_found', 'message': 'User not found'}), 200

        app_user_id = profile_res.data['id']
        supabase_admin.table('profiles').update({'personal_plan_id': new_plan_id}).eq('id', app_user_id).execute()
        logging.info(f"Updated user {app_user_id} to plan: {new_plan_id}")

        if redis_client:
            user_communities_res = supabase_admin.table('user_communities').select('community_id').eq('user_id', app_user_id).execute()
            if user_communities_res.data:
                for item in user_communities_res.data:
                    redis_client.delete(f"user_status:{app_user_id}:community:{item['community_id']}")
            redis_client.delete(f"user_status:{app_user_id}:community:none")
            logging.info(f"Invalidated Redis cache for user {app_user_id}")

        return jsonify({'status': 'success'})

    except Exception as e:
        logging.error(f"Error processing membership webhook for {whop_user_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500


@app.route('/whop/webhook/community-update', methods=['POST'])
@validate_whop_webhook
def whop_community_update_webhook():
    """
    Handles community update webhooks from Whop, specifically for member count changes.
    This is used to dynamically adjust the community's query limit.
    Example payload: { "data": { "whop_community_id": "...", "member_count": 138 } }
    """
    payload = request.get_json()
    logging.info(f"Received Whop community webhook: {payload}")

    data = payload.get('data', {})
    whop_community_id = data.get('whop_community_id')
    member_count = data.get('member_count')

    if not whop_community_id or member_count is None:
        logging.warning(f"Webhook payload missing whop_community_id or member_count: {data}")
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    try:
        # The user flow specifies a baseline of 100 queries per member.
        new_query_limit = int(member_count) * 100

        supabase_admin = get_supabase_admin_client()

        # Find the community by its Whop ID and update the query limit
        update_res = supabase_admin.table('communities').update({'query_limit': new_query_limit}).eq('whop_community_id', whop_community_id).execute()

        if not update_res.data:
             logging.warning(f"Webhook received for non-existent community with whop_community_id: {whop_community_id}")
             return jsonify({'status': 'not_found', 'message': 'Community not found'}), 200

        logging.info(f"Successfully updated query limit for community {whop_community_id} to {new_query_limit} based on {member_count} members.")

        return jsonify({'status': 'success'})

    except (ValueError, TypeError):
        logging.warning(f"Invalid member_count received: {member_count}")
        return jsonify({'status': 'error', 'message': 'Invalid member_count value'}), 400
    except Exception as e:
        logging.error(f"Error processing community webhook for {whop_community_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500


@app.route('/whop/webhook/community-plan-update', methods=['POST'])
@validate_whop_webhook
def whop_community_plan_update_webhook():
    """
    Handles community plan update webhooks from Whop.
    Example payload: { "data": { "whop_community_id": "...", "new_plan_id": "pro_community" } }
    """
    payload = request.get_json()
    logging.info(f"Received Whop community plan update webhook: {payload}")

    data = payload.get('data', {})
    whop_community_id = data.get('whop_community_id')
    new_plan_id = data.get('new_plan_id')

    if not whop_community_id or not new_plan_id:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    new_plan_details = COMMUNITY_PLANS.get(new_plan_id)
    if not new_plan_details:
        return jsonify({'status': 'error', 'message': f'Invalid plan_id: {new_plan_id}'}), 400

    try:
        supabase_admin = get_supabase_admin_client()
        community_update = {
            'plan_id': new_plan_id,
            'shared_channel_limit': new_plan_details['shared_channels_allowed']
        }
        update_res = supabase_admin.table('communities').update(community_update).eq('whop_community_id', whop_community_id).execute()

        if not update_res.data:
             logging.warning(f"Webhook received for non-existent community with whop_community_id: {whop_community_id}")
             return jsonify({'status': 'not_found', 'message': 'Community not found'}), 200

        logging.info(f"Successfully updated plan for community {whop_community_id} to {new_plan_id}.")

        # Invalidate the Redis cache for this community
        if redis_client:
            redis_client.delete(f"community_status:{update_res.data[0]['id']}")

        return jsonify({'status': 'success'})

    except Exception as e:
        logging.error(f"Error processing community plan update for {whop_community_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500


@app.before_request
def check_token_expiry():
    # Only check for expiry if a user session and all tokens exist
    if 'user' in session and session.get('expires_at') and session.get('refresh_token'):
        # Check if the token expires in the next 60 seconds
        if session['expires_at'] < (time.time() + 60):
            print("[INFO] Access token is expiring, attempting to refresh...")
            new_session = refresh_supabase_session(session.get('refresh_token'))
            if new_session:
                # Update the session with new token details
                session['access_token'] = new_session.get('access_token')
                session['refresh_token'] = new_session.get('refresh_token')
                session['expires_at'] = new_session.get('expires_at')
                print("[SUCCESS] Session refreshed successfully.")
            else:
                # If refresh fails, clear the session to force a re-login
                print("[WARNING] Failed to refresh session. Clearing session.")
                session.clear()

@app.route('/auth/callback')
def auth_callback():
    return render_template('callback.html', SUPABASE_URL=os.environ.get('SUPABASE_URL'), SUPABASE_ANON_KEY=os.environ.get('SUPABASE_ANON_KEY'))

# --- Auth: set session & ensure profile exists ---
@app.route('/auth/set-cookie', methods=['POST'])
def set_auth_cookie():
    try:
        data = request.get_json()
        access_token = data.get('access_token')
        refresh_token = data.get('refresh_token')
        expires_at = data.get('expires_at')

        if not all([access_token, refresh_token, expires_at]):
            return jsonify({'status': 'error', 'message': 'Incomplete session data provided.'}), 400

        # --- START OF FIX ---
        # 1. Initialize an anonymous Supabase client first.
        supabase = get_supabase_client()
        if not supabase:
            logging.error("Failed to initialize Supabase client. Check environment variables.")
            return jsonify({'status': 'error', 'message': 'Server configuration error.'}), 500

        # 2. Explicitly set the user's session on the client instance.
        supabase.auth.set_session(access_token, refresh_token)

        # 3. Now, get the user. This will be an authenticated call.
        user_response = supabase.auth.get_user()
        # --- END OF FIX ---

        if not user_response or not hasattr(user_response, 'user') or not user_response.user:
            logging.error(f"Failed to get user from Supabase with provided token.")
            return jsonify({'status': 'error', 'message': 'Invalid authentication token.'}), 401

        user = user_response.user

        # Ensure profile + usage stats exist
        profile = db_utils.get_profile(user.id)
        if not profile:
            db_utils.create_or_update_profile({
                'id': user.id,
                'email': user.email,
                'full_name': user.user_metadata.get('full_name'),
                'avatar_url': user.user_metadata.get('avatar_url')
            })
            db_utils.create_initial_usage_stats(user.id)

        # Save all details to the session
        session['user'] = user.model_dump()
        session['access_token'] = access_token
        session['refresh_token'] = refresh_token
        session['expires_at'] = expires_at

        return jsonify({'status': 'success', 'message': 'Session set successfully.'})
    except Exception as e:
        logging.error(f"Error in set-cookie: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal error occurred.'}), 500


# In yoppychat2/app.py

# In yoppychat2/app.py

@app.route('/whop/app')
def whop_app_entry():
    """
    Handles Whop user login, creates user and community records if they don't exist,
    and redirects to the appropriate page.
    """
    print("\n--- Whop App Entry Triggered ---")
    user_token = whop_api.get_embedded_user_token(request)
    if not user_token:
        flash("Authentication failed. Please access the app through your Whop community.", "error")
        return redirect(url_for('home'))

    whop_user = whop_api.get_user_from_token(user_token)
    whop_company = whop_api.get_current_company()

    if not whop_user or not whop_company:
        flash("Failed to verify user or company with Whop.", "error")
        return redirect(url_for('home'))

    try:
        supabase_admin = get_supabase_admin_client()
        email = whop_user.get('email')
        whop_user_id = whop_user.get('id')

        print(f"[INFO] Whop User Detected: {email} (Whop ID: {whop_user_id})")

        # --- THIS IS THE FIX ---
        # The list_users() call returns a list of users directly. We iterate over this list.
        list_of_users = supabase_admin.auth.admin.list_users()
        auth_user = next((u for u in list_of_users if u.email == email), None)
        # --- END FIX ---

        if not auth_user:
            print(f"[INFO] No existing user found for {email}. Creating new user...")
            auth_user = supabase_admin.auth.admin.create_user({
                'email': email, 'email_confirm': True, 'password': secrets.token_urlsafe(16)
            }).user
            print(f"[SUCCESS] Created new user with App ID: {auth_user.id}")

        app_user_id = str(auth_user.id)
        user_role = whop_api.get_user_role_in_company(whop_user_id)

        print(f"[INFO] User Role Determined: {user_role}")

        profile_data = {
            'id': app_user_id,
            'whop_user_id': whop_user_id,
            'full_name': whop_user.get('name'),
            'avatar_url': whop_user.get('profile_pic_url'),
            'email': email,
            'is_community_owner': user_role == 'admin'
        }

        profile = db_utils.create_or_update_profile(profile_data)
        if not profile:
            raise Exception(f"Failed to create or find profile for user {app_user_id}")

        print(f"[SUCCESS] Profile created/updated for {email}")

        db_utils.create_initial_usage_stats(app_user_id)

        whop_community_id = whop_company['id']
        community_res = supabase_admin.table('communities').select('id').eq('whop_community_id', whop_community_id).maybe_single().execute()

        community_id = None
        if community_res and community_res.data:
            community_id = community_res.data['id']
            print(f"[INFO] Found existing community in DB with ID: {community_id}")
        elif user_role == 'admin':
            print(f"[INFO] Community not found. Creating for admin user {app_user_id}...")
            new_community = db_utils.add_community({
                'whop_community_id': whop_community_id,
                'owner_user_id': app_user_id
            })
            if new_community:
                community_id = new_community['id']
                print(f"[SUCCESS] Created new community with ID: {community_id}")

        if not community_id:
            flash("This community has not been set up yet. An admin must log in first.", "error")
            return redirect(url_for('home'))

        session['user'] = auth_user.model_dump()
        session['active_community_id'] = community_id
        session['is_embedded_whop_user'] = True

        db_utils.link_user_to_community(app_user_id, community_id)

        print(f"--- Login successful for {email}. Redirecting to dashboard. ---")
        flash(f"Welcome from {whop_company.get('title', 'your community')}!", 'success')

        # For all users (owners and members), try to find the community's default channel and redirect there.
        community_res = supabase_admin.table('communities').select('default_channel_id').eq('id', community_id).single().execute()
        if community_res.data and community_res.data.get('default_channel_id'):
            channel_id = community_res.data['default_channel_id']
            channel_res = supabase_admin.table('channels').select('channel_name').eq('id', channel_id).single().execute()
            if channel_res.data and channel_res.data.get('channel_name'):
                # If a default channel exists, everyone goes to the ask page for it.
                return redirect(url_for('ask', channel_name=channel_res.data['channel_name']))

        # Fallback if no default channel is set.
        # For owners, they can set one up. For members, they must wait.
        if user_role == 'member':
            flash("Your community owner has not set a default channel yet. You will be able to chat once one is configured.", "info")
        else: # For admins/owners
            flash("Welcome! Please add a channel and set it as the default for your community members.", "info")

        return redirect(url_for('channel'))

    except Exception as e:
        logging.error(f"An error occurred during Whop login: {e}", exc_info=True)
        flash("An unexpected error occurred during login.", "error")
        return redirect(url_for('home'))

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('channel'))
    return render_template('landing.html')

@app.route('/channel', methods=['GET', 'POST'])
def channel():
    try:
        if request.method == 'POST':
            # This route is now for adding PERSONAL channels, with different logic for admins vs. regular users.
            if 'user' not in session:
                return jsonify({'status': 'error', 'message': 'Authentication required.'}), 401

            user_id = session['user']['id']
            active_community_id = session.get('active_community_id')
            user_status = get_user_status(user_id, active_community_id)

            if not user_status:
                return jsonify({'status': 'error', 'message': 'Could not verify user status.'}), 500

            # --- Dynamically apply the correct limit logic ---
            if user_status.get('is_active_community_owner'):
                # Admin logic: check against unified channel limit
                community_status = get_community_status(active_community_id)
                if not community_status:
                    return jsonify({'status': 'error', 'message': 'Could not verify community status.'}), 500

                current_total_channels = db_utils.count_channels_for_user(user_id)
                max_total_channels = community_status['limits'].get('shared_channel_limit', 0)

                if current_total_channels >= max_total_channels:
                    return jsonify({
                        'status': 'limit_reached',
                        'message': f"As a community admin, your total channel limit is {max_total_channels}. You have reached this limit."
                    }), 403
            else:
                # Regular user logic: check against personal channel limit
                max_channels = user_status['limits'].get('max_channels', 0)
                current_channels = user_status['usage'].get('channels_processed', 0)
                if max_channels != float('inf') and current_channels >= max_channels:
                    message = f"You have reached the maximum of {int(max_channels)} personal channels for your plan."
                    if user_status.get('is_whop_user'):
                        return jsonify({'status': 'limit_reached', 'message': message, 'action': 'show_upgrade_popup'}), 403
                    else:
                        return jsonify({'status': 'limit_reached', 'message': message}), 403

            # If limits are not reached, proceed with adding the channel
            def guarded_personal_channel_add():
                user_id = session['user']['id'] # re-get user_id just in case
                channel_url = request.form.get('channel_url', '').strip()
                if not channel_url:
                    return jsonify({'status': 'error', 'message': 'Channel URL is required'}), 400

                cleaned_url = clean_youtube_url(channel_url)
                existing = db_utils.find_channel_by_url(cleaned_url)

                if existing:
                    db_utils.link_user_to_channel(user_id, existing['id'])
                    if redis_client: redis_client.delete(f"user_channels:{user_id}")
                    return jsonify({'status': 'success', 'message': 'Channel added to your list.'})
                else:
                    # Personal channels are NOT shared and not linked to a community
                    new_channel = db_utils.create_channel(cleaned_url, user_id, is_shared=False, community_id=None)
                    if not new_channel:
                        return jsonify({'status': 'error', 'message': 'Could not create channel record.'}), 500

                    db_utils.link_user_to_channel(user_id, new_channel['id'])
                    task = process_channel_task.schedule(args=(new_channel['id'],), delay=1)
                    if redis_client: redis_client.delete(f"user_channels:{user_id}")
                    return jsonify({'status': 'processing', 'task_id': task.id})
            return guarded_personal_channel_add()

        return render_template(
            'channel.html',
            saved_channels=get_user_channels(),
            SUPABASE_URL=os.environ.get('SUPABASE_URL'),
            SUPABASE_ANON_KEY=os.environ.get('SUPABASE_ANON_KEY')
        )
    except Exception as e:
        logging.error(f"Error in /channel: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500

@app.route('/add_shared_channel', methods=['POST'])
@login_required
@admin_channel_limit_enforcer
def add_shared_channel():
    user_id = session['user']['id']
    community_id = session.get('active_community_id')
    channel_url = request.form.get('channel_url', '').strip()
    if not channel_url:
        return jsonify({'status': 'error', 'message': 'Channel URL is required'}), 400

    cleaned_url = clean_youtube_url(channel_url)
    existing = db_utils.find_channel_by_url(cleaned_url)

    if existing:
        pass

    new_channel = db_utils.create_channel(cleaned_url, user_id, is_shared=True, community_id=community_id)
    if not new_channel:
        return jsonify({'status': 'error', 'message': 'Could not create shared channel record.'}), 500

    task = process_channel_task.schedule(args=(new_channel['id'],), delay=1)
    return jsonify({'status': 'processing', 'task_id': task.id, 'message': 'Processing shared channel...'})

@app.route('/set-default-channel/<int:channel_id>', methods=['POST'])
@login_required
def set_default_channel(channel_id):
    user_id = session['user']['id']
    active_community_id = session.get('active_community_id')

    if not active_community_id:
        return jsonify({'status': 'error', 'message': 'No active community context.'}), 400

    supabase_admin = get_supabase_admin_client()

    # 1. Verify the current user is the owner of the active community
    community_resp = supabase_admin.table('communities').select('owner_user_id').eq('id', active_community_id).single().execute()
    if not community_resp.data or str(community_resp.data['owner_user_id']) != str(user_id):
        return jsonify({'status': 'error', 'message': 'You are not the owner of this community.'}), 403

    # 2. Verify the channel being set as default is a shared channel within that community
    channel_resp = supabase_admin.table('channels').select('id').eq('id', channel_id).eq('is_shared', True).eq('community_id', active_community_id).single().execute()
    if not channel_resp.data:
        return jsonify({'status': 'error', 'message': 'This channel is not a shared channel in your community.'}), 403

    # 3. Update the community's default_channel_id
    supabase_admin.table('communities').update({'default_channel_id': channel_id}).eq('id', active_community_id).execute()

    return jsonify({'status': 'success', 'message': 'Default channel has been updated.'})

# ---- Ask (with optional channel) ----
@app.route('/ask', defaults={'channel_name': None})
@app.route('/ask/channel/<path:channel_name>')
@login_required
def ask(channel_name):
    user_id = session['user']['id']
    access_token = session.get('access_token')
    all_user_channels = get_user_channels()
    if channel_name:
        current_channel = all_user_channels.get(channel_name)
        if not current_channel:
            flash(f"Channel '{channel_name}' not found.", 'error')
            return redirect(url_for('ask'))
        history = get_chat_history(user_id, channel_name, access_token)
    else:
        current_channel = None
        history = get_chat_history(user_id, 'general', access_token)
    return render_template(
        'ask.html',
        history=history,
        channel_name=channel_name,
        current_channel=current_channel,
        saved_channels=all_user_channels
    )


# ---- Lightweight SPA APIs ----
@app.route('/api/channel_details/<path:channel_name>')
@login_required
def get_channel_details(channel_name):
    all_user_channels = get_user_channels()
    current_channel = all_user_channels.get(channel_name)
    if not current_channel:
        return jsonify({'error': 'Channel not found or permission denied'}), 404
    return jsonify({'current_channel': current_channel, 'saved_channels': all_user_channels})

@app.route('/api/chat_history/<path:channel_name>')
@login_required
def get_chat_history_api(channel_name):
    user_id = session['user']['id']
    access_token = session.get('access_token')
    history = get_chat_history(user_id, channel_name, access_token)
    return jsonify({'history': history})

# ---- Answer streaming ----
@app.route('/stream_answer', methods=['POST'])
@login_required
@limit_enforcer('query')
def stream_answer():
    user_id = session['user']['id']
    question = request.form.get('question', '').strip()
    channel_name = request.form.get('channel_name')
    tone = request.form.get('tone', 'Casual')
    access_token = session.get('access_token')
    active_community_id = session.get('active_community_id')

    # Determine if this is a trial query. This logic is duplicated from the
    # decorator because the decorator runs *before* the route and cannot easily
    # pass this information down to the on_complete_callback.
    is_owner_in_trial = False
    if active_community_id:
        user_status = get_user_status(user_id, active_community_id)
        if user_status.get('is_active_community_owner'):
            community_status = get_community_status(active_community_id)
            if community_status and community_status['usage']['trial_queries_used'] < community_status['limits']['owner_trial_limit']:
                is_owner_in_trial = True

    # After a successful query, we need to increment the usage counter.
    # This wrapper will be called by the streaming generator upon completion.
    def on_complete_callback():
        # We only increment usage for Whop users (who have an active community).
        if active_community_id:
            db_utils.increment_community_query_usage(active_community_id, is_trial=is_owner_in_trial)

    MAX_CHAT_MESSAGES = 20
    current_channel_name_for_history = channel_name or 'general'
    history = get_chat_history(user_id, current_channel_name_for_history, access_token=access_token)

    if len(history) >= MAX_CHAT_MESSAGES:
        def limit_exceeded_stream():
            error_data = {
                'error': 'QUERY_LIMIT_REACHED',
                'message': f"You have reached the chat limit of {MAX_CHAT_MESSAGES} messages. Please use the 'Clear Chat' button to start a new conversation."
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
        return Response(limit_exceeded_stream(), mimetype='text/event-stream')

    chat_history_for_prompt = ''
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

    channel_data = None
    video_ids = None
    if channel_name:
        all_user_channels = get_user_channels()
        channel_data = all_user_channels.get(channel_name)
        if channel_data:
            video_ids = {v['video_id'] for v in channel_data.get('videos', [])}

    stream = answer_question_stream(
        question_for_prompt=final_question_with_history,
        question_for_search=question,
        channel_data=channel_data,
        video_ids=video_ids,
        user_id=user_id,
        access_token=access_token,
        tone=tone,
        on_complete=on_complete_callback
    )
    return Response(stream, mimetype='text/event-stream')

# ---- Channel admin: delete & refresh ----
@app.route('/delete_channel/<int:channel_id>', methods=['POST'])
@login_required
def delete_channel_route(channel_id):
    user_id = session['user']['id']
    supabase_admin = get_supabase_admin_client()
    try:
        # Ensure user owns/has link to this channel
        supabase_admin.table('user_channels').select('channel_id') \
            .eq('user_id', user_id).eq('channel_id', channel_id) \
            .limit(1).single().execute()
        # Kick off background deletion
        delete_channel_task(channel_id, user_id)
        return jsonify({'status': 'success', 'message': 'Channel deletion has been started in the background.'})
    except APIError as e:
        if 'PGRST116' in e.message:
            return jsonify({'status': 'error', 'message': 'Channel not found or you do not have permission.'}), 404
        logging.error(f"API Error on channel deletion for {channel_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'A database error occurred.'}), 500
    except Exception as e:
        logging.error(f"Error initiating deletion for channel {channel_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An error occurred while starting the deletion process.'}), 500

@app.route('/refresh_channel/<int:channel_id>', methods=['POST'])
@login_required
def refresh_channel_route(channel_id):
    user_id = session['user']['id']
    access_token = session.get('access_token')
    try:
        supabase = get_supabase_client(access_token)
        supabase.table('user_channels') \
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

# ---- Telegram webhook ----
@app.route('/telegram/webhook/<webhook_secret>', methods=['POST'])
def telegram_webhook(webhook_secret):
    config = load_config()
    token = config.get('telegram_bot_token')
    if not token:
        logging.error('Webhook received but TELEGRAM_BOT_TOKEN is not configured.')
        return 'Configuration error', 500
    expected_secret = token.split(':')[-1][:10]
    header_secret = request.headers.get('X-Telegram-Bot-Api-Secret-Token')
    if not (secrets.compare_digest(webhook_secret, expected_secret) and header_secret and secrets.compare_digest(header_secret, expected_secret)):
        logging.warning('Unauthorized webhook access attempt.')
        return 'Unauthorized', 403
    update = request.get_json()
    process_telegram_update_task(update)
    return jsonify({'status': 'ok'})

# ---- Subscriber count formatting filter ----
@app.template_filter('format_subscribers')
def format_subscribers_filter(value):
    try:
        num = int(value)
        if num < 1000:
            return str(num)
        if num < 1_000_000:
            k_value = f"{num / 1000:.1f}"
            return k_value.replace('.0', '') + 'K'
        m_value = f"{num / 1_000_000:.1f}"
        return m_value.replace('.0', '') + 'M'
    except (ValueError, TypeError):
        return ''

# ---- Task progress ----
@app.route('/task_result/<task_id>')
@login_required
def task_result(task_id):
    if redis_client:
        progress_data = redis_client.get(f"task_progress:{task_id}")
        if progress_data:
            return jsonify(json.loads(progress_data))
    try:
        result = huey.result(task_id, preserve=True)
        if result is not None:
            return jsonify({'status': 'complete', 'progress': 100, 'message': str(result)})
    except TaskException as e:
        logging.error(f"Task {task_id} failed: {e}")
        return jsonify({'status': 'failed', 'progress': 0, 'message': str(e)})
    return jsonify({'status': 'processing', 'progress': 5, 'message': 'Task is starting...'})

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
# ---- Misc ----
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/integrations', methods=['GET', 'POST'])
def integrations():
    if 'user' not in session:
        return render_template('integrations.html', user=None)

    user_id = session['user']['id']
    supabase_admin = get_supabase_admin_client()

    # POST request handling
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'connect_personal':
            connection_code = secrets.token_hex(8)
            data_to_store = {
                'app_user_id': user_id,
                'telegram_chat_id': 0,
                'connection_code': connection_code,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'is_active': False
            }
            supabase_admin.table('telegram_connections').upsert(data_to_store, on_conflict='app_user_id').execute()
        elif action == 'disconnect_personal':
            supabase_admin.table('telegram_connections').delete().eq('app_user_id', user_id).execute()
        elif action == 'disconnect_group':
            channel_id = request.form.get('channel_id')
            supabase_admin.table('group_connections').delete().eq('linked_channel_id', channel_id).execute()
            flash('Telegram group successfully disconnected.', 'success')

        return redirect(url_for('integrations', channel_id=request.form.get('channel_id')))

    # GET request handling
    # Personal connection
    personal_connection_status = 'not_connected'
    telegram_username = None
    personal_connection_code = None
    existing_response = supabase_admin.table('telegram_connections').select('*').eq('app_user_id', user_id).limit(1).execute()
    if existing_response.data:
        existing_data = existing_response.data[0]
        if existing_data.get('is_active'):
            personal_connection_status = 'connected'
            telegram_username = existing_data.get('telegram_username', 'N/A')
        else:
            personal_connection_status = 'code_generated'
            personal_connection_code = existing_data.get('connection_code')

    # Group connection
    group_connection_status = 'not_connected'
    group_details = None
    group_connection_code = None
    selected_channel_id = request.args.get('channel_id', type=int)
    if selected_channel_id:
        response = supabase_admin.table('group_connections').select('*').eq('linked_channel_id', selected_channel_id).eq('is_active', True).limit(1).execute()
        if response.data:
            group_connection_status = 'connected'
            group_details = response.data[0]
        else:
            group_connection_status = 'code_generated'
            connection_code = secrets.token_hex(10)
            supabase_admin.table('group_connections').upsert({
                'owner_user_id': user_id,
                'linked_channel_id': selected_channel_id,
                'connection_code': connection_code,
                'is_active': False
            }, on_conflict='linked_channel_id').execute()
            group_connection_code = connection_code

    token, _ = get_bot_token_and_url()
    bot_username = token.split(':')[0] if token else 'YourBot'

    return render_template(
        'integrations.html',
        user=session.get('user'),
        saved_channels=get_user_channels(),
        personal_connection_status=personal_connection_status,
        telegram_username=telegram_username,
        personal_connection_code=personal_connection_code,
        group_connection_status=group_connection_status,
        group_details=group_details,
        group_connection_code=group_connection_code,
        selected_channel_id=selected_channel_id,
        bot_username=bot_username
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html', saved_channels=get_user_channels())

@app.route('/terms')
def terms():
    return render_template('terms.html', saved_channels=get_user_channels())

@app.route('/api/toggle_channel_privacy/<int:channel_id>', methods=['POST'])
@login_required
def toggle_channel_privacy(channel_id):
    """
    Allows a community owner to toggle a channel between personal and shared.
    """
    user_id = session['user']['id']
    supabase_admin = get_supabase_admin_client()

    try:
        # 1. Fetch the channel to get its community_id
        channel_res = supabase_admin.table('channels').select('community_id, is_shared, user_id').eq('id', channel_id).single().execute()
        if not channel_res.data:
            return jsonify({'status': 'error', 'message': 'Channel not found.'}), 404

        channel_data = channel_res.data
        community_id = channel_data.get('community_id')

        # A channel must belong to a community to be toggled. Also, only the original creator can toggle it.
        if not community_id or str(channel_data.get('user_id')) != str(user_id):
             return jsonify({'status': 'error', 'message': 'This action is not allowed for this channel.'}), 403

        # 2. Verify the current user is the owner of that community
        community_res = supabase_admin.table('communities').select('owner_user_id').eq('id', community_id).single().execute()
        if not community_res.data or str(community_res.data.get('owner_user_id')) != str(user_id):
            return jsonify({'status': 'error', 'message': 'You are not the owner of this community.'}), 403

        # 3. If shared, check if it's the default channel. Default channels cannot be made private.
        new_is_shared = not channel_data['is_shared']
        if not new_is_shared: # If trying to make it private
            community_details = supabase_admin.table('communities').select('default_channel_id').eq('id', community_id).single().execute()
            if community_details.data and community_details.data.get('default_channel_id') == channel_id:
                return jsonify({'status': 'error', 'message': 'You cannot make a default channel private. Set a different default channel first.'}), 400

        # 4. Toggle the is_shared status
        update_res = supabase_admin.table('channels').update({'is_shared': new_is_shared}).eq('id', channel_id).execute()

        if not update_res.data:
            raise Exception("Failed to update channel privacy.")

        return jsonify({
            'status': 'success',
            'message': f"Channel is now {'shared' if new_is_shared else 'personal'}.",
            'is_shared': new_is_shared
        })

    except Exception as e:
        logging.error(f"Error toggling privacy for channel {channel_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
