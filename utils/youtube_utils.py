# In utils/youtube_utils.py


import os
import re
import logging
import time
from datetime import datetime
from urllib.parse import urlparse, urlunparse
import concurrent.futures
from typing import List, Dict, Optional
import isodate
from typing import List, Dict, Optional, Union # <-- Added Optional and Union
# --- New Imports for the API ---
from googleapiclient.discovery import build
from dotenv import load_dotenv

# --- Existing Imports to Keep ---
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import requests

# --- Setup ---
load_dotenv()
log = logging.getLogger(__name__)
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
youtube_api = None
if YOUTUBE_API_KEY:
    try:
        youtube_api = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        print("Successfully initialized YouTube Data API client.")
    except Exception as e:
        log.error(f"Failed to initialize YouTube API client: {e}")
else:
    log.warning("YOUTUBE_API_KEY not found. API-related functions will not work.")


# ==================================================================
# SECTION 1: ORIGINAL HELPER FUNCTIONS (PRESERVED)
# ==================================================================

def get_transcript(video_id: str) -> Optional[str]:
    """
    Fetches a transcript for a given video_id using a two-step process.

    1. Primary Method (Web Scraping): It first tries to scrape the transcript
       from youtubetotranscript.com. This is fast but can break if the
       website's layout changes.

    2. Fallback Method (API): If scraping fails for any reason, it uses the
       youtube_transcript_api library, which is more reliable but might
       not always find a transcript.

    Args:
        video_id (str): The unique identifier for the YouTube video.

    Returns:
        str | None: The full transcript text as a single string, or None
                    if both methods fail to retrieve a transcript.
    """
    # --- METHOD 1: WEB SCRAPING (Primary) ---
    try:
        print(f"[{video_id}] Trying Method 1: Web Scraping...")
        transcript_url = f"https://youtubetotranscript.com/transcript?v={video_id}"
        response = requests.get(transcript_url, timeout=30)
        response.raise_for_status()  # Will raise an error for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')

        # The website's structure might change, so we check for common containers
        transcript_container = soup.find('div', class_='-mt-4') or soup.find('article')
        if not transcript_container:
            raise ValueError("Transcript container not found on the website.")

        # Filter out sponsored content or unwanted text
        filter_keywords = [
            "SponsorBlock", "Recapio", "Author :", "free prompts", "on steroids", "magic words"
        ]

        paragraphs = transcript_container.find_all('p')
        transcript_lines = [
            p.get_text(" ", strip=True)
            for p in paragraphs
            if p.get_text(" ", strip=True) and not any(kw in p.get_text() for kw in filter_keywords)
        ]

        if not transcript_lines:
            raise ValueError("No valid transcript text found after scraping.")

        print(f"[{video_id}] SUCCESS: Transcript found via web scraping.")
        return "\n".join(transcript_lines)

    except Exception as e:
        print(f"[{video_id}] Method 1 (Scraping) failed: {e}")
        print(f"[{video_id}] Trying Method 2: API Fallback...")

        # --- METHOD 2: YOUTUBE TRANSCRIPT API (Fallback) ---
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en','hi'])
            print(f"[{video_id}] SUCCESS: Transcript found via API.")
            return "\n".join([segment['text'] for segment in transcript])
            

        except Exception as api_e:
            log.error(f"[{video_id}] Method 2 (API) also failed: {api_e}", exc_info=False)
            return None # Return None if both methods fail

def is_youtube_video_url(url: str) -> bool:
    """Checks if a URL is a valid YouTube video URL."""
    video_pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)'
    return re.match(video_pattern, url) is not None

def clean_youtube_url(url):
    """Removes tracking parameters from a YouTube URL for consistency."""
    parsed = urlparse(url)
    clean_url = urlunparse((
        parsed.scheme, parsed.netloc, parsed.path, '', '', ''
    ))
    return clean_url


# ==================================================================
# SECTION 2: NEW API-BASED FUNCTIONS (WITH FIX)
# ==================================================================

def get_channel_details_by_url(channel_url: str):
    """
    [CORRECTED] Fetches channel details using a URL, with a robust fallback.
    """
    if not youtube_api:
        raise ConnectionError("YouTube API client is not initialized. Check your API key.")
        
    patterns = [
        r'(?:youtube\.com/channel/)([^/?&]+)',
        r'(?:youtube\.com/c/)([^/?&]+)',
        r'(?:youtube\.com/@)([^/?&]+)',
        r'(?:youtube\.com/user/)([^/?&]+)'
    ]
    
    channel_identifier = None
    identifier_type = None

    for pattern in patterns:
        match = re.search(pattern, channel_url)
        if match:
            channel_identifier = match.group(1)
            if '/channel/' in pattern:
                identifier_type = 'id'
            else:
                identifier_type = 'forUsername'
            break
            
    if not channel_identifier:
        raise ValueError("Could not extract a valid channel ID or custom name from the URL.")

    # First, try a direct lookup.
    response = None
    try:
        request_params = {'part': "snippet,contentDetails,statistics"}
        if identifier_type == 'id':
            request_params['id'] = channel_identifier
        else:
            request_params['forUsername'] = channel_identifier
            
        request = youtube_api.channels().list(**request_params)
        response = request.execute()
    except Exception as e:
        log.warning(f"Direct API lookup failed with an exception: {e}")

    # If the direct lookup was successful and returned items, we're done.
    if response and response.get('items'):
        return response['items'][0]

    # If the first attempt failed or returned no items, we MUST fall back to search.
    log.warning(f"Direct lookup for '{channel_identifier}' failed or was empty. Trying search as a fallback...")
    search_request = youtube_api.search().list(
        part="snippet",
        q=channel_identifier,
        type="channel",
        maxResults=1
    )
    search_response = search_request.execute()
    if not search_response.get('items'):
        raise ValueError(f"YouTube channel not found for identifier via search: {channel_identifier}")
    
    # Now get the full details using the channel ID from the search result.
    channel_id = search_response['items'][0]['snippet']['channelId']
    details_request = youtube_api.channels().list(
        part="snippet,contentDetails,statistics",
        id=channel_id
    )
    details_response = details_request.execute()
    return details_response['items'][0]


# ==================================================================
# SECTION 3: REFACTORED CORE FUNCTIONS
# ==================================================================

def extract_channel_videos(youtube_api_client, channel_url, max_videos=50):
    """
    Extracts video URLs, thumbnail, and subscriber count from a YouTube channel
    using the official YouTube Data API. Now includes a robust handle lookup.
    """
    try:
        # Step 1: Parse the identifier (handle, username, or ID) from the URL
        match = re.search(r'(?:channel/|c/|@|user/)([^/?\s]+)', channel_url)
        if not match:
            log.error(f"Could not parse a channel identifier from URL: {channel_url}")
            return [], '', 0
        
        identifier = match.group(1)
        channel_id = None

        # --- START: FIX FOR @handle LOOKUP ---
        # If the identifier is not a standard channel ID, use the Search API to find it.
        if not identifier.startswith('UC'):
            print(f"Identifier '{identifier}' is not a channel ID. Using Search API to find it...")
            search_response = youtube_api_client.search().list(
                q=identifier,
                type='channel',
                part='id',
                maxResults=1
            ).execute()
            
            if not search_response.get('items'):
                log.error(f"Search API could not find a channel for identifier: {identifier}")
                return [], '', 0
            
            channel_id = search_response['items'][0]['id']['channelId']
            print(f"Found Channel ID: {channel_id}")
        else:
            # If the identifier is already a channel ID (starts with UC)
            channel_id = identifier
        # --- END: FIX FOR @handle LOOKUP ---

        # Step 2: Get channel details using the definitive channel_id
        response = youtube_api_client.channels().list(
            part="snippet,contentDetails,statistics",
            id=channel_id # Now we use the reliable channel ID
        ).execute()

        if not response.get('items'):
            log.error(f"Could not find channel details for ID: {channel_id}")
            return [], '', 0

        channel_item = response['items'][0]
        uploads_playlist_id = channel_item['contentDetails']['relatedPlaylists']['uploads']
        subscriber_count = int(channel_item['statistics'].get('subscriberCount', 0))
        channel_thumbnail = channel_item['snippet']['thumbnails']['high']['url']

        # Step 3: Fetch videos from the uploads playlist (this part remains the same)
        video_ids = []
        next_page_token = None
        while len(video_ids) < max_videos:
            playlist_response = youtube_api_client.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=min(50, max_videos - len(video_ids)),
                pageToken=next_page_token
            ).execute()

            for item in playlist_response.get('items', []):
                video_ids.append(item['contentDetails']['videoId'])

            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break

        video_urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
        return video_urls, channel_thumbnail, subscriber_count

    except Exception as e:
        log.error(f"Error extracting videos via API: {e}", exc_info=True)
        return [], '', 0

def get_video_transcripts(youtube_api_client, video_urls: List[str], max_videos: int = 50, progress_callback=None) -> List[Dict]:
    """
    Processes video URLs by fetching metadata in batches from the YouTube API
    and then fetching transcripts in parallel.
    """
    start_time = time.perf_counter()
    urls_to_process = video_urls[:max_videos]
    print(f"Starting API-based transcript extraction for {len(urls_to_process)} videos...")

    # --- PHASE 1: Fetch Video Metadata in Batches via API ---
    video_metadata_list = []
    video_ids = [re.search(r'v=([a-zA-Z0-9_-]+)', url).group(1) for url in urls_to_process if re.search(r'v=([a-zA-Z0-9_-]+)', url)]
    
    # The API's video->list endpoint can take up to 50 IDs at a time
    for i in range(0, len(video_ids), 50):
        chunk_ids = video_ids[i:i + 50]
        try:
            print(f"--- Phase 1: Fetching metadata for batch of {len(chunk_ids)} videos ---")
            response = youtube_api_client.videos().list(
                part="snippet,contentDetails", # snippet has title, desc, etc. contentDetails has duration.
                id=",".join(chunk_ids)
            ).execute()
            video_metadata_list.extend(response.get('items', []))
        except Exception as e:
            log.error(f"API error fetching video details for batch {i//50 + 1}: {e}")

    # --- PHASE 2: Fetch Transcripts in Parallel (No change here) ---
    print(f"\n--- Phase 2: Fetching transcripts in parallel for {len(video_metadata_list)} videos ---")

    def fetch_transcript_worker(info_dict: Dict) -> Optional[Dict]:
        video_id = info_dict.get('id')
        snippet = info_dict.get('snippet', {})
        content_details = info_dict.get('contentDetails', {})
        
        try:
            transcript_text = get_transcript(video_id)
            if not transcript_text:
                return None # Skip videos without transcripts

            # Parse duration from ISO 8601 format (e.g., 'PT1M35S')
            duration_iso = content_details.get('duration', 'PT0S')
            duration_seconds = isodate.parse_duration(duration_iso).total_seconds()

            video_data = {
                'video_id': video_id,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'title': snippet.get('title'),
                'uploader': snippet.get('channelTitle'),
                'description': snippet.get('description'),
                'duration': duration_seconds,
                'upload_date': snippet.get('publishedAt', '').split('T')[0], # Format as YYYY-MM-DD
                'transcript': transcript_text
            }
            print(f"✅ Successfully processed transcript for: {video_data['title'][:50]}...")
            return video_data
        except Exception as e:
            log.error(f"❌ Worker failed for video ID {video_id}: {e}", exc_info=False)
            return None

    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_video = {executor.submit(fetch_transcript_worker, meta): meta for meta in video_metadata_list}
        # ... (rest of the parallel processing logic remains the same) ...
        for future in concurrent.futures.as_completed(future_to_video):
            result = future.result()
            if result:
                final_results.append(result)

    duration = time.perf_counter() - start_time
    print(f"\n[PERFORMANCE] Completed in {duration:.2f} seconds.")
    return final_results


# ==================================================================
# SECTION 4: TEST SCRIPT (WITH FIX)
# ==================================================================

if __name__ == '__main__':
    # --- Step 1: Initialize the YouTube API Client ---
    # It's best practice to get the key from an environment variable.
    # Make sure you have set this environment variable in your system.
    API_KEY = os.environ.get("YOUTUBE_API_KEY")
    if not API_KEY:
        print("ERROR: YOUTUBE_API_KEY environment variable not set.")
        exit() # Exit the script if the key is not found
        
    try:
        print("Initializing YouTube Data API client...")
        youtube_api_client = build('youtube', 'v3', developerKey=API_KEY)
        print("Successfully initialized YouTube Data API client.")
    except Exception as e:
        print(f"Failed to initialize YouTube API client: {e}")
        exit()

    # --- Step 2: Get User Input ---
    channel_url_input = input("Please enter the YouTube channel URL to test: ")

    if channel_url_input:
        # --- Step 3: Extract Videos and Channel Info ---
        test_max_videos = 50 # Using a smaller number for faster testing
        print(f"\n--- Extracting the first {test_max_videos} videos from: {channel_url_input} ---\n")

        # THIS IS THE FIX: Pass the 'youtube_api_client' as the first argument
        video_urls, thumbnail_url, subscriber_count = extract_channel_videos(
            youtube_api_client, 
            channel_url_input, 
            max_videos=test_max_videos
        )

        if video_urls:
            print(f"--- SUCCESS: Found {len(video_urls)} videos ---")
            print(f"Channel Thumbnail URL: {thumbnail_url}")
            print(f"Subscriber Count: {subscriber_count}")
            
            # --- Step 4: Fetch Transcripts ---
            print(f"\n--- Fetching transcripts for the {len(video_urls)} extracted videos... ---\n")
            
            # THIS IS ALSO FIXED: Pass the 'youtube_api_client' here as well
            transcripts_data = get_video_transcripts(
                youtube_api_client, 
                video_urls, 
                max_videos=test_max_videos
            )

            if transcripts_data:
                print("\n--- FINAL SUCCESS: All transcripts received ---")
                for i, transcript in enumerate(transcripts_data):
                    print(f"\n  Video {i+1}: {transcript.get('title')}")
                    print(f"  Upload Date: {transcript.get('upload_date')}")
            else:
                print("\n--- Test FAILED at Step 4: Could not fetch any transcripts. ---")
        else:
            print("\n--- Test FAILED at Step 3: Could not find any videos for that channel. ---")
    else:
        print("No URL entered. Exiting test.")