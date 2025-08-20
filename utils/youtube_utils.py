# In utils/youtube_utils.py


import os
import re
import logging
import time
from datetime import datetime
from urllib.parse import urlparse, urlunparse
import concurrent.futures
import isodate
from typing import List, Dict, Optional, Union, Tuple
from googleapiclient.discovery import build
from dotenv import load_dotenv
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
# SECTION 1: CORE HELPER FUNCTIONS
# ==================================================================

def get_transcript(video_id: str) -> Optional[str]:
    """
    Fetches a transcript for a given video_id using a two-step process:
    1. Web Scraping (fast, primary)
    2. youtube_transcript_api (slower, fallback)
    """
    # --- METHOD 1: WEB SCRAPING (Primary) ---
    try:
        transcript_url = f"https://youtubetotranscript.com/transcript?v={video_id}"
        response = requests.get(transcript_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        transcript_container = soup.find('div', class_='-mt-4') or soup.find('article')
        if not transcript_container: raise ValueError("Transcript container not found.")
        filter_keywords = ["SponsorBlock", "Recapio", "Author :", "free prompts", "on steroids"]
        paragraphs = transcript_container.find_all('p')
        transcript_lines = [p.get_text(" ", strip=True) for p in paragraphs if p.get_text(" ", strip=True) and not any(kw in p.get_text() for kw in filter_keywords)]
        if not transcript_lines: raise ValueError("No valid transcript text found.")
        return "\n".join(transcript_lines)
    except Exception:
        # --- METHOD 2: YOUTUBE TRANSCRIPT API (Fallback) ---
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en','hi'])
            return "\n".join([segment['text'] for segment in transcript])
        except Exception as api_e:
            log.error(f"[{video_id}] Both scraping and API failed: {api_e}", exc_info=False)
            return None

def _fetch_transcript_worker(info_dict: Dict) -> Optional[Dict]:
    """
    Internal worker function for fetching a single transcript in a parallel thread.
    """
    video_id = info_dict.get('id')
    snippet = info_dict.get('snippet', {})
    content_details = info_dict.get('contentDetails', {})
    try:
        transcript_text = get_transcript(video_id)
        if not transcript_text:
            return None # Skip videos where no transcript could be found
        duration_iso = content_details.get('duration', 'PT0S')
        duration_seconds = isodate.parse_duration(duration_iso).total_seconds()
        video_data = {
            'video_id': video_id,
            'url': f"https://www.youtube.com/watch?v={video_id}",
            'title': snippet.get('title'),
            'uploader': snippet.get('channelTitle'),
            'description': snippet.get('description'),
            'duration': duration_seconds,
            'upload_date': snippet.get('publishedAt', '').split('T')[0],
            'transcript': transcript_text
        }
        print(f"✅ Successfully processed transcript for: {video_data['title'][:50]}...")
        return video_data
    except Exception as e:
        log.error(f"❌ Worker failed for video ID {video_id}: {e}", exc_info=False)
        return None

# ==================================================================
# SECTION 2: ROBUST CHANNEL & URL PROCESSING FUNCTIONS
# ==================================================================

def get_transcripts_from_channel(
    youtube_api_client,
    channel_url: str,
    target_video_count: int = 50,
    min_duration_seconds: int = 61,
    max_videos_to_scan: int = 500
) -> Tuple[List[Dict], str, int]:
    """
    Intelligently finds and processes a target number of long-form videos from a channel.
    """
    start_time = time.perf_counter()

    # --- Step 1: Find Channel ID and Details ---
    try:
        print(f"--- Step 1: Getting channel details for {channel_url} ---")
        match = re.search(r'(?:channel/|c/|@|user/)([^/?\s]+)', channel_url)
        if not match: raise ValueError("Could not parse a channel identifier from URL.")
        identifier = match.group(1)

        search_response = youtube_api_client.search().list(q=identifier, type='channel', part='id', maxResults=1).execute()
        if not search_response.get('items'): raise ValueError(f"Search API could not find a channel for identifier: {identifier}")
        channel_id = search_response['items'][0]['id']['channelId']

        channel_response = youtube_api_client.channels().list(part="snippet,contentDetails,statistics", id=channel_id).execute()
        if not channel_response.get('items'): raise ValueError(f"Could not find channel details for ID: {channel_id}")

        channel_item = channel_response['items'][0]
        uploads_playlist_id = channel_item['contentDetails']['relatedPlaylists']['uploads']
        subscriber_count = int(channel_item['statistics'].get('subscriberCount', 0))
        channel_thumbnail = channel_item['snippet']['thumbnails']['high']['url']
        print(f"Found Channel: {channel_item['snippet']['title']} ({subscriber_count} subscribers)")

    except Exception as e:
        log.error(f"Failed to get channel details: {e}", exc_info=True)
        return [], '', 0

    # --- Step 2: Scan for long-form videos ---
    long_form_metadata = []
    next_page_token = None
    videos_scanned = 0
    print(f"\n--- Step 2: Scanning for {target_video_count} videos longer than {min_duration_seconds}s (max scan: {max_videos_to_scan}) ---")

    while len(long_form_metadata) < target_video_count and videos_scanned < max_videos_to_scan:
        playlist_response = youtube_api_client.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        video_ids_chunk = [item['contentDetails']['videoId'] for item in playlist_response.get('items', [])]
        if not video_ids_chunk: break

        videos_scanned += len(video_ids_chunk)

        video_details_response = youtube_api_client.videos().list(
            part="snippet,contentDetails",
            id=",".join(video_ids_chunk)
        ).execute()

        for video_meta in video_details_response.get('items', []):
            duration_iso = video_meta.get('contentDetails', {}).get('duration', 'PT0S')
            try:
                duration_seconds = isodate.parse_duration(duration_iso).total_seconds()
                if duration_seconds > min_duration_seconds:
                    long_form_metadata.append(video_meta)
                    if len(long_form_metadata) >= target_video_count:
                        break
            except isodate.ISO8601Error:
                continue

        print(f"\rScanned: {videos_scanned} videos | Found: {len(long_form_metadata)} long-form videos", end="")
        if len(long_form_metadata) >= target_video_count: break

        next_page_token = playlist_response.get('nextPageToken')
        if not next_page_token: break

    print(f"\nScan complete. Found {len(long_form_metadata)} long-form videos to process.")

    # --- Step 3: Fetch transcripts in parallel ---
    if not long_form_metadata:
        return [], channel_thumbnail, subscriber_count

    print(f"\n--- Step 3: Fetching transcripts in parallel for {len(long_form_metadata)} videos ---")
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_video = {executor.submit(_fetch_transcript_worker, meta): meta for meta in long_form_metadata}
        for future in concurrent.futures.as_completed(future_to_video):
            result = future.result()
            if result:
                final_results.append(result)

    duration = time.perf_counter() - start_time
    print(f"\n[PERFORMANCE] Completed entire process in {duration:.2f} seconds.")

    return final_results, channel_thumbnail, subscriber_count

def get_transcripts_from_urls(
    youtube_api_client,
    video_urls: List[str],
    min_duration_seconds: int = 61
) -> List[Dict]:
    """
    Processes a specific list of video URLs, filters them for long-form content,
    and fetches their transcripts. Ideal for sync tasks.
    """
    print(f"--- Processing {len(video_urls)} URLs to find long-form content ---")
    video_ids = [match.group(1) for url in video_urls if (match := re.search(r"v=([a-zA-Z0-9_-]+)", url))]
    if not video_ids:
        return []

    # --- Step 1: Get metadata for all videos at once ---
    video_metadata_list = []
    for i in range(0, len(video_ids), 50): # Process in chunks of 50
        chunk_ids = video_ids[i:i + 50]
        try:
            response = youtube_api_client.videos().list(
                part="snippet,contentDetails",
                id=",".join(chunk_ids)
            ).execute()
            video_metadata_list.extend(response.get('items', []))
        except Exception as e:
            log.error(f"API error fetching video details for a chunk: {e}")

    # --- Step 2: Filter for long-form content ---
    long_form_metadata = []
    for video_meta in video_metadata_list:
        duration_iso = video_meta.get('contentDetails', {}).get('duration', 'PT0S')
        try:
            if isodate.parse_duration(duration_iso).total_seconds() > min_duration_seconds:
                long_form_metadata.append(video_meta)
        except isodate.ISO8601Error:
            continue
    
    print(f"Found {len(long_form_metadata)} new long-form videos to process.")
    if not long_form_metadata:
        return []

    # --- Step 3: Fetch transcripts in parallel ---
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_video = {executor.submit(_fetch_transcript_worker, meta): meta for meta in long_form_metadata}
        for future in concurrent.futures.as_completed(future_to_video):
            result = future.result()
            if result:
                final_results.append(result)
    
    return final_results

# ==================================================================
# SECTION 3: OTHER UTILITY FUNCTIONS
# ==================================================================
def is_youtube_video_url(url: str) -> bool:
    """Checks if a URL is a valid YouTube video URL."""
    video_pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)'
    return re.match(video_pattern, url) is not None

def clean_youtube_url(url):
    """Removes tracking parameters from a YouTube URL for consistency."""
    parsed = urlparse(url)
    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '',''))
    return clean_url