# qa_utils.py
import os
import json
import time
import logging
import numpy as np
import requests
from functools import lru_cache
from typing import Iterator, List, Optional, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from datetime import datetime
from . import prompts
from .supabase_client import get_supabase_client, get_supabase_admin_client
import re
import threading
from . import prompts
from .intent_classifier import classify_intent
from utils.supabase_client import get_supabase_admin_client
import tiktoken


# Load environment variables from .env file
load_dotenv()
cross_encoder = None
_cross_encoder_lock = threading.Lock()  # Prevent race condition in multi-threaded context

DEFAULT_REQUEST_TIMEOUT = 30  # seconds
REQUEST_RETRY_COUNT = 2
REQUEST_RETRY_BACKOFF = 1.0  # seconds

def _get_api_key(provider: str) -> Optional[str]:
    """Gets the appropriate API key from environment variables."""
    key_map = {
        'openai': 'OPENAI_API_KEY',
        'gemini': 'GEMINI_API_KEY',
        'groq': 'GROQ_API_KEY'
    }
    env_var = key_map.get(provider)
    return os.environ.get(env_var) if env_var else None

# --- Helper network request with retries ---
def _post_with_retries(url: str, json_payload: dict, headers: dict = None, timeout: int = DEFAULT_REQUEST_TIMEOUT) -> Optional[requests.Response]:
    attempt = 0
    while attempt <= REQUEST_RETRY_COUNT:
        try:
            response = requests.post(url, json=json_payload, headers=headers or {}, timeout=timeout)
            return response
        except requests.RequestException as e:
            logging.warning(f"Request to {url} failed on attempt {attempt + 1}/{REQUEST_RETRY_COUNT + 1}: {e}")
            attempt += 1
            time.sleep(REQUEST_RETRY_BACKOFF * attempt)
    logging.error(f"All attempts to {url} failed.")
    return None

# --- Provider-Specific Embedding Functions ---
def _create_openai_embedding(texts: List[str], model: str, api_key: str) -> Optional[List[Optional[np.ndarray]]]:
    """
    Returns a list of numpy arrays (float32) aligned with input texts.
    If an item failed, the corresponding position will be None.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(input=texts, model=model)
        # response.data should be a list aligned to `texts`
        embeddings = []
        for d in response.data:
            emb = np.array(d.embedding).astype('float32') if hasattr(d, 'embedding') or 'embedding' in d else None
            embeddings.append(emb)
        return embeddings
    except Exception as e:
        logging.error(f"Failed to create OpenAI batch embedding: {e}", exc_info=True)
        return [None] * len(texts)

def _create_groq_embedding(texts: List[str], model: str, api_key: str) -> Optional[List[Optional[np.ndarray]]]:
    logging.warning("Groq embedding function is not implemented; returning None placeholders.")
    return [None] * len(texts)

def _create_gemini_embedding(texts: List[str], model: str, api_key: str) -> Optional[List[Optional[np.ndarray]]]:
    """
    Uses Google Gemini embed_content. Normalizes output to a list aligned with inputs.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_name = f"models/{model}" if not model.startswith('models/') else model
        # Gemini batching: prefer sending the whole list if supported
        result = genai.embed_content(model=model_name, content=texts, task_type="retrieval_document")
        # result may contain 'embedding' as a list-of-lists or single list. Normalize it.
        embeddings = []
        if isinstance(result, dict) and 'embedding' in result:
            emb_data = result['embedding']
            # If single embedding for a single text:
            if isinstance(emb_data, (list, tuple)) and all(isinstance(x, (float, int)) for x in emb_data) and len(texts) == 1:
                embeddings = [np.array(emb_data).astype('float32')]
            elif isinstance(emb_data, list) and len(emb_data) == len(texts):
                # already aligned
                for embedding in emb_data:
                    embeddings.append(np.array(embedding).astype('float32') if embedding is not None else None)
            else:
                # Unknown shape; try to coerce
                for embedding in emb_data:
                    embeddings.append(np.array(embedding).astype('float32') if embedding is not None else None)
        else:
            logging.error("Unexpected Gemini response format when creating embeddings.")
            return [None] * len(texts)
        # Ensure alignment with input length
        if len(embeddings) != len(texts):
            # pad or truncate to match
            embeddings = (embeddings + [None] * len(texts))[:len(texts)]
        return embeddings
    except Exception as e:
        logging.error(f"Failed to create Gemini batch embedding: {e}", exc_info=True)
        return [None] * len(texts)

def _create_ollama_embedding(texts: List[str], model: str, ollama_url: str) -> Optional[List[Optional[np.ndarray]]]:
    """
    Calls Ollama embeddings endpoint. Returns list aligned with `texts`.
    Tries to batch if possible; otherwise falls back to single requests but preserves order with None placeholders.
    """
    if not ollama_url:
        logging.error("OLLAMA_URL is not provided.")
        return [None] * len(texts)

    try:
        # Try batch request (if API supports list)
        payload = {"model": model, "prompt": texts}
        response = _post_with_retries(f"{ollama_url}/api/embeddings", {"model": model, "prompt": texts}, timeout=DEFAULT_REQUEST_TIMEOUT)
        if response and response.status_code == 200:
            try:
                parsed = response.json()
                # Expecting either {'embedding': [...]} or {'embeddings': [...]} etc.
                if 'embeddings' in parsed and isinstance(parsed['embeddings'], list):
                    embeddings = []
                    for emb in parsed['embeddings']:
                        embeddings.append(np.array(emb).astype('float32') if emb is not None else None)
                    # Align length
                    if len(embeddings) != len(texts):
                        embeddings = (embeddings + [None] * len(texts))[:len(texts)]
                    return embeddings
                elif 'embedding' in parsed:
                    # If single embedding returned for whole batch and only one text
                    emb = parsed['embedding']
                    if len(texts) == 1:
                        return [np.array(emb).astype('float32')]
                    else:
                        # Unexpected single embedding for multiple texts
                        logging.error("Ollama returned single embedding for multiple prompts.")
                        return [None] * len(texts)
            except ValueError:
                logging.error("Could not parse JSON from Ollama response.")
        # If batch request didn't work, fallback to per-item
        embeddings = []
        for text in texts:
            resp = _post_with_retries(f"{ollama_url}/api/embeddings", {"model": model, "prompt": text}, timeout=DEFAULT_REQUEST_TIMEOUT)
            if resp and resp.status_code == 200:
                try:
                    json_data = resp.json()
                    emb = json_data.get('embedding') or json_data.get('embeddings')
                    # emb might be list or list-of-lists
                    if isinstance(emb, list) and all(isinstance(x, (float, int)) for x in emb):
                        embeddings.append(np.array(emb).astype('float32'))
                    elif isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], (list, tuple)):
                        # embeddings returned as nested list: take first
                        embeddings.append(np.array(emb[0]).astype('float32'))
                    else:
                        embeddings.append(None)
                except ValueError:
                    logging.error("Could not parse JSON from Ollama chunk response.")
                    embeddings.append(None)
            else:
                embeddings.append(None)
        return embeddings
    except Exception as e:
        logging.error(f"Exception during Ollama embedding: {e}", exc_info=True)
        return [None] * len(texts)

EMBEDDING_PROVIDER_MAP = {
    'openai': _create_openai_embedding,
    'gemini': _create_gemini_embedding,
    'ollama': _create_ollama_embedding,
    'groq': _create_groq_embedding
}

# In talktoyoutuber - v11/utils/qa_utils.py

def get_routed_context(question: str, channel_data: Optional[dict], user_id: str, access_token: str):
    """
    Intelligently routes the user's question to the appropriate retrieval strategy.
    It first checks for simple, high-precision cases (e.g., identity) and then uses
    an ML classifier for more nuanced routing.
    """
    question_lower = question.lower()
    video_ids = {v['video_id'] for v in channel_data.get('videos', [])} if channel_data else None

    # --- Priority 1: High-precision keyword matching for identity ---
    identity_keywords = ['who are you', 'what is your name', 'introduce yourself']
    if any(keyword in question_lower for keyword in identity_keywords):
        print("Intent Detected (Keyword): Identity. Prepending intro context.")
        identity_context = []
        if channel_data:
            creator_name = channel_data.get('channel_name', 'the creator')
            summary = channel_data.get('summary', 'a content creator who makes videos on YouTube.')
            identity_context.append({
                'video_title': 'Introduction',
                'chunk_text': f"My name is {creator_name}. I run this channel where {summary}",
                'video_url': channel_data.get('channel_url', '#'),
                'video_id': 'intro_chunk'
            })
        # We still search for other relevant chunks to provide a comprehensive answer
        additional_chunks = search_and_rerank_chunks(question, user_id, access_token, video_ids)
        return identity_context + additional_chunks

    # --- Priority 2: ML-based Intent Classification for nuanced routing ---
    intent = classify_intent(question)
    print(f"Query intent classified as: {intent}")

    if intent == "Latest Updates":
        print("Routing to: Hybrid Search with strong temporal bias.")
        # A bias of 2.0 makes the recency score twice as important.
        return search_and_rerank_chunks(question, user_id, access_token, video_ids, temporal_bias=2.0)

    # For all other intents, use the standard hybrid search for now.
    # This provides a clean structure to add more specific logic later (e.g., personality bias).
    else:
        print(f"Routing to: Hybrid Search (standard for intent: {intent}).")
        return search_and_rerank_chunks(question, user_id, access_token, video_ids, temporal_bias=1.0)

# --- Provider-Specific LLM STREAMING FUNCTIONS ---
def _get_openai_answer_stream(prompt: str, model: str, api_key: str, **kwargs):
    try:
        import openai
        base_url = kwargs.get('base_url')
        temperature = kwargs.get('temperature', 1)
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response_stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=temperature,
            stream=True
        )
        for chunk in response_stream:
            # compatibility with different SDK response shapes
            content = None
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
            elif 'choices' in chunk and chunk['choices'][0].get('delta', {}).get('content'):
                content = chunk['choices'][0]['delta']['content']
            if content:
                yield content
    except Exception as e:
        logging.error(f"Failed to get OpenAI stream: {e}", exc_info=True)
        yield "Error: Could not get a response from the provider."

def _get_groq_answer_stream(prompt: str, model: str, api_key: str, **kwargs):
    try:
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        data = {'model': model, 'messages': [{"role": "user", "content": prompt}], 'max_tokens': 1024, 'temperature': 1, 'stream': True}
        with requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=data, stream=True, timeout=DEFAULT_REQUEST_TIMEOUT) as response:
            for raw in response.iter_lines():
                if raw and raw.startswith(b'data: '):
                    chunk_data = raw.decode('utf-8')[6:].strip()
                    if chunk_data != '[DONE]':
                        try:
                            json_data = json.loads(chunk_data)
                            content = json_data['choices'][0]['delta'].get('content')
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        logging.error(f"Failed to get Groq stream: {e}", exc_info=True)
        yield "Error: Could not get a response from the provider."

def _get_gemini_answer_stream(prompt: str, model: str, api_key: str, **kwargs):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model)
        response_stream = gemini_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=1024, temperature=0.7), stream=True)
        for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
    except Exception as e:
        logging.error(f"Failed to get Gemini stream: {e}", exc_info=True)
        yield "Error: Could not get a response from the provider."

def _get_ollama_answer_stream(prompt: str, model: str, ollama_url: str, **kwargs):
    try:
        response = requests.post(f"{ollama_url}/api/chat", json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}, timeout=DEFAULT_REQUEST_TIMEOUT, stream=True)
        for chunk in response.iter_lines():
            if chunk:
                try:
                    json_data = json.loads(chunk)
                    # Ollama stream shape might differ; try multiple fallbacks
                    content = None
                    if isinstance(json_data, dict):
                        content = json_data.get('message', {}).get('content') or json_data.get('content') or json_data.get('text')
                    if content:
                        yield content
                except ValueError:
                    continue
    except Exception as e:
        logging.error(f"Failed to get Ollama stream: {e}", exc_info=True)
        yield "Error: Could not get a response from the provider."

LLM_STREAM_PROVIDER_MAP = {
    'openai': _get_openai_answer_stream,
    'groq': _get_groq_answer_stream,
    'gemini': _get_gemini_answer_stream,
    'ollama': _get_ollama_answer_stream
}

@lru_cache(maxsize=10)
def get_channel_corpus(user_id: str, video_ids_tuple: tuple, access_token: str):
    """
    Fetches all text chunks for a given set of video IDs to build a corpus for BM25.
    Uses LRU cache to avoid re-fetching for the same channel.
    The access_token is part of the cache key to ensure data isolation.
    """
    if not video_ids_tuple:
        return []

    video_ids = list(video_ids_tuple)
    logging.info(f"Corpus cache miss for user {user_id}. Fetching corpus for {len(video_ids)} videos.")

    supabase = get_supabase_client(access_token=access_token) if access_token else get_supabase_admin_client()

    # Fetch all chunks for these video_ids. A production system might need pagination here.
    # For now, we'll set a generous limit.
    response = supabase.table('embeddings').select('metadata').in_('video_id', video_ids).limit(10000).execute()

    if not getattr(response, 'data', None):
        logging.warning(f"Could not fetch corpus for videos: {video_ids}")
        return []

    corpus_docs = [item['metadata'] for item in response.data]

    logging.info(f"Successfully fetched {len(corpus_docs)} documents for the corpus for user {user_id}.")
    return corpus_docs

def rerank_with_cross_encoder(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-ranks chunks using a Cross-Encoder model, lazy-loaded on first call.
    Thread-safe lazy initialization is used to avoid race conditions.
    """
    global cross_encoder
    if cross_encoder is None:
        with _cross_encoder_lock:
            if cross_encoder is None:
                try:
                    print("Loading Cross-Encoder model for the first time...")
                    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                    print("Cross-Encoder model loaded successfully.")
                except Exception as e:
                    logging.warning(f"Could not load Cross-Encoder model: {e}. Re-ranking will be disabled.")
                    cross_encoder = 'failed_to_load'
    if cross_encoder == 'failed_to_load' or not chunks:
        return chunks

    start_time = time.perf_counter()
    print(f"Re-ranking {len(chunks)} chunks for query: '{query[:50]}...'")
    pairs = [[query, chunk.get('chunk_text', '')] for chunk in chunks]
    try:
        scores = cross_encoder.predict(pairs)
    except Exception as e:
        logging.error(f"Cross-Encoder prediction failed: {e}", exc_info=True)
        return chunks
    for chunk, score in zip(chunks, scores):
        chunk['relevance_score'] = float(score)
    sorted_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)
    end_time = time.perf_counter()
    print(f"[TIME_LOG] Re-ranking with Cross-Encoder took {end_time - start_time:.4f} seconds.")
    return sorted_chunks

def search_and_rerank_chunks(query: str, user_id: str, access_token: str, video_ids: Optional[set] = None, temporal_bias: float = 1.0):
    total_start_time = time.perf_counter()
    
    # --- 1. Hybrid Search: BM25 (Keyword) + Semantic ---

    # 1a. Get Corpus for BM25 (cached)
    corpus_docs = []
    if video_ids:
        video_ids_tuple = tuple(sorted(list(video_ids)))
        corpus_docs = get_channel_corpus(user_id, video_ids_tuple, access_token)

    # 1b. Perform BM25 Search
    bm25_results = []
    if corpus_docs:
        bm25_start_time = time.perf_counter()
        corpus_texts = [doc.get('chunk_text', '') for doc in corpus_docs]
        tokenized_corpus = [doc.split(" ") for doc in corpus_texts]

        if tokenized_corpus:
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split(" ")
            bm25_scores = bm25.get_scores(tokenized_query)

            for doc, score in zip(corpus_docs, bm25_scores):
                if score > 0.1:  # Threshold to reduce noise
                    result_item = doc.copy()
                    result_item['bm25_score'] = score
                    bm25_results.append(result_item)

            bm25_results = sorted(bm25_results, key=lambda x: x['bm25_score'], reverse=True)[:25]
            bm25_end_time = time.perf_counter()
            print(f"[TIME_LOG] BM25 Search took {bm25_end_time - bm25_start_time:.4f}s, found {len(bm25_results)} candidates.")

    # 1c. Perform Semantic Search
    def create_query_embedding(query_text: str):
        provider = os.environ.get('EMBED_PROVIDER', 'openai')
        model = os.environ.get('EMBED_MODEL')
        api_key = _get_api_key(provider)
        ollama_url = os.environ.get('OLLAMA_URL')
        embedding_function = EMBEDDING_PROVIDER_MAP.get(provider)
        if provider == 'ollama':
            embeddings = embedding_function([query_text], model, ollama_url=ollama_url)
        else:
            embeddings = embedding_function([query_text], model, api_key=api_key)
        return embeddings[0] if embeddings else None

    semantic_results = []
    try:
        query_embedding = create_query_embedding(query)
        if query_embedding is not None:
            supabase = get_supabase_client(access_token=access_token) if access_token else get_supabase_admin_client()
            match_params = {
                'query_embedding': query_embedding.tolist(),
                'match_threshold': float(os.environ.get('MATCH_THRESHOLD', 0.4)),
                'match_count': int(os.environ.get('MATCH_COUNT', 50)),
                'p_video_ids': list(video_ids) if video_ids else None
            }
            rpc_start_time = time.perf_counter()
            response = supabase.rpc('match_embeddings', match_params).execute()
            rpc_end_time = time.perf_counter()
            print(f"[TIME_LOG] Supabase 'match_embeddings' RPC call took {rpc_end_time - rpc_start_time:.4f} seconds.")

            if getattr(response, 'data', None):
                for row in response.data:
                    chunk_data = row.get('metadata') or {}
                    chunk_data['similarity_score'] = row.get('similarity')
                    semantic_results.append(chunk_data)
                print(f"SUCCESS: Received {len(semantic_results)} semantic results from Supabase.")
        else:
            logging.error("Failed to create query embedding.")

    except Exception as e:
        if 'JWT expired' in str(e): return "JWT_EXPIRED"
        logging.error(f"Error in semantic search part of hybrid search: {e}", exc_info=True)

    # --- 2. Combine, Score, and Prepare for Reranking ---

    # 2a. Combine Semantic and BM25 results, ensuring uniqueness
    combined_chunks = {}
    for chunk in semantic_results:
        chunk_id = (chunk.get('video_id'), chunk.get('chunk_index'))
        combined_chunks[chunk_id] = chunk
    for chunk in bm25_results:
        chunk_id = (chunk.get('video_id'), chunk.get('chunk_index'))
        if chunk_id in combined_chunks:
            combined_chunks[chunk_id]['bm25_score'] = chunk.get('bm25_score', 0.0)
        else:
            combined_chunks[chunk_id] = chunk

    # 2b. Apply Temporal Scoring (Recency Boost)
    w_semantic = 0.5
    w_bm25 = 0.3
    w_recency = 0.2

    # Normalize BM25 scores for stable hybrid scoring
    max_bm25_score = max([c.get('bm25_score', 0.0) for c in combined_chunks.values()]) if bm25_results else 1.0

    scored_chunks = []
    for chunk in combined_chunks.values():
        # Temporal decay score
        recency_score = 0.0
        upload_date_str = chunk.get('upload_date')
        if upload_date_str:
            try:
                upload_date = datetime.fromisoformat(upload_date_str.replace('Z', '+00:00'))
                days_since_upload = (datetime.now(datetime.now().astimezone().tzinfo) - upload_date).days
                half_life = 180  # Score halves every 180 days
                decay_rate = -0.693 / half_life
                recency_score = np.exp(decay_rate * max(0, days_since_upload))
            except (ValueError, TypeError):
                recency_score = 0.0
        
        # Normalized BM25
        normalized_bm25 = chunk.get('bm25_score', 0.0) / max_bm25_score if max_bm25_score > 0 else 0.0
        
        # Hybrid score for pre-ranking
        chunk['hybrid_score'] = (
            w_semantic * chunk.get('similarity_score', 0.0) +
            w_bm25 * normalized_bm25 +
            w_recency * recency_score * temporal_bias
        )
        scored_chunks.append(chunk)

    # Sort by hybrid score to select the best candidates for the expensive cross-encoder
    scored_chunks.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
    
    # --- 3. Rerank with Cross-Encoder ---
    CHUNKS_TO_RERANK = int(os.environ.get('CHUNKS_TO_RERANK', 25))
    chunks_for_reranker = scored_chunks[:CHUNKS_TO_RERANK]
    print(f"Passing the top {len(chunks_for_reranker)} hybrid results to the re-ranker.")

    if os.environ.get('ENABLE_RERANKING', 'true').lower() == 'true':
        reranked_results = rerank_with_cross_encoder(query, chunks_for_reranker)
    else:
        print("Re-ranking is disabled via environment variable. Skipping.")
        reranked_results = chunks_for_reranker
        # If not reranking, sort by hybrid score
        reranked_results.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)

    # --- 4. Diversify and Finalize Results ---
    top_k = int(os.environ.get('TOP_K', 5))
    final_results = []
    video_counts = {}
    for chunk in reranked_results:
        video_id = chunk.get('video_id')
        if video_counts.get(video_id, 0) < 2:
            final_results.append(chunk)
            video_counts[video_id] = video_counts.get(video_id, 0) + 1
        if len(final_results) >= top_k:
            break

    print(f"Selected {len(final_results)} diverse, highly relevant chunks for the context.")
    total_end_time = time.perf_counter()
    print(f"[TIME_LOG] Total search_and_rerank_chunks took {total_end_time - total_start_time:.4f} seconds.")
    return final_results

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Counts the number of tokens in a text string using tiktoken.
    Falls back to a default encoder if the model name is not recognized.
    """
    try:
        # Get the encoding for a specific model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # If the model is not found, use a general-purpose encoding
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

def _expand_query_with_history(question: str, history: str) -> str:
    """
    Uses a fast LLM to rewrite a conversational query into a standalone query
    that incorporates context from the chat history.
    """
    if not history.strip():
        return question

    print("Expanding query with chat history...")

    QUERY_EXPANSION_PROMPT = """Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that can be understood without the chat history.
Your goal is to make the question self-contained, so it can be used for a search query.
Do not answer the question, simply rewrite it.
If the follow-up question is already standalone, just return it as is.

CONVERSATION HISTORY:
---
{chat_history}
---

FOLLOW-UP QUESTION: "{question}"

STANDALONE QUESTION:"""

    try:
        provider = "groq"
        model = "llama3-8b-8192"
        api_key = _get_api_key(provider)

        if not api_key:
            logging.error("No API key found for query expansion (Groq). Falling back to original question.")
            return question

        prompt = QUERY_EXPANSION_PROMPT.format(chat_history=history, question=question)

        base_url = 'https://api.groq.com/openai/v1'

        expanded_query = _get_openai_answer_non_stream(prompt, model, api_key, base_url=base_url, temperature=0.0)

        if expanded_query and expanded_query.strip():
            print(f"Expanded query: '{expanded_query.strip()}'")
            return expanded_query.strip().strip('"')
        else:
            print("Query expansion returned empty result. Falling back to original question.")
            return question

    except Exception as e:
        logging.error(f"Error in _expand_query_with_history: {e}. Falling back to original question.")
        return question

def answer_question_stream(question_for_prompt: str, question_for_search: str, channel_data: dict = None, video_ids: set = None, user_id: str = None, access_token: str = None, tone: str = 'Casual') -> Iterator[str]:
    """
    Finds relevant context from documents and streams an answer to the user's question.
    """
    from tasks import post_answer_processing_task
    
    total_request_start_time = time.perf_counter()

    # --- Configuration Logging ---
    llm_provider = os.environ.get('LLM_PROVIDER', 'groq')
    llm_model = os.environ.get('MODEL_NAME', 'Not Set')
    embed_provider = os.environ.get('EMBED_PROVIDER', 'openai')
    embed_model = os.environ.get('EMBED_MODEL', 'Not Set')
    api_key = _get_api_key(llm_provider)
    masked_api_key = f"{api_key[:5]}...{api_key[-4:]}" if api_key else "Not Set"
    base_url = os.environ.get('OPENAI_API_BASE_URL', 'Default')

    print("--- Answering Question with the following configuration ---")
    print(f"  LLM Provider:         {llm_provider}")
    print(f"  LLM Model:            {llm_model}")
    print(f"  Embedding Provider:   {embed_provider}")
    print(f"  Embedding Model:      {embed_model}")
    print(f"  API Key Used:         {masked_api_key}")
    if llm_provider == 'openai' and base_url != 'Default':
        print(f"  OpenAI Base URL:      {base_url}")
    print("---------------------------------------------------------")

    # --- Separate chat history from the original question ---
    chat_history_for_prompt = ""
    original_question = question_for_prompt 
    history_marker = "Now, answer this new question, considering the history as context:\n"
    if history_marker in question_for_prompt:
        parts = question_for_prompt.split(history_marker)
        history_section = parts[0]
        original_question = parts[1]
        chat_history_for_prompt = history_section.replace("Given the following conversation history:\n", "").replace("--- End History ---\n\n", "")
    print(f"Answering question for user {user_id}: '{original_question[:100]}...'")
    
    if not user_id:
        yield "data: {\"error\": \"User not identified. Please log in.\"}\n\n"
        return

    # --- Expand the search query with conversation history for better context ---
    if chat_history_for_prompt:
        search_query = _expand_query_with_history(question_for_search, chat_history_for_prompt)
    else:
        search_query = question_for_search

    relevant_chunks = get_routed_context(search_query, channel_data, user_id, access_token)

    if relevant_chunks == "JWT_EXPIRED":
        yield 'data: {"error": "JWT_EXPIRED"}\n\n'
        return
    
    if not relevant_chunks:
        yield "data: {\"answer\": \"I couldn't find any relevant information in the documents to answer your question.\"}\n\n"
        yield "data: [DONE]\n\n"
        return

    # --- The rest of the function for processing and streaming the answer ---
    sources_dict = {}
    for chunk in relevant_chunks:
        try:
            url = chunk.get('video_url')
            if url and url not in sources_dict:
                sources_dict[url] = {'title': chunk.get('video_title', 'Unknown Title'), 'url': url}
        except Exception:
            continue
    formatted_sources = sorted(list(sources_dict.values()), key=lambda s: s['title'])
    yield f"data: {json.dumps({'sources': formatted_sources})}\n\n"

    context_parts = [f"From video '{chunk.get('video_title', 'Unknown')}' (uploaded on {chunk.get('upload_date', 'N/A')}): {chunk.get('chunk_text', '')}" for chunk in relevant_chunks]
    context = '\n\n'.join(context_parts)
    
    if channel_data:
        creator_name = channel_data.get('creator_name', channel_data.get('channel_name', 'the creator'))
        
        if tone == 'Factual':
            prompt_template = prompts.FACTUAL_PERSONA_PROMPT
            print("Using Factual Persona Prompt")
        else:
            prompt_template = prompts.CASUAL_PERSONA_PROMPT
            print("Using Casual Persona Prompt")

        prompt = prompt_template.format(
            creator_name=creator_name, 
            context=context, 
            question=original_question,
            chat_history=chat_history_for_prompt or "This is the first message in the conversation."
        )
    else:
        prompt = prompts.NEUTRAL_ASSISTANT_PROMPT.format(context=context, question=original_question)

    # --- This block now contains all necessary variables ---
    llm_provider = os.environ.get('LLM_PROVIDER', 'groq')
    model = os.environ.get('MODEL_NAME')
    api_key = _get_api_key(llm_provider)
    ollama_url = os.environ.get('OLLAMA_URL')
    openai_base_url = os.environ.get('OPENAI_API_BASE_URL') # <-- ensured defined
    temperature = float(os.environ.get('LLM_TEMPERATURE', 0.7))
    prompt_token_count = count_tokens(prompt, model)
    print(f"  Prompt Token Count:     {prompt_token_count}")
    
    stream_function = LLM_STREAM_PROVIDER_MAP.get(llm_provider)
    if not stream_function:
        yield "data: {\"answer\": \"Error: The selected LLM provider does not support streaming.\"}\n\n"
        yield "data: [DONE]\n\n"
        return

    full_answer = ""
    llm_stream_start_time = time.perf_counter()
    first_token_time_logged = False
    
    stream_kwargs = {
        'api_key': api_key,
        'ollama_url': ollama_url,
        'base_url': openai_base_url,
        'temperature': temperature
    }

    for chunk in stream_function(prompt, model, **stream_kwargs):
        if not first_token_time_logged:
            first_token_end_time = time.perf_counter()
            print(f"[TIME_LOG] LLM time to first token: {first_token_end_time - llm_stream_start_time:.4f} seconds.")
            first_token_time_logged = True
            
        full_answer += chunk
        yield f"data: {json.dumps({'answer': chunk})}\n\n"

    llm_stream_end_time = time.perf_counter()
    if not first_token_time_logged and not full_answer:
        print("[TIME_LOG] LLM stream produced no output.")
    else:
        print(f"[TIME_LOG] Full LLM stream generation took {llm_stream_end_time - llm_stream_start_time:.4f} seconds.")
    
    yield "data: [DONE]\n\n"
    
    if full_answer and "Error:" not in full_answer:
        try:
            post_answer_processing_task(
                user_id=user_id,
                channel_name=channel_data.get('channel_name', 'general') if channel_data else 'general',
                question=original_question,
                answer=full_answer,
                sources=formatted_sources
            )
        except Exception as e:
            logging.error(f"post_answer_processing_task failed: {e}", exc_info=True)
    
    total_request_end_time = time.perf_counter()
    print(f"[TIME_LOG] Total answer_question_stream request (end-to-end) took {total_request_end_time - total_request_start_time:.4f} seconds.")

def _get_openai_answer_non_stream(prompt: str, model: str, api_key: str, **kwargs):
    """Gets a single, non-streamed response from an OpenAI-compatible API."""
    try:
        import openai
        base_url = kwargs.get('base_url')
        temperature = kwargs.get('temperature', 0.2) # Lower temp for factual extraction
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=temperature,
            stream=False
        )
        # Handle possible different response shapes
        content = ""
        if hasattr(response.choices[0].message, 'content'):
            content = response.choices[0].message.content
        elif isinstance(response.choices[0], dict) and response.choices[0].get('message', {}).get('content'):
            content = response.choices[0]['message']['content']
        return content or ""
    except Exception as e:
        logging.error(f"Failed to get OpenAI non-stream response: {e}", exc_info=True)
        return ""
    
def extract_topics_from_text(text_sample: str) -> list:
    print('llm called')
    llm_provider = os.environ.get('TOPIC_LLM_PROVIDER', os.environ.get('LLM_PROVIDER', 'openai'))
    model = os.environ.get('TOPIC_MODEL_NAME', os.environ.get('MODEL_NAME'))
    api_key = _get_api_key(llm_provider)

    # --- START: FIX ---
    # Correctly determine the base_url for different providers
    if llm_provider == 'groq':
        base_url = 'https://api.groq.com/openai/v1'
    else:
        base_url = os.environ.get('OPENAI_API_BASE_URL', None)
    # --- END: FIX ---

    if not all([llm_provider, model, api_key]):
        print("Topic extraction LLM not fully configured.")
        return []

    prompt = prompts.TOPIC_EXTRACTION_PROMPT.format(context=text_sample)
    print(prompt)
    topic_string = _get_openai_answer_non_stream(prompt, model, api_key, base_url=base_url, temperature=0.3, max_tokens=200)
    print(topic_string)
    if topic_string:
        # Improved parsing to handle different AI model outputs
        cleaned_string = re.sub(r'^\s*topics:\s*', '', topic_string, flags=re.IGNORECASE).strip()
        topics = [t.strip() for t in cleaned_string.split(',') if t.strip()]
        print(topics)
        return topics
    
    return []

def generate_channel_summary(text_sample: str) -> str:
    llm_provider = os.environ.get('SUMMARY_LLM_PROVIDER', os.environ.get('LLM_PROVIDER', 'openai'))
    model = os.environ.get('SUMMARY_MODEL_NAME', os.environ.get('MODEL_NAME'))
    api_key = _get_api_key(llm_provider)

    if llm_provider == 'groq':
        base_url = 'https://api.groq.com/openai/v1'
    else:
        base_url = os.environ.get('OPENAI_API_BASE_URL', None)

    if not all([llm_provider, model, api_key]):
        logging.warning("Summary LLM not fully configured.")
        return ""

    prompt = prompts.CHANNEL_SUMMARY_PROMPT.format(context=text_sample)
    summary = _get_openai_answer_non_stream(prompt, model, api_key, base_url=base_url, temperature=0.5, max_tokens=250)

    return summary.strip() if summary else ""

def _get_transcript_summary(text: str) -> str:
    """
    Uses a fast LLM to generate a concise summary of a given text.
    This is an internal utility and doesn't need to be persona-driven.
    """
    print("Generating internal summary from transcript...")
    try:
        # We use a fast and efficient model for this internal task.
        # Groq's Llama3 8B is excellent for summarization.
        provider = "groq"
        model = "llama3-8b-8192"
        api_key = _get_api_key(provider)
        
        if not api_key:
            logging.error("No API key found for internal summarizer (Groq).")
            return "I couldn't generate a summary at the moment."
            
        prompt = f"Please provide a concise, one-paragraph summary of the following video transcript excerpt. Focus on the main topics. Do not start with 'The video is about...'. Just provide the summary.\n\nTranscript:\n\"\"\"\n{text}\n\"\"\""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 250,
        }

        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        summary = response.json()['choices'][0]['message']['content']
        print("Internal summary generated successfully.")
        return summary.strip()

    except Exception as e:
        logging.error(f"Error in _get_transcript_summary: {e}")
        return "I was unable to create a summary for the video."