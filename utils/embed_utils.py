import logging
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
import os
import time
from dotenv import load_dotenv
from .supabase_client import get_supabase_admin_client
from .qa_utils import EMBEDDING_PROVIDER_MAP

# Load environment variables from .env if present
load_dotenv()

def create_and_store_embeddings(transcripts, _unused_config, user_id, progress_callback=None):
    """Create embeddings for transcript chunks in parallel and upsert to the vector store."""
    try:
        embed_provider = os.environ.get('EMBED_PROVIDER', 'openai')
        embed_model = os.environ.get('EMBED_MODEL', 'text-embedding-3-small')
        ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
        embed_api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('OPENAI_API_KEY') or os.environ.get('EMBED_API_KEY')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        all_chunks_for_embedding = []
        all_metadata = []
        total_videos = len(transcripts)

        logging.info(f"Creating embeddings for {total_videos} videos using advanced chunking...")

        for video_idx, transcript in enumerate(transcripts):
            text = transcript['transcript']
            video_title = transcript['title']
            
            chunks = text_splitter.split_text(text)
            
            logging.info(f"Processing video {video_idx + 1}/{total_videos}: {video_title[:50]}... ({len(chunks)} chunks)")

            for i, chunk in enumerate(chunks):
                enhanced_chunk = create_enhanced_chunk(chunk, transcript, i, len(chunks))
                all_chunks_for_embedding.append(enhanced_chunk)
                
                chunk_metadata = create_comprehensive_metadata(transcript, chunk, i, len(chunks))
                all_metadata.append(chunk_metadata)

        if not all_chunks_for_embedding:
            logging.warning("No chunks were created from the transcripts. Nothing to embed.")
            return False

        logging.info(f"Total chunks created: {len(all_chunks_for_embedding)}. Now creating embeddings in parallel...")

        embedding_function = EMBEDDING_PROVIDER_MAP.get(embed_provider)
        if not embedding_function:
            logging.error(f"Unsupported embedding provider selected: {embed_provider}")
            return False
        
        all_embeddings = []
        batch_size = 32
        
        text_batches = [all_chunks_for_embedding[i:i + batch_size] for i in range(0, len(all_chunks_for_embedding), batch_size)]

        def embed_batch(batch):
            """Worker function to embed a single batch of texts."""
            if embed_provider == 'ollama':
                return embedding_function(batch, embed_model, ollama_url)
            else:
                return embedding_function(batch, embed_model, embed_api_key)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # --- START: MODIFICATION FOR RATE LIMITING ---
            # Instead of submitting all jobs at once, we submit them in a loop with a delay.
            future_to_batch = {}
            for batch in text_batches:
                future_to_batch[executor.submit(embed_batch, batch)] = batch
                time.sleep(1) # Wait 1 second between submitting each batch job
            # --- END: MODIFICATION FOR RATE LIMITING ---

            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_embeddings = future.result()
                    if batch_embeddings:
                        all_embeddings.extend(batch_embeddings)
                except Exception as exc:
                    # Log the error, but continue processing other batches
                    logging.error(f'A batch generated an exception during embedding: {exc}')

        if not all_embeddings or len(all_embeddings) != len(all_chunks_for_embedding):
             logging.error(f"Embedding creation failed or produced incorrect count. Expected {len(all_chunks_for_embedding)}, got {len(all_embeddings)}. This may be due to rate limiting.")
             return False # Return False to indicate failure

        logging.info(f"Successfully created {len(all_embeddings)} embeddings. Now preparing to save to Supabase.")

        supabase = get_supabase_admin_client()
        
        vectors_to_insert = []
        for i, embedding in enumerate(all_embeddings):
            meta = all_metadata[i]
            vectors_to_insert.append({
                'user_id': user_id,
                'video_id': meta['video_id'],
                'embedding': embedding.tolist(),
                'metadata': meta
            })

        if not vectors_to_insert:
            logging.warning("No vectors to insert. Skipping database operation.")
            return True
        
        insert_batch_size = 100
        total_batches = (len(vectors_to_insert) + insert_batch_size - 1) // insert_batch_size
        
        logging.info(f"Preparing to insert {len(vectors_to_insert)} vectors in {total_batches} batches.")

        for i in range(total_batches):
            start_index = i * insert_batch_size
            end_index = start_index + insert_batch_size
            batch = vectors_to_insert[start_index:end_index]
            
            logging.info(f"Inserting batch {i + 1}/{total_batches} ({len(batch)} vectors)...")
            supabase.table('embeddings').insert(batch).execute()
            if progress_callback:
                progress_callback(i + 1, total_batches)
                
        logging.info("All batches inserted successfully.")
        return True

    except Exception as e:
        logging.error(f"Error in embedding creation process: {e}", exc_info=True)
        return False

# These helper functions are part of the file and should remain
def create_enhanced_chunk(chunk, transcript, chunk_index, total_chunks):
    video_title = transcript['title']
    uploader = transcript.get('uploader', 'Unknown')
    description = transcript.get('description', '')[:150]
    duration = transcript.get('duration', 0)
    timestamp_info = f"~{int((chunk_index / total_chunks) * duration)//60}:{int((chunk_index / total_chunks) * duration)%60:02d}" if duration > 0 else f"Part {chunk_index + 1}/{total_chunks}"
    return f"Video Title: {video_title}\nChannel: {uploader}\nTimestamp: {timestamp_info}\nContext: {description.strip() if description.strip() else 'YouTube video content'}\n\nContent: {chunk}"

def create_comprehensive_metadata(transcript, chunk, chunk_index, total_chunks):
    duration = transcript.get('duration', 0)
    estimated_start_time = int((chunk_index / total_chunks) * duration) if duration > 0 else 0
    return {
        'video_id': transcript['video_id'], 'video_title': transcript['title'], 'video_url': transcript['url'],
        'channel': transcript.get('uploader', 'Unknown'), 'video_description': transcript.get('description', '')[:300],
        'full_description': transcript.get('description', ''), 'video_duration': duration,
        'upload_date': transcript.get('upload_date', ''),
        'chunk_index': chunk_index, 'total_chunks': total_chunks, 'chunk_text': chunk, 'chunk_length': len(chunk),
        'chunk_preview': chunk[:200] + '...' if len(chunk) > 200 else chunk,
        'estimated_start_time': estimated_start_time, 'estimated_timestamp': f"{estimated_start_time//60}:{estimated_start_time%60:02d}",
        'chunk_position': f"{chunk_index + 1}/{total_chunks}", 'content_type': 'general', 'estimated_tokens': max(1, len(chunk) // 4)
    }