-- Supabase Setup Script for YoppyChat
-- This script will set up the entire database schema from scratch.

-- 1. Enable Necessary Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create Tables
-- We create channels first so communities can reference it for the default_channel_id
CREATE TABLE public.channels (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    channel_url TEXT UNIQUE NOT NULL,
    user_id UUID NOT NULL, -- We will add FK after profiles is created
    status TEXT DEFAULT 'pending' NOT NULL,
    is_shared BOOLEAN DEFAULT false NOT NULL,
    community_id UUID NULL, -- We will add FK after communities is created
    channel_name TEXT NULL,
    channel_thumbnail TEXT NULL,
    summary TEXT NULL,
    topics JSONB NULL,
    videos JSONB NULL,
    subscriber_count BIGINT NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

CREATE TABLE public.communities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    whop_community_id TEXT UNIQUE NOT NULL,
    owner_user_id UUID NOT NULL, -- We will add FK after profiles is created
    plan_id TEXT DEFAULT 'basic_community' NOT NULL,
    query_limit INTEGER DEFAULT 0 NOT NULL,
    queries_used INTEGER DEFAULT 0 NOT NULL,
    shared_channel_limit INTEGER DEFAULT 1 NOT NULL,
    trial_queries_used INTEGER DEFAULT 0 NOT NULL,
    default_channel_id BIGINT NULL REFERENCES public.channels(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

-- Add the FK for channels.community_id now that communities exists
ALTER TABLE public.channels
ADD CONSTRAINT fk_channels_community_id
FOREIGN KEY (community_id) REFERENCES public.communities(id);

CREATE TABLE public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    whop_user_id TEXT NULL,
    full_name TEXT NULL,
    avatar_url TEXT NULL,
    email TEXT NOT NULL,
    is_community_owner BOOLEAN DEFAULT false NOT NULL,
    community_id UUID NULL REFERENCES public.communities(id), -- Owner's link to their community
    personal_plan_id TEXT NULL,
    direct_subscription_plan TEXT NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

-- Now, add the remaining foreign keys
ALTER TABLE public.communities
ADD CONSTRAINT fk_communities_owner_user_id
FOREIGN KEY (owner_user_id) REFERENCES public.profiles(id);

ALTER TABLE public.channels
ADD CONSTRAINT fk_channels_user_id
FOREIGN KEY (user_id) REFERENCES public.profiles(id);

CREATE TABLE public.user_channels (
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    channel_id BIGINT NOT NULL REFERENCES public.channels(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, channel_id)
);

CREATE TABLE public.user_communities (
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    community_id UUID NOT NULL REFERENCES public.communities(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, community_id)
);

CREATE TABLE public.usage_stats (
    user_id UUID PRIMARY KEY REFERENCES public.profiles(id) ON DELETE CASCADE,
    queries_this_month INTEGER DEFAULT 0 NOT NULL,
    channels_processed INTEGER DEFAULT 0 NOT NULL,
    last_reset_date DATE DEFAULT CURRENT_DATE NOT NULL
);

CREATE TABLE public.chat_history (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    channel_name TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources JSONB NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

CREATE TABLE public.embeddings (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    channel_id BIGINT NOT NULL REFERENCES public.channels(id) ON DELETE CASCADE,
    video_id TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

CREATE TABLE public.telegram_connections (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    app_user_id UUID UNIQUE NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    telegram_chat_id BIGINT NOT NULL,
    connection_code TEXT NOT NULL,
    is_active BOOLEAN DEFAULT false NOT NULL,
    telegram_username TEXT NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

CREATE TABLE public.group_connections (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    owner_user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    linked_channel_id BIGINT UNIQUE NOT NULL REFERENCES public.channels(id) ON DELETE CASCADE,
    connection_code TEXT NOT NULL,
    is_active BOOLEAN DEFAULT false NOT NULL,
    telegram_group_id BIGINT NULL,
    telegram_group_name TEXT NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

-- 3. Create RPC Functions
CREATE OR REPLACE FUNCTION public.decrement_channel_count(p_user_id UUID)
RETURNS VOID AS $$
BEGIN
  UPDATE public.usage_stats
  SET channels_processed = channels_processed - 1
  WHERE user_id = p_user_id AND channels_processed > 0;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION public.increment_query_usage(p_community_id UUID, p_is_trial BOOLEAN)
RETURNS VOID AS $$
BEGIN
  IF p_community_id IS NOT NULL THEN
    IF p_is_trial THEN
      UPDATE public.communities
      SET trial_queries_used = trial_queries_used + 1
      WHERE id = p_community_id;
    ELSE
      UPDATE public.communities
      SET queries_used = queries_used + 1
      WHERE id = p_community_id;
    END IF;
  END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION public.increment_personal_query_usage(p_user_id UUID)
RETURNS VOID AS $$
BEGIN
  IF p_user_id IS NOT NULL THEN
    UPDATE public.usage_stats
    SET queries_this_month = queries_this_month + 1
    WHERE user_id = p_user_id;
  END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION match_embeddings (
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  p_video_ids text[]
)
RETURNS TABLE (
  id bigint,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    embeddings.id,
    embeddings.metadata,
    1 - (embeddings.embedding <=> query_embedding) AS similarity
  FROM embeddings
  WHERE (p_video_ids IS NULL OR embeddings.video_id = ANY(p_video_ids))
  AND 1 - (embeddings.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;

-- 4. Set up Row Level Security (RLS)
ALTER TABLE public.chat_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow users to see their own chat history"
ON public.chat_history FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Allow users to insert their own chat history"
ON public.chat_history FOR INSERT
WITH CHECK (auth.uid() = user_id);
