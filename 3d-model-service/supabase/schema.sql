-- Supabase Database Schema for 3D Model Generation Service
-- This schema supports user authentication, model generation tracking, and real-time updates

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users profiles table (extends auth.users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID REFERENCES auth.users(id) PRIMARY KEY,
    email TEXT NOT NULL,
    full_name TEXT,
    avatar_url TEXT,
    subscription_tier TEXT DEFAULT 'free' CHECK (subscription_tier IN ('free', 'pro', 'enterprise')),
    api_key_hash TEXT, -- For API access
    usage_quota_monthly INTEGER DEFAULT 5, -- Monthly model generation quota
    usage_count_current_month INTEGER DEFAULT 0,
    quota_reset_date TIMESTAMP WITH TIME ZONE DEFAULT (date_trunc('month', NOW()) + INTERVAL '1 month'),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model generation requests table
CREATE TABLE IF NOT EXISTS public.model_generations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    request_id TEXT UNIQUE NOT NULL, -- User-provided or generated request ID
    user_id UUID REFERENCES public.profiles(id) NOT NULL,
    vehicle_signature TEXT NOT NULL, -- e.g., 'pajero_pinin_2001'
    
    -- Request details
    graph_query JSONB NOT NULL, -- Store the original GraphQL query
    generation_options JSONB NOT NULL, -- Quality, metadata, etc.
    
    -- Status tracking
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    
    -- Results
    job_id TEXT, -- Redis job ID
    glb_url TEXT, -- S3/CDN URL to the generated GLB file
    cdn_url TEXT, -- CDN URL (if different from glb_url)
    file_size_bytes BIGINT,
    
    -- Metadata and AI context
    metadata JSONB, -- Component counts, processing time, etc.
    ai_context JSONB, -- AI-ready component data for chat integration
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Performance metrics
    queue_time_ms INTEGER, -- Time spent in queue
    processing_time_ms INTEGER, -- Time spent processing
    total_time_ms INTEGER -- Total time from request to completion
);

-- User's favorite/saved models
CREATE TABLE IF NOT EXISTS public.user_model_favorites (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) NOT NULL,
    model_generation_id UUID REFERENCES public.model_generations(id) NOT NULL,
    name TEXT, -- User-given name for the model
    description TEXT,
    tags TEXT[], -- User tags for organization
    is_public BOOLEAN DEFAULT false, -- Whether other users can view this model
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, model_generation_id)
);

-- System announcements and notifications
CREATE TABLE IF NOT EXISTS public.system_announcements (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    level TEXT DEFAULT 'info' CHECK (level IN ('info', 'warning', 'error', 'maintenance')),
    is_active BOOLEAN DEFAULT true,
    target_audience TEXT DEFAULT 'all' CHECK (target_audience IN ('all', 'free', 'pro', 'enterprise')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS public.api_usage_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id),
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    ip_address INET,
    user_agent TEXT,
    api_key_used BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_model_generations_user_id ON public.model_generations(user_id);
CREATE INDEX IF NOT EXISTS idx_model_generations_status ON public.model_generations(status);
CREATE INDEX IF NOT EXISTS idx_model_generations_vehicle_signature ON public.model_generations(vehicle_signature);
CREATE INDEX IF NOT EXISTS idx_model_generations_created_at ON public.model_generations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_generations_request_id ON public.model_generations(request_id);

CREATE INDEX IF NOT EXISTS idx_profiles_email ON public.profiles(email);
CREATE INDEX IF NOT EXISTS idx_profiles_subscription_tier ON public.profiles(subscription_tier);

CREATE INDEX IF NOT EXISTS idx_user_model_favorites_user_id ON public.user_model_favorites(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_logs_user_id ON public.api_usage_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_logs_created_at ON public.api_usage_logs(created_at DESC);

-- Row Level Security (RLS) policies
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_generations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_model_favorites ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_usage_logs ENABLE ROW LEVEL SECURITY;

-- Profiles: Users can only see and edit their own profile
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON public.profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

-- Model generations: Users can only see their own model generations
CREATE POLICY "Users can view own model generations" ON public.model_generations
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own model generations" ON public.model_generations
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own model generations" ON public.model_generations
    FOR UPDATE USING (auth.uid() = user_id);

-- Service role can access all model generations (for the API service)
CREATE POLICY "Service role can manage all model generations" ON public.model_generations
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- User model favorites: Users can only manage their own favorites
CREATE POLICY "Users can manage own favorites" ON public.user_model_favorites
    FOR ALL USING (auth.uid() = user_id);

-- System announcements: Everyone can read active announcements
CREATE POLICY "Everyone can view active announcements" ON public.system_announcements
    FOR SELECT USING (is_active = true AND (expires_at IS NULL OR expires_at > NOW()));

-- API usage logs: Users can only see their own logs
CREATE POLICY "Users can view own API usage" ON public.api_usage_logs
    FOR SELECT USING (auth.uid() = user_id);

-- Service role can manage API usage logs
CREATE POLICY "Service role can manage API usage logs" ON public.api_usage_logs
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Functions for automatic profile creation
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name', split_part(NEW.email, '@', 1))
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to automatically create profile on user signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER handle_profiles_updated_at
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_model_generations_updated_at
    BEFORE UPDATE ON public.model_generations
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

-- Function to reset monthly usage quota
CREATE OR REPLACE FUNCTION public.reset_monthly_quota()
RETURNS void AS $$
BEGIN
    UPDATE public.profiles
    SET 
        usage_count_current_month = 0,
        quota_reset_date = date_trunc('month', NOW()) + INTERVAL '1 month'
    WHERE quota_reset_date <= NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to check and increment usage quota
CREATE OR REPLACE FUNCTION public.check_and_increment_usage(user_uuid UUID)
RETURNS BOOLEAN AS $$
DECLARE
    current_usage INTEGER;
    quota_limit INTEGER;
    tier TEXT;
BEGIN
    -- Get current usage and quota
    SELECT usage_count_current_month, usage_quota_monthly, subscription_tier
    INTO current_usage, quota_limit, tier
    FROM public.profiles
    WHERE id = user_uuid;
    
    -- Check if user exists
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;
    
    -- Check quota
    IF current_usage >= quota_limit THEN
        RETURN FALSE;
    END IF;
    
    -- Increment usage
    UPDATE public.profiles
    SET usage_count_current_month = usage_count_current_month + 1
    WHERE id = user_uuid;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Real-time subscriptions setup
-- Enable real-time for model_generations table
ALTER PUBLICATION supabase_realtime ADD TABLE public.model_generations;

-- Sample data for testing (optional)
-- INSERT INTO public.system_announcements (title, message, level) VALUES
-- ('Welcome to 3D Model Generation Service', 'Start generating 3D models from electrical system diagrams!', 'info'),
-- ('Scheduled Maintenance', 'System maintenance scheduled for Sunday 2AM-4AM UTC', 'warning');

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO postgres, anon, authenticated, service_role;
GRANT ALL ON ALL TABLES IN SCHEMA public TO postgres, service_role;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO anon, authenticated;

-- Grant specific permissions for authenticated users
GRANT INSERT, UPDATE ON public.profiles TO authenticated;
GRANT INSERT, UPDATE ON public.model_generations TO authenticated;
GRANT ALL ON public.user_model_favorites TO authenticated;

-- Comments for documentation
COMMENT ON TABLE public.profiles IS 'User profiles extending Supabase auth.users with subscription and usage tracking';
COMMENT ON TABLE public.model_generations IS 'Track 3D model generation requests from queue to completion';
COMMENT ON TABLE public.user_model_favorites IS 'User-saved favorite models with custom names and tags';
COMMENT ON TABLE public.system_announcements IS 'System-wide announcements and maintenance notices';
COMMENT ON TABLE public.api_usage_logs IS 'API usage tracking for analytics and rate limiting';

COMMENT ON COLUMN public.profiles.subscription_tier IS 'User subscription level affecting quotas and features';
COMMENT ON COLUMN public.model_generations.vehicle_signature IS 'Unique identifier for vehicle type (e.g., pajero_pinin_2001)';
COMMENT ON COLUMN public.model_generations.ai_context IS 'Processed component data for AI chat integration';

-- Create a view for user dashboard data
CREATE OR REPLACE VIEW public.user_dashboard_stats AS
SELECT 
    p.id as user_id,
    p.subscription_tier,
    p.usage_count_current_month,
    p.usage_quota_monthly,
    p.quota_reset_date,
    COUNT(mg.id) as total_models_generated,
    COUNT(CASE WHEN mg.status = 'completed' THEN 1 END) as completed_models,
    COUNT(CASE WHEN mg.status = 'failed' THEN 1 END) as failed_models,
    COUNT(CASE WHEN mg.created_at > NOW() - INTERVAL '30 days' THEN 1 END) as models_last_30_days,
    COUNT(fav.id) as favorite_models_count
FROM public.profiles p
LEFT JOIN public.model_generations mg ON p.id = mg.user_id
LEFT JOIN public.user_model_favorites fav ON p.id = fav.user_id
GROUP BY p.id, p.subscription_tier, p.usage_count_current_month, 
         p.usage_quota_monthly, p.quota_reset_date;

-- Grant access to the view
GRANT SELECT ON public.user_dashboard_stats TO authenticated;