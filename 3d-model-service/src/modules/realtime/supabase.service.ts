import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { createClient, SupabaseClient, User } from '@supabase/supabase-js';
import { ConfigService } from '@nestjs/config';

export interface ModelGenerationRecord {
  id: string;
  request_id: string;
  user_id: string;
  vehicle_signature: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  glb_url?: string;
  cdn_url?: string;
  metadata?: any;
  error_message?: string;
  created_at: string;
  updated_at: string;
  completed_at?: string;
}

export interface UserProfile {
  id: string;
  email: string;
  full_name?: string;
  avatar_url?: string;
  subscription_tier?: 'free' | 'pro' | 'enterprise';
  created_at: string;
  updated_at: string;
}

@Injectable()
export class SupabaseService implements OnModuleInit {
  private readonly logger = new Logger(SupabaseService.name);
  private supabase: SupabaseClient;

  constructor(private readonly configService: ConfigService) {}

  async onModuleInit() {
    const supabaseUrl = this.configService.get<string>('SUPABASE_URL');
    const supabaseServiceKey = this.configService.get<string>('SUPABASE_SERVICE_KEY');

    if (!supabaseUrl || !supabaseServiceKey) {
      this.logger.warn('Supabase configuration missing. Supabase features will be disabled.');
      return;
    }

    this.supabase = createClient(supabaseUrl, supabaseServiceKey, {
      auth: {
        autoRefreshToken: false,
        persistSession: false
      }
    });

    this.logger.log('Supabase client initialized');

    // Test connection
    await this.testConnection();
  }

  /**
   * Test Supabase connection
   */
  private async testConnection(): Promise<void> {
    try {
      const { data, error } = await this.supabase
        .from('model_generations')
        .select('count(*)', { count: 'exact', head: true });

      if (error) {
        this.logger.error('Supabase connection test failed:', error);
      } else {
        this.logger.log('✅ Supabase connection successful');
      }
    } catch (error) {
      this.logger.error('Supabase connection test error:', error);
    }
  }

  /**
   * Verify JWT token and get user
   */
  async verifyAndGetUser(jwt: string): Promise<User | null> {
    if (!this.supabase) {
      throw new Error('Supabase not configured');
    }

    try {
      const { data: { user }, error } = await this.supabase.auth.getUser(jwt);

      if (error) {
        this.logger.warn('JWT verification failed:', error.message);
        return null;
      }

      return user;
    } catch (error) {
      this.logger.error('JWT verification error:', error);
      return null;
    }
  }

  /**
   * Get user profile from database
   */
  async getUserProfile(userId: string): Promise<UserProfile | null> {
    if (!this.supabase) {
      throw new Error('Supabase not configured');
    }

    try {
      const { data, error } = await this.supabase
        .from('profiles')
        .select('*')
        .eq('id', userId)
        .single();

      if (error) {
        this.logger.error('Failed to get user profile:', error);
        return null;
      }

      return data;
    } catch (error) {
      this.logger.error('Get user profile error:', error);
      return null;
    }
  }

  /**
   * Create or update model generation record
   */
  async upsertModelGeneration(record: Partial<ModelGenerationRecord>): Promise<ModelGenerationRecord | null> {
    if (!this.supabase) {
      throw new Error('Supabase not configured');
    }

    try {
      const { data, error } = await this.supabase
        .from('model_generations')
        .upsert({
          ...record,
          updated_at: new Date().toISOString()
        })
        .select()
        .single();

      if (error) {
        this.logger.error('Failed to upsert model generation:', error);
        return null;
      }

      this.logger.log(`Model generation record updated: ${data.request_id}`);
      return data;
    } catch (error) {
      this.logger.error('Upsert model generation error:', error);
      return null;
    }
  }

  /**
   * Update model generation progress
   */
  async updateProgress(
    requestId: string,
    progress: number,
    status: ModelGenerationRecord['status'],
    metadata?: any
  ): Promise<boolean> {
    if (!this.supabase) {
      return false;
    }

    try {
      const updateData: Partial<ModelGenerationRecord> = {
        progress,
        status,
        updated_at: new Date().toISOString()
      };

      if (metadata) {
        updateData.metadata = metadata;
      }

      if (status === 'completed') {
        updateData.completed_at = new Date().toISOString();
      }

      const { error } = await this.supabase
        .from('model_generations')
        .update(updateData)
        .eq('request_id', requestId);

      if (error) {
        this.logger.error('Failed to update progress:', error);
        return false;
      }

      return true;
    } catch (error) {
      this.logger.error('Update progress error:', error);
      return false;
    }
  }

  /**
   * Update model generation with completion data
   */
  async completeModelGeneration(
    requestId: string,
    glbUrl: string,
    cdnUrl: string,
    metadata: any
  ): Promise<boolean> {
    if (!this.supabase) {
      return false;
    }

    try {
      const { error } = await this.supabase
        .from('model_generations')
        .update({
          status: 'completed',
          progress: 100,
          glb_url: glbUrl,
          cdn_url: cdnUrl,
          metadata,
          completed_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
        .eq('request_id', requestId);

      if (error) {
        this.logger.error('Failed to complete model generation:', error);
        return false;
      }

      this.logger.log(`Model generation completed: ${requestId}`);
      return true;
    } catch (error) {
      this.logger.error('Complete model generation error:', error);
      return false;
    }
  }

  /**
   * Mark model generation as failed
   */
  async failModelGeneration(requestId: string, errorMessage: string): Promise<boolean> {
    if (!this.supabase) {
      return false;
    }

    try {
      const { error } = await this.supabase
        .from('model_generations')
        .update({
          status: 'failed',
          error_message: errorMessage,
          updated_at: new Date().toISOString()
        })
        .eq('request_id', requestId);

      if (error) {
        this.logger.error('Failed to mark model generation as failed:', error);
        return false;
      }

      this.logger.log(`Model generation failed: ${requestId} - ${errorMessage}`);
      return true;
    } catch (error) {
      this.logger.error('Fail model generation error:', error);
      return false;
    }
  }

  /**
   * Get user's model generations
   */
  async getUserModelGenerations(
    userId: string,
    limit: number = 20,
    offset: number = 0
  ): Promise<ModelGenerationRecord[]> {
    if (!this.supabase) {
      return [];
    }

    try {
      const { data, error } = await this.supabase
        .from('model_generations')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .range(offset, offset + limit - 1);

      if (error) {
        this.logger.error('Failed to get user model generations:', error);
        return [];
      }

      return data || [];
    } catch (error) {
      this.logger.error('Get user model generations error:', error);
      return [];
    }
  }

  /**
   * Get model generation by request ID
   */
  async getModelGeneration(requestId: string): Promise<ModelGenerationRecord | null> {
    if (!this.supabase) {
      return null;
    }

    try {
      const { data, error } = await this.supabase
        .from('model_generations')
        .select('*')
        .eq('request_id', requestId)
        .single();

      if (error) {
        this.logger.error('Failed to get model generation:', error);
        return null;
      }

      return data;
    } catch (error) {
      this.logger.error('Get model generation error:', error);
      return null;
    }
  }

  /**
   * Send real-time notification
   */
  async sendRealtimeUpdate(
    channel: string,
    event: string,
    payload: any
  ): Promise<boolean> {
    if (!this.supabase) {
      return false;
    }

    try {
      // Use Supabase realtime to broadcast updates
      const channelRef = this.supabase.channel(channel);
      
      await channelRef.send({
        type: 'broadcast',
        event,
        payload
      });

      return true;
    } catch (error) {
      this.logger.error('Failed to send realtime update:', error);
      return false;
    }
  }

  /**
   * Broadcast job progress update
   */
  async broadcastProgress(
    userId: string,
    requestId: string,
    progress: number,
    status: string
  ): Promise<void> {
    await this.sendRealtimeUpdate(`user:${userId}`, 'job_progress', {
      requestId,
      progress,
      status,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Broadcast job completion
   */
  async broadcastCompletion(
    userId: string,
    requestId: string,
    result: any
  ): Promise<void> {
    await this.sendRealtimeUpdate(`user:${userId}`, 'job_completed', {
      requestId,
      result,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Broadcast job failure
   */
  async broadcastFailure(
    userId: string,
    requestId: string,
    error: string
  ): Promise<void> {
    await this.sendRealtimeUpdate(`user:${userId}`, 'job_failed', {
      requestId,
      error,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Get service health
   */
  async getHealth(): Promise<{
    connected: boolean;
    url?: string;
    error?: string;
  }> {
    if (!this.supabase) {
      return {
        connected: false,
        error: 'Supabase not configured'
      };
    }

    try {
      const { data, error } = await this.supabase
        .from('model_generations')
        .select('count(*)', { count: 'exact', head: true })
        .limit(1);

      if (error) {
        return {
          connected: false,
          error: error.message
        };
      }

      return {
        connected: true,
        url: this.configService.get<string>('SUPABASE_URL')
      };
    } catch (error) {
      return {
        connected: false,
        error: error.message
      };
    }
  }

  /**
   * Check if Supabase is configured
   */
  isConfigured(): boolean {
    return !!this.supabase;
  }
}