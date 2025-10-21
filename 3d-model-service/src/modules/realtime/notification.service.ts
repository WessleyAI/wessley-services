import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { EventEmitter2, OnEvent } from '@nestjs/event-emitter';
import { SupabaseService } from './supabase.service';
import { 
  ProgressUpdate, 
  JobStartedEvent, 
  JobCompletedEvent, 
  JobFailedEvent 
} from '../jobs/progress.service';

@Injectable()
export class NotificationService implements OnModuleInit {
  private readonly logger = new Logger(NotificationService.name);
  private readonly userJobMapping = new Map<string, string>(); // requestId -> userId

  constructor(
    private readonly supabaseService: SupabaseService,
    private readonly eventEmitter: EventEmitter2
  ) {}

  async onModuleInit() {
    this.logger.log('Notification service initialized');
    
    // Subscribe to job events from the progress service
    this.setupJobEventListeners();
  }

  /**
   * Setup listeners for job events
   */
  private setupJobEventListeners(): void {
    this.eventEmitter.on('job.started', this.handleJobStarted.bind(this));
    this.eventEmitter.on('job.progress', this.handleJobProgress.bind(this));
    this.eventEmitter.on('job.completed', this.handleJobCompleted.bind(this));
    this.eventEmitter.on('job.failed', this.handleJobFailed.bind(this));
    this.eventEmitter.on('job.error', this.handleJobError.bind(this));

    this.logger.log('Job event listeners configured');
  }

  /**
   * Register user for a job request
   */
  async registerUserForJob(requestId: string, userId: string, vehicleSignature: string): Promise<void> {
    this.userJobMapping.set(requestId, userId);

    // Create initial record in Supabase
    if (this.supabaseService.isConfigured()) {
      await this.supabaseService.upsertModelGeneration({
        request_id: requestId,
        user_id: userId,
        vehicle_signature: vehicleSignature,
        status: 'queued',
        progress: 0,
        created_at: new Date().toISOString()
      });
    }

    this.logger.log(`User ${userId} registered for job ${requestId}`);
  }

  /**
   * Handle job started event
   */
  @OnEvent('job.started')
  private async handleJobStarted(event: JobStartedEvent): Promise<void> {
    const { requestId, vehicleSignature } = event;
    const userId = this.userJobMapping.get(requestId);

    this.logger.log(`Job started: ${requestId} for user: ${userId}`);

    if (!userId || !this.supabaseService.isConfigured()) {
      return;
    }

    // Update database
    await this.supabaseService.updateProgress(requestId, 0, 'processing');

    // Send real-time notification
    await this.supabaseService.broadcastProgress(userId, requestId, 0, 'Job started');
  }

  /**
   * Handle job progress event
   */
  @OnEvent('job.progress')
  private async handleJobProgress(update: ProgressUpdate): Promise<void> {
    const { requestId, progress, status } = update;
    const userId = this.userJobMapping.get(requestId);

    if (!userId || !this.supabaseService.isConfigured()) {
      return;
    }

    // Determine status based on progress
    let dbStatus: 'queued' | 'processing' | 'completed' | 'failed' = 'processing';
    if (progress >= 100) {
      dbStatus = 'completed';
    } else if (progress < 0) {
      dbStatus = 'failed';
    }

    // Update database
    await this.supabaseService.updateProgress(requestId, progress, dbStatus);

    // Send real-time notification
    await this.supabaseService.broadcastProgress(userId, requestId, progress, status);

    this.logger.debug(`Progress update sent: ${requestId} - ${progress}%`);
  }

  /**
   * Handle job completed event
   */
  @OnEvent('job.completed')
  private async handleJobCompleted(event: JobCompletedEvent): Promise<void> {
    const { requestId, result } = event;
    const userId = this.userJobMapping.get(requestId);

    this.logger.log(`Job completed: ${requestId} for user: ${userId}`);

    if (!userId || !this.supabaseService.isConfigured()) {
      // Clean up mapping even if Supabase not configured
      this.userJobMapping.delete(requestId);
      return;
    }

    // Update database with completion data
    await this.supabaseService.completeModelGeneration(
      requestId,
      result.glbUrl,
      result.cdnUrl || result.glbUrl,
      result.metadata
    );

    // Send real-time notification
    await this.supabaseService.broadcastCompletion(userId, requestId, result);

    // Clean up mapping
    this.userJobMapping.delete(requestId);

    this.logger.log(`Job completion notification sent: ${requestId}`);
  }

  /**
   * Handle job failed event
   */
  @OnEvent('job.failed')
  private async handleJobFailed(event: JobFailedEvent): Promise<void> {
    const { requestId, error } = event;
    const userId = this.userJobMapping.get(requestId);

    this.logger.error(`Job failed: ${requestId} for user: ${userId} - ${error}`);

    if (!userId || !this.supabaseService.isConfigured()) {
      // Clean up mapping even if Supabase not configured
      this.userJobMapping.delete(requestId);
      return;
    }

    // Update database
    await this.supabaseService.failModelGeneration(requestId, error);

    // Send real-time notification
    await this.supabaseService.broadcastFailure(userId, requestId, error);

    // Clean up mapping
    this.userJobMapping.delete(requestId);

    this.logger.log(`Job failure notification sent: ${requestId}`);
  }

  /**
   * Handle job error event
   */
  @OnEvent('job.error')
  private async handleJobError(update: ProgressUpdate): Promise<void> {
    const { requestId, status } = update;
    const userId = this.userJobMapping.get(requestId);

    if (!userId || !this.supabaseService.isConfigured()) {
      return;
    }

    // Send real-time error notification
    await this.supabaseService.broadcastProgress(userId, requestId, -1, status);

    this.logger.warn(`Job error notification sent: ${requestId} - ${status}`);
  }

  /**
   * Send custom notification to user
   */
  async sendUserNotification(
    userId: string,
    event: string,
    payload: any
  ): Promise<boolean> {
    if (!this.supabaseService.isConfigured()) {
      this.logger.warn('Cannot send notification - Supabase not configured');
      return false;
    }

    try {
      await this.supabaseService.sendRealtimeUpdate(`user:${userId}`, event, payload);
      this.logger.log(`Custom notification sent to user ${userId}: ${event}`);
      return true;
    } catch (error) {
      this.logger.error('Failed to send custom notification:', error);
      return false;
    }
  }

  /**
   * Broadcast system announcement
   */
  async broadcastSystemAnnouncement(message: string, level: 'info' | 'warning' | 'error' = 'info'): Promise<void> {
    if (!this.supabaseService.isConfigured()) {
      return;
    }

    try {
      await this.supabaseService.sendRealtimeUpdate('system', 'announcement', {
        message,
        level,
        timestamp: new Date().toISOString()
      });

      this.logger.log(`System announcement broadcasted: ${message}`);
    } catch (error) {
      this.logger.error('Failed to broadcast system announcement:', error);
    }
  }

  /**
   * Get active job registrations
   */
  getActiveJobRegistrations(): Array<{ requestId: string; userId: string }> {
    return Array.from(this.userJobMapping.entries()).map(([requestId, userId]) => ({
      requestId,
      userId
    }));
  }

  /**
   * Clean up old job registrations
   */
  cleanupOldRegistrations(): number {
    // This is a simple cleanup - in production you might want to check actual job status
    // For now, we'll rely on job completion/failure events to clean up
    const initialSize = this.userJobMapping.size;
    
    // Jobs that have been registered for more than 24 hours are probably stuck
    // This is a safety net - normally they should be cleaned up by events
    const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000);
    
    // Since we don't track registration time in the simple Map,
    // we'll just log the current size for monitoring
    this.logger.log(`Active job registrations: ${this.userJobMapping.size}`);
    
    return 0; // No cleanup performed in this simple implementation
  }

  /**
   * Get notification service health
   */
  getHealth(): {
    active: boolean;
    supabaseConfigured: boolean;
    activeRegistrations: number;
    eventListenersActive: boolean;
  } {
    return {
      active: true,
      supabaseConfigured: this.supabaseService.isConfigured(),
      activeRegistrations: this.userJobMapping.size,
      eventListenersActive: this.eventEmitter.listenerCount('job.started') > 0
    };
  }
}