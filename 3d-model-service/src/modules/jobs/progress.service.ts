import { Injectable, Logger } from '@nestjs/common';
import { EventEmitter2 } from '@nestjs/event-emitter';

export interface ProgressUpdate {
  requestId: string;
  progress: number;
  status: string;
  timestamp: Date;
}

export interface JobStartedEvent {
  requestId: string;
  jobId: string;
  vehicleSignature: string;
  timestamp: Date;
}

export interface JobCompletedEvent {
  requestId: string;
  result: any;
  timestamp: Date;
}

export interface JobFailedEvent {
  requestId: string;
  error: string;
  timestamp: Date;
}

@Injectable()
export class ProgressService {
  private readonly logger = new Logger(ProgressService.name);
  private readonly progressCache = new Map<string, ProgressUpdate>();

  constructor(private readonly eventEmitter: EventEmitter2) {}

  /**
   * Notify progress update
   */
  async notifyProgress(
    requestId: string,
    progress: number,
    status: string
  ): Promise<void> {
    const update: ProgressUpdate = {
      requestId,
      progress,
      status,
      timestamp: new Date()
    };

    // Cache the latest progress
    this.progressCache.set(requestId, update);

    // Emit event for real-time subscribers
    this.eventEmitter.emit('job.progress', update);

    this.logger.debug(`Progress update for ${requestId}: ${progress}% - ${status}`);
  }

  /**
   * Notify job started
   */
  async notifyJobStarted(
    requestId: string,
    jobId: string,
    vehicleSignature: string
  ): Promise<void> {
    const event: JobStartedEvent = {
      requestId,
      jobId,
      vehicleSignature,
      timestamp: new Date()
    };

    // Initial progress
    await this.notifyProgress(requestId, 0, 'Job started');

    // Emit job started event
    this.eventEmitter.emit('job.started', event);

    this.logger.log(`Job started: ${requestId} (${jobId})`);
  }

  /**
   * Notify job completed
   */
  async notifyJobCompleted(
    requestId: string,
    result: any
  ): Promise<void> {
    const event: JobCompletedEvent = {
      requestId,
      result,
      timestamp: new Date()
    };

    // Final progress
    await this.notifyProgress(requestId, 100, 'Job completed successfully');

    // Emit job completed event
    this.eventEmitter.emit('job.completed', event);

    this.logger.log(`Job completed: ${requestId}`);

    // Clean up progress cache after a delay
    setTimeout(() => {
      this.progressCache.delete(requestId);
    }, 5 * 60 * 1000); // 5 minutes
  }

  /**
   * Notify job failed
   */
  async notifyJobFailed(
    requestId: string,
    error: string
  ): Promise<void> {
    const event: JobFailedEvent = {
      requestId,
      error,
      timestamp: new Date()
    };

    // Error progress
    await this.notifyProgress(requestId, -1, `Job failed: ${error}`);

    // Emit job failed event
    this.eventEmitter.emit('job.failed', event);

    this.logger.error(`Job failed: ${requestId} - ${error}`);

    // Clean up progress cache after a delay
    setTimeout(() => {
      this.progressCache.delete(requestId);
    }, 10 * 60 * 1000); // 10 minutes for failed jobs
  }

  /**
   * Notify error during processing
   */
  async notifyError(
    requestId: string,
    error: string
  ): Promise<void> {
    const update: ProgressUpdate = {
      requestId,
      progress: -1,
      status: `Error: ${error}`,
      timestamp: new Date()
    };

    this.progressCache.set(requestId, update);
    this.eventEmitter.emit('job.error', update);

    this.logger.error(`Job error for ${requestId}: ${error}`);
  }

  /**
   * Get current progress for a request
   */
  getProgress(requestId: string): ProgressUpdate | null {
    return this.progressCache.get(requestId) || null;
  }

  /**
   * Get all active progress updates
   */
  getAllActiveProgress(): ProgressUpdate[] {
    return Array.from(this.progressCache.values())
      .filter(update => update.progress >= 0 && update.progress < 100);
  }

  /**
   * Clean up old progress entries
   */
  cleanupOldProgress(olderThanMs: number = 24 * 60 * 60 * 1000): number {
    const cutoff = new Date(Date.now() - olderThanMs);
    let cleaned = 0;

    for (const [requestId, update] of this.progressCache.entries()) {
      if (update.timestamp < cutoff) {
        this.progressCache.delete(requestId);
        cleaned++;
      }
    }

    if (cleaned > 0) {
      this.logger.log(`Cleaned up ${cleaned} old progress entries`);
    }

    return cleaned;
  }

  /**
   * Get progress statistics
   */
  getProgressStats(): {
    totalActive: number;
    totalCached: number;
    avgProgress: number;
    oldestEntry?: Date;
    newestEntry?: Date;
  } {
    const allUpdates = Array.from(this.progressCache.values());
    const activeUpdates = allUpdates.filter(u => u.progress >= 0 && u.progress < 100);

    const avgProgress = activeUpdates.length > 0
      ? activeUpdates.reduce((sum, u) => sum + u.progress, 0) / activeUpdates.length
      : 0;

    const timestamps = allUpdates.map(u => u.timestamp);
    const oldestEntry = timestamps.length > 0 ? new Date(Math.min(...timestamps.map(t => t.getTime()))) : undefined;
    const newestEntry = timestamps.length > 0 ? new Date(Math.max(...timestamps.map(t => t.getTime()))) : undefined;

    return {
      totalActive: activeUpdates.length,
      totalCached: allUpdates.length,
      avgProgress: Math.round(avgProgress),
      oldestEntry,
      newestEntry
    };
  }

  /**
   * Subscribe to progress updates for a specific request
   */
  subscribeToProgress(
    requestId: string,
    callback: (update: ProgressUpdate) => void
  ): () => void {
    const handler = (update: ProgressUpdate) => {
      if (update.requestId === requestId) {
        callback(update);
      }
    };

    this.eventEmitter.on('job.progress', handler);

    // Return unsubscribe function
    return () => {
      this.eventEmitter.off('job.progress', handler);
    };
  }

  /**
   * Subscribe to all job events
   */
  subscribeToJobEvents(callbacks: {
    onProgress?: (update: ProgressUpdate) => void;
    onStarted?: (event: JobStartedEvent) => void;
    onCompleted?: (event: JobCompletedEvent) => void;
    onFailed?: (event: JobFailedEvent) => void;
    onError?: (update: ProgressUpdate) => void;
  }): () => void {
    const handlers: Array<{ event: string; handler: Function }> = [];

    if (callbacks.onProgress) {
      const handler = callbacks.onProgress;
      this.eventEmitter.on('job.progress', handler);
      handlers.push({ event: 'job.progress', handler });
    }

    if (callbacks.onStarted) {
      const handler = callbacks.onStarted;
      this.eventEmitter.on('job.started', handler);
      handlers.push({ event: 'job.started', handler });
    }

    if (callbacks.onCompleted) {
      const handler = callbacks.onCompleted;
      this.eventEmitter.on('job.completed', handler);
      handlers.push({ event: 'job.completed', handler });
    }

    if (callbacks.onFailed) {
      const handler = callbacks.onFailed;
      this.eventEmitter.on('job.failed', handler);
      handlers.push({ event: 'job.failed', handler });
    }

    if (callbacks.onError) {
      const handler = callbacks.onError;
      this.eventEmitter.on('job.error', handler);
      handlers.push({ event: 'job.error', handler });
    }

    // Return unsubscribe function
    return () => {
      handlers.forEach(({ event, handler }) => {
        this.eventEmitter.off(event, handler);
      });
    };
  }
}