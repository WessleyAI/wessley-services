import { Injectable, Logger } from '@nestjs/common';
import { InjectQueue } from '@nestjs/bull';
import { Queue } from 'bull';
import { ModelGenerationRequestDto } from '../../common/dto/model-generation.dto';

export interface ModelGenerationJob {
  requestId: string;
  vehicleSignature: string;
  graphQuery: any;
  options: any;
  callback?: any;
  userId?: string;
  priority?: number;
}

@Injectable()
export class QueueService {
  private readonly logger = new Logger(QueueService.name);

  constructor(
    @InjectQueue('model-generation')
    private readonly modelQueue: Queue<ModelGenerationJob>
  ) {}

  /**
   * Add 3D model generation job to queue
   */
  async addModelGenerationJob(
    request: ModelGenerationRequestDto,
    userId?: string,
    priority: number = 0
  ): Promise<{ jobId: string; queuePosition: number }> {
    try {
      const jobData: ModelGenerationJob = {
        requestId: request.requestId,
        vehicleSignature: request.graphQuery.vehicleSignature,
        graphQuery: request.graphQuery,
        options: request.options,
        callback: request.callback,
        userId,
        priority
      };

      const job = await this.modelQueue.add('generate-3d-model', jobData, {
        priority: -priority, // Bull uses negative values for higher priority
        jobId: request.requestId,
        delay: 0
      });

      const queuePosition = await this.getQueuePosition(job.id);

      this.logger.log(`Added 3D model generation job: ${request.requestId} (position: ${queuePosition})`);

      return {
        jobId: job.id.toString(),
        queuePosition
      };

    } catch (error) {
      this.logger.error(`Failed to add job ${request.requestId}:`, error);
      throw error;
    }
  }

  /**
   * Get job status and details
   */
  async getJobStatus(jobId: string): Promise<{
    id: string;
    status: 'waiting' | 'active' | 'completed' | 'failed' | 'delayed' | 'paused';
    progress: number;
    data?: any;
    result?: any;
    error?: any;
    queuePosition?: number;
    estimatedTime?: number;
  }> {
    try {
      const job = await this.modelQueue.getJob(jobId);
      
      if (!job) {
        throw new Error(`Job ${jobId} not found`);
      }

      const state = await job.getState();
      const progress = job.progress();
      const queuePosition = state === 'waiting' ? await this.getQueuePosition(job.id) : undefined;

      return {
        id: job.id.toString(),
        status: state as any,
        progress: typeof progress === 'number' ? progress : 0,
        data: job.data,
        result: job.returnvalue,
        error: job.failedReason,
        queuePosition,
        estimatedTime: queuePosition ? this.estimateProcessingTime(queuePosition) : undefined
      };

    } catch (error) {
      this.logger.error(`Failed to get job status for ${jobId}:`, error);
      throw error;
    }
  }

  /**
   * Cancel a job
   */
  async cancelJob(jobId: string): Promise<boolean> {
    try {
      const job = await this.modelQueue.getJob(jobId);
      
      if (!job) {
        return false;
      }

      const state = await job.getState();
      
      if (state === 'waiting' || state === 'delayed') {
        await job.remove();
        this.logger.log(`Cancelled job: ${jobId}`);
        return true;
      } else if (state === 'active') {
        // Cannot cancel active jobs safely
        this.logger.warn(`Cannot cancel active job: ${jobId}`);
        return false;
      }

      return false;

    } catch (error) {
      this.logger.error(`Failed to cancel job ${jobId}:`, error);
      throw error;
    }
  }

  /**
   * Get queue statistics
   */
  async getQueueStats(): Promise<{
    waiting: number;
    active: number;
    completed: number;
    failed: number;
    delayed: number;
    paused: number;
  }> {
    try {
      const counts = await this.modelQueue.getJobCounts();
      return counts;
    } catch (error) {
      this.logger.error('Failed to get queue stats:', error);
      throw error;
    }
  }

  /**
   * Get failed jobs for monitoring
   */
  async getFailedJobs(limit: number = 10): Promise<Array<{
    id: string;
    data: any;
    error: string;
    failedAt: Date;
  }>> {
    try {
      const failedJobs = await this.modelQueue.getFailed(0, limit - 1);
      
      return failedJobs.map(job => ({
        id: job.id.toString(),
        data: job.data,
        error: job.failedReason || 'Unknown error',
        failedAt: new Date(job.processedOn || job.timestamp)
      }));

    } catch (error) {
      this.logger.error('Failed to get failed jobs:', error);
      throw error;
    }
  }

  /**
   * Retry a failed job
   */
  async retryJob(jobId: string): Promise<boolean> {
    try {
      const job = await this.modelQueue.getJob(jobId);
      
      if (!job) {
        return false;
      }

      await job.retry();
      this.logger.log(`Retried job: ${jobId}`);
      return true;

    } catch (error) {
      this.logger.error(`Failed to retry job ${jobId}:`, error);
      throw error;
    }
  }

  /**
   * Clean up old jobs
   */
  async cleanQueue(grace: number = 24 * 60 * 60 * 1000): Promise<{
    completed: number;
    failed: number;
  }> {
    try {
      const [completed, failed] = await Promise.all([
        this.modelQueue.clean(grace, 'completed'),
        this.modelQueue.clean(grace, 'failed')
      ]);

      this.logger.log(`Cleaned queue: ${completed.length} completed, ${failed.length} failed jobs`);

      return {
        completed: completed.length,
        failed: failed.length
      };

    } catch (error) {
      this.logger.error('Failed to clean queue:', error);
      throw error;
    }
  }

  /**
   * Get position of job in queue
   */
  private async getQueuePosition(jobId: string | number): Promise<number> {
    try {
      const waitingJobs = await this.modelQueue.getWaiting();
      const position = waitingJobs.findIndex(job => job.id === jobId);
      return position >= 0 ? position + 1 : 0;
    } catch (error) {
      this.logger.error('Failed to get queue position:', error);
      return 0;
    }
  }

  /**
   * Estimate processing time based on queue position
   */
  private estimateProcessingTime(queuePosition: number): number {
    // Estimate 2-5 minutes per model generation job
    const avgProcessingTime = 3 * 60 * 1000; // 3 minutes in ms
    return queuePosition * avgProcessingTime;
  }

  /**
   * Pause queue processing
   */
  async pauseQueue(): Promise<void> {
    await this.modelQueue.pause();
    this.logger.log('Queue paused');
  }

  /**
   * Resume queue processing
   */
  async resumeQueue(): Promise<void> {
    await this.modelQueue.resume();
    this.logger.log('Queue resumed');
  }

  /**
   * Get queue health status
   */
  async getQueueHealth(): Promise<{
    isHealthy: boolean;
    redisConnection: boolean;
    activeWorkers: number;
    queueLength: number;
    issues: string[];
  }> {
    const issues: string[] = [];
    let isHealthy = true;

    try {
      // Test Redis connection
      const stats = await this.getQueueStats();
      const redisConnection = true;

      // Check queue length
      const queueLength = stats.waiting + stats.delayed;
      if (queueLength > 100) {
        issues.push(`High queue length: ${queueLength} jobs`);
        isHealthy = false;
      }

      // Check for stuck active jobs
      if (stats.active > 5) {
        issues.push(`High number of active jobs: ${stats.active}`);
      }

      // Check failed job ratio
      const totalProcessed = stats.completed + stats.failed;
      if (totalProcessed > 0) {
        const failureRate = stats.failed / totalProcessed;
        if (failureRate > 0.1) { // More than 10% failure rate
          issues.push(`High failure rate: ${(failureRate * 100).toFixed(1)}%`);
          isHealthy = false;
        }
      }

      return {
        isHealthy,
        redisConnection,
        activeWorkers: stats.active,
        queueLength,
        issues
      };

    } catch (error) {
      return {
        isHealthy: false,
        redisConnection: false,
        activeWorkers: 0,
        queueLength: 0,
        issues: [`Redis connection failed: ${error.message}`]
      };
    }
  }
}