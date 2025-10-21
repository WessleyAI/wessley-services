import { Injectable, Logger, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { InjectQueue } from '@nestjs/bull';
import { Queue } from 'bull';
import { ModelGenerationJob } from './queue.service';
import { ProgressService } from './progress.service';

@Injectable()
export class WorkerService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(WorkerService.name);
  private isShuttingDown = false;
  private healthCheckInterval?: NodeJS.Timeout;
  private cleanupInterval?: NodeJS.Timeout;

  constructor(
    @InjectQueue('model-generation')
    private readonly modelQueue: Queue<ModelGenerationJob>,
    private readonly progressService: ProgressService
  ) {}

  async onModuleInit() {
    this.logger.log('Worker service initializing...');

    // Start health monitoring
    this.startHealthMonitoring();

    // Start periodic cleanup
    this.startPeriodicCleanup();

    this.logger.log('Worker service initialized');
  }

  async onModuleDestroy() {
    this.logger.log('Worker service shutting down...');
    this.isShuttingDown = true;

    // Clear intervals
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }

    // Gracefully close the queue
    await this.gracefulShutdown();

    this.logger.log('Worker service shut down complete');
  }

  /**
   * Get worker statistics
   */
  async getWorkerStats(): Promise<{
    isActive: boolean;
    queueName: string;
    concurrency: number;
    jobCounts: any;
    memoryUsage: NodeJS.MemoryUsage;
    uptime: number;
  }> {
    try {
      const jobCounts = await this.modelQueue.getJobCounts();
      const memoryUsage = process.memoryUsage();

      return {
        isActive: !this.isShuttingDown,
        queueName: this.modelQueue.name,
        concurrency: this.modelQueue.concurrency,
        jobCounts,
        memoryUsage,
        uptime: process.uptime()
      };

    } catch (error) {
      this.logger.error('Failed to get worker stats:', error);
      throw error;
    }
  }

  /**
   * Pause job processing
   */
  async pauseProcessing(): Promise<void> {
    try {
      await this.modelQueue.pause();
      this.logger.log('Job processing paused');
    } catch (error) {
      this.logger.error('Failed to pause processing:', error);
      throw error;
    }
  }

  /**
   * Resume job processing
   */
  async resumeProcessing(): Promise<void> {
    try {
      await this.modelQueue.resume();
      this.logger.log('Job processing resumed');
    } catch (error) {
      this.logger.error('Failed to resume processing:', error);
      throw error;
    }
  }

  /**
   * Get active jobs
   */
  async getActiveJobs(): Promise<Array<{
    id: string;
    requestId: string;
    vehicleSignature: string;
    progress: number;
    startedAt: Date;
    data: any;
  }>> {
    try {
      const activeJobs = await this.modelQueue.getActive();

      return activeJobs.map(job => ({
        id: job.id.toString(),
        requestId: job.data.requestId,
        vehicleSignature: job.data.vehicleSignature,
        progress: typeof job.progress() === 'number' ? job.progress() : 0,
        startedAt: new Date(job.processedOn || job.timestamp),
        data: job.data
      }));

    } catch (error) {
      this.logger.error('Failed to get active jobs:', error);
      throw error;
    }
  }

  /**
   * Get completed jobs (recent)
   */
  async getCompletedJobs(limit: number = 10): Promise<Array<{
    id: string;
    requestId: string;
    vehicleSignature: string;
    completedAt: Date;
    processingTime: number;
    result: any;
  }>> {
    try {
      const completedJobs = await this.modelQueue.getCompleted(0, limit - 1);

      return completedJobs.map(job => ({
        id: job.id.toString(),
        requestId: job.data.requestId,
        vehicleSignature: job.data.vehicleSignature,
        completedAt: new Date(job.finishedOn || job.timestamp),
        processingTime: (job.finishedOn || 0) - (job.processedOn || job.timestamp),
        result: job.returnvalue
      }));

    } catch (error) {
      this.logger.error('Failed to get completed jobs:', error);
      throw error;
    }
  }

  /**
   * Check worker health
   */
  async checkHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    details: {
      redis: boolean;
      queueActive: boolean;
      memoryUsage: number;
      activeJobs: number;
      queueLength: number;
      issues: string[];
    };
  }> {
    const issues: string[] = [];
    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';

    try {
      // Check Redis connection
      const jobCounts = await this.modelQueue.getJobCounts();
      const redis = true;

      // Check memory usage
      const memoryUsage = process.memoryUsage();
      const memoryUsageMB = memoryUsage.heapUsed / 1024 / 1024;

      if (memoryUsageMB > 1000) { // More than 1GB
        issues.push(`High memory usage: ${memoryUsageMB.toFixed(0)}MB`);
        status = 'degraded';
      }

      if (memoryUsageMB > 2000) { // More than 2GB
        status = 'unhealthy';
      }

      // Check queue metrics
      const activeJobs = jobCounts.active;
      const queueLength = jobCounts.waiting + jobCounts.delayed;

      if (queueLength > 50) {
        issues.push(`High queue length: ${queueLength}`);
        status = 'degraded';
      }

      if (queueLength > 100) {
        status = 'unhealthy';
      }

      if (activeJobs > 10) {
        issues.push(`Many active jobs: ${activeJobs}`);
        status = 'degraded';
      }

      // Check if queue is paused
      const queueActive = !await this.modelQueue.isPaused();
      if (!queueActive) {
        issues.push('Queue is paused');
        status = 'degraded';
      }

      return {
        status,
        details: {
          redis,
          queueActive,
          memoryUsage: memoryUsageMB,
          activeJobs,
          queueLength,
          issues
        }
      };

    } catch (error) {
      this.logger.error('Health check failed:', error);
      return {
        status: 'unhealthy',
        details: {
          redis: false,
          queueActive: false,
          memoryUsage: 0,
          activeJobs: 0,
          queueLength: 0,
          issues: [`Health check failed: ${error.message}`]
        }
      };
    }
  }

  /**
   * Force cleanup of stuck jobs
   */
  async forceCleanupStuckJobs(): Promise<{
    stuckActive: number;
    oldCompleted: number;
    oldFailed: number;
  }> {
    try {
      this.logger.log('Starting force cleanup of stuck jobs...');

      // Clean jobs stuck in active state for more than 30 minutes
      const stuckActiveJobs = await this.modelQueue.getActive();
      const thirtyMinutesAgo = Date.now() - (30 * 60 * 1000);
      
      let stuckActiveCount = 0;
      for (const job of stuckActiveJobs) {
        if ((job.processedOn || job.timestamp) < thirtyMinutesAgo) {
          await job.moveToFailed({ message: 'Job stuck in active state - force cleaned' });
          stuckActiveCount++;
        }
      }

      // Clean old completed jobs (older than 24 hours)
      const oneDayAgo = 24 * 60 * 60 * 1000;
      const oldCompleted = await this.modelQueue.clean(oneDayAgo, 'completed');
      const oldFailed = await this.modelQueue.clean(oneDayAgo, 'failed');

      this.logger.log(`Force cleanup completed: ${stuckActiveCount} stuck active, ${oldCompleted.length} old completed, ${oldFailed.length} old failed`);

      return {
        stuckActive: stuckActiveCount,
        oldCompleted: oldCompleted.length,
        oldFailed: oldFailed.length
      };

    } catch (error) {
      this.logger.error('Force cleanup failed:', error);
      throw error;
    }
  }

  /**
   * Start health monitoring
   */
  private startHealthMonitoring(): void {
    this.healthCheckInterval = setInterval(async () => {
      try {
        const health = await this.checkHealth();
        
        if (health.status === 'unhealthy') {
          this.logger.error(`Worker health check failed: ${health.details.issues.join(', ')}`);
        } else if (health.status === 'degraded') {
          this.logger.warn(`Worker health degraded: ${health.details.issues.join(', ')}`);
        }

        // Force garbage collection if memory usage is high
        if (health.details.memoryUsage > 1500 && global.gc) {
          this.logger.log('Running garbage collection due to high memory usage');
          global.gc();
        }

      } catch (error) {
        this.logger.error('Health monitoring error:', error);
      }
    }, 60000); // Every minute
  }

  /**
   * Start periodic cleanup
   */
  private startPeriodicCleanup(): void {
    this.cleanupInterval = setInterval(async () => {
      try {
        // Clean old progress updates
        this.progressService.cleanupOldProgress();

        // Clean old completed jobs (keep last 100)
        await this.modelQueue.clean(0, 'completed', 100);

        // Clean old failed jobs (keep last 50)
        await this.modelQueue.clean(0, 'failed', 50);

      } catch (error) {
        this.logger.error('Periodic cleanup error:', error);
      }
    }, 30 * 60 * 1000); // Every 30 minutes
  }

  /**
   * Graceful shutdown
   */
  private async gracefulShutdown(): Promise<void> {
    try {
      this.logger.log('Starting graceful shutdown...');

      // Wait for active jobs to complete (with timeout)
      const maxWaitTime = 5 * 60 * 1000; // 5 minutes
      const startTime = Date.now();

      while (Date.now() - startTime < maxWaitTime) {
        const activeJobs = await this.modelQueue.getActive();
        
        if (activeJobs.length === 0) {
          this.logger.log('All active jobs completed');
          break;
        }

        this.logger.log(`Waiting for ${activeJobs.length} active jobs to complete...`);
        await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
      }

      // Pause the queue to prevent new jobs
      await this.modelQueue.pause();

      this.logger.log('Graceful shutdown completed');

    } catch (error) {
      this.logger.error('Graceful shutdown error:', error);
    }
  }
}