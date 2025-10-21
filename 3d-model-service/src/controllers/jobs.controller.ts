import { Controller, Get, Post, Body, Param, Query, HttpCode, HttpStatus, UseGuards, Req } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiParam, ApiBearerAuth } from '@nestjs/swagger';
import { QueueService } from '../modules/jobs/queue.service';
import { WorkerService } from '../modules/jobs/worker.service';
import { ProgressService } from '../modules/jobs/progress.service';
import { NotificationService } from '../modules/realtime/notification.service';
import { ModelGenerationRequestDto, ModelGenerationResponseDto } from '../common/dto/model-generation.dto';
import { AuthGuard, AuthenticatedRequest, Public } from '../guards/auth.guard';

@ApiTags('Jobs')
@Controller('api/v1/jobs')
@UseGuards(AuthGuard)
@ApiBearerAuth()
export class JobsController {
  constructor(
    private readonly queueService: QueueService,
    private readonly workerService: WorkerService,
    private readonly progressService: ProgressService,
    private readonly notificationService: NotificationService
  ) {}

  @Post('generate')
  @HttpCode(HttpStatus.ACCEPTED)
  @ApiOperation({ summary: 'Submit 3D model generation job' })
  @ApiResponse({ status: 202, description: 'Job submitted successfully', type: ModelGenerationResponseDto })
  async generateModel(
    @Body() request: ModelGenerationRequestDto,
    @Req() req: AuthenticatedRequest
  ): Promise<ModelGenerationResponseDto> {
    try {
      const userId = req.user?.id;
      
      const { jobId, queuePosition } = await this.queueService.addModelGenerationJob(request, userId);

      // Register user for real-time notifications
      if (userId) {
        await this.notificationService.registerUserForJob(
          request.requestId, 
          userId, 
          request.graphQuery.vehicleSignature
        );
      }

      return {
        success: true,
        jobId,
        requestId: request.requestId,
        status: 'queued',
        queuePosition,
        estimatedTime: queuePosition * 3 * 60 * 1000, // 3 minutes per job estimate
        message: `3D model generation job queued successfully. Position: ${queuePosition}`
      };

    } catch (error) {
      return {
        success: false,
        jobId: '',
        requestId: request.requestId,
        status: 'failed',
        message: `Failed to queue job: ${error.message}`
      };
    }
  }

  @Get(':jobId/status')
  @ApiOperation({ summary: 'Get job status' })
  @ApiParam({ name: 'jobId', description: 'Job ID' })
  async getJobStatus(@Param('jobId') jobId: string) {
    try {
      const status = await this.queueService.getJobStatus(jobId);
      return {
        success: true,
        ...status
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Get(':requestId/progress')
  @ApiOperation({ summary: 'Get job progress by request ID' })
  @ApiParam({ name: 'requestId', description: 'Request ID' })
  async getProgress(@Param('requestId') requestId: string) {
    const progress = this.progressService.getProgress(requestId);
    
    if (!progress) {
      return {
        success: false,
        error: 'Progress not found'
      };
    }

    return {
      success: true,
      ...progress
    };
  }

  @Post(':jobId/cancel')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Cancel a job' })
  @ApiParam({ name: 'jobId', description: 'Job ID' })
  async cancelJob(@Param('jobId') jobId: string) {
    try {
      const cancelled = await this.queueService.cancelJob(jobId);
      return {
        success: cancelled,
        message: cancelled ? 'Job cancelled successfully' : 'Job could not be cancelled (may be active)'
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Post(':jobId/retry')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Retry a failed job' })
  @ApiParam({ name: 'jobId', description: 'Job ID' })
  async retryJob(@Param('jobId') jobId: string) {
    try {
      const retried = await this.queueService.retryJob(jobId);
      return {
        success: retried,
        message: retried ? 'Job retried successfully' : 'Job could not be retried'
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Get('queue/stats')
  @Public()
  @ApiOperation({ summary: 'Get queue statistics' })
  async getQueueStats() {
    try {
      const stats = await this.queueService.getQueueStats();
      const health = await this.queueService.getQueueHealth();
      const progressStats = this.progressService.getProgressStats();

      return {
        success: true,
        queue: stats,
        health,
        progress: progressStats
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Get('workers/stats')
  @Public()
  @ApiOperation({ summary: 'Get worker statistics' })
  async getWorkerStats() {
    try {
      const stats = await this.workerService.getWorkerStats();
      const health = await this.workerService.checkHealth();

      return {
        success: true,
        worker: stats,
        health
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Get('active')
  @ApiOperation({ summary: 'Get active jobs' })
  async getActiveJobs() {
    try {
      const activeJobs = await this.workerService.getActiveJobs();
      return {
        success: true,
        jobs: activeJobs
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Get('completed')
  @ApiOperation({ summary: 'Get recent completed jobs' })
  async getCompletedJobs(@Query('limit') limit?: string) {
    try {
      const jobs = await this.workerService.getCompletedJobs(
        limit ? parseInt(limit) : 10
      );
      return {
        success: true,
        jobs
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Get('failed')
  @ApiOperation({ summary: 'Get recent failed jobs' })
  async getFailedJobs(@Query('limit') limit?: string) {
    try {
      const jobs = await this.queueService.getFailedJobs(
        limit ? parseInt(limit) : 10
      );
      return {
        success: true,
        jobs
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Post('queue/pause')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Pause job processing' })
  async pauseQueue() {
    try {
      await this.workerService.pauseProcessing();
      return {
        success: true,
        message: 'Queue paused'
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Post('queue/resume')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Resume job processing' })
  async resumeQueue() {
    try {
      await this.workerService.resumeProcessing();
      return {
        success: true,
        message: 'Queue resumed'
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Post('queue/clean')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Clean old jobs from queue' })
  async cleanQueue(@Query('grace') grace?: string) {
    try {
      const graceMs = grace ? parseInt(grace) : 24 * 60 * 60 * 1000; // 24 hours default
      const cleaned = await this.queueService.cleanQueue(graceMs);
      return {
        success: true,
        message: `Cleaned ${cleaned.completed} completed and ${cleaned.failed} failed jobs`,
        cleaned
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  @Post('workers/cleanup')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Force cleanup stuck jobs' })
  async forceCleanup() {
    try {
      const result = await this.workerService.forceCleanupStuckJobs();
      return {
        success: true,
        message: 'Force cleanup completed',
        result
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }
}