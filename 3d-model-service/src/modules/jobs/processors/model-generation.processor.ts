import { Processor, Process, OnQueueActive, OnQueueCompleted, OnQueueFailed } from '@nestjs/bull';
import { Logger } from '@nestjs/common';
import { Job } from 'bull';
import { ModelGenerationJob } from '../queue.service';
import { GraphService } from '../../graph/graph.service';
import { SpatialService } from '../../spatial/spatial.service';
import { GeometryService } from '../../geometry/geometry.service';
import { ExportService } from '../../export/export.service';
import { StorageService } from '../../storage/storage.service';
import { ProgressService } from '../progress.service';

@Processor('model-generation')
export class ModelGenerationProcessor {
  private readonly logger = new Logger(ModelGenerationProcessor.name);

  constructor(
    private readonly graphService: GraphService,
    private readonly spatialService: SpatialService,
    private readonly geometryService: GeometryService,
    private readonly exportService: ExportService,
    private readonly storageService: StorageService,
    private readonly progressService: ProgressService
  ) {}

  @Process('generate-3d-model')
  async handleModelGeneration(job: Job<ModelGenerationJob>) {
    const { requestId, vehicleSignature, graphQuery, options } = job.data;
    
    this.logger.log(`Starting 3D model generation for request: ${requestId}`);

    try {
      // Step 1: Query electrical system data (20% progress)
      await this.updateProgress(job, 10, 'Querying electrical system data...');
      
      const electricalData = await this.graphService.queryElectricalSystem({
        vehicleSignature,
        ...graphQuery
      });

      this.logger.log(`Retrieved ${electricalData.components.length} components and ${electricalData.connections.length} connections`);

      // Step 2: Generate spatial layout (40% progress)
      await this.updateProgress(job, 30, 'Generating spatial layout...');
      
      const spatialLayout = await this.spatialService.generateLayout(electricalData, {
        quality: options.quality,
        optimizeForWeb: options.optimizeForWeb
      });

      // Step 3: Create 3D geometry (60% progress)
      await this.updateProgress(job, 50, 'Creating 3D geometry...');
      
      const scene = await this.geometryService.generateScene(spatialLayout, {
        quality: options.quality,
        includeAnimations: options.includeAnimations,
        generateLOD: options.generateLOD
      });

      // Step 4: Export to GLB (80% progress)
      await this.updateProgress(job, 70, 'Exporting to GLB format...');
      
      const glbBuffer = await this.exportService.exportToGLB(scene, {
        includeMetadata: options.includeMetadata,
        optimizeForWeb: options.optimizeForWeb,
        maxFileSize: options.maxFileSize,
        targetTriangles: options.targetTriangles
      });

      // Step 5: Upload to storage (95% progress)
      await this.updateProgress(job, 85, 'Uploading to storage...');
      
      const uploadResult = await this.storageService.uploadGLB(
        requestId,
        glbBuffer,
        {
          vehicleSignature,
          metadata: {
            componentCount: electricalData.components.length,
            connectionCount: electricalData.connections.length,
            fileSize: glbBuffer.length,
            quality: options.quality,
            generatedAt: new Date().toISOString()
          }
        }
      );

      // Step 6: Finalize (100% progress)
      await this.updateProgress(job, 100, 'Model generation complete');

      const result = {
        requestId,
        vehicleSignature,
        glbUrl: uploadResult.url,
        cdnUrl: uploadResult.cdnUrl,
        metadata: {
          componentCount: electricalData.components.length,
          connectionCount: electricalData.connections.length,
          fileSize: glbBuffer.length,
          quality: options.quality,
          spatialBounds: spatialLayout.bounds,
          processingTime: Date.now() - job.timestamp,
          generatedAt: new Date().toISOString()
        },
        aiContext: options.includeMetadata ? {
          components: electricalData.components.map(c => ({
            id: c.id,
            type: c.type,
            position: c.position,
            zone: c.zone,
            properties: c.properties
          })),
          zones: spatialLayout.zones,
          powerDistribution: electricalData.powerAnalysis
        } : undefined
      };

      this.logger.log(`3D model generation completed for request: ${requestId}`);
      this.logger.log(`File size: ${(glbBuffer.length / 1024 / 1024).toFixed(2)} MB`);
      this.logger.log(`GLB URL: ${uploadResult.url}`);

      return result;

    } catch (error) {
      this.logger.error(`3D model generation failed for request: ${requestId}`, error);
      
      // Send error notification
      await this.progressService.notifyError(requestId, error.message);
      
      throw error;
    }
  }

  @OnQueueActive()
  onActive(job: Job<ModelGenerationJob>) {
    this.logger.log(`Processing job: ${job.id} (${job.data.requestId})`);
    
    // Notify job started
    this.progressService.notifyJobStarted(
      job.data.requestId,
      job.id.toString(),
      job.data.vehicleSignature
    );
  }

  @OnQueueCompleted()
  onCompleted(job: Job<ModelGenerationJob>, result: any) {
    this.logger.log(`Job completed: ${job.id} (${job.data.requestId})`);
    
    const processingTime = Date.now() - job.timestamp;
    this.logger.log(`Processing time: ${(processingTime / 1000).toFixed(2)}s`);
    
    // Notify job completed
    this.progressService.notifyJobCompleted(
      job.data.requestId,
      result
    );

    // Send webhook if configured
    if (job.data.callback?.webhook) {
      this.sendWebhookNotification(job.data.callback.webhook, {
        event: 'completion',
        requestId: job.data.requestId,
        result,
        processingTime
      }).catch(error => {
        this.logger.error(`Webhook notification failed: ${error.message}`);
      });
    }
  }

  @OnQueueFailed()
  onFailed(job: Job<ModelGenerationJob>, error: Error) {
    this.logger.error(`Job failed: ${job.id} (${job.data.requestId})`, error);
    
    // Notify job failed
    this.progressService.notifyJobFailed(
      job.data.requestId,
      error.message
    );

    // Send webhook if configured
    if (job.data.callback?.webhook) {
      this.sendWebhookNotification(job.data.callback.webhook, {
        event: 'error',
        requestId: job.data.requestId,
        error: error.message,
        failedAt: new Date().toISOString()
      }).catch(webhookError => {
        this.logger.error(`Webhook notification failed: ${webhookError.message}`);
      });
    }
  }

  /**
   * Update job progress and notify subscribers
   */
  private async updateProgress(
    job: Job<ModelGenerationJob>, 
    progress: number, 
    status: string
  ): Promise<void> {
    await job.progress(progress);
    
    await this.progressService.notifyProgress(
      job.data.requestId,
      progress,
      status
    );

    this.logger.debug(`Job ${job.id} progress: ${progress}% - ${status}`);
  }

  /**
   * Send webhook notification
   */
  private async sendWebhookNotification(
    webhookUrl: string,
    payload: any
  ): Promise<void> {
    try {
      const response = await fetch(webhookUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'Wessley-3D-Model-Service/1.0'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Webhook failed with status: ${response.status}`);
      }

      this.logger.log(`Webhook notification sent to: ${webhookUrl}`);

    } catch (error) {
      this.logger.error(`Webhook notification failed: ${error.message}`);
      throw error;
    }
  }
}