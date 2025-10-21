import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bull';
import { QueueService } from './queue.service';
import { WorkerService } from './worker.service';
import { ProgressService } from './progress.service';
import { ModelGenerationProcessor } from './processors/model-generation.processor';
import { GraphModule } from '../graph/graph.module';
import { SpatialModule } from '../spatial/spatial.module';
import { GeometryModule } from '../geometry/geometry.module';
import { ExportModule } from '../export/export.module';
import { StorageModule } from '../storage/storage.module';
import { RealtimeModule } from '../realtime/realtime.module';

@Module({
  imports: [
    BullModule.registerQueue({
      name: 'model-generation',
      defaultJobOptions: {
        attempts: 3,
        backoff: {
          type: 'exponential',
          delay: 5000
        },
        removeOnComplete: 100,
        removeOnFail: 50
      }
    }),
    GraphModule,
    SpatialModule,
    GeometryModule,
    ExportModule,
    StorageModule,
    RealtimeModule
  ],
  providers: [
    QueueService,
    WorkerService,
    ProgressService,
    ModelGenerationProcessor
  ],
  exports: [
    QueueService,
    WorkerService,
    ProgressService
  ]
})
export class JobsModule {}