import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { BullModule } from '@nestjs/bull';
import { TerminusModule } from '@nestjs/terminus';
import { EventEmitterModule } from '@nestjs/event-emitter';

import { configuration } from './config/configuration';
import { HealthModule } from './health/health.module';
import { JobsController } from './controllers/jobs.controller';
import { GraphModule } from './modules/graph/graph.module';
import { SpatialModule } from './modules/spatial/spatial.module';
import { GeometryModule } from './modules/geometry/geometry.module';
import { ExportModule } from './modules/export/export.module';
import { StorageModule } from './modules/storage/storage.module';
import { JobsModule } from './modules/jobs/jobs.module';
import { RealtimeModule } from './modules/realtime/realtime.module';

@Module({
  imports: [
    // Configuration
    ConfigModule.forRoot({
      isGlobal: true,
      load: [configuration],
      envFilePath: ['.env.local', '.env']
    }),

    // Event emitter for real-time updates
    EventEmitterModule.forRoot(),

    // Redis/Bull for job queues
    BullModule.forRootAsync({
      useFactory: () => ({
        redis: {
          host: process.env.REDIS_HOST || 'localhost',
          port: parseInt(process.env.REDIS_PORT) || 6379,
          password: process.env.REDIS_PASSWORD,
          db: parseInt(process.env.REDIS_DB) || 0
        }
      })
    }),

    // Health checks
    TerminusModule,
    HealthModule,

    // Core business modules
    GraphModule,
    SpatialModule,
    GeometryModule,
    ExportModule,
    StorageModule,
    JobsModule,
    RealtimeModule
  ],
  controllers: [JobsController],
  providers: []
})
export class AppModule {}