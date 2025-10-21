import { Module } from '@nestjs/common';
import { TerminusModule } from '@nestjs/terminus';
import { HealthController } from './health.controller';
import { Neo4jHealthIndicator } from './indicators/neo4j.indicator';
import { RedisHealthIndicator } from './indicators/redis.indicator';
import { S3HealthIndicator } from './indicators/s3.indicator';
import { GraphModule } from '../modules/graph/graph.module';
import { StorageModule } from '../modules/storage/storage.module';

@Module({
  imports: [
    TerminusModule,
    GraphModule,
    StorageModule
  ],
  controllers: [HealthController],
  providers: [
    Neo4jHealthIndicator,
    RedisHealthIndicator,
    S3HealthIndicator
  ]
})
export class HealthModule {}