import { Module } from '@nestjs/common';
import { LayoutService } from './layout.service';
import { RoutingService } from './routing.service';
import { OptimizationService } from './optimization.service';

@Module({
  providers: [
    LayoutService,
    RoutingService,
    OptimizationService
  ],
  exports: [
    LayoutService,
    RoutingService,
    OptimizationService
  ]
})
export class SpatialModule {}