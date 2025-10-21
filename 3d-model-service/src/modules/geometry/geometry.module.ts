import { Module } from '@nestjs/common';
import { ComponentFactoryService } from './component-factory.service';
import { WireFactoryService } from './wire-factory.service';
import { MaterialService } from './material.service';
import { SceneComposerService } from './scene-composer.service';

@Module({
  providers: [
    ComponentFactoryService,
    WireFactoryService,
    MaterialService,
    SceneComposerService
  ],
  exports: [
    ComponentFactoryService,
    WireFactoryService,
    MaterialService,
    SceneComposerService
  ]
})
export class GeometryModule {}