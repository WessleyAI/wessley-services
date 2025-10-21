import { Module } from '@nestjs/common';
import { GLTFExporterService } from './gltf-exporter.service';
import { OptimizerService } from './optimizer.service';
import { MetadataService } from './metadata.service';

@Module({
  providers: [
    GLTFExporterService,
    OptimizerService,
    MetadataService
  ],
  exports: [
    GLTFExporterService,
    OptimizerService,
    MetadataService
  ]
})
export class ExportModule {}