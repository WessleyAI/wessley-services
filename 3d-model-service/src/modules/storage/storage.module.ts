import { Module } from '@nestjs/common';
import { S3Service } from './s3.service';
import { CDNService } from './cdn.service';
import { UploadService } from './upload.service';

@Module({
  providers: [
    S3Service,
    CDNService,
    UploadService
  ],
  exports: [
    S3Service,
    CDNService,
    UploadService
  ]
})
export class StorageModule {}