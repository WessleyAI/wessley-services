import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsObject, IsOptional, IsEnum, IsBoolean, IsNumber, IsArray } from 'class-validator';

export enum QualityLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high'
}

export class GraphQueryDto {
  @ApiProperty({ description: 'Vehicle signature/identifier to ensure data isolation' })
  @IsString()
  vehicleSignature: string;

  @ApiProperty({ description: 'Specific node IDs to include', required: false })
  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  nodeIds?: string[];

  @ApiProperty({ description: 'Custom Cypher query', required: false })
  @IsOptional()
  @IsString()
  cypher?: string;

  @ApiProperty({ description: 'Filter by component types', required: false })
  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  componentTypes?: string[];

  @ApiProperty({ description: 'Filter by vehicle zone', required: false })
  @IsOptional()
  @IsString()
  zoneFilter?: string;

  @ApiProperty({ description: 'Specific circuit to model', required: false })
  @IsOptional()
  @IsString()
  circuitId?: string;

  @ApiProperty({ description: 'Additional filters', required: false })
  @IsOptional()
  @IsObject()
  filters?: Record<string, any>;
}

export class ModelGenerationOptionsDto {
  @ApiProperty({ enum: QualityLevel, description: 'Output quality level' })
  @IsEnum(QualityLevel)
  quality: QualityLevel;

  @ApiProperty({ description: 'Include AI-ready metadata' })
  @IsBoolean()
  includeMetadata: boolean;

  @ApiProperty({ description: 'Optimize for web viewing' })
  @IsBoolean()
  optimizeForWeb: boolean;

  @ApiProperty({ description: 'Generate level of detail variants' })
  @IsBoolean()
  generateLOD: boolean;

  @ApiProperty({ description: 'Include component animations', required: false })
  @IsOptional()
  @IsBoolean()
  includeAnimations?: boolean;

  @ApiProperty({ description: 'Maximum file size in bytes', required: false })
  @IsOptional()
  @IsNumber()
  maxFileSize?: number;

  @ApiProperty({ description: 'Target triangle count', required: false })
  @IsOptional()
  @IsNumber()
  targetTriangles?: number;
}

export class CallbackConfigDto {
  @ApiProperty({ description: 'Webhook URL for completion notification', required: false })
  @IsOptional()
  @IsString()
  webhook?: string;

  @ApiProperty({ description: 'Events to send to webhook', required: false })
  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  events?: ('progress' | 'completion' | 'error')[];
}

export class ModelGenerationRequestDto {
  @ApiProperty({ description: 'Unique identifier for this generation request' })
  @IsString()
  requestId: string;

  @ApiProperty({ description: 'Graph query parameters' })
  @IsObject()
  graphQuery: GraphQueryDto;

  @ApiProperty({ description: 'Generation options' })
  @IsObject()
  options: ModelGenerationOptionsDto;

  @ApiProperty({ description: 'Callback configuration', required: false })
  @IsOptional()
  @IsObject()
  callback?: CallbackConfigDto;
}

export class ModelGenerationResponseDto {
  @ApiProperty({ description: 'Request success status' })
  success: boolean;

  @ApiProperty({ description: 'Job tracking ID' })
  jobId: string;

  @ApiProperty({ description: 'Original request ID' })
  requestId: string;

  @ApiProperty({ description: 'Current job status' })
  status: 'queued' | 'processing' | 'completed' | 'failed';

  @ApiProperty({ description: 'Estimated completion time in seconds', required: false })
  estimatedTime?: number;

  @ApiProperty({ description: 'Position in processing queue', required: false })
  queuePosition?: number;

  @ApiProperty({ description: 'Status message' })
  message: string;
}