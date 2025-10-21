import { ComponentEntity, ConnectionEntity, Vector3 } from '../entities/component.entity';

export interface I3DEngine {
  /**
   * Generate a 3D mesh for a component
   */
  generateComponentMesh(component: ComponentEntity, options?: MeshGenerationOptions): Promise<any>;
  
  /**
   * Generate wire harness meshes for connections
   */
  generateWireHarnesses(connections: ConnectionEntity[], options?: WireGenerationOptions): Promise<any[]>;
  
  /**
   * Compose a complete 3D scene
   */
  composeScene(components: any[], wires: any[], options?: SceneCompositionOptions): Promise<any>;
  
  /**
   * Export scene to GLB format
   */
  exportToGLB(scene: any, options?: ExportOptions): Promise<Buffer>;
}

export interface MeshGenerationOptions {
  quality: 'low' | 'medium' | 'high';
  generateLOD: boolean;
  includeMaterials: boolean;
  customGeometry?: any;
}

export interface WireGenerationOptions {
  routingAlgorithm: 'direct' | 'physics' | 'optimized';
  bendRadius: number;
  thickness: number;
  materialType: 'copper' | 'aluminum' | 'fiber';
}

export interface SceneCompositionOptions {
  includeLighting: boolean;
  includeEnvironment: boolean;
  optimizeForWeb: boolean;
  backgroundColor?: string;
}

export interface ExportOptions {
  binary: boolean;
  includeAnimations: boolean;
  embedTextures: boolean;
  compressGeometry: boolean;
  generateLOD: boolean;
  maxFileSize?: number;
}

export interface IProcessor {
  /**
   * Process/optimize spatial layout
   */
  optimize(data: any, constraints?: any): Promise<any>;
  
  /**
   * Validate processing results
   */
  validate(result: any): Promise<boolean>;
}

export interface IExporter {
  /**
   * Export to specific format
   */
  export(data: any, format: string, options?: any): Promise<Buffer>;
  
  /**
   * Get supported export formats
   */
  getSupportedFormats(): string[];
}