export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface ComponentEntity {
  id: string;
  type: string;
  name?: string;
  description?: string;
  
  // Spatial properties
  position: Vector3 | [number, number, number];
  rotation?: Vector3 | [number, number, number];
  scale?: Vector3 | [number, number, number];
  
  // Physical properties
  dimensions?: {
    width: number;
    height: number;
    depth: number;
  };
  
  // Electrical properties
  voltage?: number;
  current?: number;
  power?: number;
  resistance?: number;
  
  // Metadata
  zone?: string;
  manufacturer?: string;
  partNumber?: string;
  material?: string;
  color?: string;
  
  // Custom properties
  properties?: Record<string, any>;
  
  // Relationships
  connections?: ConnectionEntity[];
  
  // 3D specific
  meshType?: 'box' | 'cylinder' | 'sphere' | 'custom';
  bbox?: [number, number, number];
  anchor_xyz?: [number, number, number];
  anchor_zone?: string;
}

export interface ConnectionEntity {
  id: string;
  fromComponentId: string;
  toComponentId: string;
  type: 'electrical' | 'mechanical' | 'data';
  
  // Wire properties
  wireGauge?: string;
  wireColor?: string;
  wireLength?: number;
  
  // Electrical properties
  voltage?: number;
  current?: number;
  signalType?: string;
  
  // Physical routing
  routePoints?: Vector3[];
  bendRadius?: number;
  
  // Metadata
  label?: string;
  notes?: string;
  properties?: Record<string, any>;
}

export interface ElectricalSystemData {
  id: string;
  name: string;
  description?: string;
  
  components: ComponentEntity[];
  connections: ConnectionEntity[];
  
  // System metadata
  voltage?: number;
  zones?: string[];
  circuits?: CircuitEntity[];
  
  // Bounding box for the entire system
  boundingBox?: {
    min: Vector3;
    max: Vector3;
  };
  
  metadata?: Record<string, any>;
}

export interface CircuitEntity {
  id: string;
  name: string;
  type: string;
  
  voltage: number;
  maxCurrent: number;
  
  componentIds: string[];
  connectionIds: string[];
  
  properties?: Record<string, any>;
}