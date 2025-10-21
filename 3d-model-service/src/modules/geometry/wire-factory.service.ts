import { Injectable } from '@nestjs/common';
import * as THREE from 'three';
import { ConnectionEntity, Vector3 } from '../../common/entities/component.entity';
import { MaterialService } from './material.service';

export interface WireRoute {
  id: string;
  path: Vector3[];
  wireGauge: string;
  wireColor?: string;
  thickness: number;
  harnessType?: string;
}

export interface HarnessData {
  id: string;
  path: Vector3[];
  thickness: number;
  type: string;
}

export interface WireGenerationOptions {
  quality: 'low' | 'medium' | 'high';
  includeShadows: boolean;
  routingAlgorithm: 'direct' | 'physics' | 'optimized';
}

@Injectable()
export class WireFactoryService {
  constructor(private readonly materialService: MaterialService) {}

  async generateWireHarness(
    route: WireRoute,
    options: WireGenerationOptions = {
      quality: 'medium',
      includeShadows: true,
      routingAlgorithm: 'optimized'
    }
  ): Promise<THREE.Mesh> {
    if (route.path.length < 2) {
      throw new Error(`Wire route ${route.id} needs at least 2 points`);
    }

    // Convert path to THREE.Vector3 array
    const pathPoints = route.path.map(point => {
      if (Array.isArray(point)) {
        return new THREE.Vector3(point[0], point[1], point[2]);
      }
      return new THREE.Vector3(point.x, point.y, point.z);
    });

    // Create smooth curve through path points
    const curve = new THREE.CatmullRomCurve3(pathPoints);
    
    // Get quality-dependent tube segments
    const segments = this.getTubeSegments(options.quality);
    const radialSegments = this.getRadialSegments(options.quality);
    
    // Create tube geometry
    const tubeGeometry = new THREE.TubeGeometry(
      curve,
      segments,
      route.thickness / 2,
      radialSegments,
      false
    );

    // Get material based on wire color or harness type
    const material = route.wireColor
      ? this.materialService.getWireMaterial(route.wireColor)
      : this.materialService.getHarnessMaterial(route.harnessType || 'default');

    const mesh = new THREE.Mesh(tubeGeometry, material);
    
    // Configure shadows
    if (options.includeShadows) {
      mesh.castShadow = true;
    }

    // Set user data
    mesh.userData = {
      wireId: route.id,
      wireGauge: route.wireGauge,
      wireColor: route.wireColor,
      harnessType: route.harnessType,
      type: 'wire'
    };

    mesh.name = `Wire_${route.id}`;

    return mesh;
  }

  async generateHarnessMesh(
    harness: HarnessData,
    options: WireGenerationOptions = {
      quality: 'medium',
      includeShadows: true,
      routingAlgorithm: 'optimized'
    }
  ): Promise<THREE.Mesh> {
    if (harness.path.length < 2) {
      throw new Error(`Harness ${harness.id} needs at least 2 points`);
    }

    // Convert path to THREE.Vector3 array
    const pathPoints = harness.path.map(point => {
      if (Array.isArray(point)) {
        return new THREE.Vector3(point[0], point[1], point[2]);
      }
      return new THREE.Vector3(point.x, point.y, point.z);
    });

    // Create smooth curve
    const curve = new THREE.CatmullRomCurve3(pathPoints);
    
    // Get quality-dependent parameters
    const segments = this.getTubeSegments(options.quality);
    const radialSegments = this.getRadialSegments(options.quality);
    
    // Create tube geometry for harness
    const tubeGeometry = new THREE.TubeGeometry(
      curve,
      segments,
      harness.thickness / 2,
      radialSegments,
      false
    );

    // Get harness material based on type
    const material = this.materialService.getHarnessMaterial(harness.type);

    const mesh = new THREE.Mesh(tubeGeometry, material);
    
    // Configure shadows
    if (options.includeShadows) {
      mesh.castShadow = true;
    }

    // Set user data
    mesh.userData = {
      harnessId: harness.id,
      harnessType: harness.type,
      thickness: harness.thickness,
      type: 'harness'
    };

    mesh.name = `Harness_${harness.id}`;

    return mesh;
  }

  async generateDirectConnection(
    startPos: Vector3,
    endPos: Vector3,
    connection: ConnectionEntity,
    options?: WireGenerationOptions
  ): Promise<THREE.Mesh> {
    // Create simple direct connection for electrical wiring
    const start = Array.isArray(startPos) 
      ? new THREE.Vector3(startPos[0], startPos[1], startPos[2])
      : new THREE.Vector3(startPos.x, startPos.y, startPos.z);
      
    const end = Array.isArray(endPos)
      ? new THREE.Vector3(endPos[0], endPos[1], endPos[2])
      : new THREE.Vector3(endPos.x, endPos.y, endPos.z);

    // Create simple line geometry for direct connections
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    
    // Use line material for simple connections
    const material = new THREE.LineBasicMaterial({
      color: this.getWireColorHex(connection.wireColor),
      opacity: 0.8,
      transparent: true,
      linewidth: this.getWireThickness(connection.wireGauge || '18AWG')
    });

    const line = new THREE.Line(geometry, material);
    
    line.userData = {
      connectionId: connection.id,
      fromComponent: connection.fromComponentId,
      toComponent: connection.toComponentId,
      wireGauge: connection.wireGauge,
      wireColor: connection.wireColor,
      type: 'connection_wire'
    };

    line.name = `Connection_${connection.id}`;

    return line as any; // TypeScript workaround for Line vs Mesh
  }

  async generateCompleteCircuit(
    connections: ConnectionEntity[],
    componentPositions: Map<string, Vector3>,
    options?: WireGenerationOptions
  ): Promise<THREE.Group> {
    const circuitGroup = new THREE.Group();
    circuitGroup.name = 'circuit';

    for (const connection of connections) {
      const startPos = componentPositions.get(connection.fromComponentId);
      const endPos = componentPositions.get(connection.toComponentId);

      if (startPos && endPos) {
        try {
          const wire = await this.generateDirectConnection(startPos, endPos, connection, options);
          circuitGroup.add(wire);
        } catch (error) {
          console.warn(`Failed to generate wire for connection ${connection.id}:`, error.message);
        }
      }
    }

    return circuitGroup;
  }

  // Helper methods
  private getTubeSegments(quality: string): number {
    switch (quality) {
      case 'high': return 64;
      case 'medium': return 32;
      case 'low': default: return 16;
    }
  }

  private getRadialSegments(quality: string): number {
    switch (quality) {
      case 'high': return 8;
      case 'medium': return 6;
      case 'low': default: return 4;
    }
  }

  private getWireThickness(gauge: string): number {
    const gaugeMap: Record<string, number> = {
      '22AWG': 1,
      '20AWG': 1.5,
      '18AWG': 2,
      '16AWG': 2.5,
      '14AWG': 3,
      '12AWG': 3.5,
      '10AWG': 4,
      '8AWG': 5
    };
    return gaugeMap[gauge] || 2;
  }

  private getWireColorHex(wireColor?: string): number {
    if (!wireColor) return 0x666666;
    
    const colorMap: Record<string, number> = {
      red: 0xff0000,
      black: 0x000000,
      blue: 0x0000ff,
      green: 0x00ff00,
      yellow: 0xffff00,
      white: 0xffffff,
      orange: 0xff8000,
      purple: 0x800080,
      brown: 0x964B00,
      pink: 0xffc0cb,
      gray: 0x808080,
      grey: 0x808080
    };

    const normalized = wireColor.toLowerCase().replace(/[\s_-]/g, '');
    return colorMap[normalized] || 0x666666;
  }

  // Batch generation methods
  async generateWireBatch(
    routes: WireRoute[],
    options?: WireGenerationOptions
  ): Promise<THREE.Object3D[]> {
    const wires: THREE.Object3D[] = [];
    
    for (const route of routes) {
      try {
        const wire = await this.generateWireHarness(route, options);
        wires.push(wire);
      } catch (error) {
        console.warn(`Failed to generate wire for route ${route.id}:`, error.message);
      }
    }
    
    return wires;
  }

  async generateHarnessBatch(
    harnesses: HarnessData[],
    options?: WireGenerationOptions
  ): Promise<THREE.Object3D[]> {
    const meshes: THREE.Object3D[] = [];
    
    for (const harness of harnesses) {
      try {
        const mesh = await this.generateHarnessMesh(harness, options);
        meshes.push(mesh);
      } catch (error) {
        console.warn(`Failed to generate harness ${harness.id}:`, error.message);
      }
    }
    
    return meshes;
  }
}