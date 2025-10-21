import { Injectable } from '@nestjs/common';
import * as THREE from 'three';
import { ComponentEntity } from '../../common/entities/component.entity';
import { MaterialService } from './material.service';

export interface ComponentMeshOptions {
  quality: 'low' | 'medium' | 'high';
  generateHitZone: boolean;
  includeShadows: boolean;
}

@Injectable()
export class ComponentFactoryService {
  constructor(private readonly materialService: MaterialService) {}

  async generateComponentMesh(
    component: ComponentEntity,
    options: ComponentMeshOptions = {
      quality: 'medium',
      generateHitZone: true,
      includeShadows: true
    }
  ): Promise<THREE.Object3D> {
    const { geometry, needsHitZone } = this.createGeometry(component, options.quality);
    const material = this.materialService.getComponentMaterial(component.type);
    
    const mesh = new THREE.Mesh(geometry, material);
    
    // Set position
    const position = Array.isArray(component.position) 
      ? component.position 
      : [component.position.x, component.position.y, component.position.z];
    mesh.position.set(...position);
    
    // Set rotation if provided
    if (component.rotation) {
      const rotation = Array.isArray(component.rotation)
        ? component.rotation
        : [component.rotation.x, component.rotation.y, component.rotation.z];
      mesh.rotation.set(...rotation);
    }
    
    // Set scale if provided
    if (component.scale) {
      const scale = Array.isArray(component.scale)
        ? component.scale
        : [component.scale.x, component.scale.y, component.scale.z];
      mesh.scale.set(...scale);
    }
    
    // Configure shadows
    if (options.includeShadows) {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }
    
    // Set user data for identification
    mesh.userData = {
      componentId: component.id,
      componentType: component.type,
      component: component
    };
    
    mesh.name = `Component_${component.id}`;
    
    // Create hit zone for small components if needed
    if (needsHitZone && options.generateHitZone) {
      return this.createWithHitZone(mesh, component);
    }
    
    return mesh;
  }

  private createGeometry(component: ComponentEntity, quality: string): { geometry: THREE.BufferGeometry, needsHitZone: boolean } {
    const bbox = component.bbox || component.dimensions 
      ? [component.dimensions.width, component.dimensions.height, component.dimensions.depth]
      : [0.05, 0.05, 0.025];
    
    let geometry: THREE.BufferGeometry;
    let needsHitZone = false;
    
    // Get quality-dependent segment counts
    const segments = this.getSegmentCount(quality);
    
    switch (component.type) {
      case 'fuse':
        geometry = new THREE.CylinderGeometry(
          bbox[0] / 2, 
          bbox[0] / 2, 
          bbox[2], 
          segments.cylinder
        );
        break;
        
      case 'relay':
        geometry = new THREE.BoxGeometry(bbox[0], bbox[1], bbox[2]);
        break;
        
      case 'connector':
        geometry = new THREE.BoxGeometry(bbox[0], bbox[1], bbox[2]);
        break;
        
      case 'bus':
        geometry = new THREE.BoxGeometry(bbox[0], bbox[1], bbox[2]);
        break;
        
      case 'ground_point':
        geometry = new THREE.SphereGeometry(0.04, segments.sphere, segments.sphere / 2);
        break;
        
      case 'splice':
        geometry = this.createSpliceGeometry(component, segments);
        needsHitZone = true;
        break;
        
      case 'pin':
        geometry = new THREE.CylinderGeometry(0.005, 0.005, 0.015, segments.pin);
        needsHitZone = true;
        break;
        
      case 'component':
      default:
        geometry = new THREE.BoxGeometry(bbox[0], bbox[1], bbox[2]);
        break;
    }
    
    return { geometry, needsHitZone };
  }

  private createSpliceGeometry(component: ComponentEntity, segments: any): THREE.BufferGeometry {
    // Create a small junction box for splice connections
    const size = 0.02;
    const connections = component.connections?.length || 2;
    
    if (connections <= 2) {
      // Simple cylinder for 2-way splice
      return new THREE.CylinderGeometry(size/2, size/2, size, segments.cylinder);
    } else if (connections <= 4) {
      // Cross-shaped for 3-4 way splice
      return new THREE.BoxGeometry(size, size, size);
    } else {
      // Sphere for complex multi-way splice
      return new THREE.SphereGeometry(size, segments.sphere, segments.sphere / 2);
    }
  }

  private createWithHitZone(visualMesh: THREE.Mesh, component: ComponentEntity): THREE.Group {
    const group = new THREE.Group();
    group.name = `ComponentGroup_${component.id}`;
    
    // Create invisible hit zone for better interaction
    const hitZoneGeometry = new THREE.SphereGeometry(0.025, 8, 6);
    const hitZoneMaterial = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0,
      visible: false
    });
    
    const hitZone = new THREE.Mesh(hitZoneGeometry, hitZoneMaterial);
    hitZone.position.copy(visualMesh.position);
    hitZone.userData = {
      ...visualMesh.userData,
      visualMesh: visualMesh,
      isHitZone: true
    };
    hitZone.name = `HitZone_${component.id}`;
    
    group.add(visualMesh);
    group.add(hitZone);
    
    // Set group userData
    group.userData = visualMesh.userData;
    
    return group;
  }

  private getSegmentCount(quality: string) {
    switch (quality) {
      case 'high':
        return {
          cylinder: 16,
          sphere: 16,
          pin: 8
        };
      case 'medium':
        return {
          cylinder: 8,
          sphere: 12,
          pin: 6
        };
      case 'low':
      default:
        return {
          cylinder: 6,
          sphere: 8,
          pin: 4
        };
    }
  }

  async generateBatch(
    components: ComponentEntity[],
    options?: ComponentMeshOptions
  ): Promise<THREE.Object3D[]> {
    const meshes: THREE.Object3D[] = [];
    
    for (const component of components) {
      try {
        const mesh = await this.generateComponentMesh(component, options);
        meshes.push(mesh);
      } catch (error) {
        console.warn(`Failed to generate mesh for component ${component.id}:`, error.message);
        // Continue with other components
      }
    }
    
    return meshes;
  }
}