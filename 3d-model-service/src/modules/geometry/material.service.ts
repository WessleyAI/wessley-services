import { Injectable } from '@nestjs/common';
import * as THREE from 'three';

export interface MaterialConfig {
  color: number;
  opacity: number;
  transparent: boolean;
  metalness?: number;
  roughness?: number;
  emissive?: number;
}

@Injectable()
export class MaterialService {
  private materials: Map<string, THREE.Material> = new Map();
  private originalMaterials: Map<THREE.Object3D, THREE.Material> = new Map();

  constructor() {
    this.initializeMaterials();
  }

  private initializeMaterials() {
    // Component materials (extracted from enhanced-r3f-viewer)
    const componentMaterials = {
      default: { color: 0x4fc3f7, opacity: 0.8, transparent: true },
      fuse: { color: 0xff9800, opacity: 0.8, transparent: true },
      relay: { color: 0x4caf50, opacity: 0.8, transparent: true },
      bus: { color: 0xe91e63, opacity: 0.8, transparent: true },
      ground_point: { color: 0x424242, opacity: 0.8, transparent: true },
      connector: { color: 0x9c27b0, opacity: 0.8, transparent: true },
      splice: { color: 0xf44336, opacity: 0.9, transparent: true },
      pin: { color: 0xffc107, opacity: 0.9, transparent: true },
      component: { color: 0x4fc3f7, opacity: 0.8, transparent: true }
    };

    // Wire/harness materials
    const harnessMaterials = {
      engine: { color: 0xff5722, opacity: 0.6, transparent: true },
      dash: { color: 0x2196f3, opacity: 0.6, transparent: true },
      floor: { color: 0x4caf50, opacity: 0.6, transparent: true },
      door_left: { color: 0x9c27b0, opacity: 0.6, transparent: true },
      door_right: { color: 0xff9800, opacity: 0.6, transparent: true },
      tailgate: { color: 0x795548, opacity: 0.6, transparent: true },
      default_wire: { color: 0x666666, opacity: 0.7, transparent: true }
    };

    // Special materials
    const specialMaterials = {
      selected: { color: 0xffeb3b, opacity: 1.0, transparent: false },
      highlighted: { color: 0x00ff00, opacity: 0.9, transparent: true },
      error: { color: 0xff0000, opacity: 0.9, transparent: true },
      invisible: { color: 0x000000, opacity: 0, transparent: true }
    };

    // Create component materials
    Object.entries(componentMaterials).forEach(([key, config]) => {
      this.materials.set(`component_${key}`, this.createMaterial(config));
    });

    // Create harness materials
    Object.entries(harnessMaterials).forEach(([key, config]) => {
      this.materials.set(`harness_${key}`, this.createMaterial(config));
    });

    // Create special materials
    Object.entries(specialMaterials).forEach(([key, config]) => {
      this.materials.set(`special_${key}`, this.createMaterial(config));
    });

    // Create wire color materials (common wire colors)
    const wireColors = {
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

    Object.entries(wireColors).forEach(([color, hex]) => {
      this.materials.set(`wire_${color}`, this.createMaterial({
        color: hex,
        opacity: 0.8,
        transparent: true
      }));
    });
  }

  private createMaterial(config: MaterialConfig): THREE.Material {
    // Use MeshLambertMaterial for good performance with lighting
    return new THREE.MeshLambertMaterial({
      color: config.color,
      opacity: config.opacity,
      transparent: config.transparent,
      side: THREE.DoubleSide
    });
  }

  getComponentMaterial(componentType: string): THREE.Material {
    const materialKey = `component_${componentType}`;
    return this.materials.get(materialKey) || this.materials.get('component_default')!;
  }

  getHarnessMaterial(harnessType: string): THREE.Material {
    // Determine harness type from ID or zone
    let materialType = 'default_wire';
    
    if (harnessType.includes('dash')) materialType = 'dash';
    else if (harnessType.includes('floor')) materialType = 'floor';
    else if (harnessType.includes('door') && harnessType.includes('L')) materialType = 'door_left';
    else if (harnessType.includes('door') && harnessType.includes('R')) materialType = 'door_right';
    else if (harnessType.includes('tailgate')) materialType = 'tailgate';
    else if (harnessType.includes('engine')) materialType = 'engine';

    const materialKey = `harness_${materialType}`;
    return this.materials.get(materialKey) || this.materials.get('harness_default_wire')!;
  }

  getWireMaterial(wireColor?: string): THREE.Material {
    if (wireColor) {
      const normalizedColor = wireColor.toLowerCase().replace(/[\s_-]/g, '');
      const materialKey = `wire_${normalizedColor}`;
      const material = this.materials.get(materialKey);
      if (material) return material;
    }
    
    return this.materials.get('harness_default_wire')!;
  }

  getSpecialMaterial(type: 'selected' | 'highlighted' | 'error' | 'invisible'): THREE.Material {
    return this.materials.get(`special_${type}`)!;
  }

  // Material state management for hover/selection effects
  storeOriginalMaterial(object: THREE.Object3D, material: THREE.Material) {
    this.originalMaterials.set(object, material);
  }

  getOriginalMaterial(object: THREE.Object3D): THREE.Material | undefined {
    return this.originalMaterials.get(object);
  }

  restoreOriginalMaterial(object: THREE.Object3D) {
    const original = this.originalMaterials.get(object);
    if (original && object instanceof THREE.Mesh) {
      object.material = original;
    }
  }

  // Apply temporary material (for hover/selection)
  applyTemporaryMaterial(object: THREE.Object3D, materialType: 'selected' | 'highlighted' | 'error') {
    if (object instanceof THREE.Mesh) {
      // Store original if not already stored
      if (!this.originalMaterials.has(object)) {
        this.storeOriginalMaterial(object, object.material as THREE.Material);
      }
      
      object.material = this.getSpecialMaterial(materialType);
    }
  }

  // Create custom material with specific properties
  createCustomMaterial(config: MaterialConfig): THREE.Material {
    if (config.metalness !== undefined || config.roughness !== undefined) {
      // Use PBR material for metallic/rough surfaces
      return new THREE.MeshStandardMaterial({
        color: config.color,
        opacity: config.opacity,
        transparent: config.transparent,
        metalness: config.metalness || 0,
        roughness: config.roughness || 0.5,
        emissive: config.emissive || 0x000000
      });
    } else {
      return this.createMaterial(config);
    }
  }

  // Get all available material types
  getAvailableMaterials(): string[] {
    return Array.from(this.materials.keys());
  }

  // Cleanup
  dispose() {
    this.materials.forEach(material => material.dispose());
    this.materials.clear();
    this.originalMaterials.clear();
  }
}