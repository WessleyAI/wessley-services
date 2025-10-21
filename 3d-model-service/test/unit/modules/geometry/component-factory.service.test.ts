import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ComponentFactoryService } from '@/modules/geometry/component-factory.service';
import { ComponentEntity } from '@/common/entities/component.entity';
import * as THREE from 'three';

// Mock Three.js objects
vi.mock('three', () => ({
  BoxGeometry: vi.fn(() => ({ type: 'BoxGeometry' })),
  CylinderGeometry: vi.fn(() => ({ type: 'CylinderGeometry' })),
  SphereGeometry: vi.fn(() => ({ type: 'SphereGeometry' })),
  PlaneGeometry: vi.fn(() => ({ type: 'PlaneGeometry' })),
  MeshStandardMaterial: vi.fn(() => ({ type: 'MeshStandardMaterial' })),
  Mesh: vi.fn(() => ({ type: 'Mesh', position: { set: vi.fn() }, userData: {} })),
  Vector3: vi.fn(() => ({ x: 0, y: 0, z: 0 })),
}));

describe('ComponentFactoryService', () => {
  let service: ComponentFactoryService;

  beforeEach(() => {
    service = new ComponentFactoryService();
    vi.clearAllMocks();
  });

  describe('createComponent', () => {
    it('should create relay component with correct geometry', () => {
      const component: ComponentEntity = {
        id: 'relay_001',
        name: 'Main Relay',
        type: 'relay',
        vehicleSignature: 'test_vehicle',
        position: { x: 10, y: 20, z: 30 },
        specifications: {}
      };

      const result = service.createComponent(component);

      expect(THREE.BoxGeometry).toHaveBeenCalledWith(20, 15, 10);
      expect(result.position.set).toHaveBeenCalledWith(10, 20, 30);
      expect(result.userData).toEqual({
        componentId: 'relay_001',
        componentType: 'relay',
        vehicleSignature: 'test_vehicle',
        specifications: {}
      });
    });

    it('should create fuse component with cylinder geometry', () => {
      const component: ComponentEntity = {
        id: 'fuse_001',
        name: 'Main Fuse',
        type: 'fuse',
        vehicleSignature: 'test_vehicle',
        position: { x: 0, y: 0, z: 0 },
        specifications: { rating: '30A' }
      };

      service.createComponent(component);

      expect(THREE.CylinderGeometry).toHaveBeenCalledWith(3, 3, 15, 8);
    });

    it('should create sensor component with sphere geometry', () => {
      const component: ComponentEntity = {
        id: 'sensor_001',
        name: 'Temperature Sensor',
        type: 'sensor',
        vehicleSignature: 'test_vehicle',
        position: { x: 5, y: 5, z: 5 },
        specifications: {}
      };

      service.createComponent(component);

      expect(THREE.SphereGeometry).toHaveBeenCalledWith(8, 16, 12);
    });

    it('should create ECU component with large box geometry', () => {
      const component: ComponentEntity = {
        id: 'ecu_001',
        name: 'Engine Control Unit',
        type: 'ecu',
        vehicleSignature: 'test_vehicle',
        position: { x: 0, y: 0, z: 0 },
        specifications: {}
      };

      service.createComponent(component);

      expect(THREE.BoxGeometry).toHaveBeenCalledWith(80, 60, 20);
    });

    it('should use default geometry for unknown component types', () => {
      const component: ComponentEntity = {
        id: 'unknown_001',
        name: 'Unknown Component',
        type: 'unknown_type',
        vehicleSignature: 'test_vehicle',
        position: { x: 0, y: 0, z: 0 },
        specifications: {}
      };

      service.createComponent(component);

      expect(THREE.BoxGeometry).toHaveBeenCalledWith(12, 10, 8);
    });

    it('should throw error for missing position data', () => {
      const component: ComponentEntity = {
        id: 'comp_001',
        name: 'Component',
        type: 'relay',
        vehicleSignature: 'test_vehicle',
        position: null as any,
        specifications: {}
      };

      expect(() => {
        service.createComponent(component);
      }).toThrow('Component position is required for 3D generation');
    });

    it('should throw error for invalid position coordinates', () => {
      const component: ComponentEntity = {
        id: 'comp_001',
        name: 'Component',
        type: 'relay',
        vehicleSignature: 'test_vehicle',
        position: { x: NaN, y: 0, z: 0 },
        specifications: {}
      };

      expect(() => {
        service.createComponent(component);
      }).toThrow('Invalid position coordinates');
    });
  });

  describe('getComponentColor', () => {
    it('should return correct colors for component types', () => {
      expect(service.getComponentColor('relay')).toBe(0x4A90E2);
      expect(service.getComponentColor('fuse')).toBe(0xF5A623);
      expect(service.getComponentColor('connector')).toBe(0x7ED321);
      expect(service.getComponentColor('sensor')).toBe(0xD0021B);
      expect(service.getComponentColor('ecu')).toBe(0x9013FE);
    });

    it('should return default color for unknown types', () => {
      expect(service.getComponentColor('unknown')).toBe(0xCCCCCC);
      expect(service.getComponentColor('')).toBe(0xCCCCCC);
      expect(service.getComponentColor(null as any)).toBe(0xCCCCCC);
    });
  });

  describe('createMaterial', () => {
    it('should create material with correct properties', () => {
      const component: ComponentEntity = {
        id: 'comp_001',
        name: 'Component',
        type: 'relay',
        vehicleSignature: 'test_vehicle',
        position: { x: 0, y: 0, z: 0 },
        specifications: {}
      };

      service.createComponent(component);

      expect(THREE.MeshStandardMaterial).toHaveBeenCalledWith({
        color: 0x4A90E2,
        metalness: 0.3,
        roughness: 0.7,
        name: 'material_relay'
      });
    });
  });

  describe('batch component creation', () => {
    it('should create multiple components efficiently', () => {
      const components: ComponentEntity[] = [
        {
          id: 'relay_001',
          name: 'Relay 1',
          type: 'relay',
          vehicleSignature: 'test_vehicle',
          position: { x: 0, y: 0, z: 0 },
          specifications: {}
        },
        {
          id: 'fuse_001',
          name: 'Fuse 1',
          type: 'fuse',
          vehicleSignature: 'test_vehicle',
          position: { x: 10, y: 0, z: 0 },
          specifications: {}
        }
      ];

      const results = service.createComponents(components);

      expect(results).toHaveLength(2);
      expect(THREE.BoxGeometry).toHaveBeenCalledTimes(1);
      expect(THREE.CylinderGeometry).toHaveBeenCalledTimes(1);
    });

    it('should handle errors in batch creation gracefully', () => {
      const components: ComponentEntity[] = [
        {
          id: 'valid_001',
          name: 'Valid Component',
          type: 'relay',
          vehicleSignature: 'test_vehicle',
          position: { x: 0, y: 0, z: 0 },
          specifications: {}
        },
        {
          id: 'invalid_001',
          name: 'Invalid Component',
          type: 'relay',
          vehicleSignature: 'test_vehicle',
          position: null as any,
          specifications: {}
        }
      ];

      expect(() => {
        service.createComponents(components);
      }).toThrow('Failed to create component invalid_001');
    });
  });

  describe('performance optimization', () => {
    it('should reuse materials for same component types', () => {
      const components: ComponentEntity[] = [
        {
          id: 'relay_001',
          name: 'Relay 1',
          type: 'relay',
          vehicleSignature: 'test_vehicle',
          position: { x: 0, y: 0, z: 0 },
          specifications: {}
        },
        {
          id: 'relay_002',
          name: 'Relay 2',
          type: 'relay',
          vehicleSignature: 'test_vehicle',
          position: { x: 10, y: 0, z: 0 },
          specifications: {}
        }
      ];

      service.createComponents(components);

      // Material should only be created once for same type
      expect(THREE.MeshStandardMaterial).toHaveBeenCalledTimes(1);
    });

    it('should validate component data before expensive operations', () => {
      const invalidComponent: ComponentEntity = {
        id: '',
        name: '',
        type: 'relay',
        vehicleSignature: '',
        position: { x: 0, y: 0, z: 0 },
        specifications: {}
      };

      expect(() => {
        service.createComponent(invalidComponent);
      }).toThrow('Component ID and vehicle signature are required');
    });
  });
});