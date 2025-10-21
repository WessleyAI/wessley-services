import { ComponentEntity } from '@/common/entities/component.entity';
import { WireConnectionEntity } from '@/common/entities/wire-connection.entity';

export const mockVehicleSignature = 'test_vehicle_pajero_2001';

export const mockComponents: ComponentEntity[] = [
  {
    id: 'relay_main_001',
    name: 'Main Power Relay',
    type: 'relay',
    vehicleSignature: mockVehicleSignature,
    position: { x: 100, y: 200, z: 300 },
    specifications: {
      voltage: 12,
      current: 30,
      manufacturer: 'Bosch',
      partNumber: 'BO-REL-001'
    }
  },
  {
    id: 'fuse_main_001',
    name: 'Main Fuse 30A',
    type: 'fuse',
    vehicleSignature: mockVehicleSignature,
    position: { x: 150, y: 200, z: 300 },
    specifications: {
      rating: '30A',
      type: 'blade',
      color: 'green'
    }
  },
  {
    id: 'sensor_temp_001',
    name: 'Engine Temperature Sensor',
    type: 'sensor',
    vehicleSignature: mockVehicleSignature,
    position: { x: 200, y: 250, z: 350 },
    specifications: {
      sensorType: 'temperature',
      range: '-40 to 150°C',
      resistance: '2.3kΩ'
    }
  },
  {
    id: 'ecu_engine_001',
    name: 'Engine Control Unit',
    type: 'ecu',
    vehicleSignature: mockVehicleSignature,
    position: { x: 300, y: 400, z: 200 },
    specifications: {
      processor: 'ARM Cortex-M4',
      memory: '2MB Flash',
      canBus: true
    }
  },
  {
    id: 'connector_main_001',
    name: 'Main Harness Connector',
    type: 'connector',
    vehicleSignature: mockVehicleSignature,
    position: { x: 250, y: 300, z: 250 },
    specifications: {
      pinCount: 24,
      type: 'AMP Superseal',
      waterproof: true
    }
  }
];

export const mockWireConnections: WireConnectionEntity[] = [
  {
    id: 'wire_001',
    startComponent: mockComponents[0], // relay
    endComponent: mockComponents[1],   // fuse
    wireGauge: '2.5mm²',
    wireColor: 'red',
    length: 250,
    signalType: 'power'
  },
  {
    id: 'wire_002',
    startComponent: mockComponents[1], // fuse
    endComponent: mockComponents[4],   // connector
    wireGauge: '1.5mm²',
    wireColor: 'blue',
    length: 180,
    signalType: 'power'
  },
  {
    id: 'wire_003',
    startComponent: mockComponents[4], // connector
    endComponent: mockComponents[2],   // sensor
    wireGauge: '0.75mm²',
    wireColor: 'yellow',
    length: 120,
    signalType: 'signal'
  },
  {
    id: 'wire_004',
    startComponent: mockComponents[4], // connector
    endComponent: mockComponents[3],   // ecu
    wireGauge: '1.0mm²',
    wireColor: 'green',
    length: 200,
    signalType: 'data'
  }
];

export const mockCircuitData = {
  id: 'circuit_main_001',
  name: 'Main Power Circuit',
  vehicleSignature: mockVehicleSignature,
  voltage: 12.0,
  maxCurrent: 30.0,
  components: [mockComponents[0], mockComponents[1], mockComponents[4]]
};

export const mockSpatialLayout = {
  vehicleSignature: mockVehicleSignature,
  components: mockComponents,
  connections: mockWireConnections,
  boundingBox: {
    min: { x: 100, y: 200, z: 200 },
    max: { x: 300, y: 400, z: 350 },
    size: { x: 200, y: 200, z: 150 }
  }
};

export const mockValidationResult = {
  vehicleSignature: mockVehicleSignature,
  totalComponents: 5,
  componentsWithSpatialData: 5,
  componentsWithoutSpatialData: 0,
  totalConnections: 4,
  unpoweredComponents: 0,
  orphanedConnectors: 0,
  integrityScore: 100
};

export function createMockComponent(overrides: Partial<ComponentEntity> = {}): ComponentEntity {
  return {
    id: 'mock_component_001',
    name: 'Mock Component',
    type: 'connector',
    vehicleSignature: mockVehicleSignature,
    position: { x: 0, y: 0, z: 0 },
    specifications: {},
    ...overrides
  };
}

export function createMockWireConnection(overrides: Partial<WireConnectionEntity> = {}): WireConnectionEntity {
  return {
    id: 'mock_wire_001',
    startComponent: mockComponents[0],
    endComponent: mockComponents[1],
    wireGauge: '1.5mm²',
    wireColor: 'black',
    length: 100,
    signalType: 'power',
    ...overrides
  };
}