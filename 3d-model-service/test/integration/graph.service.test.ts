import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Test, TestingModule } from '@nestjs/testing';
import { ConfigModule } from '@nestjs/config';
import { GraphService } from '@/modules/graph/graph.service';
import { Neo4jService } from '@/modules/graph/neo4j.service';
import { QueryBuilder } from '@/modules/graph/query.builder';
import neo4j from 'neo4j-driver';

describe('GraphService Integration', () => {
  let service: GraphService;
  let neo4jService: Neo4jService;
  let module: TestingModule;
  let driver: neo4j.Driver;

  const testVehicleSignature = 'test_integration_vehicle';

  beforeEach(async () => {
    module = await Test.createTestingModule({
      imports: [
        ConfigModule.forRoot({
          isGlobal: true,
          envFilePath: '.env.test',
        }),
      ],
      providers: [
        GraphService,
        Neo4jService,
        QueryBuilder,
      ],
    }).compile();

    service = module.get<GraphService>(GraphService);
    neo4jService = module.get<Neo4jService>(Neo4jService);

    // Initialize driver
    driver = neo4j.driver(
      process.env.NEO4J_URI!,
      neo4j.auth.basic(process.env.NEO4J_USERNAME!, process.env.NEO4J_PASSWORD!)
    );

    await neo4jService.onModuleInit();
    await setupTestData();
  });

  afterEach(async () => {
    await cleanupTestData();
    await neo4jService.onModuleDestroy();
    await module.close();
  });

  async function setupTestData() {
    const session = driver.session();
    
    try {
      // Create test vehicle
      await session.run(`
        CREATE (v:Vehicle {
          signature: $vehicleSignature,
          make: 'TestMake',
          model: 'TestModel',
          year: 2023
        })
      `, { vehicleSignature: testVehicleSignature });

      // Create test components
      await session.run(`
        CREATE 
        (c1:Component {
          id: 'test_relay_001',
          name: 'Test Main Relay',
          type: 'relay',
          vehicle_signature: $vehicleSignature,
          position: [100.0, 200.0, 300.0],
          specifications: {voltage: 12, current: 30}
        }),
        (c2:Component {
          id: 'test_fuse_001',
          name: 'Test Main Fuse',
          type: 'fuse',
          vehicle_signature: $vehicleSignature,
          position: [150.0, 200.0, 300.0],
          specifications: {rating: '30A'}
        }),
        (c3:Component {
          id: 'test_sensor_001',
          name: 'Test Temperature Sensor',
          type: 'sensor',
          vehicle_signature: $vehicleSignature,
          position: [200.0, 250.0, 350.0],
          specifications: {type: 'temperature'}
        })
      `, { vehicleSignature: testVehicleSignature });

      // Create test connections
      await session.run(`
        MATCH 
        (c1:Component {id: 'test_relay_001', vehicle_signature: $vehicleSignature}),
        (c2:Component {id: 'test_fuse_001', vehicle_signature: $vehicleSignature}),
        (c3:Component {id: 'test_sensor_001', vehicle_signature: $vehicleSignature})
        CREATE 
        (c1)-[:CONNECTS_TO {wire_gauge: '2.5mm²', wire_color: 'red'}]->(c2),
        (c2)-[:CONNECTS_TO {wire_gauge: '1.5mm²', wire_color: 'blue'}]->(c3)
      `, { vehicleSignature: testVehicleSignature });

      // Create test circuit
      await session.run(`
        CREATE (circuit:Circuit {
          id: 'test_circuit_001',
          name: 'Test Main Circuit',
          vehicle_signature: $vehicleSignature,
          voltage: 12.0,
          max_current: 30.0
        })
      `, { vehicleSignature: testVehicleSignature });

      // Link components to circuit
      await session.run(`
        MATCH 
        (circuit:Circuit {id: 'test_circuit_001', vehicle_signature: $vehicleSignature}),
        (c1:Component {id: 'test_relay_001', vehicle_signature: $vehicleSignature}),
        (c2:Component {id: 'test_fuse_001', vehicle_signature: $vehicleSignature})
        CREATE 
        (c1)-[:PART_OF {role: 'control'}]->(circuit),
        (c2)-[:PART_OF {role: 'protection'}]->(circuit)
      `, { vehicleSignature: testVehicleSignature });

    } finally {
      await session.close();
    }
  }

  async function cleanupTestData() {
    const session = driver.session();
    
    try {
      await session.run(`
        MATCH (n {vehicle_signature: $vehicleSignature})
        DETACH DELETE n
      `, { vehicleSignature: testVehicleSignature });

      await session.run(`
        MATCH (v:Vehicle {signature: $vehicleSignature})
        DELETE v
      `, { vehicleSignature: testVehicleSignature });
    } finally {
      await session.close();
    }
  }

  describe('getComponentsWithSpatialData', () => {
    it('should retrieve components with spatial coordinates', async () => {
      const result = await service.getComponentsWithSpatialData(testVehicleSignature);

      expect(result).toHaveLength(3);
      expect(result[0]).toMatchObject({
        id: 'test_relay_001',
        name: 'Test Main Relay',
        type: 'relay',
        vehicleSignature: testVehicleSignature,
        position: { x: 100, y: 200, z: 300 },
        specifications: { voltage: 12, current: 30 }
      });
    });

    it('should only return components for specified vehicle', async () => {
      // Create component for different vehicle
      const session = driver.session();
      await session.run(`
        CREATE (c:Component {
          id: 'other_vehicle_component',
          name: 'Other Vehicle Component',
          type: 'relay',
          vehicle_signature: 'other_vehicle',
          position: [0.0, 0.0, 0.0]
        })
      `);
      await session.close();

      const result = await service.getComponentsWithSpatialData(testVehicleSignature);

      expect(result).toHaveLength(3);
      expect(result.every(c => c.vehicleSignature === testVehicleSignature)).toBe(true);
    });

    it('should handle empty results gracefully', async () => {
      const result = await service.getComponentsWithSpatialData('nonexistent_vehicle');

      expect(result).toEqual([]);
    });
  });

  describe('getWireConnections', () => {
    it('should retrieve wire connections with spatial data', async () => {
      const result = await service.getWireConnections(testVehicleSignature);

      expect(result).toHaveLength(2);
      
      const connection = result.find(c => 
        c.startComponent.id === 'test_relay_001' && 
        c.endComponent.id === 'test_fuse_001'
      );

      expect(connection).toMatchObject({
        startComponent: {
          id: 'test_relay_001',
          position: { x: 100, y: 200, z: 300 }
        },
        endComponent: {
          id: 'test_fuse_001',
          position: { x: 150, y: 200, z: 300 }
        },
        wireGauge: '2.5mm²',
        wireColor: 'red'
      });
    });

    it('should enforce vehicle signature isolation', async () => {
      const result = await service.getWireConnections(testVehicleSignature);

      // All components in connections should belong to the same vehicle
      result.forEach(connection => {
        expect(connection.startComponent.vehicleSignature).toBe(testVehicleSignature);
        expect(connection.endComponent.vehicleSignature).toBe(testVehicleSignature);
      });
    });
  });

  describe('analyzeCircuit', () => {
    it('should analyze circuit with component relationships', async () => {
      const result = await service.analyzeCircuit(testVehicleSignature, 'Test Main Circuit');

      expect(result).toMatchObject({
        circuit: {
          id: 'test_circuit_001',
          name: 'Test Main Circuit',
          vehicleSignature: testVehicleSignature,
          voltage: 12.0,
          maxCurrent: 30.0
        }
      });

      expect(result.components).toHaveLength(2);
      expect(result.components.some(c => c.id === 'test_relay_001')).toBe(true);
      expect(result.components.some(c => c.id === 'test_fuse_001')).toBe(true);
    });

    it('should handle nonexistent circuits', async () => {
      const result = await service.analyzeCircuit(testVehicleSignature, 'Nonexistent Circuit');

      expect(result).toBeNull();
    });
  });

  describe('validateSystemIntegrity', () => {
    it('should validate system integrity and return statistics', async () => {
      const result = await service.validateSystemIntegrity(testVehicleSignature);

      expect(result).toMatchObject({
        vehicleSignature: testVehicleSignature,
        totalComponents: 3,
        componentsWithSpatialData: 3,
        totalConnections: 2,
        unpoweredComponents: expect.any(Number),
        orphanedConnectors: expect.any(Number)
      });

      expect(result.totalComponents).toBeGreaterThan(0);
      expect(result.componentsWithSpatialData).toBe(result.totalComponents);
    });

    it('should detect integrity issues', async () => {
      // Create component without spatial data
      const session = driver.session();
      await session.run(`
        CREATE (c:Component {
          id: 'test_no_position',
          name: 'Component Without Position',
          type: 'relay',
          vehicle_signature: $vehicleSignature
        })
      `, { vehicleSignature: testVehicleSignature });
      await session.close();

      const result = await service.validateSystemIntegrity(testVehicleSignature);

      expect(result.totalComponents).toBe(4);
      expect(result.componentsWithSpatialData).toBe(3);
      expect(result.componentsWithoutSpatialData).toBe(1);
    });
  });

  describe('performance and scalability', () => {
    it('should handle large datasets efficiently', async () => {
      const session = driver.session();
      
      // Create 100 test components
      const createQuery = `
        UNWIND range(1, 100) as i
        CREATE (c:Component {
          id: 'perf_test_' + i,
          name: 'Performance Test Component ' + i,
          type: 'connector',
          vehicle_signature: $vehicleSignature,
          position: [i * 10.0, 0.0, 0.0]
        })
      `;
      
      await session.run(createQuery, { vehicleSignature: testVehicleSignature });
      await session.close();

      const startTime = Date.now();
      const result = await service.getComponentsWithSpatialData(testVehicleSignature);
      const endTime = Date.now();

      expect(result).toHaveLength(103); // 3 original + 100 new
      expect(endTime - startTime).toBeLessThan(5000); // Should complete within 5 seconds
    });

    it('should use connection pooling effectively', async () => {
      // Multiple concurrent requests should not cause connection issues
      const promises = Array.from({ length: 10 }, () =>
        service.getComponentsWithSpatialData(testVehicleSignature)
      );

      const results = await Promise.all(promises);

      results.forEach(result => {
        expect(result).toHaveLength(3);
      });
    });
  });

  describe('error handling', () => {
    it('should handle database connection errors gracefully', async () => {
      // Temporarily close the connection
      await neo4jService.onModuleDestroy();

      await expect(
        service.getComponentsWithSpatialData(testVehicleSignature)
      ).rejects.toThrow();

      // Reconnect for cleanup
      await neo4jService.onModuleInit();
    });

    it('should validate input parameters', async () => {
      await expect(
        service.getComponentsWithSpatialData('')
      ).rejects.toThrow('Vehicle signature is required');

      await expect(
        service.analyzeCircuit(testVehicleSignature, '')
      ).rejects.toThrow('Circuit name is required');
    });

    it('should handle malformed data gracefully', async () => {
      // Create component with invalid position data
      const session = driver.session();
      await session.run(`
        CREATE (c:Component {
          id: 'test_invalid_position',
          name: 'Invalid Position Component',
          type: 'relay',
          vehicle_signature: $vehicleSignature,
          position: 'invalid_position_data'
        })
      `, { vehicleSignature: testVehicleSignature });
      await session.close();

      // Should handle gracefully and exclude invalid data
      const result = await service.getComponentsWithSpatialData(testVehicleSignature);
      
      expect(result.some(c => c.id === 'test_invalid_position')).toBe(false);
    });
  });
});