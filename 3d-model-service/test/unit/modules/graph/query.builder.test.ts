import { describe, it, expect, beforeEach } from 'vitest';
import { QueryBuilder } from '@/modules/graph/query.builder';

describe('QueryBuilder', () => {
  let queryBuilder: QueryBuilder;

  beforeEach(() => {
    queryBuilder = new QueryBuilder();
  });

  describe('buildComponentQuery', () => {
    it('should build basic component query with vehicle signature', () => {
      const result = queryBuilder.buildComponentQuery('test_vehicle_2001');
      
      expect(result.query).toContain('MATCH (c:Component {vehicle_signature: $vehicleSignature})');
      expect(result.parameters).toEqual({ vehicleSignature: 'test_vehicle_2001' });
    });

    it('should include spatial filtering when requested', () => {
      const result = queryBuilder.buildComponentQuery('test_vehicle_2001', {
        includeSpatialData: true,
      });
      
      expect(result.query).toContain('c.position IS NOT NULL');
    });

    it('should filter by component type when specified', () => {
      const result = queryBuilder.buildComponentQuery('test_vehicle_2001', {
        componentType: 'relay',
      });
      
      expect(result.query).toContain('c.type = $componentType');
      expect(result.parameters).toEqual({
        vehicleSignature: 'test_vehicle_2001',
        componentType: 'relay',
      });
    });

    it('should include connections when requested', () => {
      const result = queryBuilder.buildComponentQuery('test_vehicle_2001', {
        includeConnections: true,
      });
      
      expect(result.query).toContain('OPTIONAL MATCH (c)-[r:CONNECTS_TO]->(connected:Component)');
      expect(result.query).toContain('connected.vehicle_signature = $vehicleSignature');
    });

    it('should apply limit when specified', () => {
      const result = queryBuilder.buildComponentQuery('test_vehicle_2001', {
        limit: 50,
      });
      
      expect(result.query).toContain('LIMIT $limit');
      expect(result.parameters).toEqual({
        vehicleSignature: 'test_vehicle_2001',
        limit: 50,
      });
    });
  });

  describe('buildWireHarnessQuery', () => {
    it('should build wire harness query with vehicle signature isolation', () => {
      const result = queryBuilder.buildWireHarnessQuery('test_vehicle_2001');
      
      expect(result.query).toContain('MATCH (c1:Component {vehicle_signature: $vehicleSignature})-[r:CONNECTS_TO]->(c2:Component {vehicle_signature: $vehicleSignature})');
      expect(result.parameters).toEqual({ vehicleSignature: 'test_vehicle_2001' });
    });

    it('should require spatial data for both components', () => {
      const result = queryBuilder.buildWireHarnessQuery('test_vehicle_2001');
      
      expect(result.query).toContain('c1.position IS NOT NULL AND c2.position IS NOT NULL');
    });

    it('should include wire properties in return', () => {
      const result = queryBuilder.buildWireHarnessQuery('test_vehicle_2001');
      
      expect(result.query).toContain('wire_gauge: r.wire_gauge');
      expect(result.query).toContain('wire_color: r.wire_color');
    });
  });

  describe('buildCircuitAnalysisQuery', () => {
    it('should build circuit analysis query with power tracing', () => {
      const result = queryBuilder.buildCircuitAnalysisQuery('test_vehicle_2001', 'headlight_circuit');
      
      expect(result.query).toContain('MATCH (circuit:Circuit {vehicle_signature: $vehicleSignature, name: $circuitName})');
      expect(result.query).toContain('OPTIONAL MATCH path = (source:Component)-[:POWERED_BY*..5]->(c)');
      expect(result.parameters).toEqual({
        vehicleSignature: 'test_vehicle_2001',
        circuitName: 'headlight_circuit',
      });
    });

    it('should filter power sources correctly', () => {
      const result = queryBuilder.buildCircuitAnalysisQuery('test_vehicle_2001', 'test_circuit');
      
      expect(result.query).toContain("source.type IN ['battery', 'alternator', 'power_supply']");
    });
  });

  describe('buildSystemValidationQuery', () => {
    it('should build comprehensive system validation query', () => {
      const result = queryBuilder.buildSystemValidationQuery('test_vehicle_2001');
      
      expect(result.query).toContain('MATCH (v:Vehicle {signature: $vehicleSignature})');
      expect(result.query).toContain('unpowered:Component');
      expect(result.query).toContain('no_position:Component');
      expect(result.query).toContain('orphan_connector:Connector');
      expect(result.parameters).toEqual({ vehicleSignature: 'test_vehicle_2001' });
    });

    it('should exclude power sources from unpowered component check', () => {
      const result = queryBuilder.buildSystemValidationQuery('test_vehicle_2001');
      
      expect(result.query).toContain("unpowered.type NOT IN ['battery', 'alternator', 'ground']");
    });
  });

  describe('vehicle signature isolation', () => {
    it('should always include vehicle signature in component queries', () => {
      const testCases = [
        queryBuilder.buildComponentQuery('test_vehicle'),
        queryBuilder.buildWireHarnessQuery('test_vehicle'),
        queryBuilder.buildSystemValidationQuery('test_vehicle'),
      ];

      testCases.forEach((result) => {
        expect(result.query).toContain('vehicle_signature: $vehicleSignature');
        expect(result.parameters.vehicleSignature).toBe('test_vehicle');
      });
    });

    it('should prevent cross-vehicle data leakage in connection queries', () => {
      const result = queryBuilder.buildWireHarnessQuery('secure_vehicle');
      
      // Both components in connection must have same vehicle signature
      const vehicleMatches = result.query.match(/vehicle_signature: \$vehicleSignature/g);
      expect(vehicleMatches).toHaveLength(2);
    });
  });

  describe('error handling', () => {
    it('should throw error for empty vehicle signature', () => {
      expect(() => {
        queryBuilder.buildComponentQuery('');
      }).toThrow('Vehicle signature is required');
    });

    it('should throw error for null vehicle signature', () => {
      expect(() => {
        queryBuilder.buildComponentQuery(null as any);
      }).toThrow('Vehicle signature is required');
    });

    it('should validate circuit name in circuit analysis', () => {
      expect(() => {
        queryBuilder.buildCircuitAnalysisQuery('test_vehicle', '');
      }).toThrow('Circuit name is required');
    });
  });

  describe('query optimization', () => {
    it('should use indexed fields in WHERE clauses', () => {
      const result = queryBuilder.buildComponentQuery('test_vehicle', {
        componentType: 'relay',
      });
      
      // Should use indexed vehicle_signature and type fields
      expect(result.query).toContain('vehicle_signature: $vehicleSignature');
      expect(result.query).toContain('c.type = $componentType');
    });

    it('should limit result sets to prevent performance issues', () => {
      const result = queryBuilder.buildComponentQuery('test_vehicle');
      
      // Should have a reasonable default limit
      expect(result.query).toContain('LIMIT');
    });
  });
});