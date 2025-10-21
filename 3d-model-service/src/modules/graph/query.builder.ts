import { Injectable } from '@nestjs/common';
import { GraphQueryDto } from '../../common/dto/model-generation.dto';

export interface CypherQuery {
  cypher: string;
  parameters: Record<string, any>;
}

@Injectable()
export class QueryBuilder {
  
  /**
   * Helper method to add vehicle signature filtering to custom Cypher queries
   */
  private addVehicleSignatureFilter(cypher: string, vehicleSignature: string): string {
    // Simple regex-based approach to add vehicle_signature filters
    // This is a basic implementation - in production, consider using a proper Cypher parser
    
    // Add vehicle_signature to MATCH clauses for Components, Circuits, and Harnesses
    let enhancedCypher = cypher
      .replace(/MATCH\s+\((\w+):Component\b/gi, `MATCH ($1:Component {vehicle_signature: $vehicleSignature}`)
      .replace(/MATCH\s+\((\w+):Circuit\b/gi, `MATCH ($1:Circuit {vehicle_signature: $vehicleSignature}`)
      .replace(/MATCH\s+\((\w+):Harness\b/gi, `MATCH ($1:Harness {vehicle_signature: $vehicleSignature}`);
    
    return enhancedCypher;
  }
  
  /**
   * Build a Cypher query from GraphQueryDto
   */
  buildElectricalSystemQuery(query: GraphQueryDto): CypherQuery {
    if (query.cypher) {
      // If custom Cypher is provided, ensure vehicle signature filtering
      const enhancedCypher = this.addVehicleSignatureFilter(query.cypher, query.vehicleSignature);
      return {
        cypher: enhancedCypher,
        parameters: { vehicleSignature: query.vehicleSignature, ...(query.filters || {}) }
      };
    }

    let cypherParts: string[] = [];
    let parameters: Record<string, any> = { vehicleSignature: query.vehicleSignature };
    let whereConditions: string[] = [];

    // Base query to get components and their relationships
    cypherParts.push('MATCH (c:Component)');

    // CRITICAL: Always filter by vehicle signature first
    whereConditions.push('c.vehicle_signature = $vehicleSignature');

    // Add specific node ID filtering
    if (query.nodeIds && query.nodeIds.length > 0) {
      whereConditions.push('c.id IN $nodeIds');
      parameters.nodeIds = query.nodeIds;
    }

    // Add component type filtering
    if (query.componentTypes && query.componentTypes.length > 0) {
      whereConditions.push('c.type IN $componentTypes');
      parameters.componentTypes = query.componentTypes;
    }

    // Add zone filtering
    if (query.zoneFilter) {
      whereConditions.push('c.anchor_zone = $zoneFilter');
      parameters.zoneFilter = query.zoneFilter;
    }

    // Add circuit filtering
    if (query.circuitId) {
      cypherParts.push('MATCH (circuit:Circuit {id: $circuitId, vehicle_signature: $vehicleSignature})-[:CONTAINS]->(c)');
      parameters.circuitId = query.circuitId;
    }

    // Add custom filters
    if (query.filters) {
      Object.entries(query.filters).forEach(([key, value], index) => {
        const paramName = `filter${index}`;
        whereConditions.push(`c.${key} = $${paramName}`);
        parameters[paramName] = value;
      });
    }

    // Add WHERE clause - always includes vehicle signature
    cypherParts.push(`WHERE ${whereConditions.join(' AND ')}`);

    // Get connections - ensure connected components are from same vehicle
    cypherParts.push(
      'OPTIONAL MATCH (c)-[r:CONNECTS_TO|POWERS|CONTROLS]->(target:Component)',
      'WHERE target.vehicle_signature = $vehicleSignature OR target IS NULL',
      'RETURN c, collect({relationship: r, target: target}) as connections'
    );

    return {
      cypher: cypherParts.join('\n'),
      parameters
    };
  }

  /**
   * Build query to get all components in a specific circuit
   */
  buildCircuitQuery(circuitId: string, vehicleSignature: string): CypherQuery {
    return {
      cypher: `
        MATCH (circuit:Circuit {id: $circuitId, vehicle_signature: $vehicleSignature})
        MATCH (circuit)-[:CONTAINS]->(component:Component {vehicle_signature: $vehicleSignature})
        OPTIONAL MATCH (component)-[r:CONNECTS_TO|POWERS|CONTROLS]->(target:Component {vehicle_signature: $vehicleSignature})
        WHERE target.id IS NULL OR (circuit)-[:CONTAINS]->(target)
        RETURN circuit, component, collect({relationship: r, target: target}) as connections
      `,
      parameters: { circuitId, vehicleSignature }
    };
  }

  /**
   * Build query to get component spatial data
   */
  buildSpatialDataQuery(componentIds: string[], vehicleSignature: string): CypherQuery {
    return {
      cypher: `
        MATCH (c:Component {vehicle_signature: $vehicleSignature})
        WHERE c.id IN $componentIds
        RETURN c.id as id,
               c.type as type,
               c.anchor_xyz as position,
               c.bbox_m as bbox,
               c.anchor_zone as zone,
               c.rotation as rotation,
               c.scale as scale,
               c.properties as properties
      `,
      parameters: { componentIds, vehicleSignature }
    };
  }

  /**
   * Build query to get wire harness routing data
   */
  buildHarnessDataQuery(vehicleSignature: string, zoneFilter?: string): CypherQuery {
    let cypher = `
      MATCH (h:Harness {vehicle_signature: $vehicleSignature})
    `;
    
    const parameters: Record<string, any> = { vehicleSignature };
    
    if (zoneFilter) {
      cypher += ' WHERE h.anchor_zone = $zoneFilter';
      parameters.zoneFilter = zoneFilter;
    }
    
    cypher += `
      RETURN h.id as id,
             h.path_xyz as path,
             h.thickness as thickness,
             h.type as type,
             h.anchor_zone as zone
    `;

    return { cypher, parameters };
  }

  /**
   * Build query to get complete electrical topology
   */
  buildElectricalTopologyQuery(vehicleSignature: string, systemId?: string): CypherQuery {
    let cypher = `
      MATCH (c:Component {vehicle_signature: $vehicleSignature})
    `;
    
    const parameters: Record<string, any> = { vehicleSignature };
    
    if (systemId) {
      cypher += '-[:PART_OF]->(system:ElectricalSystem {id: $systemId, vehicle_signature: $vehicleSignature})';
      parameters.systemId = systemId;
    }
    
    cypher += `
      OPTIONAL MATCH (c)-[r:CONNECTS_TO]->(target:Component {vehicle_signature: $vehicleSignature})
      OPTIONAL MATCH (c)-[p:POWERED_BY]->(power:Component {vehicle_signature: $vehicleSignature})
      OPTIONAL MATCH (c)-[ctrl:CONTROLS]->(controlled:Component {vehicle_signature: $vehicleSignature})
      
      RETURN c as component,
             collect(DISTINCT {type: 'connection', relationship: r, target: target}) +
             collect(DISTINCT {type: 'power', relationship: p, target: power}) +
             collect(DISTINCT {type: 'control', relationship: ctrl, target: controlled}) as relationships
    `;

    return { cypher, parameters };
  }

  /**
   * Build query to get power distribution analysis
   */
  buildPowerAnalysisQuery(componentId: string, vehicleSignature: string): CypherQuery {
    return {
      cypher: `
        MATCH path = (source:Component {vehicle_signature: $vehicleSignature})-[:POWERS*]->(c:Component {id: $componentId, vehicle_signature: $vehicleSignature})
        WITH path, source
        MATCH (c)-[:CONNECTS_TO*0..2]->(downstream:Component {vehicle_signature: $vehicleSignature})
        RETURN source as powerSource,
               nodes(path) as powerPath,
               collect(DISTINCT downstream) as downstreamComponents,
               length(path) as pathLength
        ORDER BY pathLength
      `,
      parameters: { componentId, vehicleSignature }
    };
  }

  /**
   * Build query to find shortest path between components
   */
  buildShortestPathQuery(fromId: string, toId: string, vehicleSignature: string, maxDepth: number = 5): CypherQuery {
    return {
      cypher: `
        MATCH (from:Component {id: $fromId, vehicle_signature: $vehicleSignature}), (to:Component {id: $toId, vehicle_signature: $vehicleSignature})
        MATCH path = shortestPath((from)-[:CONNECTS_TO*1..${maxDepth}]-(to))
        WHERE ALL(node in nodes(path) WHERE node.vehicle_signature = $vehicleSignature)
        RETURN path,
               length(path) as pathLength,
               [node in nodes(path) | {id: node.id, type: node.type, zone: node.anchor_zone}] as pathNodes
      `,
      parameters: { fromId, toId, vehicleSignature }
    };
  }

  /**
   * Build query to get components by zone with spatial ordering
   */
  buildZoneComponentsQuery(zone: string, vehicleSignature: string): CypherQuery {
    return {
      cypher: `
        MATCH (c:Component {anchor_zone: $zone, vehicle_signature: $vehicleSignature})
        WHERE c.anchor_xyz IS NOT NULL
        RETURN c,
               c.anchor_xyz[0] as x,
               c.anchor_xyz[1] as y,
               c.anchor_xyz[2] as z
        ORDER BY x, y, z
      `,
      parameters: { zone, vehicleSignature }
    };
  }

  /**
   * Build query to analyze circuit load and capacity
   */
  buildCircuitLoadAnalysisQuery(vehicleSignature: string, circuitId?: string): CypherQuery {
    let cypher = `
      MATCH (circuit:Circuit {vehicle_signature: $vehicleSignature})
    `;
    
    const parameters: Record<string, any> = { vehicleSignature };
    
    if (circuitId) {
      cypher += ' WHERE circuit.id = $circuitId';
      parameters.circuitId = circuitId;
    }
    
    cypher += `
      MATCH (circuit)-[:CONTAINS]->(component:Component {vehicle_signature: $vehicleSignature})
      WHERE component.current_draw IS NOT NULL
      RETURN circuit.id as circuitId,
             circuit.name as circuitName,
             circuit.max_current as maxCurrent,
             circuit.voltage as voltage,
             sum(component.current_draw) as totalCurrentDraw,
             count(component) as componentCount,
             collect({id: component.id, type: component.type, current: component.current_draw}) as components
    `;

    return { cypher, parameters };
  }

  /**
   * Build query to get system-wide statistics
   */
  buildSystemStatsQuery(vehicleSignature: string): CypherQuery {
    return {
      cypher: `
        MATCH (c:Component {vehicle_signature: $vehicleSignature})
        OPTIONAL MATCH (circuit:Circuit {vehicle_signature: $vehicleSignature})
        OPTIONAL MATCH (h:Harness {vehicle_signature: $vehicleSignature})
        OPTIONAL MATCH (c1:Component {vehicle_signature: $vehicleSignature})-[r:CONNECTS_TO]->(c2:Component {vehicle_signature: $vehicleSignature})
        
        RETURN count(DISTINCT c) as totalComponents,
               count(DISTINCT circuit) as totalCircuits,
               count(DISTINCT h) as totalHarnesses,
               count(DISTINCT r) as totalConnections,
               collect(DISTINCT c.type) as componentTypes,
               collect(DISTINCT c.anchor_zone) as zones
      `,
      parameters: { vehicleSignature }
    };
  }
}