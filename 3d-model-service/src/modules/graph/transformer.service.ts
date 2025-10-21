import { Injectable } from '@nestjs/common';
import * as neo4j from 'neo4j-driver';
import { ComponentEntity, ConnectionEntity, ElectricalSystemData, CircuitEntity, Vector3 } from '../../common/entities/component.entity';

@Injectable()
export class TransformerService {

  /**
   * Transform Neo4j query results to ElectricalSystemData
   */
  transformToElectricalSystem(
    records: neo4j.Record[],
    systemId: string = 'generated'
  ): ElectricalSystemData {
    const components: ComponentEntity[] = [];
    const connections: ConnectionEntity[] = [];
    const connectionSet = new Set<string>(); // To avoid duplicates

    for (const record of records) {
      const componentNode = record.get('c');
      const connectionData = record.get('connections') || [];

      // Transform component
      const component = this.transformComponent(componentNode);
      components.push(component);

      // Transform connections
      for (const connData of connectionData) {
        if (connData.relationship && connData.target) {
          const connection = this.transformConnection(
            connData.relationship,
            component.id,
            connData.target
          );
          
          // Use a unique key to avoid duplicates
          const connectionKey = `${connection.fromComponentId}-${connection.toComponentId}-${connection.type}`;
          if (!connectionSet.has(connectionKey)) {
            connections.push(connection);
            connectionSet.add(connectionKey);
          }
        }
      }
    }

    // Calculate system bounding box
    const boundingBox = this.calculateBoundingBox(components);

    // Extract unique zones
    const zones = [...new Set(components.map(c => c.zone).filter(Boolean))] as string[];

    return {
      id: systemId,
      name: `Electrical System ${systemId}`,
      components,
      connections,
      zones,
      boundingBox,
      metadata: {
        componentCount: components.length,
        connectionCount: connections.length,
        zoneCount: zones.length
      }
    };
  }

  /**
   * Transform Neo4j node to ComponentEntity
   */
  transformComponent(node: neo4j.Node): ComponentEntity {
    const properties = node.properties;
    
    // Convert Neo4j integers to regular numbers
    const convertInt = (value: any) => neo4j.isInt(value) ? value.toNumber() : value;

    // Extract position data
    let position: Vector3 | [number, number, number] = [0, 0, 0];
    if (properties.anchor_xyz) {
      const coords = Array.isArray(properties.anchor_xyz) 
        ? properties.anchor_xyz.map(convertInt)
        : [0, 0, 0];
      position = [coords[0] || 0, coords[1] || 0, coords[2] || 0];
    } else if (properties.position) {
      position = properties.position;
    }

    // Extract bounding box
    let bbox: [number, number, number] | undefined;
    if (properties.bbox_m) {
      const bboxData = Array.isArray(properties.bbox_m)
        ? properties.bbox_m.map(convertInt)
        : [0.05, 0.05, 0.025];
      bbox = [bboxData[0] || 0.05, bboxData[1] || 0.05, bboxData[2] || 0.025];
    }

    // Extract rotation and scale
    let rotation: [number, number, number] | undefined;
    if (properties.rotation) {
      const rot = Array.isArray(properties.rotation)
        ? properties.rotation.map(convertInt)
        : [0, 0, 0];
      rotation = [rot[0] || 0, rot[1] || 0, rot[2] || 0];
    }

    let scale: [number, number, number] | undefined;
    if (properties.scale) {
      const scl = Array.isArray(properties.scale)
        ? properties.scale.map(convertInt)
        : [1, 1, 1];
      scale = [scl[0] || 1, scl[1] || 1, scl[2] || 1];
    }

    const component: ComponentEntity = {
      id: properties.id || node.identity.toNumber().toString(),
      type: properties.type || properties.node_type || 'component',
      name: properties.name || properties.label,
      description: properties.description,
      position,
      rotation,
      scale,
      bbox,
      anchor_xyz: position as [number, number, number],
      anchor_zone: properties.anchor_zone || properties.zone,
      
      // Physical properties
      dimensions: bbox ? {
        width: bbox[0],
        height: bbox[1], 
        depth: bbox[2]
      } : undefined,
      
      // Electrical properties
      voltage: convertInt(properties.voltage),
      current: convertInt(properties.current) || convertInt(properties.current_draw),
      power: convertInt(properties.power),
      resistance: convertInt(properties.resistance),
      
      // Metadata
      zone: properties.zone || properties.anchor_zone,
      manufacturer: properties.manufacturer,
      partNumber: properties.part_number || properties.partNumber,
      material: properties.material,
      color: properties.color,
      
      // Store all properties for reference
      properties: this.cleanProperties(properties)
    };

    return component;
  }

  /**
   * Transform Neo4j relationship to ConnectionEntity
   */
  transformConnection(
    relationship: neo4j.Relationship,
    fromComponentId: string,
    targetNode: neo4j.Node
  ): ConnectionEntity {
    const relProps = relationship.properties;
    const targetProps = targetNode.properties;
    const convertInt = (value: any) => neo4j.isInt(value) ? value.toNumber() : value;

    const connection: ConnectionEntity = {
      id: relationship.identity.toNumber().toString(),
      fromComponentId,
      toComponentId: targetProps.id || targetNode.identity.toNumber().toString(),
      type: this.mapRelationshipType(relationship.type),
      
      // Wire properties
      wireGauge: relProps.wire_gauge || relProps.wireGauge,
      wireColor: relProps.wire_color || relProps.wireColor || relProps.color,
      wireLength: convertInt(relProps.wire_length || relProps.wireLength),
      
      // Electrical properties
      voltage: convertInt(relProps.voltage),
      current: convertInt(relProps.current),
      signalType: relProps.signal_type || relProps.signalType,
      
      // Physical routing
      routePoints: this.extractRoutePoints(relProps.route_points || relProps.routePoints),
      bendRadius: convertInt(relProps.bend_radius || relProps.bendRadius),
      
      // Metadata
      label: relProps.label || relProps.name,
      notes: relProps.notes || relProps.description,
      properties: this.cleanProperties(relProps)
    };

    return connection;
  }

  /**
   * Transform circuit data from Neo4j
   */
  transformCircuit(records: neo4j.Record[]): CircuitEntity[] {
    const circuitsMap = new Map<string, CircuitEntity>();
    
    for (const record of records) {
      const circuitNode = record.get('circuit');
      const componentNode = record.get('component');
      
      if (circuitNode) {
        const circuitId = circuitNode.properties.id;
        
        if (!circuitsMap.has(circuitId)) {
          const convertInt = (value: any) => neo4j.isInt(value) ? value.toNumber() : value;
          
          circuitsMap.set(circuitId, {
            id: circuitId,
            name: circuitNode.properties.name || circuitId,
            type: circuitNode.properties.type || 'electrical',
            voltage: convertInt(circuitNode.properties.voltage) || 12,
            maxCurrent: convertInt(circuitNode.properties.max_current) || 15,
            componentIds: [],
            connectionIds: [],
            properties: this.cleanProperties(circuitNode.properties)
          });
        }
        
        // Add component to circuit
        if (componentNode) {
          const circuit = circuitsMap.get(circuitId)!;
          const componentId = componentNode.properties.id;
          if (!circuit.componentIds.includes(componentId)) {
            circuit.componentIds.push(componentId);
          }
        }
      }
    }
    
    return Array.from(circuitsMap.values());
  }

  /**
   * Calculate bounding box for all components
   */
  private calculateBoundingBox(components: ComponentEntity[]): { min: Vector3; max: Vector3 } {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (const component of components) {
      const pos = Array.isArray(component.position) 
        ? component.position 
        : [component.position.x, component.position.y, component.position.z];
      
      const [x, y, z] = pos;
      
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
    }

    // Handle case where no components have valid positions
    if (!isFinite(minX)) {
      return {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0, y: 0, z: 0 }
      };
    }

    return {
      min: { x: minX, y: minY, z: minZ },
      max: { x: maxX, y: maxY, z: maxZ }
    };
  }

  /**
   * Map Neo4j relationship types to standard types
   */
  private mapRelationshipType(relType: string): 'electrical' | 'mechanical' | 'data' {
    switch (relType.toUpperCase()) {
      case 'CONNECTS_TO':
      case 'POWERS':
      case 'POWERED_BY':
        return 'electrical';
      case 'MOUNTED_TO':
      case 'ATTACHED_TO':
        return 'mechanical';
      case 'COMMUNICATES_WITH':
      case 'SENDS_DATA':
        return 'data';
      default:
        return 'electrical'; // Default assumption
    }
  }

  /**
   * Extract route points from Neo4j data
   */
  private extractRoutePoints(routeData: any): Vector3[] | undefined {
    if (!routeData) return undefined;
    
    if (Array.isArray(routeData)) {
      return routeData.map(point => {
        if (Array.isArray(point) && point.length >= 3) {
          return { x: point[0], y: point[1], z: point[2] };
        }
        return point;
      });
    }
    
    return undefined;
  }

  /**
   * Clean Neo4j properties by converting special types
   */
  private cleanProperties(properties: Record<string, any>): Record<string, any> {
    const cleaned: Record<string, any> = {};
    
    for (const [key, value] of Object.entries(properties)) {
      if (neo4j.isInt(value)) {
        cleaned[key] = value.toNumber();
      } else if (neo4j.isDate(value) || neo4j.isDateTime(value) || neo4j.isTime(value)) {
        cleaned[key] = value.toString();
      } else if (Array.isArray(value)) {
        cleaned[key] = value.map(item => 
          neo4j.isInt(item) ? item.toNumber() : item
        );
      } else {
        cleaned[key] = value;
      }
    }
    
    return cleaned;
  }

  /**
   * Extract harness data from Neo4j results
   */
  transformHarnessData(records: neo4j.Record[]): any[] {
    return records.map(record => {
      const convertInt = (value: any) => neo4j.isInt(value) ? value.toNumber() : value;
      
      return {
        id: record.get('id'),
        path: record.get('path'),
        thickness: convertInt(record.get('thickness')) || 0.02,
        type: record.get('type') || 'default',
        zone: record.get('zone')
      };
    });
  }
}