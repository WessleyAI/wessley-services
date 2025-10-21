import { Injectable } from '@nestjs/common';
import { Neo4jService } from './neo4j.service';
import { QueryBuilder } from './query.builder';
import { TransformerService } from './transformer.service';
import { GraphQueryDto } from '../../common/dto/model-generation.dto';
import { ComponentEntity, ElectricalSystemData, CircuitEntity } from '../../common/entities/component.entity';

@Injectable()
export class GraphService {
  constructor(
    private readonly neo4jService: Neo4jService,
    private readonly queryBuilder: QueryBuilder,
    private readonly transformerService: TransformerService
  ) {}

  /**
   * Query electrical system data from Neo4j and transform to standard format
   */
  async queryElectricalSystem(query: GraphQueryDto): Promise<ElectricalSystemData> {
    const { cypher, parameters } = this.queryBuilder.buildElectricalSystemQuery(query);
    
    console.log('Executing Neo4j query:', cypher);
    console.log('Parameters:', parameters);
    
    const result = await this.neo4jService.run(cypher, parameters);
    
    if (result.records.length === 0) {
      throw new Error('No electrical system data found for the given query');
    }
    
    const electricalData = this.transformerService.transformToElectricalSystem(
      result.records,
      query.circuitId || 'generated'
    );
    
    console.log(`✅ Retrieved ${electricalData.components.length} components and ${electricalData.connections.length} connections`);
    
    return electricalData;
  }

  /**
   * Get a specific component by ID
   */
  async getComponentById(componentId: string, vehicleSignature: string): Promise<ComponentEntity | null> {
    const result = await this.neo4jService.run(
      'MATCH (c:Component {id: $componentId, vehicle_signature: $vehicleSignature}) RETURN c',
      { componentId, vehicleSignature }
    );
    
    if (result.records.length === 0) {
      return null;
    }
    
    const componentNode = result.records[0].get('c');
    return this.transformerService.transformComponent(componentNode);
  }

  /**
   * Get all components in a specific zone
   */
  async getComponentsByZone(zone: string, vehicleSignature: string): Promise<ComponentEntity[]> {
    const { cypher, parameters } = this.queryBuilder.buildZoneComponentsQuery(zone, vehicleSignature);
    const result = await this.neo4jService.run(cypher, parameters);
    
    return result.records.map(record => {
      const componentNode = record.get('c');
      return this.transformerService.transformComponent(componentNode);
    });
  }

  /**
   * Get circuit information with all components
   */
  async getCircuitData(circuitId: string, vehicleSignature: string): Promise<{ circuit: CircuitEntity; components: ComponentEntity[] }> {
    const { cypher, parameters } = this.queryBuilder.buildCircuitQuery(circuitId, vehicleSignature);
    const result = await this.neo4jService.run(cypher, parameters);
    
    if (result.records.length === 0) {
      throw new Error(`Circuit ${circuitId} not found for vehicle ${vehicleSignature}`);
    }
    
    const circuits = this.transformerService.transformCircuit(result.records);
    const circuit = circuits[0];
    
    // Get components
    const components = result.records
      .map(record => record.get('component'))
      .filter(Boolean)
      .map(node => this.transformerService.transformComponent(node));
    
    return { circuit, components };
  }

  /**
   * Analyze power distribution for a component
   */
  async analyzePowerDistribution(componentId: string, vehicleSignature: string): Promise<any> {
    const { cypher, parameters } = this.queryBuilder.buildPowerAnalysisQuery(componentId, vehicleSignature);
    const result = await this.neo4jService.run(cypher, parameters);
    
    return result.records.map(record => ({
      powerSource: this.transformerService.transformComponent(record.get('powerSource')),
      pathLength: this.neo4jService.convertInteger(record.get('pathLength')),
      powerPath: record.get('powerPath').map((node: any) => 
        this.transformerService.transformComponent(node)
      ),
      downstreamComponents: record.get('downstreamComponents').map((node: any) =>
        this.transformerService.transformComponent(node)
      )
    }));
  }

  /**
   * Find shortest path between two components
   */
  async findShortestPath(fromId: string, toId: string, vehicleSignature: string, maxDepth: number = 5): Promise<any> {
    const { cypher, parameters } = this.queryBuilder.buildShortestPathQuery(fromId, toId, vehicleSignature, maxDepth);
    const result = await this.neo4jService.run(cypher, parameters);
    
    if (result.records.length === 0) {
      return null;
    }
    
    const record = result.records[0];
    return {
      pathLength: this.neo4jService.convertInteger(record.get('pathLength')),
      pathNodes: record.get('pathNodes'),
      path: record.get('path')
    };
  }

  /**
   * Get circuit load analysis
   */
  async getCircuitLoadAnalysis(vehicleSignature: string, circuitId?: string): Promise<any[]> {
    const { cypher, parameters } = this.queryBuilder.buildCircuitLoadAnalysisQuery(vehicleSignature, circuitId);
    const result = await this.neo4jService.run(cypher, parameters);
    
    return result.records.map(record => ({
      circuitId: record.get('circuitId'),
      circuitName: record.get('circuitName'),
      maxCurrent: this.neo4jService.convertInteger(record.get('maxCurrent')),
      voltage: this.neo4jService.convertInteger(record.get('voltage')),
      totalCurrentDraw: this.neo4jService.convertInteger(record.get('totalCurrentDraw')),
      componentCount: this.neo4jService.convertInteger(record.get('componentCount')),
      loadPercentage: (this.neo4jService.convertInteger(record.get('totalCurrentDraw')) / 
                      this.neo4jService.convertInteger(record.get('maxCurrent'))) * 100,
      components: record.get('components')
    }));
  }

  /**
   * Get system-wide statistics
   */
  async getSystemStatistics(vehicleSignature: string): Promise<any> {
    const { cypher, parameters } = this.queryBuilder.buildSystemStatsQuery(vehicleSignature);
    const result = await this.neo4jService.run(cypher, parameters);
    
    if (result.records.length === 0) {
      return null;
    }
    
    const record = result.records[0];
    return {
      vehicleSignature,
      totalComponents: this.neo4jService.convertInteger(record.get('totalComponents')),
      totalCircuits: this.neo4jService.convertInteger(record.get('totalCircuits')),
      totalHarnesses: this.neo4jService.convertInteger(record.get('totalHarnesses')),
      totalConnections: this.neo4jService.convertInteger(record.get('totalConnections')),
      componentTypes: record.get('componentTypes'),
      zones: record.get('zones')
    };
  }

  /**
   * Get harness routing data
   */
  async getHarnessData(vehicleSignature: string, zoneFilter?: string): Promise<any[]> {
    const { cypher, parameters } = this.queryBuilder.buildHarnessDataQuery(vehicleSignature, zoneFilter);
    const result = await this.neo4jService.run(cypher, parameters);
    
    return this.transformerService.transformHarnessData(result.records);
  }

  /**
   * Validate electrical system data
   */
  async validateElectricalSystem(vehicleSignature: string, systemId?: string): Promise<{ isValid: boolean; issues: string[] }> {
    const issues: string[] = [];
    
    // Check for orphaned components
    const orphanedResult = await this.neo4jService.run(`
      MATCH (c:Component {vehicle_signature: $vehicleSignature})
      WHERE NOT (c)-[:CONNECTS_TO|POWERS|POWERED_BY]-()
      RETURN count(c) as orphanedCount
    `, { vehicleSignature });
    
    const orphanedCount = this.neo4jService.convertInteger(orphanedResult.records[0].get('orphanedCount'));
    if (orphanedCount > 0) {
      issues.push(`Found ${orphanedCount} orphaned components with no connections`);
    }
    
    // Check for missing spatial data
    const missingSpatialResult = await this.neo4jService.run(`
      MATCH (c:Component {vehicle_signature: $vehicleSignature})
      WHERE c.anchor_xyz IS NULL
      RETURN count(c) as missingSpatialCount
    `, { vehicleSignature });
    
    const missingSpatialCount = this.neo4jService.convertInteger(missingSpatialResult.records[0].get('missingSpatialCount'));
    if (missingSpatialCount > 0) {
      issues.push(`Found ${missingSpatialCount} components without spatial coordinates`);
    }
    
    // Check for circuit overloads
    const circuitAnalysis = await this.getCircuitLoadAnalysis(vehicleSignature);
    const overloadedCircuits = circuitAnalysis.filter(circuit => circuit.loadPercentage > 100);
    if (overloadedCircuits.length > 0) {
      issues.push(`Found ${overloadedCircuits.length} circuits with load > 100%`);
    }
    
    return {
      isValid: issues.length === 0,
      issues
    };
  }

  /**
   * Health check for Neo4j connection
   */
  async healthCheck(): Promise<boolean> {
    return this.neo4jService.healthCheck();
  }

  /**
   * Get database statistics
   */
  async getDatabaseStats(): Promise<any> {
    return this.neo4jService.getDatabaseStats();
  }
}