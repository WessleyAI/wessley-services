import { Injectable, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as neo4j from 'neo4j-driver';

@Injectable()
export class Neo4jService implements OnModuleInit, OnModuleDestroy {
  private driver: neo4j.Driver;
  private session: neo4j.Session;

  constructor(private configService: ConfigService) {}

  async onModuleInit() {
    const uri = this.configService.get<string>('neo4j.uri');
    const username = this.configService.get<string>('neo4j.username');
    const password = this.configService.get<string>('neo4j.password');
    const database = this.configService.get<string>('neo4j.database');

    this.driver = neo4j.driver(uri, neo4j.auth.basic(username, password), {
      maxConnectionPoolSize: this.configService.get<number>('neo4j.maxConnectionPoolSize'),
      connectionTimeout: this.configService.get<number>('neo4j.connectionTimeout')
    });

    // Verify connectivity
    await this.verifyConnectivity();
    
    console.log('✅ Neo4j connection established');
  }

  async onModuleDestroy() {
    if (this.session) {
      await this.session.close();
    }
    if (this.driver) {
      await this.driver.close();
    }
    console.log('🔌 Neo4j connection closed');
  }

  private async verifyConnectivity(): Promise<void> {
    const session = this.driver.session();
    try {
      await session.run('RETURN 1 as test');
    } catch (error) {
      throw new Error(`Failed to connect to Neo4j: ${error.message}`);
    } finally {
      await session.close();
    }
  }

  async run(cypher: string, parameters: Record<string, any> = {}): Promise<neo4j.QueryResult> {
    const session = this.driver.session({
      database: this.configService.get<string>('neo4j.database')
    });

    try {
      const result = await session.run(cypher, parameters);
      return result;
    } catch (error) {
      console.error('Neo4j query error:', error);
      throw error;
    } finally {
      await session.close();
    }
  }

  async runTransaction<T>(
    work: (tx: neo4j.Transaction) => Promise<T>
  ): Promise<T> {
    const session = this.driver.session({
      database: this.configService.get<string>('neo4j.database')
    });

    try {
      return await session.executeWrite(work);
    } catch (error) {
      console.error('Neo4j transaction error:', error);
      throw error;
    } finally {
      await session.close();
    }
  }

  async runReadTransaction<T>(
    work: (tx: neo4j.Transaction) => Promise<T>
  ): Promise<T> {
    const session = this.driver.session({
      database: this.configService.get<string>('neo4j.database'),
      defaultAccessMode: neo4j.session.READ
    });

    try {
      return await session.executeRead(work);
    } catch (error) {
      console.error('Neo4j read transaction error:', error);
      throw error;
    } finally {
      await session.close();
    }
  }

  // Health check for the service
  async healthCheck(): Promise<boolean> {
    try {
      await this.verifyConnectivity();
      return true;
    } catch (error) {
      console.error('Neo4j health check failed:', error);
      return false;
    }
  }

  // Get database statistics
  async getDatabaseStats(): Promise<any> {
    const result = await this.run(`
      CALL apoc.meta.stats() YIELD labelCount, relTypeCount, propertyKeyCount, nodeCount, relCount
      RETURN labelCount, relTypeCount, propertyKeyCount, nodeCount, relCount
    `);

    if (result.records.length > 0) {
      const record = result.records[0];
      return {
        nodeCount: record.get('nodeCount').toNumber(),
        relationshipCount: record.get('relCount').toNumber(),
        labelCount: record.get('labelCount').toNumber(),
        relationshipTypeCount: record.get('relTypeCount').toNumber(),
        propertyKeyCount: record.get('propertyKeyCount').toNumber()
      };
    }

    return null;
  }

  // Helper method to convert Neo4j integers to regular numbers
  convertInteger(value: any): number {
    if (neo4j.isInt(value)) {
      return value.toNumber();
    }
    return value;
  }

  // Helper method to convert Neo4j records to plain objects
  recordToObject(record: neo4j.Record): Record<string, any> {
    const obj: Record<string, any> = {};
    record.keys.forEach(key => {
      const value = record.get(key);
      if (neo4j.isInt(value)) {
        obj[key] = value.toNumber();
      } else if (neo4j.isDate(value) || neo4j.isDateTime(value) || neo4j.isTime(value)) {
        obj[key] = value.toString();
      } else if (value && typeof value === 'object' && value.properties) {
        // Node or Relationship
        obj[key] = {
          ...value.properties,
          identity: value.identity?.toNumber(),
          labels: value.labels,
          type: value.type
        };
      } else {
        obj[key] = value;
      }
    });
    return obj;
  }

  // Helper method to convert multiple records to objects
  recordsToObjects(records: neo4j.Record[]): Record<string, any>[] {
    return records.map(record => this.recordToObject(record));
  }
}