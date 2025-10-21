#!/usr/bin/env node

/**
 * Update existing Neo4j data to add vehicle signatures without reimporting
 */

const neo4j = require('neo4j-driver');

// Configuration
const NEO4J_URI = process.env.NEO4J_URI || 'bolt://localhost:7687';
const NEO4J_USERNAME = process.env.NEO4J_USERNAME || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || 'password';

// Vehicle signature for existing data
const VEHICLE_SIGNATURE = process.env.VEHICLE_SIGNATURE || 'pajero_pinin_2001';

class VehicleSignatureUpdater {
  constructor() {
    this.driver = null;
  }

  async connect() {
    console.log('🔌 Connecting to Neo4j...');
    this.driver = neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USERNAME, NEO4J_PASSWORD));
    
    const session = this.driver.session();
    try {
      await session.run('RETURN 1');
      console.log('✅ Connected to Neo4j successfully');
    } finally {
      await session.close();
    }
  }

  async disconnect() {
    if (this.driver) {
      await this.driver.close();
      console.log('🔌 Disconnected from Neo4j');
    }
  }

  async updateNodes() {
    console.log('🔄 Adding vehicle signatures to existing nodes...');
    const session = this.driver.session();
    
    try {
      // Update all Components
      const componentResult = await session.run(`
        MATCH (c:Component)
        WHERE c.vehicle_signature IS NULL
        SET c.vehicle_signature = $vehicleSignature
        RETURN count(c) as updatedComponents
      `, { vehicleSignature: VEHICLE_SIGNATURE });
      
      const componentsUpdated = componentResult.records[0].get('updatedComponents').toNumber();
      console.log(`📦 Updated ${componentsUpdated} components`);

      // Update all Harnesses
      const harnessResult = await session.run(`
        MATCH (h:Harness)
        WHERE h.vehicle_signature IS NULL
        SET h.vehicle_signature = $vehicleSignature
        RETURN count(h) as updatedHarnesses
      `, { vehicleSignature: VEHICLE_SIGNATURE });
      
      const harnessesUpdated = harnessResult.records[0].get('updatedHarnesses').toNumber();
      console.log(`🔗 Updated ${harnessesUpdated} harnesses`);

      // Update all Circuits
      const circuitResult = await session.run(`
        MATCH (c:Circuit)
        WHERE c.vehicle_signature IS NULL
        SET c.vehicle_signature = $vehicleSignature
        RETURN count(c) as updatedCircuits
      `, { vehicleSignature: VEHICLE_SIGNATURE });
      
      const circuitsUpdated = circuitResult.records[0].get('updatedCircuits').toNumber();
      console.log(`⚡ Updated ${circuitsUpdated} circuits`);

      // Update all other nodes (Wires, Locations, etc.)
      const otherResult = await session.run(`
        MATCH (n)
        WHERE n.vehicle_signature IS NULL 
        AND NOT n:Component 
        AND NOT n:Harness 
        AND NOT n:Circuit
        SET n.vehicle_signature = $vehicleSignature
        RETURN count(n) as updatedOthers
      `, { vehicleSignature: VEHICLE_SIGNATURE });
      
      const othersUpdated = otherResult.records[0].get('updatedOthers').toNumber();
      console.log(`🔧 Updated ${othersUpdated} other nodes`);

      return {
        components: componentsUpdated,
        harnesses: harnessesUpdated,
        circuits: circuitsUpdated,
        others: othersUpdated,
        total: componentsUpdated + harnessesUpdated + circuitsUpdated + othersUpdated
      };

    } finally {
      await session.close();
    }
  }

  async updateRelationships() {
    console.log('🔗 Adding vehicle signatures to existing relationships...');
    const session = this.driver.session();
    
    try {
      const result = await session.run(`
        MATCH ()-[r]-()
        WHERE r.vehicle_signature IS NULL
        SET r.vehicle_signature = $vehicleSignature
        RETURN count(r) as updatedRelationships
      `, { vehicleSignature: VEHICLE_SIGNATURE });
      
      const relationshipsUpdated = result.records[0].get('updatedRelationships').toNumber();
      console.log(`🔗 Updated ${relationshipsUpdated} relationships`);
      
      return relationshipsUpdated;

    } finally {
      await session.close();
    }
  }

  async createIndexes() {
    console.log('🏗️ Creating vehicle signature indexes...');
    const session = this.driver.session();
    
    try {
      const commands = [
        // Vehicle signature indexes for fast filtering
        'CREATE INDEX component_vehicle_sig_idx IF NOT EXISTS FOR (c:Component) ON (c.vehicle_signature)',
        'CREATE INDEX harness_vehicle_sig_idx IF NOT EXISTS FOR (h:Harness) ON (h.vehicle_signature)',
        'CREATE INDEX circuit_vehicle_sig_idx IF NOT EXISTS FOR (c:Circuit) ON (c.vehicle_signature)',
        'CREATE INDEX vehicle_signature_idx IF NOT EXISTS FOR (n) ON (n.vehicle_signature)'
      ];
      
      for (const command of commands) {
        try {
          await session.run(command);
        } catch (error) {
          if (!error.message.includes('already exists') && !error.message.includes('equivalent')) {
            console.warn(`Index warning: ${error.message}`);
          }
        }
      }
      
      console.log('✅ Created vehicle signature indexes');
      
    } finally {
      await session.close();
    }
  }

  async verifyUpdate() {
    console.log('🔍 Verifying vehicle signature update...');
    const session = this.driver.session();
    
    try {
      // Check nodes without vehicle signatures
      const noSigResult = await session.run(`
        MATCH (n)
        WHERE n.vehicle_signature IS NULL
        RETURN count(n) as nodesWithoutSignature
      `);
      
      const nodesWithoutSignature = noSigResult.records[0].get('nodesWithoutSignature').toNumber();
      
      // Check relationships without vehicle signatures
      const noSigRelResult = await session.run(`
        MATCH ()-[r]-()
        WHERE r.vehicle_signature IS NULL
        RETURN count(r) as relationshipsWithoutSignature
      `);
      
      const relationshipsWithoutSignature = noSigRelResult.records[0].get('relationshipsWithoutSignature').toNumber();

      // Count by vehicle signature
      const countResult = await session.run(`
        MATCH (n)
        RETURN n.vehicle_signature as vehicleSignature, count(n) as nodeCount
        ORDER BY vehicleSignature
      `);
      
      console.log('\n📊 VERIFICATION RESULTS');
      console.log('='.repeat(50));
      console.log(`Nodes without vehicle signature: ${nodesWithoutSignature}`);
      console.log(`Relationships without vehicle signature: ${relationshipsWithoutSignature}`);
      
      console.log('\nNodes by vehicle signature:');
      countResult.records.forEach(record => {
        const vehicleSignature = record.get('vehicleSignature') || 'NULL';
        const count = record.get('nodeCount').toNumber();
        console.log(`  📱 ${vehicleSignature}: ${count} nodes`);
      });

      return {
        nodesWithoutSignature,
        relationshipsWithoutSignature,
        isComplete: nodesWithoutSignature === 0 && relationshipsWithoutSignature === 0
      };

    } finally {
      await session.close();
    }
  }

  async run() {
    try {
      await this.connect();
      
      console.log(`🏷️ Adding vehicle signature: "${VEHICLE_SIGNATURE}"`);
      
      // Update nodes
      const nodeStats = await this.updateNodes();
      
      // Update relationships
      const relationshipStats = await this.updateRelationships();
      
      // Create indexes
      await this.createIndexes();
      
      // Verify the update
      const verification = await this.verifyUpdate();
      
      console.log('\n🎉 UPDATE COMPLETE!');
      console.log('='.repeat(50));
      console.log(`✅ Updated ${nodeStats.total} nodes`);
      console.log(`✅ Updated ${relationshipStats} relationships`);
      console.log(`✅ Vehicle signature: "${VEHICLE_SIGNATURE}"`);
      
      if (verification.isComplete) {
        console.log('✅ All data now has vehicle signatures');
        console.log('✅ Ready for vehicle-isolated 3D model generation');
      } else {
        console.log('⚠️ Some data still missing vehicle signatures');
        console.log(`   Nodes: ${verification.nodesWithoutSignature}`);
        console.log(`   Relationships: ${verification.relationshipsWithoutSignature}`);
      }
      
    } catch (error) {
      console.error('❌ Update failed:', error);
      process.exit(1);
    } finally {
      await this.disconnect();
    }
  }
}

// Run the update
if (require.main === module) {
  const updater = new VehicleSignatureUpdater();
  updater.run().catch(console.error);
}

module.exports = VehicleSignatureUpdater;