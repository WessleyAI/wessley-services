#!/usr/bin/env node

/**
 * Compare what's in the database vs what's in the GraphML file
 */

const neo4j = require('neo4j-driver');
const fs = require('fs');
const path = require('path');
const { DOMParser } = require('xmldom');

// Configuration
const NEO4J_URI = process.env.NEO4J_URI || 'bolt://localhost:7687';
const NEO4J_USERNAME = process.env.NEO4J_USERNAME || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || 'password';
const GRAPHML_FILE = '../../../n8n_workflow/demo/model.xml';

class DataComparison {
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

  analyzeGraphML() {
    console.log('📄 Analyzing GraphML file...');
    const xmlContent = fs.readFileSync(path.join(__dirname, GRAPHML_FILE), 'utf8');
    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlContent, 'text/xml');

    const nodes = doc.getElementsByTagName('node');
    const edges = doc.getElementsByTagName('edge');

    console.log(`📊 GraphML contains: ${nodes.length} nodes, ${edges.length} edges`);

    // Check what attributes are actually used
    const attributes = new Set();
    for (let i = 0; i < nodes.length; i++) {
      const dataElements = nodes[i].getElementsByTagName('data');
      for (let j = 0; j < dataElements.length; j++) {
        const key = dataElements[j].getAttribute('key');
        if (key) attributes.add(key);
      }
    }

    console.log('📋 Attributes found in GraphML:');
    Array.from(attributes).sort().forEach(attr => console.log(`  📝 ${attr}`));

    return { nodeCount: nodes.length, edgeCount: edges.length, attributes: Array.from(attributes) };
  }

  async analyzeDatabaseContent() {
    console.log('\n💾 Analyzing database content...');
    const session = this.driver.session();

    try {
      // Get all nodes
      const nodeResult = await session.run(`
        MATCH (n)
        RETURN 
          count(n) as totalNodes,
          collect(DISTINCT labels(n)[0]) as nodeTypes,
          collect(DISTINCT n.vehicle_signature) as vehicleSignatures
      `);

      const nodeRecord = nodeResult.records[0];
      
      // Get component details
      const componentResult = await session.run(`
        MATCH (c:Component)
        RETURN 
          count(c) as totalComponents,
          count(CASE WHEN c.anchor_xyz IS NOT NULL THEN 1 END) as withSpatial,
          count(CASE WHEN c.anchor_zone IS NOT NULL THEN 1 END) as withZone,
          count(CASE WHEN c.type IS NOT NULL THEN 1 END) as withType,
          count(CASE WHEN c.canonical_id IS NOT NULL THEN 1 END) as withCanonicalId,
          count(CASE WHEN c.code_id IS NOT NULL THEN 1 END) as withCodeId,
          collect(DISTINCT c.type)[0..5] as sampleTypes,
          collect(DISTINCT c.anchor_zone)[0..5] as sampleZones
      `);

      const compRecord = componentResult.records[0];

      // Get relationship count
      const relResult = await session.run(`
        MATCH ()-[r]->()
        RETURN count(r) as totalRelationships, collect(DISTINCT type(r)) as relationshipTypes
      `);

      const relRecord = relResult.records[0];

      console.log('\n📊 DATABASE CONTENT');
      console.log('='.repeat(50));
      console.log(`Total Nodes: ${nodeRecord.get('totalNodes').toNumber()}`);
      console.log(`Node Types: ${nodeRecord.get('nodeTypes').join(', ')}`);
      console.log(`Vehicle Signatures: ${nodeRecord.get('vehicleSignatures').join(', ')}`);
      console.log(`Total Relationships: ${relRecord.get('totalRelationships').toNumber()}`);
      console.log(`Relationship Types: ${relRecord.get('relationshipTypes').join(', ')}`);

      console.log('\n🔧 COMPONENT DATA QUALITY');
      console.log('='.repeat(50));
      console.log(`Total Components: ${compRecord.get('totalComponents').toNumber()}`);
      console.log(`With anchor_xyz: ${compRecord.get('withSpatial').toNumber()}`);
      console.log(`With anchor_zone: ${compRecord.get('withZone').toNumber()}`);
      console.log(`With type: ${compRecord.get('withType').toNumber()}`);
      console.log(`With canonical_id: ${compRecord.get('withCanonicalId').toNumber()}`);
      console.log(`With code_id: ${compRecord.get('withCodeId').toNumber()}`);
      console.log(`Sample Types: ${compRecord.get('sampleTypes').join(', ')}`);
      console.log(`Sample Zones: ${compRecord.get('sampleZones').join(', ')}`);

      return {
        totalNodes: nodeRecord.get('totalNodes').toNumber(),
        totalComponents: compRecord.get('totalComponents').toNumber(),
        withSpatial: compRecord.get('withSpatial').toNumber(),
        withZone: compRecord.get('withZone').toNumber(),
        sampleTypes: compRecord.get('sampleTypes'),
        sampleZones: compRecord.get('sampleZones')
      };

    } finally {
      await session.close();
    }
  }

  async run() {
    try {
      await this.connect();
      
      const graphmlData = this.analyzeGraphML();
      const dbData = await this.analyzeDatabaseContent();

      console.log('\n🔍 COMPARISON RESULTS');
      console.log('='.repeat(50));
      console.log(`GraphML Nodes: ${graphmlData.nodeCount}`);
      console.log(`Database Nodes: ${dbData.totalNodes}`);
      console.log(`Match: ${graphmlData.nodeCount === dbData.totalNodes ? '✅' : '❌'}`);
      
      console.log(`\nGraphML Edges: ${graphmlData.edgeCount}`);
      console.log(`Database Relationships: estimated from import`);

      console.log('\n🎯 KEY FINDINGS');
      console.log('='.repeat(50));
      
      if (dbData.withSpatial === 0) {
        console.log('❌ NO SPATIAL DATA: All components missing anchor_xyz coordinates');
        console.log('📍 Need to add spatial coordinates for 3D modeling');
      } else {
        console.log(`✅ Spatial data: ${dbData.withSpatial}/${dbData.totalComponents} components`);
      }

      if (dbData.withZone > 0) {
        console.log(`✅ Zone data: ${dbData.withZone}/${dbData.totalComponents} components have zones`);
        console.log('📍 Can generate spatial coordinates from zones');
      }

      console.log('\n💡 RECOMMENDATION');
      console.log('='.repeat(50));
      if (dbData.withSpatial === 0 && dbData.withZone > 0) {
        console.log('🔧 RUN safe-spatial-update.js to add missing coordinates');
        console.log('✅ This will not delete any data, only add spatial coordinates');
      } else if (dbData.totalNodes !== graphmlData.nodeCount) {
        console.log('⚠️  Data count mismatch - may need to re-import');
      } else {
        console.log('✅ Database looks complete and ready');
      }

    } catch (error) {
      console.error('❌ Comparison failed:', error);
    } finally {
      await this.disconnect();
    }
  }
}

if (require.main === module) {
  const comparison = new DataComparison();
  comparison.run().catch(console.error);
}

module.exports = DataComparison;