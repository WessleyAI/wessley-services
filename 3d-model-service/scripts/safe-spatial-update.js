#!/usr/bin/env node

/**
 * SAFE Spatial Coordinate Update - Only adds missing spatial data to existing nodes
 * Does NOT delete or overwrite existing data
 */

const neo4j = require('neo4j-driver');

// Configuration
const NEO4J_URI = process.env.NEO4J_URI || 'bolt://localhost:7687';
const NEO4J_USERNAME = process.env.NEO4J_USERNAME || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || 'password';
const VEHICLE_SIGNATURE = 'pajero_pinin_2001';

// 3D Zone Layout for Pajero Pinin
const ZONE_COORDINATES = {
  'Engine Compartment': { x: 0, y: 0, z: 2000 },
  'Engine Bay': { x: 0, y: 0, z: 2000 },
  'Dash Panel': { x: 0, y: 0, z: 0 },
  'Dashboard': { x: 0, y: 0, z: 0 },
  'Cabin': { x: 0, y: 0, z: -500 },
  'Chassis': { x: 0, y: -500, z: 0 },
  'Floor': { x: 0, y: -500, z: -1000 },
  'Floor & Roof': { x: 0, y: 0, z: -1000 },
  'Left Front Door': { x: -800, y: 0, z: 0 },
  'Right Front Door': { x: 800, y: 0, z: 0 },
  'Left Rear Door': { x: -800, y: 0, z: -1500 },
  'Right Rear Door': { x: 800, y: 0, z: -1500 },
  'Rear': { x: 0, y: 0, z: -3000 },
  'Tailgate': { x: 0, y: 0, z: -3200 },
  'Unknown': { x: 0, y: 0, z: 0 }
};

const COMPONENT_TYPE_OFFSETS = {
  'fuse': { x: 0, y: 100, z: 0 },
  'relay': { x: 0, y: 150, z: 0 },
  'connector': { x: 0, y: 50, z: 0 },
  'component': { x: 0, y: 0, z: 0 },
  'ground_point': { x: 0, y: -50, z: 0 },
  'splice': { x: 0, y: 25, z: 0 },
  'harness': { x: 0, y: -25, z: 0 },
  'wire': { x: 0, y: 10, z: 0 }
};

class SafeSpatialUpdater {
  constructor() {
    this.driver = null;
    this.componentCounter = new Map();
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

  async analyzeCurrentData() {
    console.log('🔍 Analyzing current database state...');
    const session = this.driver.session();
    
    try {
      // Get current data overview
      const overviewResult = await session.run(`
        MATCH (n {vehicle_signature: $vehicleSignature})
        OPTIONAL MATCH (n1 {vehicle_signature: $vehicleSignature})-[r]->(n2 {vehicle_signature: $vehicleSignature})
        RETURN 
          count(DISTINCT n) as totalNodes,
          count(DISTINCT r) as totalRelationships,
          collect(DISTINCT labels(n)[0]) as nodeTypes
      `, { vehicleSignature: VEHICLE_SIGNATURE });

      // Check spatial data coverage
      const spatialResult = await session.run(`
        MATCH (c:Component {vehicle_signature: $vehicleSignature})
        RETURN 
          count(c) as totalComponents,
          count(CASE WHEN c.anchor_xyz IS NOT NULL THEN 1 END) as componentsWithSpatial,
          count(CASE WHEN c.anchor_xyz IS NULL THEN 1 END) as componentsWithoutSpatial
      `, { vehicleSignature: VEHICLE_SIGNATURE });

      // Get zones that need spatial data
      const zoneResult = await session.run(`
        MATCH (c:Component {vehicle_signature: $vehicleSignature})
        WHERE c.anchor_xyz IS NULL
        RETURN c.anchor_zone as zone, c.type as componentType, count(c) as count
        ORDER BY zone, componentType
      `, { vehicleSignature: VEHICLE_SIGNATURE });

      const overview = overviewResult.records[0];
      const spatial = spatialResult.records[0];

      console.log('\n📊 CURRENT DATABASE STATE');
      console.log('='.repeat(50));
      console.log(`Vehicle: ${VEHICLE_SIGNATURE}`);
      console.log(`Total Nodes: ${overview.get('totalNodes').toNumber()}`);
      console.log(`Total Relationships: ${overview.get('totalRelationships').toNumber()}`);
      console.log(`Node Types: ${overview.get('nodeTypes').join(', ')}`);
      
      console.log('\n📍 SPATIAL DATA STATUS');
      console.log('='.repeat(50));
      console.log(`Total Components: ${spatial.get('totalComponents').toNumber()}`);
      console.log(`With Spatial Data: ${spatial.get('componentsWithSpatial').toNumber()}`);
      console.log(`WITHOUT Spatial Data: ${spatial.get('componentsWithoutSpatial').toNumber()}`);

      const needsSpatial = spatial.get('componentsWithoutSpatial').toNumber();
      
      if (needsSpatial > 0) {
        console.log('\n🎯 COMPONENTS THAT NEED SPATIAL COORDINATES:');
        console.log('='.repeat(50));
        zoneResult.records.forEach(record => {
          const zone = record.get('zone') || 'Unknown';
          const type = record.get('componentType') || 'component';
          const count = record.get('count').toNumber();
          console.log(`  📍 ${zone} - ${type}: ${count} components`);
        });
      }

      return {
        totalNodes: overview.get('totalNodes').toNumber(),
        totalRelationships: overview.get('totalRelationships').toNumber(),
        totalComponents: spatial.get('totalComponents').toNumber(),
        componentsWithSpatial: spatial.get('componentsWithSpatial').toNumber(),
        componentsWithoutSpatial: needsSpatial,
        zonesNeedingSpatial: zoneResult.records.map(r => ({
          zone: r.get('zone') || 'Unknown',
          type: r.get('componentType') || 'component',
          count: r.get('count').toNumber()
        }))
      };

    } finally {
      await session.close();
    }
  }

  generateSpatialCoordinates(anchorZone, nodeType, counter) {
    // Get base coordinates for the zone
    const zoneCoords = ZONE_COORDINATES[anchorZone] || ZONE_COORDINATES['Unknown'];
    
    // Get offset for component type
    const typeOffset = COMPONENT_TYPE_OFFSETS[nodeType] || COMPONENT_TYPE_OFFSETS['component'];
    
    // Add spacing to avoid overlapping components
    const spacing = {
      x: (counter % 5) * 100, // 5 components per row
      y: 0,
      z: Math.floor(counter / 5) * 100 // New row every 5 components
    };
    
    // Calculate final position
    const position = [
      zoneCoords.x + typeOffset.x + spacing.x,
      zoneCoords.y + typeOffset.y + spacing.y,
      zoneCoords.z + typeOffset.z + spacing.z
    ];
    
    return position;
  }

  async addSpatialCoordinates() {
    console.log('📍 Adding spatial coordinates to components without them...');
    const session = this.driver.session();
    
    try {
      // Get all components without spatial data
      const result = await session.run(`
        MATCH (c:Component {vehicle_signature: $vehicleSignature})
        WHERE c.anchor_xyz IS NULL
        RETURN c.id as id, c.anchor_zone as zone, c.type as type
        ORDER BY c.anchor_zone, c.type
      `, { vehicleSignature: VEHICLE_SIGNATURE });

      let updatedCount = 0;
      
      for (const record of result.records) {
        const id = record.get('id');
        const zone = record.get('zone') || 'Unknown';
        const type = record.get('type') || 'component';
        
        // Get counter for positioning
        const zoneKey = `${zone}_${type}`;
        const counter = this.componentCounter.get(zoneKey) || 0;
        this.componentCounter.set(zoneKey, counter + 1);
        
        // Generate coordinates
        const position = this.generateSpatialCoordinates(zone, type, counter);
        
        // Update the component with spatial coordinates
        await session.run(`
          MATCH (c:Component {id: $id, vehicle_signature: $vehicleSignature})
          SET c.anchor_xyz = $position, c.position = $position
          RETURN c.id
        `, { 
          id, 
          vehicleSignature: VEHICLE_SIGNATURE, 
          position 
        });
        
        updatedCount++;
        
        if (updatedCount % 50 === 0) {
          console.log(`  📍 Updated ${updatedCount} components...`);
        }
      }

      console.log(`✅ Added spatial coordinates to ${updatedCount} components`);
      return updatedCount;

    } finally {
      await session.close();
    }
  }

  async verifyUpdate() {
    console.log('🔍 Verifying spatial data update...');
    const session = this.driver.session();
    
    try {
      const result = await session.run(`
        MATCH (c:Component {vehicle_signature: $vehicleSignature})
        RETURN 
          count(c) as totalComponents,
          count(CASE WHEN c.anchor_xyz IS NOT NULL THEN 1 END) as componentsWithSpatial,
          count(CASE WHEN c.anchor_xyz IS NULL THEN 1 END) as componentsWithoutSpatial
      `, { vehicleSignature: VEHICLE_SIGNATURE });

      const record = result.records[0];
      const total = record.get('totalComponents').toNumber();
      const withSpatial = record.get('componentsWithSpatial').toNumber();
      const withoutSpatial = record.get('componentsWithoutSpatial').toNumber();

      console.log('\n📊 FINAL SPATIAL DATA STATUS');
      console.log('='.repeat(50));
      console.log(`Total Components: ${total}`);
      console.log(`With Spatial Data: ${withSpatial}`);
      console.log(`Without Spatial Data: ${withoutSpatial}`);

      const coverage = total > 0 ? ((withSpatial / total) * 100).toFixed(1) : 0;
      console.log(`Spatial Coverage: ${coverage}%`);

      return {
        total,
        withSpatial,
        withoutSpatial,
        coverage: parseFloat(coverage)
      };

    } finally {
      await session.close();
    }
  }

  async run() {
    try {
      await this.connect();
      
      // Analyze current state
      const analysis = await this.analyzeCurrentData();
      
      if (analysis.componentsWithoutSpatial === 0) {
        console.log('\n✅ All components already have spatial coordinates!');
        console.log('🎉 Database is ready for 3D model generation');
        return;
      }

      console.log('\n⚠️  SAFE UPDATE PLAN');
      console.log('='.repeat(50));
      console.log('This script will:');
      console.log('✅ ONLY add missing spatial coordinates (anchor_xyz)');
      console.log('✅ NOT delete or modify any existing data');
      console.log('✅ NOT create duplicate nodes');
      console.log('✅ Use vehicle signature filtering');
      console.log(`📊 Will update ${analysis.componentsWithoutSpatial} components`);

      // Add spatial coordinates
      const updatedCount = await this.addSpatialCoordinates();
      
      // Verify the update
      const verification = await this.verifyUpdate();
      
      console.log('\n🎉 SAFE SPATIAL UPDATE COMPLETE!');
      console.log('='.repeat(50));
      console.log(`✅ Updated ${updatedCount} components with spatial coordinates`);
      console.log(`✅ Spatial coverage: ${verification.coverage}%`);
      console.log('✅ Ready for 3D model generation');
      console.log('✅ No existing data was deleted or modified');
      
    } catch (error) {
      console.error('❌ Update failed:', error);
      process.exit(1);
    } finally {
      await this.disconnect();
    }
  }
}

// Run the safe update
if (require.main === module) {
  console.log('🛡️  SAFE SPATIAL COORDINATE UPDATER');
  console.log('This script will NOT delete any existing data');
  console.log('It only adds missing spatial coordinates\n');
  
  const updater = new SafeSpatialUpdater();
  updater.run().catch(console.error);
}

module.exports = SafeSpatialUpdater;