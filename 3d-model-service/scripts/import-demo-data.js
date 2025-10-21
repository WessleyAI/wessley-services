#!/usr/bin/env node

/**
 * Import demo NDJSON data into Neo4j
 * Converts the existing electrical system data to Neo4j graph format
 */

const neo4j = require('neo4j-driver');
const fs = require('fs');
const path = require('path');

// Configuration
const NEO4J_URI = process.env.NEO4J_URI || 'bolt://localhost:7687';
const NEO4J_USERNAME = process.env.NEO4J_USERNAME || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || 'password';

// Paths to demo data
const DEMO_DATA_PATH = '../../../n8n_workflow/demo';
const DATA_FILES = [
  'pajero_electrical_system.ndjson'
];

async function main() {
  const driver = neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USERNAME, NEO4J_PASSWORD));
  
  try {
    console.log('🔌 Connecting to Neo4j...');
    await verifyConnection(driver);
    console.log('✅ Connected to Neo4j successfully');

    // Clear existing data
    console.log('🧹 Clearing existing data...');
    await clearDatabase(driver);

    // Import NDJSON data
    for (const file of DATA_FILES) {
      const filePath = path.join(__dirname, DEMO_DATA_PATH, file);
      console.log(`📥 Importing ${file}...`);
      await importNDJSONFile(driver, filePath);
    }

    // Create indexes for performance
    console.log('🏗️ Creating indexes...');
    await createIndexes(driver);

    // Create spatial coordinates (extracted from demo viewer)
    console.log('📍 Adding spatial coordinates...');
    await addSpatialCoordinates(driver);

    // Verify import
    const stats = await getImportStats(driver);
    console.log('📊 Import Statistics:');
    console.log(`   Nodes: ${stats.nodeCount}`);
    console.log(`   Relationships: ${stats.relationshipCount}`);
    console.log(`   Node Types: ${stats.nodeTypes.join(', ')}`);

    console.log('🎉 Data import completed successfully!');

  } catch (error) {
    console.error('❌ Import failed:', error);
    process.exit(1);
  } finally {
    await driver.close();
  }
}

async function verifyConnection(driver) {
  const session = driver.session();
  try {
    await session.run('RETURN 1');
  } finally {
    await session.close();
  }
}

async function clearDatabase(driver) {
  const session = driver.session();
  try {
    await session.run('MATCH (n) DETACH DELETE n');
  } finally {
    await session.close();
  }
}

async function importNDJSONFile(driver, filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.trim().split('\n');
  
  const nodes = [];
  const edges = [];
  
  // Parse NDJSON lines
  for (const line of lines) {
    try {
      const record = JSON.parse(line);
      if (record.type === 'node') {
        nodes.push(record);
      } else if (record.type === 'edge') {
        edges.push(record);
      }
    } catch (error) {
      console.warn(`Failed to parse line: ${line.substring(0, 100)}...`);
    }
  }

  console.log(`   Found ${nodes.length} nodes and ${edges.length} edges`);

  // Import nodes
  await importNodes(driver, nodes);
  
  // Import edges
  await importEdges(driver, edges);
}

async function importNodes(driver, nodes) {
  const session = driver.session();
  
  try {
    for (const node of nodes) {
      const props = node.properties || {};
      const nodeType = props.node_type || 'unknown';
      
      // Create node with appropriate label
      const label = mapNodeTypeToLabel(nodeType);
      
      await session.run(`
        CREATE (n:${label} {
          id: $id,
          type: $type,
          node_type: $node_type,
          canonical_id: $canonical_id,
          code_id: $code_id,
          anchor_zone: $anchor_zone,
          zone: $zone,
          properties: $properties,
          created_at: datetime()
        })
      `, {
        id: node.id,
        type: nodeType,
        node_type: nodeType,
        canonical_id: props.canonical_id,
        code_id: props.code_id,
        anchor_zone: props.anchor_zone,
        zone: props.anchor_zone, // Use anchor_zone as zone for compatibility
        properties: props
      });
    }
  } finally {
    await session.close();
  }
}

async function importEdges(driver, edges) {
  const session = driver.session();
  
  try {
    for (const edge of edges) {
      const props = edge.properties || {};
      
      // Create relationship
      await session.run(`
        MATCH (from {id: $from_id})
        MATCH (to {id: $to_id})
        CREATE (from)-[r:CONNECTS_TO {
          id: $id,
          relationship_type: $rel_type,
          wire_gauge: $wire_gauge,
          wire_color: $wire_color,
          properties: $properties,
          created_at: datetime()
        }]->(to)
      `, {
        from_id: edge.from || edge.source,
        to_id: edge.to || edge.target,
        id: edge.id,
        rel_type: props.relationship_type || 'connection',
        wire_gauge: props.wire_gauge,
        wire_color: props.wire_color,
        properties: props
      });
    }
  } finally {
    await session.close();
  }
}

function mapNodeTypeToLabel(nodeType) {
  const labelMap = {
    'component': 'Component',
    'fuse': 'Component',
    'relay': 'Component', 
    'connector': 'Component',
    'ground_point': 'Component',
    'ground_plane': 'Component',
    'splice': 'Component',
    'pin': 'Component',
    'harness': 'Harness',
    'location': 'Location',
    'circuit': 'Circuit',
    'wire': 'Wire'
  };
  
  return labelMap[nodeType] || 'Node';
}

async function createIndexes(driver) {
  const session = driver.session();
  
  try {
    const indexes = [
      'CREATE INDEX component_id_idx IF NOT EXISTS FOR (c:Component) ON (c.id)',
      'CREATE INDEX component_type_idx IF NOT EXISTS FOR (c:Component) ON (c.type)',
      'CREATE INDEX component_zone_idx IF NOT EXISTS FOR (c:Component) ON (c.zone)',
      'CREATE INDEX harness_id_idx IF NOT EXISTS FOR (h:Harness) ON (h.id)',
      'CREATE INDEX location_id_idx IF NOT EXISTS FOR (l:Location) ON (l.id)',
      'CREATE CONSTRAINT component_id_unique IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE'
    ];
    
    for (const indexQuery of indexes) {
      try {
        await session.run(indexQuery);
      } catch (error) {
        // Index might already exist, continue
        console.warn(`Index creation warning: ${error.message}`);
      }
    }
  } finally {
    await session.close();
  }
}

async function addSpatialCoordinates(driver) {
  const session = driver.session();
  
  try {
    // Add spatial coordinates based on zone and component type
    // These coordinates are derived from the enhanced-r3f-viewer positioning
    const spatialUpdates = [
      // Engine bay components
      { zone: 'Engine Compartment', component: 'Battery', coords: [0.5, -0.8, 0.8], bbox: [0.3, 0.2, 0.2] },
      { zone: 'Engine Compartment', component: 'Alternator', coords: [0.8, -0.6, 0.6], bbox: [0.15, 0.15, 0.15] },
      { zone: 'Engine Compartment', component: 'Starter', coords: [1.2, -0.4, 0.5], bbox: [0.12, 0.12, 0.2] },
      { zone: 'Engine Compartment', component: 'Ignition', coords: [1.0, -0.2, 0.7], bbox: [0.08, 0.08, 0.1] },
      
      // Dash panel components  
      { zone: 'Dash Panel', component: 'ECU', coords: [2.1, 0.0, 0.9], bbox: [0.2, 0.15, 0.05] },
      { zone: 'Dash Panel', component: 'A/T-ECU', coords: [2.1, 0.3, 0.9], bbox: [0.15, 0.1, 0.05] },
      
      // Ground points
      { zone: 'Engine Compartment', type: 'ground_point', coords: [0.2, -0.5, 0.1], bbox: [0.04, 0.04, 0.04] },
      { zone: 'Dash Panel', type: 'ground_point', coords: [2.0, 0.0, 0.1], bbox: [0.04, 0.04, 0.04] },
      { zone: 'Rear Cargo/Tailgate', type: 'ground_point', coords: [4.0, 0.0, 0.1], bbox: [0.04, 0.04, 0.04] }
    ];
    
    for (const update of spatialUpdates) {
      let query = 'MATCH (c:Component) WHERE c.zone = $zone';
      let params = { zone: update.zone };
      
      if (update.component) {
        query += ' AND (c.canonical_id CONTAINS $component OR c.code_id CONTAINS $component)';
        params.component = update.component;
      } else if (update.type) {
        query += ' AND c.type = $type';
        params.type = update.type;
      }
      
      query += `
        SET c.anchor_xyz = $coords,
            c.bbox_m = $bbox,
            c.position = $coords
        RETURN c.id, c.canonical_id
      `;
      
      params.coords = update.coords;
      params.bbox = update.bbox;
      
      const result = await session.run(query, params);
      if (result.records.length > 0) {
        console.log(`   Updated coordinates for: ${result.records.map(r => r.get('c.canonical_id')).join(', ')}`);
      }
    }
  } finally {
    await session.close();
  }
}

async function getImportStats(driver) {
  const session = driver.session();
  
  try {
    const result = await session.run(`
      MATCH (n)
      OPTIONAL MATCH ()-[r]->()
      RETURN 
        count(DISTINCT n) as nodeCount,
        count(DISTINCT r) as relationshipCount,
        collect(DISTINCT labels(n)[0]) as nodeTypes
    `);
    
    const record = result.records[0];
    return {
      nodeCount: record.get('nodeCount').toNumber(),
      relationshipCount: record.get('relationshipCount').toNumber(),
      nodeTypes: record.get('nodeTypes').filter(Boolean)
    };
  } finally {
    await session.close();
  }
}

// Run the import
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { main };