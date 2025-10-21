#!/usr/bin/env node

/**
 * Complete GraphML Import with Spatial Coordinate Generation
 * Clears database and re-imports all GraphML data with proper spatial coordinates
 */

const neo4j = require('neo4j-driver');
const fs = require('fs');
const path = require('path');
const { DOMParser } = require('xmldom');

// Configuration
const NEO4J_URI = process.env.NEO4J_URI || 'bolt://localhost:7687';
const NEO4J_USERNAME = process.env.NEO4J_USERNAME || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || 'password';
const VEHICLE_SIGNATURE = process.env.VEHICLE_SIGNATURE || 'pajero_pinin_2001';
const GRAPHML_FILE = '../../../n8n_workflow/demo/model.xml';

// 3D Zone Layout for Pajero Pinin (realistic spatial arrangement)
const ZONE_COORDINATES = {
  // Engine Compartment (front of vehicle)
  'Engine Compartment': { x: 0, y: 0, z: 2000 },
  'Engine Bay': { x: 0, y: 0, z: 2000 },
  
  // Dashboard and Cabin
  'Dash Panel': { x: 0, y: 0, z: 0 },
  'Dashboard': { x: 0, y: 0, z: 0 },
  'Cabin': { x: 0, y: 0, z: -500 },
  
  // Chassis and Floor
  'Chassis': { x: 0, y: -500, z: 0 },
  'Floor': { x: 0, y: -500, z: -1000 },
  'Floor & Roof': { x: 0, y: 0, z: -1000 },
  
  // Doors
  'Left Front Door': { x: -800, y: 0, z: 0 },
  'Right Front Door': { x: 800, y: 0, z: 0 },
  'Left Rear Door': { x: -800, y: 0, z: -1500 },
  'Right Rear Door': { x: 800, y: 0, z: -1500 },
  
  // Rear
  'Rear': { x: 0, y: 0, z: -3000 },
  'Tailgate': { x: 0, y: 0, z: -3200 },
  
  // Default for unknown zones
  'Unknown': { x: 0, y: 0, z: 0 }
};

// Component type offsets within zones
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

class CompleteGraphMLImporter {
  constructor() {
    this.driver = null;
    this.nodes = [];
    this.edges = [];
    this.keyDefinitions = new Map();
    this.componentCounter = new Map(); // For positioning multiple components in same zone
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

  parseGraphML(filePath) {
    console.log(`📄 Reading GraphML file: ${filePath}`);
    
    const xmlContent = fs.readFileSync(filePath, 'utf8');
    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlContent, 'text/xml');

    // Parse key definitions
    this.parseKeyDefinitions(doc);

    // Parse nodes
    const nodeElements = doc.getElementsByTagName('node');
    console.log(`🔍 Found ${nodeElements.length} nodes`);
    
    for (let i = 0; i < nodeElements.length; i++) {
      const node = this.extractNodeData(nodeElements[i]);
      this.nodes.push(node);
    }

    // Parse edges
    const edgeElements = doc.getElementsByTagName('edge');
    console.log(`🔍 Found ${edgeElements.length} edges`);
    
    for (let i = 0; i < edgeElements.length; i++) {
      const edge = this.extractEdgeData(edgeElements[i]);
      this.edges.push(edge);
    }

    console.log(`✅ Parsed ${this.nodes.length} nodes and ${this.edges.length} edges`);
  }

  parseKeyDefinitions(doc) {
    const keyElements = doc.getElementsByTagName('key');
    for (let i = 0; i < keyElements.length; i++) {
      const key = keyElements[i];
      const id = key.getAttribute('id');
      const attrName = key.getAttribute('attr.name');
      const attrType = key.getAttribute('attr.type');
      const forElement = key.getAttribute('for');
      
      this.keyDefinitions.set(id, {
        name: attrName,
        type: attrType,
        for: forElement
      });
    }
    console.log(`📋 Found ${this.keyDefinitions.size} key definitions`);
  }

  extractNodeData(nodeElement) {
    const nodeData = {
      id: nodeElement.getAttribute('id'),
      properties: {}
    };

    // Extract all data elements
    const dataElements = nodeElement.getElementsByTagName('data');
    for (let i = 0; i < dataElements.length; i++) {
      const dataElement = dataElements[i];
      const key = dataElement.getAttribute('key');
      const value = dataElement.textContent?.trim() || '';
      
      if (key && value) {
        const keyDef = this.keyDefinitions.get(key);
        const propName = keyDef?.name || key;
        nodeData.properties[propName] = this.convertValue(value, keyDef?.type);
      }
    }

    return nodeData;
  }

  extractEdgeData(edgeElement) {
    const edgeData = {
      id: edgeElement.getAttribute('id') || `${edgeElement.getAttribute('source')}_to_${edgeElement.getAttribute('target')}`,
      source: edgeElement.getAttribute('source'),
      target: edgeElement.getAttribute('target'),
      properties: {}
    };

    // Extract all data elements
    const dataElements = edgeElement.getElementsByTagName('data');
    for (let i = 0; i < dataElements.length; i++) {
      const dataElement = dataElements[i];
      const key = dataElement.getAttribute('key');
      const value = dataElement.textContent?.trim() || '';
      
      if (key && value) {
        const keyDef = this.keyDefinitions.get(key);
        const propName = keyDef?.name || key;
        edgeData.properties[propName] = this.convertValue(value, keyDef?.type);
      }
    }

    return edgeData;
  }

  convertValue(value, type) {
    switch (type) {
      case 'int':
        return parseInt(value, 10);
      case 'double':
      case 'float':
        return parseFloat(value);
      case 'boolean':
        return value.toLowerCase() === 'true';
      default:
        return value;
    }
  }

  generateSpatialCoordinates(node) {
    const nodeType = node.properties.node_type || 'component';
    const anchorZone = node.properties.anchor_zone || 'Unknown';
    
    // Get base coordinates for the zone
    const zoneCoords = ZONE_COORDINATES[anchorZone] || ZONE_COORDINATES['Unknown'];
    
    // Get offset for component type
    const typeOffset = COMPONENT_TYPE_OFFSETS[nodeType] || COMPONENT_TYPE_OFFSETS['component'];
    
    // Get counter for this zone to spread components
    const zoneKey = `${anchorZone}_${nodeType}`;
    const counter = this.componentCounter.get(zoneKey) || 0;
    this.componentCounter.set(zoneKey, counter + 1);
    
    // Add some spacing between similar components in same zone
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

  async clearDatabase() {
    console.log('🧹 Clearing existing data...');
    const session = this.driver.session();
    try {
      await session.run('MATCH (n) DETACH DELETE n');
      console.log('✅ Database cleared');
    } finally {
      await session.close();
    }
  }

  async importNodes() {
    console.log(`📥 Importing ${this.nodes.length} nodes with spatial coordinates...`);
    const session = this.driver.session();
    
    try {
      for (const node of this.nodes) {
        const nodeType = node.properties.node_type || 'Unknown';
        const label = this.mapNodeTypeToLabel(nodeType);
        
        // Generate spatial coordinates
        const position = this.generateSpatialCoordinates(node);
        
        // Prepare properties with spatial data
        const props = {
          id: node.id,
          type: nodeType,
          vehicle_signature: VEHICLE_SIGNATURE,
          anchor_xyz: position,
          position: position, // Duplicate for convenience
          ...node.properties,
          created_at: new Date().toISOString()
        };

        // Create node with dynamic label
        const query = `
          CREATE (n:${label} $props)
          RETURN n.id as id
        `;
        
        await session.run(query, { props });
      }
      
      console.log(`✅ Imported ${this.nodes.length} nodes with spatial coordinates`);
    } finally {
      await session.close();
    }
  }

  async importEdges() {
    console.log(`📥 Importing ${this.edges.length} edges...`);
    const session = this.driver.session();
    
    try {
      for (const edge of this.edges) {
        const relType = this.mapRelationshipType(edge.properties.relationship || 'CONNECTS_TO');
        
        // Prepare properties
        const props = {
          id: edge.id,
          vehicle_signature: VEHICLE_SIGNATURE,
          ...edge.properties,
          created_at: new Date().toISOString()
        };

        // Create relationship ensuring both nodes belong to same vehicle
        const query = `
          MATCH (from {id: $fromId, vehicle_signature: $vehicleSignature})
          MATCH (to {id: $toId, vehicle_signature: $vehicleSignature})
          CREATE (from)-[r:${relType} $props]->(to)
          RETURN r.id as id
        `;
        
        try {
          await session.run(query, {
            fromId: edge.source,
            toId: edge.target,
            vehicleSignature: VEHICLE_SIGNATURE,
            props
          });
        } catch (error) {
          console.warn(`Failed to create edge ${edge.id}: ${error.message}`);
        }
      }
      
      console.log(`✅ Imported ${this.edges.length} edges`);
    } finally {
      await session.close();
    }
  }

  async createIndexes() {
    console.log('🏗️ Creating indexes and constraints...');
    const session = this.driver.session();
    
    try {
      const commands = [
        // Composite constraints with vehicle_signature for data isolation
        'CREATE CONSTRAINT component_vehicle_id_unique IF NOT EXISTS FOR (c:Component) REQUIRE (c.id, c.vehicle_signature) IS UNIQUE',
        'CREATE CONSTRAINT harness_vehicle_id_unique IF NOT EXISTS FOR (h:Harness) REQUIRE (h.id, h.vehicle_signature) IS UNIQUE',
        'CREATE CONSTRAINT location_vehicle_id_unique IF NOT EXISTS FOR (l:Location) REQUIRE (l.id, l.vehicle_signature) IS UNIQUE',
        'CREATE CONSTRAINT circuit_vehicle_id_unique IF NOT EXISTS FOR (c:Circuit) REQUIRE (c.id, c.vehicle_signature) IS UNIQUE',
        
        // Performance indexes
        'CREATE INDEX component_vehicle_sig_idx IF NOT EXISTS FOR (c:Component) ON (c.vehicle_signature)',
        'CREATE INDEX component_type_idx IF NOT EXISTS FOR (c:Component) ON (c.type)',
        'CREATE INDEX component_zone_idx IF NOT EXISTS FOR (c:Component) ON (c.anchor_zone)',
        'CREATE INDEX spatial_idx IF NOT EXISTS FOR (c:Component) ON (c.anchor_xyz)',
        'CREATE INDEX wire_color_idx IF NOT EXISTS FOR (w:Wire) ON (w.color)',
        'CREATE INDEX wire_gauge_idx IF NOT EXISTS FOR (w:Wire) ON (w.gauge)'
      ];
      
      for (const command of commands) {
        try {
          await session.run(command);
        } catch (error) {
          if (!error.message.includes('already exists') && !error.message.includes('equivalent')) {
            console.warn(`Index/constraint warning: ${error.message}`);
          }
        }
      }
      
      console.log('✅ Created indexes and constraints');
    } finally {
      await session.close();
    }
  }

  mapNodeTypeToLabel(nodeType) {
    const labelMap = {
      'component': 'Component',
      'fuse': 'Component',
      'relay': 'Component',
      'connector': 'Component',
      'ground_point': 'Component',
      'ground_plane': 'Component',
      'splice': 'Component',
      'pin': 'Component',
      'terminal': 'Component',
      'harness': 'Harness',
      'wire': 'Wire',
      'location': 'Location',
      'circuit': 'Circuit',
      'power_rail': 'PowerRail',
      'unknown': 'Node'
    };
    
    return labelMap[nodeType.toLowerCase()] || 'Component';
  }

  mapRelationshipType(relationship) {
    const relMap = {
      'connected_to': 'CONNECTS_TO',
      'powers': 'POWERS',
      'powered_by': 'POWERED_BY',
      'controls': 'CONTROLS',
      'controlled_by': 'CONTROLLED_BY',
      'grounds_to': 'GROUNDS_TO',
      'wire_to_ground': 'GROUNDS_TO',
      'ground_to_plane': 'GROUNDS_TO',
      'routes_through': 'ROUTES_THROUGH',
      'contains': 'CONTAINS',
      'part_of': 'PART_OF'
    };
    
    return relMap[relationship?.toLowerCase()] || 'CONNECTS_TO';
  }

  async getImportStats() {
    console.log('📊 Gathering import statistics...');
    const session = this.driver.session();
    
    try {
      const result = await session.run(`
        MATCH (n {vehicle_signature: $vehicleSignature})
        OPTIONAL MATCH (n1 {vehicle_signature: $vehicleSignature})-[r]->(n2 {vehicle_signature: $vehicleSignature})
        RETURN 
          count(DISTINCT n) as nodeCount,
          count(DISTINCT r) as relationshipCount,
          collect(DISTINCT labels(n)[0]) as nodeTypes,
          collect(DISTINCT type(r)) as relationshipTypes
      `, { vehicleSignature: VEHICLE_SIGNATURE });
      
      const record = result.records[0];
      const stats = {
        nodeCount: record.get('nodeCount').toNumber(),
        relationshipCount: record.get('relationshipCount').toNumber(),
        nodeTypes: record.get('nodeTypes').filter(Boolean),
        relationshipTypes: record.get('relationshipTypes').filter(Boolean)
      };

      // Check spatial data coverage
      const spatialResult = await session.run(`
        MATCH (c:Component {vehicle_signature: $vehicleSignature})
        WHERE c.anchor_xyz IS NOT NULL
        RETURN count(c) as spatialComponents
      `, { vehicleSignature: VEHICLE_SIGNATURE });
      
      const spatialComponents = spatialResult.records[0].get('spatialComponents').toNumber();

      console.log('\n📈 COMPLETE IMPORT STATISTICS');
      console.log('='.repeat(50));
      console.log(`Vehicle: ${VEHICLE_SIGNATURE}`);
      console.log(`Total Nodes: ${stats.nodeCount}`);
      console.log(`Total Relationships: ${stats.relationshipCount}`);
      console.log(`Components with Spatial Data: ${spatialComponents}`);
      console.log(`Node Types: ${stats.nodeTypes.join(', ')}`);
      console.log(`Relationship Types: ${stats.relationshipTypes.join(', ')}`);

      return { ...stats, spatialComponents };
    } finally {
      await session.close();
    }
  }

  async run() {
    try {
      await this.connect();

      // Parse GraphML file
      const graphmlPath = path.join(__dirname, GRAPHML_FILE);
      if (!fs.existsSync(graphmlPath)) {
        throw new Error(`GraphML file not found: ${graphmlPath}`);
      }
      
      this.parseGraphML(graphmlPath);

      // Clear and reimport
      await this.clearDatabase();
      await this.importNodes();
      await this.importEdges();
      await this.createIndexes();

      // Show final statistics
      const stats = await this.getImportStats();

      console.log('\n🎉 COMPLETE GRAPHML IMPORT FINISHED!');
      console.log('🌐 Neo4j Browser: http://localhost:7474');
      console.log('👤 Username: neo4j');
      console.log('🔑 Password: password');
      console.log(`📊 ${stats.spatialComponents} components ready for 3D modeling`);

    } catch (error) {
      console.error('❌ Import failed:', error);
      process.exit(1);
    } finally {
      await this.disconnect();
    }
  }
}

// Run the complete import
if (require.main === module) {
  const importer = new CompleteGraphMLImporter();
  importer.run().catch(console.error);
}

module.exports = CompleteGraphMLImporter;