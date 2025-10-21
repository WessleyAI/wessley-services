# Knowledge Graph Service Architecture

## Overview
**Language**: Python (FastAPI + Neo4j)  
**Primary Function**: Store, query, and analyze electrical system relationships and component data  
**Status**: ✅ **Production Ready** (Integrated with 3D Model Service)

## Core Responsibilities
- Store component specifications and electrical relationships
- Maintain circuit topology and connection data
- Provide graph traversal and electrical system analysis
- Support complex queries for 3D model generation
- Manage vehicle signature-based data isolation
- Handle GraphML imports and data validation
- Provide analytics and insights on electrical systems

## Technology Stack
```json
{
  "framework": "FastAPI",
  "language": "Python 3.11+",
  "database": "Neo4j 5.x",
  "driver": "neo4j-python-driver",
  "query_language": "Cypher",
  "data_format": "GraphML + JSON",
  "cache": "Redis",
  "monitoring": "Prometheus + Grafana",
  "testing": "pytest + testcontainers",
  "deployment": "Docker + Kubernetes"
}
```

## Service Architecture
```
src/
├── models/
│   ├── nodes/
│   │   ├── component.py             # Component node model
│   │   ├── circuit.py               # Circuit node model
│   │   ├── zone.py                  # Physical zone model
│   │   ├── connector.py             # Connector node model
│   │   └── vehicle.py               # Vehicle metadata model
│   ├── relationships/
│   │   ├── connects_to.py           # Physical connections
│   │   ├── powered_by.py            # Power relationships
│   │   ├── controls.py              # Control relationships
│   │   ├── located_in.py            # Spatial relationships
│   │   └── part_of.py               # Hierarchical relationships
│   └── schemas/
│       ├── component_schema.py      # Component validation schemas
│       ├── circuit_schema.py        # Circuit validation schemas
│       └── import_schema.py         # Data import schemas
├── services/
│   ├── graph_service.py             # Core graph operations
│   ├── query_service.py             # Complex query operations
│   ├── import_service.py            # Data import and validation
│   ├── analysis_service.py          # Electrical system analysis
│   ├── spatial_service.py           # Spatial data management
│   └── vehicle_service.py           # Vehicle-specific operations
├── repositories/
│   ├── component_repository.py      # Component CRUD operations
│   ├── circuit_repository.py        # Circuit CRUD operations
│   ├── relationship_repository.py   # Relationship management
│   └── analytics_repository.py      # Analytics and reporting
├── queries/
│   ├── component_queries.py         # Component-specific Cypher queries
│   ├── circuit_queries.py           # Circuit analysis queries
│   ├── spatial_queries.py           # Spatial relationship queries
│   ├── analytics_queries.py         # Analytics and reporting queries
│   └── optimization_queries.py      # Performance-optimized queries
├── api/
│   ├── components.py                # Component management endpoints
│   ├── circuits.py                  # Circuit analysis endpoints
│   ├── vehicles.py                  # Vehicle data endpoints
│   ├── analysis.py                  # System analysis endpoints
│   ├── import_export.py             # Data import/export endpoints
│   └── health.py                    # Health check endpoints
├── importers/
│   ├── graphml_importer.py          # GraphML file processing
│   ├── json_importer.py             # JSON data import
│   ├── csv_importer.py              # CSV data import
│   └── validation_service.py        # Data validation and cleanup
└── utils/
    ├── neo4j_utils.py               # Neo4j client utilities
    ├── cypher_builder.py            # Dynamic query building
    ├── data_validation.py           # Data integrity validation
    └── performance_utils.py         # Query optimization utilities
```

## Neo4j Database Schema

### Node Types
```cypher
// Component nodes - electrical components in the system
CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE;
CREATE INDEX component_vehicle IF NOT EXISTS FOR (c:Component) ON (c.vehicle_signature);
CREATE INDEX component_type IF NOT EXISTS FOR (c:Component) ON (c.type);

(:Component {
  id: "string",                    // Unique component identifier
  vehicle_signature: "string",     // Vehicle isolation key
  type: "string",                  // relay, fuse, sensor, etc.
  name: "string",                  // Human-readable name
  part_number: "string",           // Manufacturer part number
  manufacturer: "string",          // Component manufacturer
  specifications: "map",           // Technical specifications
  position: "map",                 // 3D position {x, y, z}
  zone: "string",                  // Physical location zone
  voltage_rating: "float",         // Voltage rating (V)
  current_rating: "float",         // Current rating (A)
  created_at: "datetime",
  updated_at: "datetime"
})

// Circuit nodes - electrical circuits
CREATE CONSTRAINT circuit_id IF NOT EXISTS FOR (c:Circuit) REQUIRE c.id IS UNIQUE;
CREATE INDEX circuit_vehicle IF NOT EXISTS FOR (c:Circuit) ON (c.vehicle_signature);

(:Circuit {
  id: "string",
  vehicle_signature: "string",
  name: "string",                  // Circuit name (e.g., "Headlight Circuit")
  voltage: "float",                // Operating voltage
  max_current: "float",            // Maximum current capacity
  circuit_type: "string",          // power, signal, ground
  protection: "string",            // Fuse/breaker type
  created_at: "datetime",
  updated_at: "datetime"
})

// Zone nodes - physical zones in the vehicle
CREATE CONSTRAINT zone_id IF NOT EXISTS FOR (z:Zone) REQUIRE z.id IS UNIQUE;

(:Zone {
  id: "string",
  vehicle_signature: "string",
  name: "string",                  // Engine bay, dashboard, etc.
  bounds: "map",                   // 3D bounds {min_x, max_x, min_y, max_y, min_z, max_z}
  access_level: "string",          // easy, moderate, difficult
  environmental_conditions: "map", // temperature, humidity, vibration
  created_at: "datetime"
})

// Connector nodes - physical connectors
CREATE CONSTRAINT connector_id IF NOT EXISTS FOR (c:Connector) REQUIRE c.id IS UNIQUE;

(:Connector {
  id: "string",
  vehicle_signature: "string",
  type: "string",                  // terminal, plug, socket
  pin_count: "integer",           // Number of pins/terminals
  connector_family: "string",      // Connector standard/family
  position: "map",                // 3D position
  created_at: "datetime"
})

// Vehicle nodes - vehicle metadata
CREATE CONSTRAINT vehicle_signature IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.signature IS UNIQUE;

(:Vehicle {
  signature: "string",             // Unique vehicle identifier
  make: "string",                  // Vehicle manufacturer
  model: "string",                 // Vehicle model
  year: "integer",                 // Model year
  engine: "string",                // Engine specification
  market: "string",                // Geographic market
  created_at: "datetime",
  metadata: "map"                  // Additional vehicle data
})
```

### Relationship Types
```cypher
// Physical electrical connections
(:Component)-[:CONNECTS_TO {
  wire_gauge: "string",            // Wire gauge (e.g., "2.5mm²")
  wire_color: "string",            // Wire color code
  wire_length: "float",            // Estimated length (mm)
  connection_type: "string",       // direct, via_connector, splice
  signal_type: "string",           // power, ground, signal, data
  created_at: "datetime"
}]->(:Component)

// Power supply relationships
(:Component)-[:POWERED_BY {
  power_type: "string",            // battery, alternator, regulated
  voltage: "float",                // Supply voltage
  max_current: "float",            // Maximum current draw
  always_on: "boolean",            // Constant power vs switched
  created_at: "datetime"
}]->(:Component)

// Control relationships
(:Component)-[:CONTROLS {
  control_type: "string",          // switch, relay, pwm, data
  signal_voltage: "float",         // Control signal voltage
  response_time: "float",          // Response time (ms)
  created_at: "datetime"
}]->(:Component)

// Spatial relationships
(:Component)-[:LOCATED_IN {
  position_type: "string",         // mounted, floating, integrated
  accessibility: "string",         // easy, moderate, difficult
  created_at: "datetime"
}]->(:Zone)

// Circuit membership
(:Component)-[:PART_OF {
  role: "string",                  // source, load, protection, control
  is_critical: "boolean",          // Critical component flag
  created_at: "datetime"
}]->(:Circuit)

// Connector relationships
(:Component)-[:USES_CONNECTOR {
  pin_assignment: "map",           // Pin to function mapping
  connector_side: "string",        // male, female, either
  created_at: "datetime"
}]->(:Connector)
```

## Core Query Operations

### 1. Vehicle-Isolated Component Retrieval
```python
# Get all components for a specific vehicle
async def get_vehicle_components(vehicle_signature: str):
    query = """
    MATCH (c:Component {vehicle_signature: $signature})
    OPTIONAL MATCH (c)-[r:CONNECTS_TO]->(connected:Component)
    RETURN c, collect({relationship: r, component: connected}) as connections
    ORDER BY c.type, c.name
    """
    return await execute_query(query, signature=vehicle_signature)

# Get components by type with spatial data
async def get_components_by_type(vehicle_signature: str, component_type: str):
    query = """
    MATCH (c:Component {vehicle_signature: $signature, type: $type})
    OPTIONAL MATCH (c)-[:LOCATED_IN]->(z:Zone)
    RETURN c, z
    ORDER BY c.name
    """
    return await execute_query(query, signature=vehicle_signature, type=component_type)
```

### 2. Circuit Analysis
```python
# Analyze complete electrical circuit
async def analyze_circuit(vehicle_signature: str, circuit_name: str):
    query = """
    MATCH (circuit:Circuit {vehicle_signature: $signature, name: $circuit_name})
    MATCH (c:Component)-[:PART_OF]->(circuit)
    OPTIONAL MATCH path = (source:Component)-[:POWERED_BY*..5]->(c)
    WHERE source.type IN ['battery', 'alternator', 'power_supply']
    OPTIONAL MATCH (c)-[conn:CONNECTS_TO]->(other:Component)-[:PART_OF]->(circuit)
    RETURN circuit, c, path, collect({connection: conn, component: other}) as circuit_connections
    """
    return await execute_query(query, signature=vehicle_signature, circuit_name=circuit_name)

# Find power distribution paths
async def trace_power_path(vehicle_signature: str, from_component: str, to_component: str):
    query = """
    MATCH (start:Component {vehicle_signature: $signature, id: $from_id})
    MATCH (end:Component {vehicle_signature: $signature, id: $to_id})
    MATCH path = shortestPath((start)-[:CONNECTS_TO|POWERED_BY*..10]->(end))
    RETURN path, length(path) as path_length
    ORDER BY path_length
    LIMIT 5
    """
    return await execute_query(query, signature=vehicle_signature, from_id=from_component, to_id=to_component)
```

### 3. Spatial Queries for 3D Generation
```python
# Get components with spatial coordinates for 3D model generation
async def get_spatial_layout(vehicle_signature: str):
    query = """
    MATCH (c:Component {vehicle_signature: $signature})
    WHERE c.position IS NOT NULL
    OPTIONAL MATCH (c)-[:LOCATED_IN]->(z:Zone)
    OPTIONAL MATCH (c)-[conn:CONNECTS_TO]->(other:Component)
    WHERE other.vehicle_signature = $signature AND other.position IS NOT NULL
    RETURN c, z, collect({
        connection: conn,
        target_component: other,
        target_position: other.position
    }) as connections
    ORDER BY z.name, c.type, c.name
    """
    return await execute_query(query, signature=vehicle_signature)

# Get wire harness routing data
async def get_wire_harness_data(vehicle_signature: str):
    query = """
    MATCH (c1:Component {vehicle_signature: $signature})-[conn:CONNECTS_TO]->(c2:Component {vehicle_signature: $signature})
    WHERE c1.position IS NOT NULL AND c2.position IS NOT NULL
    RETURN c1, c2, conn,
           distance(point({x: c1.position.x, y: c1.position.y, z: c1.position.z}),
                   point({x: c2.position.x, y: c2.position.y, z: c2.position.z})) as wire_length
    ORDER BY wire_length DESC
    """
    return await execute_query(query, signature=vehicle_signature)
```

### 4. System Analysis and Validation
```python
# Validate electrical system integrity
async def validate_electrical_system(vehicle_signature: str):
    query = """
    MATCH (v:Vehicle {signature: $signature})
    
    // Count components without power source
    OPTIONAL MATCH (unpowered:Component {vehicle_signature: $signature})
    WHERE NOT (unpowered)-[:POWERED_BY]->()
    AND unpowered.type NOT IN ['battery', 'alternator', 'ground']
    
    // Count components without spatial data
    OPTIONAL MATCH (no_position:Component {vehicle_signature: $signature})
    WHERE no_position.position IS NULL
    
    // Count orphaned connectors
    OPTIONAL MATCH (orphan_connector:Connector {vehicle_signature: $signature})
    WHERE NOT (orphan_connector)<-[:USES_CONNECTOR]-()
    
    RETURN v,
           count(DISTINCT unpowered) as unpowered_components,
           count(DISTINCT no_position) as components_without_position,
           count(DISTINCT orphan_connector) as orphaned_connectors
    """
    return await execute_query(query, signature=vehicle_signature)

# Get system statistics
async def get_system_statistics(vehicle_signature: str):
    query = """
    MATCH (v:Vehicle {signature: $signature})
    
    OPTIONAL MATCH (c:Component {vehicle_signature: $signature})
    OPTIONAL MATCH (circuit:Circuit {vehicle_signature: $signature})
    OPTIONAL MATCH (z:Zone {vehicle_signature: $signature})
    OPTIONAL MATCH ()-[conn:CONNECTS_TO {vehicle_signature: $signature}]->()
    
    RETURN v,
           count(DISTINCT c) as total_components,
           count(DISTINCT circuit) as total_circuits,
           count(DISTINCT z) as total_zones,
           count(DISTINCT conn) as total_connections,
           collect(DISTINCT c.type) as component_types
    """
    return await execute_query(query, signature=vehicle_signature)
```

## API Endpoints

### Component Management
```python
# Get all components for a vehicle
GET /api/v1/vehicles/:signature/components
GET /api/v1/vehicles/:signature/components?type=relay&zone=engine_bay

# Get specific component with relationships
GET /api/v1/components/:id
GET /api/v1/components/:id/connections
GET /api/v1/components/:id/circuits

# Component CRUD operations
POST /api/v1/components
PUT /api/v1/components/:id
DELETE /api/v1/components/:id
```

### Circuit Analysis
```python
# Get all circuits for a vehicle
GET /api/v1/vehicles/:signature/circuits

# Analyze specific circuit
GET /api/v1/circuits/:id/analysis
GET /api/v1/circuits/:id/components
GET /api/v1/circuits/:id/power-flow

# Circuit management
POST /api/v1/circuits
PUT /api/v1/circuits/:id
DELETE /api/v1/circuits/:id
```

### Spatial Data for 3D Generation
```python
# Get spatial layout for 3D model generation
GET /api/v1/vehicles/:signature/spatial-layout
GET /api/v1/vehicles/:signature/wire-harnesses
GET /api/v1/vehicles/:signature/zones

# Update spatial coordinates
PUT /api/v1/components/:id/position
POST /api/v1/vehicles/:signature/spatial-batch-update
```

### System Analysis
```python
# System validation and health checks
GET /api/v1/vehicles/:signature/validation
GET /api/v1/vehicles/:signature/statistics
GET /api/v1/vehicles/:signature/integrity-report

# Performance analytics
GET /api/v1/analytics/query-performance
GET /api/v1/analytics/data-quality
```

### Data Import/Export
```python
# Import data from various formats
POST /api/v1/import/graphml
POST /api/v1/import/json
POST /api/v1/import/csv

# Export data
GET /api/v1/vehicles/:signature/export?format=graphml
GET /api/v1/vehicles/:signature/export?format=json
```

## Integration with 3D Model Service

### Data Retrieval Interface
```python
# Standardized interface for 3D Model Service
class GraphDataProvider:
    async def get_components_with_spatial_data(self, vehicle_signature: str):
        """Get all components with 3D coordinates for model generation"""
        pass
    
    async def get_wire_connections(self, vehicle_signature: str):
        """Get wire connection data for harness routing"""
        pass
    
    async def get_component_specifications(self, component_ids: List[str]):
        """Get detailed specifications for 3D mesh generation"""
        pass
    
    async def validate_vehicle_data_completeness(self, vehicle_signature: str):
        """Validate that vehicle has sufficient data for 3D generation"""
        pass
```

### Real-time Data Sync
```python
# Notify 3D Model Service of data changes
async def notify_data_change(vehicle_signature: str, change_type: str, affected_components: List[str]):
    webhook_payload = {
        "vehicle_signature": vehicle_signature,
        "change_type": change_type,  # created, updated, deleted
        "affected_components": affected_components,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await http_client.post(
        f"{MODEL_SERVICE_URL}/api/v1/webhooks/graph-data-change",
        json=webhook_payload
    )
```

## Performance Characteristics
- **Query Response Time**: < 100ms for typical component queries
- **Complex Analysis**: < 500ms for full circuit analysis
- **Spatial Queries**: < 200ms for 3D layout retrieval
- **Data Import**: 1000+ components per minute
- **Concurrent Queries**: 500+ simultaneous operations
- **Database Size**: Supports 100,000+ components per vehicle

## Environment Configuration
```bash
# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=wessley

# Cache
REDIS_URL=redis://localhost:6379/3

# Service Integration
MODEL_SERVICE_URL=http://localhost:3001
LEARNING_SERVICE_URL=http://localhost:3002
SEMANTIC_SERVICE_URL=http://localhost:3003

# Data Import
MAX_IMPORT_FILE_SIZE=100MB
ENABLE_AUTO_SPATIAL_GENERATION=true
DEFAULT_VEHICLE_SIGNATURE=pajero_pinin_2001

# Service Configuration
PORT=3004
LOG_LEVEL=info
ENABLE_QUERY_LOGGING=true
```

## Data Quality & Validation

### Automated Validation Rules
```python
validation_rules = {
    "component_validation": [
        "vehicle_signature_required",
        "unique_component_id",
        "valid_component_type",
        "specifications_schema_valid"
    ],
    "spatial_validation": [
        "position_coordinates_valid",
        "position_within_vehicle_bounds",
        "no_component_overlap"
    ],
    "electrical_validation": [
        "voltage_ratings_consistent",
        "current_capacity_adequate",
        "power_source_reachable"
    ],
    "relationship_validation": [
        "bidirectional_connections_consistent",
        "circuit_continuity_maintained",
        "connector_pin_assignments_valid"
    ]
}
```

## Future Enhancements

### Phase 1: Advanced Analytics
- Electrical load analysis and optimization
- Fault detection and diagnostic capabilities
- Performance modeling and simulation

### Phase 2: AI Integration
- Automated spatial coordinate generation
- Intelligent component placement suggestions
- Predictive maintenance recommendations

### Phase 3: Multi-Vehicle Support
- Cross-vehicle component analysis
- Family/platform-based comparisons
- Component standardization insights

This service provides the foundational knowledge graph capabilities that power the entire Wessley.ai electrical system analysis platform.