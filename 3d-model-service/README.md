# 3D Model Generation Service

Production-ready service for converting Neo4j electrical system knowledge graphs into interactive GLB 3D models.

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install dependencies
npm install

# Start Neo4j (Docker)
docker run -d \
  --name neo4j-wessley \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["graph-data-science", "apoc"]' \
  neo4j:5.12-community

# Start Redis (Docker)
docker run -d \
  --name redis-wessley \
  -p 6379:6379 \
  redis:7-alpine
```

### 2. Import Sample Data

```bash
# Import Pajero Pinin electrical system from GraphML
node scripts/import-graphml.js

# Verify import in Neo4j Browser
# http://localhost:7474 (neo4j/password)
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env.development

# Update with your settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
REDIS_HOST=localhost
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
```

### 4. Start the Service

```bash
# Development mode
npm run start:dev

# Production mode
npm run build
npm run start:prod
```

## 📋 API Endpoints

### Generate 3D Model
```http
POST /api/v1/models/generate
Content-Type: application/json
Authorization: Bearer <supabase-jwt>

{
  "requestId": "model_123",
  "graphQuery": {
    "componentTypes": ["fuse", "relay", "connector"],
    "zoneFilter": "Engine Compartment"
  },
  "options": {
    "quality": "high",
    "includeMetadata": true,
    "optimizeForWeb": true
  }
}
```

### Get Model Status
```http
GET /api/v1/models/{requestId}
Authorization: Bearer <supabase-jwt>
```

### Download GLB File
```http
GET /api/v1/models/{requestId}/download
Authorization: Bearer <supabase-jwt>
```

## 🏗️ Architecture

### Core Modules

- **GraphModule**: Neo4j integration and data transformation
- **SpatialModule**: 3D positioning and layout algorithms  
- **GeometryModule**: Three.js mesh generation and materials
- **ExportModule**: GLB optimization and export
- **StorageModule**: S3/CDN file management
- **JobsModule**: Redis-based async processing
- **RealtimeModule**: Supabase integration for live updates

### Data Flow

```
Neo4j Graph → GraphService → SpatialService → GeometryService → ExportService → S3 → GLB URL
                ↓
            Supabase Real-time Updates → Web Client
```

## 🔧 Development

### Sample Queries

```cypher
// Get all components in engine bay
MATCH (c:Component {anchor_zone: "Engine Compartment"})
RETURN c.id, c.type, c.canonical_id

// Find power distribution paths  
MATCH path = (battery:Component {code_id: "Battery"})-[:POWERS*]->(component:Component)
RETURN path

// Get components by type
MATCH (c:Component)
WHERE c.type IN ["fuse", "relay"]
RETURN c.id, c.type, c.anchor_zone
```

### Testing the Service

```bash
# Run unit tests
npm test

# Run integration tests  
npm run test:e2e

# Generate test coverage
npm run test:cov
```

### Example 3D Model Generation

```javascript
const response = await fetch('http://localhost:3001/api/v1/models/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_SUPABASE_JWT'
  },
  body: JSON.stringify({
    requestId: 'demo_' + Date.now(),
    graphQuery: {
      componentTypes: ['fuse', 'relay', 'connector', 'component'],
      zoneFilter: 'Engine Compartment'
    },
    options: {
      quality: 'high',
      includeMetadata: true,
      optimizeForWeb: true,
      generateLOD: true
    }
  })
});

const result = await response.json();
console.log('Model generation started:', result.jobId);
```

## 🚀 Production Deployment

### Option 1: Neo4j AuraDB (Recommended)

```bash
# 1. Create AuraDB instance at https://neo4j.com/cloud/aura/
# 2. Import data using Neo4j Browser or scripts
# 3. Update environment variables

NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-aura-password
```

### Option 2: Self-Hosted Neo4j

```yaml
# docker-compose.production.yml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.12-enterprise
    environment:
      - NEO4J_AUTH=neo4j/your-secure-password
      - NEO4J_dbms_memory_heap_initial_size=1G
      - NEO4J_dbms_memory_heap_max_size=4G
      - NEO4J_dbms_memory_pagecache_size=2G
    volumes:
      - neo4j_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"
```

### Option 3: Cloud Integration with Supabase

The service is designed to complement Supabase:

- **Authentication**: Uses Supabase JWT tokens
- **Real-time**: Progress updates via Supabase channels  
- **Storage**: Can use Supabase Storage as backup
- **Database**: Metadata stored in Supabase PostgreSQL
- **API**: Follows Supabase REST conventions

```typescript
// Web app integration
const { data: { session } } = await supabase.auth.getSession();

const response = await fetch('/api/v1/models/generate', {
  headers: {
    'Authorization': `Bearer ${session.access_token}`
  },
  body: JSON.stringify(modelRequest)
});
```

## 📊 Monitoring

### Health Checks

```bash
# Service health
curl http://localhost:3001/api/v1/health

# Neo4j connectivity  
curl http://localhost:3001/api/v1/health/neo4j

# System metrics
curl http://localhost:3001/api/v1/metrics
```

### Key Metrics

- Model generation time (p95, p99)
- Queue depth and processing rate
- Memory usage and GC performance
- Neo4j query performance
- S3 upload success rate

## 🤝 Integration Examples

### With React Three Fiber

```typescript
// Load generated GLB in React
import { useLoader } from '@react-three/fiber';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

function ElectricalSystemModel({ modelUrl }: { modelUrl: string }) {
  const gltf = useLoader(GLTFLoader, modelUrl);
  
  return (
    <primitive 
      object={gltf.scene} 
      userData={gltf.userData}
    />
  );
}
```

### With AI Chat Agent

```typescript
// AI agent queries 3D model metadata
const modelData = await fetch(`/api/v1/models/${modelId}`);
const { metadata } = await modelData.json();

// Use spatial data to answer questions
const componentPosition = metadata.aiContext.components
  .find(c => c.id === 'starter_relay')?.position;

chatAgent.respond(`The starter relay is located at coordinates ${componentPosition}`);
```

This service provides a complete pipeline from Neo4j electrical system data to production-ready 3D models for web visualization and AI-powered assistance.