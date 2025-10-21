# Vehicle Electrical System Graph Service

A graph database service for managing vehicle electrical system data using Neo4j and FastAPI.

## Architecture

This service implements a comprehensive architecture for managing vehicle electrical systems:

- **Node Models**: Components, Circuits, Vehicles, Zones, Connectors
- **Relationship Models**: Connections, Controls, Location, Part-of relationships
- **Services**: Graph operations, Analysis, Import/Export, Spatial processing
- **Repositories**: Data access layer with CRUD operations
- **APIs**: RESTful endpoints for all operations

## Features

- Vehicle data management with signature-based isolation
- Component and circuit analysis
- Spatial 3D positioning and wire routing
- Multi-format data import (GraphML, JSON, CSV)
- Performance monitoring and caching
- Comprehensive validation and error handling

## Quick Start

### Prerequisites

- Python 3.11+
- Neo4j database
- Docker (optional)

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd graph-service
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your Neo4j credentials
```

4. Run the service
```bash
uvicorn src.main:app --reload
```

### Docker

```bash
docker build -t graph-service .
docker run -p 8000:8000 graph-service
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

- `GET /health` - Service health check
- `GET /vehicles` - List vehicles
- `GET /vehicles/{signature}/components` - Get vehicle components
- `POST /import-export/graphml` - Import GraphML data
- `GET /analysis/{signature}/comprehensive` - Comprehensive analysis

## Configuration

Environment variables (see `src/config.py`):

- `NEO4J_URI` - Neo4j connection URI
- `NEO4J_USERNAME` - Neo4j username
- `NEO4J_PASSWORD` - Neo4j password
- `NEO4J_DATABASE` - Neo4j database name
- `ENVIRONMENT` - Application environment

## Development

### Project Structure

```
src/
├── api/                    # FastAPI routers
├── models/                 # Data models
│   ├── nodes/             # Node models
│   ├── relationships/     # Relationship models
│   └── schemas/           # Pydantic schemas
├── services/              # Business logic
├── repositories/          # Data access layer
├── queries/               # Cypher queries
├── importers/             # Data importers
├── utils/                 # Utilities
├── config.py              # Configuration
└── main.py               # FastAPI application
```

### Testing

```bash
# Run tests (when test suite is added)
pytest
```

### Performance Monitoring

The service includes built-in performance monitoring. Access metrics via:

```bash
GET /info
```

## Data Models

### Vehicle Signature

All data is isolated by `vehicle_signature` - a unique identifier for each vehicle's electrical system.

### Component Model

```python
{
    "id": "component_123",
    "vehicle_signature": "toyota_camry_2020",
    "type": "relay",
    "name": "Main Relay",
    "position": [x, y, z],
    "voltage": "12V",
    "current_rating": "30A"
}
```

### Connection Model

```python
{
    "from": "component_1",
    "to": "component_2",
    "vehicle_signature": "toyota_camry_2020",
    "wire_color": "red",
    "wire_gauge": "14awg",
    "signal_type": "power"
}
```

## Import Formats

### GraphML
Standard GraphML format with custom attributes for electrical components.

### JSON
```json
{
    "components": [...],
    "connections": [...],
    "metadata": {
        "vehicle_signature": "...",
        "source_file": "..."
    }
}
```

### CSV
Separate CSV files for components, connections, and circuits with configurable column mapping.

## License

[License information]