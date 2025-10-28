# Wessley Ingestion Service

Data ingestion service for processing electrical schematics and technical documents into structured graph data.

## Features

- **Multi-format processing**: PDF, PNG, JPG, SVG document support
- **OCR engines**: Tesseract, DeepSeek, Mistral with late fusion
- **Schematic parsing**: Symbol detection, wire routing, netlist generation
- **Graph storage**: Neo4j for relationships, Qdrant for semantic search
- **Real-time updates**: Supabase channels for job progress
- **Scalable processing**: Redis queue with background workers

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Development Setup

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd apps/services/ingestion-service
   make setup
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services**:
   ```bash
   make run
   ```

4. **View API documentation**:
   - OpenAPI docs: http://localhost:8080/docs
   - Health check: http://localhost:8080/healthz

### Production Deployment

See `docker-compose.yml` for production configuration with:
- Redis for job queuing
- Neo4j for graph storage
- Qdrant for vector search
- MinIO for artifact storage
- Prometheus for monitoring

## API Endpoints

### Core Operations

- `POST /v1/ingestions` - Create ingestion job
- `GET /v1/ingestions/{job_id}` - Get job status/results
- `POST /v1/benchmarks/run` - Run performance benchmarks

### Health Checks

- `GET /healthz` - Service health status
- `GET /readyz` - Service readiness check

## Architecture

### Processing Pipeline

1. **Pre-processing**: PDF rasterization, image cleanup
2. **OCR**: Multi-engine text extraction with confidence scoring
3. **Schematic Analysis**: Symbol detection, wire tracing, netlist generation
4. **Persistence**: Graph storage in Neo4j, embeddings in Qdrant
5. **Artifacts**: Export to JSON, GraphML, debug overlays

### Data Models

- **TextSpan**: OCR results with coordinates and confidence
- **Component**: Electronic components with pins and attributes
- **Netlist**: Wire connections and electrical nets
- **Job**: Processing status and artifacts

## Development

### Local Development

```bash
# Install dependencies
make dev-install

# Run linting
make lint

# Run tests
make test

# Start in development mode
make run-dev
```

### Testing

- Unit tests: `make test-unit`
- Integration tests: `make test-integration`
- Benchmarks: `make bench`

### Code Quality

- **Linting**: Ruff for code style and imports
- **Type checking**: mypy for static analysis
- **Testing**: pytest with coverage reporting
- **Pre-commit**: Automated quality checks

## Milestones

- [x] **M1**: Service skeleton with FastAPI, Redis queue, health endpoints
- [ ] **M2**: Multi-engine OCR with preprocessing pipelines
- [ ] **M3**: Schematic parsing (symbols, nets, components)
- [ ] **M4**: Neo4j/Qdrant persistence and indexing
- [ ] **M5**: Observability, rate limiting, security hardening

## Configuration

### Environment Variables

Key configuration options:

```bash
# Application
APP_ENV=dev|staging|prod
PORT=8080

# Authentication
SUPABASE_JWT_PUBLIC_KEY=...
REQUIRE_AUTH=true

# Storage
SUPABASE_URL=...
S3_BUCKET=wessley-ingestions

# Queue
REDIS_URL=redis://localhost:6379

# OCR Providers
DEEPSEEK_API_KEY=...
MISTRAL_API_KEY=...

# Datastores
NEO4J_URI=bolt://localhost:7687
QDRANT_URL=http://localhost:6333

# Features
FEATURE_SCHEMATIC_PARSE=true
OCR_ENGINES=tesseract,deepseek
MAX_PAGES=50
```

### Docker Compose Services

- **ingestion-service**: Main FastAPI application
- **redis**: Job queue and caching
- **neo4j**: Graph database for component relationships
- **qdrant**: Vector database for semantic search
- **minio**: S3-compatible storage for artifacts
- **prometheus**: Metrics collection (optional)

## Performance

### Benchmarks

Run performance tests across engines and datasets:

```bash
make bench
```

Metrics tracked:
- **CER/WER**: Character/Word Error Rate for OCR
- **Symbol Detection**: Precision/Recall for component recognition
- **Connectivity Accuracy**: Correct net memberships
- **E2E F1**: Weighted composite score

### DoD Gates

- M3 requires ≥0.80 E2E F1 on clean datasets
- ≥0.65 E2E F1 on noisy scanned documents

## Security

- **Authentication**: Supabase JWT validation on all endpoints
- **Authorization**: Row-level security for user data access
- **Input validation**: Pydantic schemas with sanitization
- **Rate limiting**: Per-user quotas and request size caps
- **Secrets management**: Environment-based configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `make lint test`
5. Submit a pull request

## License

Copyright © 2024 Wessley.ai