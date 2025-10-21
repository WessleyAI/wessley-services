# Semantic Search Service

High-performance semantic search service for electrical components and technical documentation. Provides intelligent search capabilities using vector embeddings, natural language processing, and machine learning.

## Features

### Core Search Capabilities
- **Universal Search**: Multi-collection semantic search across components, documentation, and knowledge graph entities
- **Natural Language Queries**: Process technical queries in plain English
- **Intelligent Filtering**: Advanced filtering by component type, vehicle model, electrical specifications
- **Hybrid Search**: Combines vector similarity with keyword matching for optimal results

### Advanced Features
- **Component Recommendations**: AI-powered suggestions for similar, compatible, and alternative components
- **Documentation Search**: Semantic search through technical manuals, repair guides, and service documentation
- **Chat Context Enhancement**: Enriches AI chat responses with relevant technical context
- **Real-time Indexing**: Automatic indexing of new components and documentation

### Performance & Reliability
- **Vector Embeddings**: Uses sentence-transformers for high-quality semantic representations
- **Caching**: Redis-based caching for fast response times
- **Async Architecture**: Fully asynchronous for high concurrency
- **Health Monitoring**: Comprehensive health checks and metrics

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and Setup**
   ```bash
   cd apps/services/semantic-service
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start Services**
   ```bash
   docker-compose up -d
   ```

3. **Verify Installation**
   ```bash
   curl http://localhost:8003/ping
   curl http://localhost:8003/search/health
   ```

### Manual Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Start External Services**
   ```bash
   # Qdrant vector database
   docker run -p 6333:6333 qdrant/qdrant:v1.7.0
   
   # Redis cache
   docker run -p 6379:6379 redis:7-alpine
   
   # Neo4j graph database
   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/wessley123 neo4j:5.15-community
   ```

3. **Run Service**
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Documentation

### Search Endpoints

#### Universal Search
```http
POST /search/universal
Content-Type: application/json

{
  "query": "fuel pump relay for Honda Civic",
  "search_type": "hybrid",
  "collections": ["components", "documentation"],
  "limit": 20,
  "similarity_threshold": 0.7,
  "vehicle_signature": "2019_honda_civic_sedan"
}
```

#### Component Search
```http
POST /search/components
Content-Type: application/json

{
  "query": "12V relay high current",
  "vehicle_signature": "2019_honda_civic_sedan",
  "component_types": ["relay"],
  "voltage_range": {"min": 10, "max": 14},
  "limit": 10
}
```

#### Documentation Search
```http
POST /search/documentation
Content-Type: application/json

{
  "query": "troubleshoot fuel pump relay",
  "document_types": ["troubleshooting", "repair_guide"],
  "vehicle_models": ["honda_civic"],
  "limit": 5
}
```

### Enhancement Endpoints

#### Chat Context Enhancement
```http
POST /search/chat/enhance
Content-Type: application/json

{
  "user_query": "Why is my fuel pump not working?",
  "vehicle_signature": "2019_honda_civic_sedan",
  "chat_history": [
    {"role": "user", "content": "I'm having car troubles"},
    {"role": "assistant", "content": "I can help diagnose the issue"}
  ],
  "max_components": 5,
  "max_documentation": 3
}
```

#### Component Recommendations
```http
GET /search/recommendations/comp_fuel_pump_relay_001?limit=5&similarity_threshold=0.8
```

### Management Endpoints

#### Index Components
```http
POST /search/index/components
Content-Type: application/json

{
  "vehicle_signature": "2019_honda_civic_sedan",
  "force_reindex": false,
  "batch_size": 100,
  "include_relationships": true
}
```

#### Health Check
```http
GET /search/health
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `development` |
| `DEBUG` | Enable debug mode | `false` |
| `HOST` | Service host | `0.0.0.0` |
| `PORT` | Service port | `8000` |
| `QDRANT_URL` | Qdrant vector database URL | `http://localhost:6333` |
| `REDIS_URL` | Redis cache URL | `redis://localhost:6379/0` |
| `NEO4J_URI` | Neo4j graph database URI | `bolt://localhost:7687` |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | - |
| `SENTENCE_TRANSFORMERS_MODEL` | Embedding model | `all-MiniLM-L6-v2` |

### Search Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `VECTOR_SEARCH_TIMEOUT` | Vector search timeout (seconds) | `30` |
| `CACHE_TTL_SECONDS` | Cache TTL (seconds) | `3600` |
| `MAX_SEARCH_RESULTS` | Maximum search results | `100` |
| `DEFAULT_SIMILARITY_THRESHOLD` | Default similarity threshold | `0.7` |

## Development

### Project Structure
```
src/
├── api/                 # FastAPI routes and endpoints
├── config/             # Configuration and settings
├── core/               # Core utilities (logging, etc.)
├── models/             # Pydantic data models
├── services/           # Business logic services
│   ├── embedding.py    # Embedding generation
│   ├── vector_store.py # Qdrant integration
│   ├── search_service.py # Core search logic
│   ├── recommendation_service.py # Recommendations
│   └── documentation_service.py # Documentation search
└── main.py             # FastAPI application entry point
```

### Running Tests
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
```

## Deployment

### Production Deployment

1. **Build Docker Image**
   ```bash
   docker build -t semantic-search-service:latest .
   ```

2. **Deploy with Docker Compose**
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

3. **Health Check**
   ```bash
   curl http://your-domain:8003/search/health
   ```

### Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests.

### Performance Tuning

- **Embedding Cache**: Increase `CACHE_TTL_SECONDS` for better performance
- **Batch Size**: Tune `MAX_EMBEDDING_BATCH_SIZE` based on available memory
- **Connection Pools**: Adjust Redis and Neo4j connection pool sizes
- **Vector Collections**: Optimize Qdrant collection settings for your data size

## Monitoring

### Health Endpoints
- `/ping` - Simple health check
- `/search/health` - Comprehensive service health
- `/search/analytics` - Search analytics and metrics

### Logging
Structured JSON logging with correlation IDs for request tracing.

### Metrics
- Search response times
- Cache hit rates
- Error rates
- Embedding generation performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: https://docs.wessley.ai
- Issues: https://github.com/wessley/semantic-search-service/issues
- Email: support@wessley.ai