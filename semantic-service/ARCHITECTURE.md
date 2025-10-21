# Semantic Search Service Architecture

## Overview
**Language**: Python (FastAPI + Qdrant)  
**Primary Function**: Vector-based search for components and documentation with natural language understanding  
**Status**: 🔄 **Future Implementation**

## Core Responsibilities
- Store component embeddings for similarity search and recommendation
- Enable natural language component queries and technical assistance
- Support documentation and manual semantic search
- Provide recommendations for similar parts and systems
- Enrich chat context with relevant technical information
- Index and search electrical system documentation
- Support multi-language technical terminology

## Technology Stack
```json
{
  "framework": "FastAPI",
  "language": "Python 3.11+",
  "vector_db": "Qdrant",
  "embeddings": "sentence-transformers + OpenAI",
  "nlp": "spaCy + transformers",
  "cache": "Redis",
  "search": "Elasticsearch (hybrid search)",
  "monitoring": "Prometheus + Grafana",
  "testing": "pytest + pytest-asyncio",
  "deployment": "Docker + Kubernetes"
}
```

## Service Architecture
```
src/
├── embeddings/
│   ├── generators/
│   │   ├── component_embedder.py    # Component specification embeddings
│   │   ├── document_embedder.py     # Technical documentation embeddings
│   │   ├── image_embedder.py        # Visual component embeddings
│   │   └── multimodal_embedder.py   # Combined text+image embeddings
│   ├── models/
│   │   ├── automotive_bert.py       # Fine-tuned BERT for automotive domain
│   │   ├── technical_embedder.py    # Specialized technical term embeddings
│   │   └── multilingual_embedder.py # Multi-language support
│   └── processors/
│       ├── text_processor.py        # Text preprocessing and cleaning
│       ├── spec_parser.py           # Component specification parsing
│       └── document_parser.py       # Technical document processing
├── search/
│   ├── vector_search.py             # Qdrant vector similarity search
│   ├── hybrid_search.py             # Combined vector + keyword search
│   ├── recommendation.py            # Component recommendation engine
│   └── query_expansion.py           # Query understanding and expansion
├── indexing/
│   ├── component_indexer.py         # Component specification indexing
│   ├── document_indexer.py          # Documentation indexing
│   ├── knowledge_indexer.py         # Knowledge graph entity indexing
│   └── batch_processor.py           # Batch indexing operations
├── services/
│   ├── search_service.py            # Core search functionality
│   ├── recommendation_service.py    # Recommendation algorithms
│   ├── chat_enhancement_service.py  # Chat context enrichment
│   └── indexing_service.py          # Data indexing coordination
├── api/
│   ├── search.py                    # Search endpoints
│   ├── recommendations.py           # Recommendation endpoints
│   ├── chat.py                      # Chat enhancement endpoints
│   ├── admin.py                     # Administrative endpoints
│   └── health.py                    # Health check endpoints
├── models/
│   ├── search_models.py             # Pydantic models for search
│   ├── component_models.py          # Component data models
│   └── document_models.py           # Document data models
└── utils/
    ├── qdrant_utils.py              # Qdrant client utilities
    ├── embedding_utils.py           # Embedding processing utilities
    └── text_utils.py                # Text processing utilities
```

## Vector Database Schema (Qdrant)

### Collections
```python
# Component Collection
{
  "collection_name": "components",
  "vector_size": 768,  # sentence-transformers output size
  "distance": "Cosine",
  "payload_schema": {
    "component_id": "str",
    "vehicle_signature": "str",
    "component_type": "str",
    "manufacturer": "str",
    "part_number": "str",
    "specifications": "dict",
    "description": "str",
    "categories": "list[str]",
    "voltage_rating": "float",
    "current_rating": "float",
    "indexed_at": "datetime"
  }
}

# Documentation Collection
{
  "collection_name": "documentation",
  "vector_size": 768,
  "distance": "Cosine",
  "payload_schema": {
    "document_id": "str",
    "document_type": "str",  # manual, datasheet, standard
    "title": "str",
    "content_chunk": "str",
    "page_number": "int",
    "vehicle_models": "list[str]",
    "component_types": "list[str]",
    "keywords": "list[str]",
    "language": "str",
    "indexed_at": "datetime"
  }
}

# Knowledge Graph Entities Collection
{
  "collection_name": "kg_entities",
  "vector_size": 768,
  "distance": "Cosine",
  "payload_schema": {
    "entity_id": "str",
    "entity_type": "str",  # component, circuit, system
    "label": "str",
    "description": "str",
    "relationships": "list[str]",
    "properties": "dict",
    "vehicle_signature": "str",
    "indexed_at": "datetime"
  }
}
```

## Core Search Capabilities

### 1. Natural Language Component Search
```python
# Example: "12V relay for headlight circuit with 30A capacity"
async def search_components(query: str, filters: Dict = None):
    # Parse natural language query
    parsed_query = await parse_technical_query(query)
    
    # Generate query embedding
    query_embedding = await generate_embedding(query)
    
    # Search vector database
    results = await qdrant_client.search(
        collection_name="components",
        query_vector=query_embedding,
        limit=20,
        query_filter=build_qdrant_filter(filters, parsed_query)
    )
    
    # Re-rank results based on technical specifications
    return await rerank_by_specifications(results, parsed_query)
```

### 2. Component Recommendation
```python
# Find similar components
async def get_similar_components(component_id: str, limit: int = 10):
    # Get component embedding
    component = await get_component_by_id(component_id)
    component_embedding = await generate_embedding(component.description)
    
    # Find similar components
    similar = await qdrant_client.search(
        collection_name="components",
        query_vector=component_embedding,
        limit=limit + 1,  # +1 to exclude self
        query_filter={"must_not": {"key": "component_id", "match": {"value": component_id}}}
    )
    
    return similar[1:]  # Exclude the original component
```

### 3. Documentation Search
```python
# Semantic documentation search
async def search_documentation(query: str, document_types: List[str] = None):
    query_embedding = await generate_embedding(query)
    
    filters = {}
    if document_types:
        filters["document_type"] = {"any": document_types}
    
    results = await qdrant_client.search(
        collection_name="documentation",
        query_vector=query_embedding,
        limit=50,
        query_filter=filters
    )
    
    # Group by document and rank by relevance
    return await group_and_rank_documentation(results)
```

### 4. Chat Context Enhancement
```python
# Enrich chat queries with relevant context
async def enhance_chat_context(user_query: str, model_metadata: Dict):
    # Search for relevant components
    relevant_components = await search_components(user_query, {
        "vehicle_signature": model_metadata.get("vehicle_signature")
    })
    
    # Search for relevant documentation
    relevant_docs = await search_documentation(user_query)
    
    # Get spatial context from 3D model
    spatial_context = await get_spatial_context(user_query, model_metadata)
    
    return {
        "components": relevant_components[:5],
        "documentation": relevant_docs[:3],
        "spatial_context": spatial_context,
        "enhanced_query": await expand_query(user_query)
    }
```

## API Endpoints

### Search APIs
```python
# Natural language component search
POST /api/v1/search/components
{
  "query": "12V relay for lighting circuit",
  "filters": {
    "vehicle_signature": "pajero_pinin_2001",
    "voltage_rating": {"min": 12, "max": 14},
    "component_type": ["relay", "switch"]
  },
  "limit": 20,
  "include_similar": true
}

# Component recommendations
GET /api/v1/components/:id/similar?limit=10

# Documentation search
POST /api/v1/search/documentation
{
  "query": "headlight wiring diagram",
  "document_types": ["manual", "schematic"],
  "vehicle_models": ["pajero_pinin_2001"],
  "language": "en"
}

# Hybrid search (vector + keyword)
POST /api/v1/search/hybrid
{
  "query": "starter motor relay location",
  "search_type": "both",  # vector, keyword, both
  "collections": ["components", "documentation"],
  "boost_factors": {"exact_match": 2.0, "semantic": 1.0}
}
```

### Chat Enhancement APIs
```python
# Enhance chat context
POST /api/v1/chat/enhance
{
  "user_query": "Where is the starter relay located?",
  "model_metadata": {
    "model_id": "model_123",
    "vehicle_signature": "pajero_pinin_2001",
    "glb_url": "https://cdn.wessley.ai/models/model_123.glb"
  },
  "chat_history": [...]
}

# Get contextual suggestions
POST /api/v1/chat/suggestions
{
  "current_context": "User is looking at the engine bay fuse box",
  "vehicle_signature": "pajero_pinin_2001",
  "suggestion_types": ["related_components", "common_questions", "troubleshooting"]
}
```

### Indexing Management APIs
```python
# Index component data
POST /api/v1/index/components
{
  "vehicle_signature": "pajero_pinin_2001",
  "force_reindex": false,
  "batch_size": 100
}

# Index documentation
POST /api/v1/index/documentation
{
  "document_urls": ["s3://docs/manual.pdf"],
  "document_type": "manual",
  "language": "en"
}

# Get indexing status
GET /api/v1/index/status

# Search statistics
GET /api/v1/search/analytics
```

## Integration Points

### With 3D Model Service (TypeScript/NestJS)
```typescript
// 3D Model Service calls Semantic Service for chat enhancement
interface ISemanticService {
  searchComponents(query: string, vehicleSignature: string): Promise<Component[]>;
  getSimilarComponents(componentId: string): Promise<Component[]>;
  enhanceChatContext(query: string, model3D: GLBMetadata): Promise<ChatContext>;
  searchDocumentation(query: string, filters?: SearchFilters): Promise<Document[]>;
}
```

### With Neo4j Knowledge Graph
```python
# Index knowledge graph entities for semantic search
async def index_knowledge_graph_entities():
    query = """
    MATCH (n)
    WHERE n.vehicle_signature IS NOT NULL
    RETURN n.id as id, labels(n) as types, n.name as name, 
           n.description as description, properties(n) as properties
    """
    
    async for record in neo4j_session.run(query):
        entity_text = build_entity_description(record)
        embedding = await generate_embedding(entity_text)
        
        await qdrant_client.upsert(
            collection_name="kg_entities",
            points=[{
                "id": record["id"],
                "vector": embedding,
                "payload": {
                    "entity_id": record["id"],
                    "entity_type": record["types"][0],
                    "label": record["name"],
                    "description": record["description"],
                    "properties": record["properties"]
                }
            }]
        )
```

### With Learning Service
```python
# Share search patterns for ML improvement
async def log_search_interaction(query: str, results: List, user_action: str):
    interaction = {
        "query": query,
        "results_count": len(results),
        "user_action": user_action,  # clicked, ignored, refined
        "timestamp": datetime.utcnow(),
        "session_id": get_session_id()
    }
    
    # Send to Learning Service for pattern analysis
    await learning_service.log_search_pattern(interaction)
```

## Performance Characteristics
- **Search Response Time**: < 200ms for vector searches
- **Indexing Throughput**: 1000+ documents per minute
- **Vector Storage**: 100M+ embeddings with sub-second retrieval
- **Memory Usage**: < 4GB for embedding models
- **Concurrent Searches**: 1000+ simultaneous queries
- **Index Update Latency**: < 5 seconds for new components

## Environment Configuration
```bash
# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-key
QDRANT_COLLECTION_SIZE=100000

# Embedding Models
SENTENCE_TRANSFORMERS_MODEL=all-MiniLM-L6-v2
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_API_KEY=your-hf-key

# Search Configuration
DEFAULT_SEARCH_LIMIT=20
MAX_SEARCH_LIMIT=100
SIMILARITY_THRESHOLD=0.7

# Cache
REDIS_URL=redis://localhost:6379/2
CACHE_TTL_SECONDS=3600

# External Services
NEO4J_URL=bolt://localhost:7687
MODEL_SERVICE_URL=http://localhost:3001
LEARNING_SERVICE_URL=http://localhost:3002

# Service Configuration
PORT=3003
LOG_LEVEL=info
ENABLE_HYBRID_SEARCH=true
```

## Advanced Features

### 1. Multi-modal Search
```python
# Search by component image + text description
async def multimodal_search(image_data: bytes, text_query: str):
    # Generate image embedding
    image_embedding = await generate_image_embedding(image_data)
    
    # Generate text embedding
    text_embedding = await generate_embedding(text_query)
    
    # Combine embeddings (weighted average)
    combined_embedding = combine_embeddings(image_embedding, text_embedding, weights=[0.6, 0.4])
    
    return await qdrant_client.search(
        collection_name="components",
        query_vector=combined_embedding,
        limit=20
    )
```

### 2. Query Understanding
```python
# Parse technical queries with NLP
async def parse_technical_query(query: str):
    doc = nlp_model(query)
    
    extracted = {
        "component_types": extract_component_mentions(doc),
        "specifications": extract_technical_specs(doc),
        "actions": extract_action_verbs(doc),
        "locations": extract_spatial_references(doc),
        "intent": classify_query_intent(doc)
    }
    
    return extracted
```

### 3. Knowledge Graph Enhanced Search
```python
# Use graph relationships to improve search
async def graph_enhanced_search(query: str, vehicle_signature: str):
    # Get initial semantic matches
    semantic_results = await search_components(query)
    
    # Expand with graph relationships
    enhanced_results = []
    for result in semantic_results:
        # Get connected components from Neo4j
        connected = await get_connected_components(result.component_id)
        enhanced_results.extend(connected)
    
    # Re-rank combined results
    return await rerank_with_graph_context(enhanced_results, query)
```

## Future Enhancements

### Phase 1: Foundation (Current)
- Basic vector search with sentence transformers
- Component and documentation indexing
- RESTful search APIs

### Phase 2: Advanced NLP
- Domain-specific fine-tuned models
- Multi-language support
- Advanced query understanding

### Phase 3: Multi-modal AI
- Image-based component search
- Voice query support
- Video content indexing

### Phase 4: Intelligent Assistant
- Conversational search interface
- Proactive recommendations
- Context-aware suggestions

## Monitoring & Analytics
- **Search Analytics**: Query patterns, result quality, user behavior
- **Performance Monitoring**: Response times, throughput, error rates
- **Model Performance**: Embedding quality, search relevance metrics
- **Usage Metrics**: Popular searches, component access patterns

This service will provide intelligent search and recommendation capabilities to enhance the user experience across the entire Wessley.ai platform.