# Knowledge Scraper Service

Automotive knowledge acquisition from forums, YouTube, and parts catalogs.

## Overview

This service scrapes automotive knowledge from public sources to build a comprehensive knowledge base for the Wessley.ai platform. The scraped data populates:

- **Qdrant**: ~100M semantic vectors for RAG search
- **Neo4j**: ~1M component relationship nodes
- **PostgreSQL**: ~10M structured metadata rows

## Data Sources

| Source | Content | Volume | Priority |
|--------|---------|--------|----------|
| Reddit | r/MechanicAdvice, r/cartalk, r/AskMechanics | ~5M posts | P0 |
| Brand Forums | BimmerPost, ToyotaNation, HondaTech, etc. | ~50M posts | P0 |
| YouTube | Repair video transcripts | ~1M videos | P1 |
| iFixit | Repair guides (CC licensed) | ~100K guides | P1 |
| Parts Catalogs | RockAuto, AutoZone | ~10M parts | P1 |
| NHTSA | Recalls, complaints, TSBs | ~1M records | P2 |

## Setup

```bash
# Install dependencies
make setup

# Start infrastructure (Neo4j, Qdrant, Redis)
make start

# Run the service
make run
```

## Configuration

Copy `.env.example` to `.env` and configure:

- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET`: Reddit API credentials
- `OPENAI_API_KEY`: For LLM-based knowledge extraction
- `NEO4J_*`: Graph database connection
- `QDRANT_*`: Vector database connection

## API Endpoints

### Scraping

- `POST /v1/scrape` - Start a scraping job
- `GET /v1/jobs/{job_id}` - Check job status

### Knowledge Retrieval

- `GET /v1/knowledge/search` - Semantic search over scraped knowledge
- `GET /v1/knowledge/component/{type}` - Get component information
- `GET /v1/knowledge/vehicle/{make}/{model}/{year}` - Vehicle-specific knowledge

### Health

- `GET /ping` - Liveness check
- `GET /health` - Readiness check with dependencies

## Testing

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests (requires Docker services)
make test-integration
```

## Architecture

```
src/
├── api/                 # FastAPI routes
│   └── scraper_routes.py
├── core/                # Core utilities
│   ├── config.py        # Settings (Pydantic)
│   ├── logging.py       # Structured logging
│   └── schemas.py       # Pydantic models
├── scrapers/            # Data source scrapers
│   ├── base.py          # Base scraper interface
│   ├── reddit.py        # Reddit scraper (PRAW)
│   ├── forum.py         # Generic forum scraper
│   ├── youtube.py       # YouTube transcript scraper
│   └── parts.py         # Parts catalog scraper
├── services/            # Business logic
│   ├── extraction.py    # LLM knowledge extraction
│   ├── persistence.py   # Storage operations
│   └── normalization.py # Data normalization
├── jobs/                # Job queue
│   └── queue.py         # Redis-based job queue
└── main.py              # FastAPI app entry
```

## Legal Considerations

- Respects robots.txt
- Rate limits all requests (configurable)
- Caches aggressively to reduce load
- iFixit content is CC-BY-NC-SA licensed
- NHTSA data is public domain
- Forum scraping: used only for training, not republishing
