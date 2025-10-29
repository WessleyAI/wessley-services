# 🚀 Quick Setup Guide - Wessley.ai Ingestion Service

This guide will help you set up the complete local development environment for the Wessley.ai automotive electronics ingestion service with semantic search and continual learning capabilities.

## 📋 Prerequisites

- **Docker & Docker Compose** - Container orchestration
- **Python 3.11+** - Application runtime
- **Poetry** - Python dependency management
- **Node.js** (optional) - For Supabase CLI

## ⚡ Quick Start (5 minutes)

### 1. **System Setup** (one-time)
```bash
# Install system dependencies and setup environment
make setup
```

This will:
- ✅ Check all system dependencies
- ✅ Install Poetry if needed
- ✅ Install Python dependencies
- ✅ Create `.env` file from template
- ✅ Make scripts executable

### 2. **Start Infrastructure**
```bash
# Start all required services (Redis, Neo4j, Qdrant, MinIO)
make start
```

This will:
- 🐳 Start Docker containers
- 🔗 Initialize Neo4j constraints
- 🔍 Create Qdrant collections
- 🪣 Setup MinIO buckets
- ⚡ Validate all services are healthy

### 3. **Start Application**
```bash
# Start the ingestion service API
make start-app
```

The API will be available at: **http://localhost:8080**

### 4. **Verify Everything Works**
```bash
# Check health of all services
make health

# Test the semantic search
curl "http://localhost:8080/v1/search?q=starter+relay"

# Run evaluation benchmarks
make eval
```

## 🛠️ Manual Setup (if you prefer step-by-step)

<details>
<summary>Click to expand manual setup instructions</summary>

### Prerequisites Installation

**macOS (with Homebrew):**
```bash
# Install Docker Desktop from https://docker.com/get-started
# Install dependencies
brew install python@3.11 poetry node tesseract redis

# Install Supabase CLI
npm install -g @supabase/cli
```

**Ubuntu/Debian:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3-pip nodejs npm tesseract-ocr redis-tools curl netcat-openbsd

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install Supabase CLI
npm install -g @supabase/cli
```

### Service Setup

1. **Clone and Navigate:**
```bash
cd wessley.ai/apps/services/ingestion-service
```

2. **Environment Configuration:**
```bash
cp .env.example .env
# Edit .env with your preferences (defaults work for local dev)
```

3. **Install Python Dependencies:**
```bash
poetry install
```

4. **Start Infrastructure Services:**
```bash
./scripts/start-local.sh
```

5. **Start Application:**
```bash
./scripts/run-dev.sh
```

</details>

## 🎯 What You Get

After successful setup, you'll have:

| Service | URL | Purpose |
|---------|-----|---------|
| **Main API** | http://localhost:8080 | FastAPI service with semantic search |
| **API Docs** | http://localhost:8080/docs | Interactive API documentation |
| **Neo4j Browser** | http://localhost:7474 | Graph database (neo4j/password123) |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | Vector database |
| **MinIO Console** | http://localhost:9001 | S3-compatible storage (minioadmin/minioadmin123) |
| **Supabase Studio** | http://localhost:54323 | Database admin (if Supabase CLI installed) |

## 🧪 Testing the System

### 1. **Health Check**
```bash
make health
# Should show all services as ✅ healthy
```

### 2. **API Tests**
```bash
# Basic health
curl http://localhost:8080/healthz

# Semantic search
curl "http://localhost:8080/v1/search?q=ECU+power+circuit"

# Search suggestions
curl "http://localhost:8080/v1/search/suggestions?q=relay"

# List available benchmarks
curl http://localhost:8080/v1/evaluation/benchmarks
```

### 3. **Run Evaluations**
```bash
# Run all benchmarks via CLI
poetry run python -m src.evaluation.cli run

# Run specific benchmark
poetry run python -m src.evaluation.cli run --benchmark component_identification

# Run evaluation via API
curl -X POST "http://localhost:8080/v1/evaluation/run" \
  -H "Content-Type: application/json" \
  -d '{"benchmark_suites": ["component_identification"]}'
```

## 🔧 Development Commands

```bash
# Development workflow
make help                 # Show all available commands
make test                 # Run all tests
make lint                 # Check code quality
make format               # Format code
make ci                   # Run full CI checks

# Service management
make start                # Start all services
make stop                 # Stop all services
make restart              # Restart services
make logs                 # View application logs
make health               # Check service health

# Database operations
make db-reset             # Reset all databases (DESTRUCTIVE!)
make db-neo4j            # Connect to Neo4j shell
make db-redis            # Connect to Redis CLI

# Evaluation
make eval                 # Run default benchmarks
make eval-all             # Run all benchmarks
make eval-component       # Run component identification only
```

## 🐛 Troubleshooting

### Common Issues

**Docker not starting:**
```bash
# Check Docker is running
docker info

# If Docker Desktop, restart it
# On Linux: sudo systemctl restart docker
```

**Port conflicts:**
```bash
# Check what's using a port
lsof -i :8080
lsof -i :6379

# Kill process if needed
kill -9 <PID>
```

**Poetry not found:**
```bash
# Add to PATH (after installing)
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

**Services not healthy:**
```bash
# Check specific service logs
docker-compose logs redis
docker-compose logs neo4j
docker-compose logs qdrant
docker-compose logs minio

# Restart specific service
docker-compose restart <service-name>
```

### Reset Everything
```bash
# Nuclear option - reset all data and containers
make clean
docker system prune -a -f
make setup
make start
```

## 📚 Next Steps

1. **Explore the API**: Visit http://localhost:8080/docs
2. **Run Evaluations**: `make eval` to see the benchmarking system
3. **Add Test Data**: Upload automotive schematics via the API
4. **Monitor Performance**: Check the evaluation metrics and system health
5. **Develop Features**: The system is ready for development!

## 🆘 Getting Help

- **Check service health**: `make health`
- **View logs**: `make logs`
- **Reset and try again**: `make clean && make setup && make start`
- **Makefile help**: `make help`

---

**🎉 You're all set!** The automotive electronics semantic search and learning system is ready for development and testing.