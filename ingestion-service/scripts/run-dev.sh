#!/bin/bash
set -e

echo "🚀 Starting Wessley.ai Ingestion Service - Development Mode"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env exists
if [ ! -f .env ]; then
    print_error ".env file not found. Please run ./scripts/start-local.sh first"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    print_error "Poetry is not installed. Install with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

print_status "Installing Python dependencies..."
poetry install

# Check if required services are running
print_status "Checking infrastructure dependencies..."

services_ok=true

# Check Redis
if ! redis-cli -u ${REDIS_URL} ping > /dev/null 2>&1; then
    print_error "Redis is not accessible at ${REDIS_URL}"
    services_ok=false
fi

# Check Neo4j
neo4j_host=$(echo ${NEO4J_URI} | sed 's|bolt://||' | cut -d: -f1)
neo4j_port=$(echo ${NEO4J_URI} | sed 's|bolt://||' | cut -d: -f2)
if ! nc -z ${neo4j_host} ${neo4j_port} 2>/dev/null; then
    print_error "Neo4j is not accessible at ${NEO4J_URI}"
    services_ok=false
fi

# Check Qdrant
qdrant_url=${QDRANT_URL}
if ! curl -sf ${qdrant_url}/health > /dev/null 2>&1; then
    print_error "Qdrant is not accessible at ${qdrant_url}"
    services_ok=false
fi

if [ "$services_ok" = false ]; then
    print_error "Some infrastructure services are not running."
    echo "Please run ./scripts/start-local.sh first to start the required services."
    exit 1
fi

print_success "All infrastructure dependencies are available"

# Optional: Run any database migrations
# print_status "Running database migrations..."
# poetry run alembic upgrade head

# Create necessary directories
mkdir -p data logs benchmarks/results fixtures

print_status "Starting the ingestion service..."
echo ""
echo "🌐 Service will be available at: http://localhost:${PORT:-8080}"
echo "📊 Health check: http://localhost:${PORT:-8080}/healthz"
echo "📝 API docs: http://localhost:${PORT:-8080}/docs"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Start the application with hot reload
poetry run uvicorn src.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --reload \
    --reload-dir src \
    --log-level info