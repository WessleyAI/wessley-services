#!/bin/bash
set -e

echo "🚀 Starting Wessley.ai Ingestion Service - Local Infrastructure"
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env exists, if not copy from example
if [ ! -f .env ]; then
    print_warning ".env file not found, copying from .env.example"
    cp .env.example .env
    print_warning "Please edit .env file with your configuration"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_status "Stopping any existing containers..."
docker compose down

print_status "Starting Docker services..."
docker compose up -d redis neo4j qdrant minio

print_status "Waiting for services to be ready..."
sleep 15

# Check service health
print_status "Checking service health..."

# Check Redis
if docker compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis is ready"
else
    print_warning "Redis is not responding yet"
fi

# Check Neo4j (may take longer to start)
print_status "Waiting for Neo4j to be ready..."
for i in {1..30}; do
    if docker compose exec -T neo4j cypher-shell -u neo4j -p password123 "RETURN 1" > /dev/null 2>&1; then
        print_success "Neo4j is ready"
        break
    else
        if [ $i -eq 30 ]; then
            print_warning "Neo4j is taking longer than expected to start"
        else
            sleep 2
        fi
    fi
done

# Check Qdrant
if curl -sf http://localhost:6333/health > /dev/null 2>&1; then
    print_success "Qdrant is ready"
else
    print_warning "Qdrant health check failed"
fi

# Check MinIO
if curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; then
    print_success "MinIO is ready"
else
    print_warning "MinIO health check failed"
fi

# Setup MinIO bucket
print_status "Setting up MinIO bucket..."
docker compose run --rm minio-setup > /dev/null 2>&1 || print_warning "MinIO setup may have failed"

# Initialize Neo4j constraints and indexes
print_status "Setting up Neo4j constraints and indexes..."
docker compose exec -T neo4j cypher-shell -u neo4j -p password123 \
  "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE;" 2>/dev/null || true

docker compose exec -T neo4j cypher-shell -u neo4j -p password123 \
  "CREATE INDEX net_name IF NOT EXISTS FOR (n:Net) ON (n.name);" 2>/dev/null || true

docker compose exec -T neo4j cypher-shell -u neo4j -p password123 \
  "CREATE INDEX vehicle_signature IF NOT EXISTS FOR (v:Vehicle) ON (v.make, v.model, v.year);" 2>/dev/null || true

print_success "Neo4j constraints and indexes created"

# Create Qdrant collection
print_status "Setting up Qdrant collection..."
curl -X PUT "http://localhost:6333/collections/wessley_docs" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    },
    "optimizers_config": {
      "default_segment_number": 2
    }
  }' > /dev/null 2>&1 || print_warning "Qdrant collection setup may have failed"

print_success "Qdrant collection 'wessley_docs' created"

# Optional: Start Supabase if CLI is available
if command -v supabase &> /dev/null; then
    print_status "Starting Supabase local instance..."
    supabase start > /dev/null 2>&1 || print_warning "Supabase start failed (this is optional for basic functionality)"
    print_success "Supabase started (if available)"
else
    print_warning "Supabase CLI not found. Install with: npm install -g @supabase/cli"
fi

echo ""
echo "================================================================"
print_success "Local infrastructure is ready!"
echo "================================================================"
echo ""
echo "🌐 Access points:"
echo "  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)"
echo "  - Neo4j Browser: http://localhost:7474 (neo4j/password123)"
echo "  - Qdrant Dashboard: http://localhost:6333/dashboard"
echo "  - Redis: localhost:6379"
if command -v supabase &> /dev/null; then
    echo "  - Supabase Studio: http://localhost:54323"
fi
echo ""
echo "📝 Next steps:"
echo "  1. Review and update .env file if needed"
echo "  2. Install Python dependencies: poetry install"
echo "  3. Start the application: ./scripts/run-dev.sh"
echo "  4. Check health: ./scripts/health-check.sh"
echo ""
echo "🔧 Useful commands:"
echo "  - View logs: docker-compose logs -f [service]"
echo "  - Stop services: docker-compose down"
echo "  - Reset data: docker-compose down -v"
echo ""