#!/bin/bash
set -e

echo "🚀 Starting Wessley.ai Services (Simple Mode)"
echo "============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

print_status "Stopping any existing containers..."
docker compose down 2>/dev/null || true

print_status "Starting Redis..."
docker compose up -d redis
sleep 5

print_status "Starting Neo4j..."
docker compose up -d neo4j
sleep 15

print_status "Starting Qdrant..."
docker compose up -d qdrant
sleep 5

print_status "Starting MinIO..."
docker compose up -d minio
sleep 10

print_status "Setting up MinIO bucket..."
docker compose run --rm minio-setup 2>/dev/null || print_warning "MinIO setup may have failed"

print_status "Checking service health..."

# Wait for Neo4j to be ready
print_status "Waiting for Neo4j to be ready..."
for i in {1..20}; do
    if docker compose exec -T neo4j cypher-shell -u neo4j -p password123 "RETURN 1" > /dev/null 2>&1; then
        print_success "Neo4j is ready"
        break
    else
        if [ $i -eq 20 ]; then
            print_warning "Neo4j is taking longer than expected"
        else
            sleep 3
        fi
    fi
done

# Setup Neo4j constraints
print_status "Setting up Neo4j constraints..."
docker compose exec -T neo4j cypher-shell -u neo4j -p password123 \
  "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE;" 2>/dev/null || true

docker compose exec -T neo4j cypher-shell -u neo4j -p password123 \
  "CREATE INDEX net_name IF NOT EXISTS FOR (n:Net) ON (n.name);" 2>/dev/null || true

# Setup Qdrant collection
print_status "Setting up Qdrant collection..."
sleep 5
curl -X PUT "http://localhost:6333/collections/wessley_docs" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
  }' > /dev/null 2>&1 || print_warning "Qdrant collection setup may have failed"

print_success "Qdrant collection created"

echo ""
echo "============================================="
print_success "Services are starting up!"
echo "============================================="
echo ""
echo "🌐 Access points:"
echo "  - Neo4j Browser: http://localhost:7474 (neo4j/password123)"
echo "  - Qdrant Dashboard: http://localhost:6333/dashboard"
echo "  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)"
echo "  - Redis: localhost:6379"
echo ""
echo "📝 Next step: ./scripts/run-dev.sh to start the application"
echo ""
echo "🔍 Check status: ./scripts/health-check.sh"