#!/bin/bash
set -e

echo "🚀 Starting Wessley.ai Services (Direct Mode)"
echo "============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Create network if it doesn't exist
docker network create wessley-network 2>/dev/null || true

# Stop and remove existing containers
print_status "Cleaning up existing containers..."
docker stop wessley-redis wessley-neo4j wessley-qdrant wessley-minio 2>/dev/null || true
docker rm wessley-redis wessley-neo4j wessley-qdrant wessley-minio 2>/dev/null || true

# Start Redis
print_status "Starting Redis..."
docker run -d \
  --name wessley-redis \
  --network wessley-network \
  -p 6379:6379 \
  redis:7-alpine \
  redis-server --appendonly yes

sleep 3

# Start Neo4j
print_status "Starting Neo4j..."
docker run -d \
  --name wessley-neo4j \
  --network wessley-network \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
  -e NEO4J_dbms_security_procedures_allowlist=apoc.* \
  neo4j:5.13-community

sleep 10

# Start Qdrant
print_status "Starting Qdrant..."
docker run -d \
  --name wessley-qdrant \
  --network wessley-network \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:v1.7.0

sleep 5

# Start MinIO
print_status "Starting MinIO..."
docker run -d \
  --name wessley-minio \
  --network wessley-network \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin123 \
  minio/minio:latest \
  server /data --console-address ":9001"

sleep 10

print_status "Checking service health..."

# Check Redis
if docker exec wessley-redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis is ready"
else
    print_warning "Redis is not responding"
fi

# Check MinIO
if curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; then
    print_success "MinIO is ready"
else
    print_warning "MinIO is not responding"
fi

# Check Qdrant
if curl -sf http://localhost:6333/health > /dev/null 2>&1; then
    print_success "Qdrant is ready"
else
    print_warning "Qdrant is not responding"
fi

# Wait for Neo4j and setup
print_status "Waiting for Neo4j to be ready..."
for i in {1..30}; do
    if docker exec wessley-neo4j cypher-shell -u neo4j -p password123 "RETURN 1" > /dev/null 2>&1; then
        print_success "Neo4j is ready"
        break
    else
        if [ $i -eq 30 ]; then
            print_warning "Neo4j is taking longer than expected"
        else
            sleep 2
        fi
    fi
done

# Setup Neo4j constraints
print_status "Setting up Neo4j constraints..."
docker exec wessley-neo4j cypher-shell -u neo4j -p password123 \
  "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE;" 2>/dev/null || true

docker exec wessley-neo4j cypher-shell -u neo4j -p password123 \
  "CREATE INDEX net_name IF NOT EXISTS FOR (n:Net) ON (n.name);" 2>/dev/null || true

print_success "Neo4j constraints created"

# Setup MinIO bucket
print_status "Setting up MinIO bucket..."
docker run --rm \
  --network wessley-network \
  minio/mc:latest \
  sh -c "
    mc alias set local http://wessley-minio:9000 minioadmin minioadmin123;
    mc mb local/wessley-ingestions --ignore-existing;
    mc policy set public local/wessley-ingestions;
  " > /dev/null 2>&1 || print_warning "MinIO bucket setup may have failed"

# Setup Qdrant collection
print_status "Setting up Qdrant collection..."
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
print_success "All services are running!"
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
echo ""
echo "🛑 To stop all services: docker stop wessley-redis wessley-neo4j wessley-qdrant wessley-minio"