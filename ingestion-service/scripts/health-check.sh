#!/bin/bash

echo "🔍 Wessley.ai Health Check"
echo "========================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_service() {
    local name=$1
    local url=$2
    local type=${3:-http}
    
    if [ "$type" = "redis" ]; then
        if redis-cli -u "$url" ping > /dev/null 2>&1; then
            echo -e "✅ ${GREEN}$name${NC} - $url"
            return 0
        else
            echo -e "❌ ${RED}$name${NC} - $url"
            return 1
        fi
    elif [ "$type" = "tcp" ]; then
        host=$(echo $url | cut -d: -f1)
        port=$(echo $url | cut -d: -f2)
        if nc -z $host $port 2>/dev/null; then
            echo -e "✅ ${GREEN}$name${NC} - $url"
            return 0
        else
            echo -e "❌ ${RED}$name${NC} - $url"
            return 1
        fi
    else
        if curl -sf "$url" > /dev/null 2>&1; then
            echo -e "✅ ${GREEN}$name${NC} - $url"
            return 0
        else
            echo -e "❌ ${RED}$name${NC} - $url"
            return 1
        fi
    fi
}

echo "Checking infrastructure services..."
echo ""

# Load .env if available
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Infrastructure Services
total=0
healthy=0

# API Service
((total++))
if check_service "API Service" "http://localhost:${PORT:-8080}/healthz"; then
    ((healthy++))
fi

# Redis
((total++))
if check_service "Redis" "${REDIS_URL:-redis://localhost:6379}" "redis"; then
    ((healthy++))
fi

# Neo4j Web Interface
((total++))
if check_service "Neo4j Browser" "http://localhost:7474"; then
    ((healthy++))
fi

# Neo4j Bolt (TCP check)
((total++))
neo4j_host=$(echo ${NEO4J_URI:-bolt://localhost:7687} | sed 's|bolt://||' | cut -d: -f1)
neo4j_port=$(echo ${NEO4J_URI:-bolt://localhost:7687} | sed 's|bolt://||' | cut -d: -f2)
if check_service "Neo4j Bolt" "${neo4j_host}:${neo4j_port}" "tcp"; then
    ((healthy++))
fi

# Qdrant
((total++))
if check_service "Qdrant" "${QDRANT_URL:-http://localhost:6333}"; then
    ((healthy++))
fi

# MinIO
((total++))
if check_service "MinIO API" "http://localhost:9000/minio/health/live"; then
    ((healthy++))
fi

# MinIO Console
((total++))
if check_service "MinIO Console" "http://localhost:9001"; then
    ((healthy++))
fi

# Supabase (optional)
if curl -sf "http://localhost:54321" > /dev/null 2>&1; then
    ((total++))
    if check_service "Supabase" "http://localhost:54321"; then
        ((healthy++))
    fi
fi

echo ""
echo "========================="
if [ $healthy -eq $total ]; then
    echo -e "🎉 ${GREEN}All services healthy${NC} ($healthy/$total)"
    
    echo ""
    echo "🌐 Quick access links:"
    echo "  - API Docs: http://localhost:${PORT:-8080}/docs"
    echo "  - Health: http://localhost:${PORT:-8080}/healthz"
    echo "  - Neo4j: http://localhost:7474"
    echo "  - Qdrant: http://localhost:6333/dashboard"
    echo "  - MinIO: http://localhost:9001"
    if [ $total -gt 7 ]; then
        echo "  - Supabase: http://localhost:54323"
    fi
    
    echo ""
    echo "🧪 Test the system:"
    echo "  curl http://localhost:${PORT:-8080}/healthz"
    echo "  curl \"http://localhost:${PORT:-8080}/v1/search?q=starter+relay\""
    
elif [ $healthy -gt 0 ]; then
    echo -e "⚠️  ${YELLOW}Partially healthy${NC} ($healthy/$total)"
    echo "Some services are not responding. Check the logs:"
    echo "  docker compose logs [service-name]"
else
    echo -e "💥 ${RED}All services down${NC} ($healthy/$total)"
    echo "Start the infrastructure with:"
    echo "  ./scripts/start-local.sh"
fi

echo ""

exit $((total - healthy))