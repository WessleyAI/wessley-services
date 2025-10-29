"""
Health check and readiness probe implementations.
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False

try:
    from supabase import Client as SupabaseClient
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: float
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "version": self.version,
            "components": {
                name: {
                    "status": comp.status.value,
                    "message": comp.message,
                    "response_time_ms": comp.response_time_ms,
                    "metadata": comp.metadata or {}
                }
                for name, comp in self.components.items()
            }
        }


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._redis_client = None
        self._supabase_client = None
        self._neo4j_driver = None
        self._qdrant_client = None
    
    def set_redis_client(self, client):
        """Set Redis client for health checks."""
        self._redis_client = client
    
    def set_supabase_client(self, client):
        """Set Supabase client for health checks."""
        self._supabase_client = client
    
    def set_neo4j_driver(self, driver):
        """Set Neo4j driver for health checks."""
        self._neo4j_driver = driver
    
    def set_qdrant_client(self, client):
        """Set Qdrant client for health checks."""
        self._qdrant_client = client
    
    async def check_redis(self) -> ComponentHealth:
        """Check Redis connectivity and performance."""
        if not AIOREDIS_AVAILABLE:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.DEGRADED,
                message="aioredis not available"
            )
        
        if not self._redis_client:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.DEGRADED,
                message="Redis client not configured"
            )
        
        try:
            start_time = time.time()
            await self._redis_client.ping()
            response_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = await self._redis_client.info()
            memory_used = info.get('used_memory_human', 'unknown')
            connected_clients = info.get('connected_clients', 0)
            
            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis is responding",
                response_time_ms=response_time,
                metadata={
                    "memory_used": memory_used,
                    "connected_clients": connected_clients
                }
            )
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}"
            )
    
    async def check_supabase(self) -> ComponentHealth:
        """Check Supabase connectivity."""
        if not SUPABASE_AVAILABLE:
            return ComponentHealth(
                name="supabase",
                status=HealthStatus.DEGRADED,
                message="supabase client not available"
            )
        
        if not self._supabase_client:
            return ComponentHealth(
                name="supabase",
                status=HealthStatus.DEGRADED,
                message="Supabase client not configured"
            )
        
        try:
            start_time = time.time()
            # Simple health check - attempt to select from a system table
            response = self._supabase_client.table('ingestion_jobs').select('count').limit(1).execute()
            response_time = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                name="supabase",
                status=HealthStatus.HEALTHY,
                message="Supabase is responding",
                response_time_ms=response_time,
                metadata={
                    "connection": "ok"
                }
            )
        except Exception as e:
            self.logger.error(f"Supabase health check failed: {e}")
            return ComponentHealth(
                name="supabase",
                status=HealthStatus.UNHEALTHY,
                message=f"Supabase connection failed: {str(e)}"
            )
    
    async def check_neo4j(self) -> ComponentHealth:
        """Check Neo4j connectivity."""
        if not self._neo4j_driver:
            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.DEGRADED,
                message="Neo4j driver not configured - persistence degraded"
            )
        
        try:
            start_time = time.time()
            async with self._neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as health")
                await result.consume()
            response_time = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.HEALTHY,
                message="Neo4j is responding",
                response_time_ms=response_time
            )
        except Exception as e:
            self.logger.error(f"Neo4j health check failed: {e}")
            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.DEGRADED,
                message=f"Neo4j connection failed: {str(e)} - persistence degraded"
            )
    
    async def check_qdrant(self) -> ComponentHealth:
        """Check Qdrant connectivity."""
        if not self._qdrant_client:
            return ComponentHealth(
                name="qdrant",
                status=HealthStatus.DEGRADED,
                message="Qdrant client not configured - semantic search degraded"
            )
        
        try:
            start_time = time.time()
            # Check if we can get collection info
            collections = await self._qdrant_client.get_collections()
            response_time = (time.time() - start_time) * 1000
            
            collection_count = len(collections.collections) if hasattr(collections, 'collections') else 0
            
            return ComponentHealth(
                name="qdrant",
                status=HealthStatus.HEALTHY,
                message="Qdrant is responding",
                response_time_ms=response_time,
                metadata={
                    "collections": collection_count
                }
            )
        except Exception as e:
            self.logger.error(f"Qdrant health check failed: {e}")
            return ComponentHealth(
                name="qdrant",
                status=HealthStatus.DEGRADED,
                message=f"Qdrant connection failed: {str(e)} - semantic search degraded"
            )
    
    async def check_disk_space(self) -> ComponentHealth:
        """Check available disk space."""
        try:
            import shutil
            
            # Check disk space for current directory
            total, used, free = shutil.disk_usage("/")
            
            free_gb = free // (1024**3)
            total_gb = total // (1024**3)
            used_percent = (used / total) * 100
            
            # Warn if less than 2GB free or more than 90% used
            if free_gb < 2 or used_percent > 90:
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {free_gb}GB free ({used_percent:.1f}% used)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_gb}GB free"
            
            return ComponentHealth(
                name="disk",
                status=status,
                message=message,
                metadata={
                    "free_gb": free_gb,
                    "total_gb": total_gb,
                    "used_percent": round(used_percent, 1)
                }
            )
        except Exception as e:
            self.logger.error(f"Disk space check failed: {e}")
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk space check failed: {str(e)}"
            )
    
    async def check_memory(self) -> ComponentHealth:
        """Check memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            available_gb = memory.available // (1024**3)
            
            # Warn if more than 85% used or less than 1GB available
            if used_percent > 85 or available_gb < 1:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {used_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage OK: {used_percent:.1f}% used"
            
            return ComponentHealth(
                name="memory",
                status=status,
                message=message,
                metadata={
                    "used_percent": round(used_percent, 1),
                    "available_gb": available_gb,
                    "total_gb": memory.total // (1024**3)
                }
            )
        except ImportError:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.DEGRADED,
                message="psutil not available - cannot check memory"
            )
        except Exception as e:
            self.logger.error(f"Memory check failed: {e}")
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}"
            )
    
    async def check_all(self, include_external: bool = True) -> SystemHealth:
        """Perform all health checks."""
        checks = [
            self.check_disk_space(),
            self.check_memory(),
        ]
        
        if include_external:
            checks.extend([
                self.check_redis(),
                self.check_supabase(),
                self.check_neo4j(),
                self.check_qdrant(),
            ])
        
        # Run all checks concurrently
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        components = {}
        overall_status = HealthStatus.HEALTHY
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check exception: {result}")
                continue
            
            if isinstance(result, ComponentHealth):
                components[result.name] = result
                
                # Determine overall status
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        return SystemHealth(
            status=overall_status,
            components=components,
            timestamp=time.time()
        )


# Global health checker instance
health_checker = HealthChecker()