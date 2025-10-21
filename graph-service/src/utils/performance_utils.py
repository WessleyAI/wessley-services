"""Performance utilities for graph operations"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    operation: str
    duration: float
    timestamp: datetime
    parameters: Dict[str, Any]
    result_count: Optional[int] = None
    memory_usage: Optional[float] = None
    error: Optional[str] = None

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.operation_counts: Dict[str, int] = {}
        self.total_durations: Dict[str, float] = {}
    
    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric"""
        self.metrics.append(metric)
        
        # Update operation statistics
        if metric.operation not in self.operation_counts:
            self.operation_counts[metric.operation] = 0
            self.total_durations[metric.operation] = 0.0
        
        self.operation_counts[metric.operation] += 1
        self.total_durations[metric.operation] += metric.duration
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        if operation not in self.operation_counts:
            return {"error": "Operation not found"}
        
        count = self.operation_counts[operation]
        total_duration = self.total_durations[operation]
        avg_duration = total_duration / count if count > 0 else 0
        
        # Get recent metrics for this operation
        recent_metrics = [m for m in self.metrics[-100:] if m.operation == operation]
        recent_durations = [m.duration for m in recent_metrics]
        
        return {
            "operation": operation,
            "total_calls": count,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "recent_calls": len(recent_metrics),
            "recent_avg_duration": sum(recent_durations) / len(recent_durations) if recent_durations else 0,
            "min_duration": min(recent_durations) if recent_durations else 0,
            "max_duration": max(recent_durations) if recent_durations else 0
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all operations"""
        return {
            operation: self.get_operation_stats(operation)
            for operation in self.operation_counts.keys()
        }
    
    def get_slow_operations(self, threshold: float = 1.0) -> List[PerformanceMetrics]:
        """Get operations that took longer than threshold seconds"""
        return [m for m in self.metrics if m.duration > threshold]
    
    def clear_metrics(self):
        """Clear all recorded metrics"""
        self.metrics.clear()
        self.operation_counts.clear()
        self.total_durations.clear()

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(operation_name: str = None, include_params: bool = False):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            parameters = {}
            if include_params:
                parameters = {
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
            
            error = None
            result_count = None
            
            try:
                result = await func(*args, **kwargs)
                
                # Try to extract result count
                if isinstance(result, (list, tuple)):
                    result_count = len(result)
                elif isinstance(result, dict) and 'count' in result:
                    result_count = result['count']
                
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                metric = PerformanceMetrics(
                    operation=op_name,
                    duration=duration,
                    timestamp=datetime.utcnow(),
                    parameters=parameters,
                    result_count=result_count,
                    error=error
                )
                
                performance_monitor.record_metric(metric)
                
                # Log slow operations
                if duration > 5.0:
                    logger.warning(f"Slow operation: {op_name} took {duration:.2f}s")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            parameters = {}
            if include_params:
                parameters = {
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
            
            error = None
            result_count = None
            
            try:
                result = func(*args, **kwargs)
                
                if isinstance(result, (list, tuple)):
                    result_count = len(result)
                elif isinstance(result, dict) and 'count' in result:
                    result_count = result['count']
                
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                metric = PerformanceMetrics(
                    operation=op_name,
                    duration=duration,
                    timestamp=datetime.utcnow(),
                    parameters=parameters,
                    result_count=result_count,
                    error=error
                )
                
                performance_monitor.record_metric(metric)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class BatchProcessor:
    """Utility for processing large datasets in batches"""
    
    @staticmethod
    async def process_in_batches(data: List[Any], batch_size: int, 
                               processor: Callable, max_concurrent: int = 5) -> List[Any]:
        """Process data in batches with concurrency control"""
        if not data:
            return []
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch):
            async with semaphore:
                return await processor(batch)
        
        # Create batches
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        
        # Process batches concurrently
        batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results

class QueryOptimizer:
    """Query optimization utilities"""
    
    @staticmethod
    def should_use_index(field_name: str, operation: str) -> bool:
        """Determine if an index should be used for a field/operation"""
        # Fields that typically benefit from indexes
        indexed_fields = {'id', 'vehicle_signature', 'type', 'name'}
        
        # Operations that benefit from indexes
        index_operations = {'=', 'IN', 'STARTS WITH', 'ENDS WITH'}
        
        return field_name in indexed_fields and operation in index_operations
    
    @staticmethod
    def estimate_selectivity(field_name: str, value: Any) -> float:
        """Estimate selectivity of a filter condition"""
        # This is a simplified estimation
        if field_name == 'vehicle_signature':
            return 0.1  # Very selective
        elif field_name == 'id':
            return 0.01  # Highly selective
        elif field_name == 'type':
            return 0.2  # Moderately selective
        else:
            return 0.5  # Default selectivity
    
    @staticmethod
    def optimize_query_order(conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder query conditions for optimal performance"""
        # Sort by estimated selectivity (most selective first)
        return sorted(conditions, 
                     key=lambda c: QueryOptimizer.estimate_selectivity(c.get('field', ''), c.get('value')))

class CacheManager:
    """Simple in-memory cache for frequently accessed data"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if datetime.utcnow() > entry['expires']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['created'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'value': value,
            'created': datetime.utcnow(),
            'expires': datetime.utcnow() + timedelta(seconds=self.ttl_seconds)
        }
    
    def delete(self, key: str):
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        now = datetime.utcnow()
        expired_count = sum(1 for entry in self.cache.values() 
                          if now > entry['expires'])
        
        return {
            "total_entries": len(self.cache),
            "expired_entries": expired_count,
            "valid_entries": len(self.cache) - expired_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }

# Global cache instance
query_cache = CacheManager()

def cache_result(key_func: Callable = None, ttl: int = 300):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = query_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            query_cache.set(cache_key, result)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            cached_result = query_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = func(*args, **kwargs)
            query_cache.set(cache_key, result)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator