"""
Prometheus metrics collection for the ingestion service.
"""
import time
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from functools import wraps

from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)


class MetricsCollector:
    """Centralized metrics collection for the ingestion service."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.logger = logging.getLogger(__name__)
        
        # Request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests by method and status',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Job processing metrics
        self.jobs_total = Counter(
            'ingestion_jobs_total',
            'Total ingestion jobs by status',
            ['status', 'user_id'],
            registry=self.registry
        )
        
        self.job_duration_seconds = Histogram(
            'ingestion_job_duration_seconds',
            'Job processing duration in seconds by stage',
            ['stage', 'status'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600],
            registry=self.registry
        )
        
        self.job_queue_size = Gauge(
            'ingestion_job_queue_size',
            'Number of jobs in queue by priority',
            ['priority'],
            registry=self.registry
        )
        
        # OCR metrics
        self.ocr_operations_total = Counter(
            'ocr_operations_total',
            'Total OCR operations by engine and status',
            ['engine', 'status'],
            registry=self.registry
        )
        
        self.ocr_accuracy = Histogram(
            'ocr_accuracy_ratio',
            'OCR accuracy (CER/WER) by engine',
            ['engine', 'metric_type'],
            buckets=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
            registry=self.registry
        )
        
        self.ocr_pages_processed = Counter(
            'ocr_pages_processed_total',
            'Total pages processed by OCR engine',
            ['engine'],
            registry=self.registry
        )
        
        # Schematic analysis metrics
        self.components_detected = Counter(
            'schematic_components_detected_total',
            'Total components detected by type',
            ['component_type'],
            registry=self.registry
        )
        
        self.nets_extracted = Counter(
            'schematic_nets_extracted_total',
            'Total nets extracted by page',
            ['project_id'],
            registry=self.registry
        )
        
        self.schematic_accuracy = Histogram(
            'schematic_analysis_accuracy',
            'Schematic analysis structural accuracy',
            buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99],
            registry=self.registry
        )
        
        # Persistence metrics
        self.persistence_operations = Counter(
            'persistence_operations_total',
            'Total persistence operations by backend and status',
            ['backend', 'operation', 'status'],
            registry=self.registry
        )
        
        self.persistence_duration = Histogram(
            'persistence_operation_duration_seconds',
            'Persistence operation duration by backend',
            ['backend', 'operation'],
            registry=self.registry
        )
        
        self.storage_bytes_written = Counter(
            'storage_bytes_written_total',
            'Total bytes written to storage by backend',
            ['backend', 'artifact_type'],
            registry=self.registry
        )
        
        # System resource metrics
        self.memory_usage_bytes = Gauge(
            'process_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'process_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        self.disk_usage_bytes = Gauge(
            'disk_usage_bytes',
            'Disk usage in bytes by mount point',
            ['mountpoint', 'fstype'],
            registry=self.registry
        )
        
        # External service metrics
        self.external_service_requests = Counter(
            'external_service_requests_total',
            'Total requests to external services',
            ['service', 'operation', 'status'],
            registry=self.registry
        )
        
        self.external_service_duration = Histogram(
            'external_service_request_duration_seconds',
            'External service request duration',
            ['service', 'operation'],
            registry=self.registry
        )
        
        # Database connection pools
        self.db_connections_active = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database'],
            registry=self.registry
        )
        
        self.db_connections_idle = Gauge(
            'database_connections_idle',
            'Idle database connections',
            ['database'],
            registry=self.registry
        )
        
        # Error tracking
        self.errors_total = Counter(
            'errors_total',
            'Total errors by type and component',
            ['error_type', 'component', 'severity'],
            registry=self.registry
        )
        
        # User quota metrics
        self.user_quota_usage = Gauge(
            'user_quota_usage_ratio',
            'User quota usage ratio',
            ['user_id', 'quota_type'],
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'ingestion_service_info',
            'Information about the ingestion service',
            registry=self.registry
        )
        
        # Set service info
        self.service_info.info({
            'version': '1.0.0',
            'python_version': 'unknown',
            'build_date': 'unknown'
        })
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_job_started(self, user_id: str):
        """Record job start."""
        self.jobs_total.labels(status='started', user_id=user_id).inc()
    
    def record_job_completed(self, user_id: str, duration: float, stage: str = 'total'):
        """Record job completion."""
        self.jobs_total.labels(status='completed', user_id=user_id).inc()
        self.job_duration_seconds.labels(stage=stage, status='completed').observe(duration)
    
    def record_job_failed(self, user_id: str, duration: float, stage: str = 'unknown'):
        """Record job failure."""
        self.jobs_total.labels(status='failed', user_id=user_id).inc()
        self.job_duration_seconds.labels(stage=stage, status='failed').observe(duration)
    
    def record_ocr_operation(self, engine: str, status: str, pages: int = 1, cer: float = None, wer: float = None):
        """Record OCR operation metrics."""
        self.ocr_operations_total.labels(engine=engine, status=status).inc()
        self.ocr_pages_processed.labels(engine=engine).inc(pages)
        
        if cer is not None:
            self.ocr_accuracy.labels(engine=engine, metric_type='cer').observe(cer)
        if wer is not None:
            self.ocr_accuracy.labels(engine=engine, metric_type='wer').observe(wer)
    
    def record_component_detection(self, component_type: str, count: int = 1):
        """Record component detection."""
        self.components_detected.labels(component_type=component_type).inc(count)
    
    def record_nets_extraction(self, project_id: str, count: int):
        """Record nets extraction."""
        self.nets_extracted.labels(project_id=project_id).inc(count)
    
    def record_schematic_accuracy(self, accuracy: float):
        """Record schematic analysis accuracy."""
        self.schematic_accuracy.observe(accuracy)
    
    def record_persistence_operation(self, backend: str, operation: str, status: str, duration: float):
        """Record persistence operation."""
        self.persistence_operations.labels(
            backend=backend,
            operation=operation,
            status=status
        ).inc()
        
        self.persistence_duration.labels(
            backend=backend,
            operation=operation
        ).observe(duration)
    
    def record_storage_write(self, backend: str, artifact_type: str, bytes_written: int):
        """Record storage write operation."""
        self.storage_bytes_written.labels(
            backend=backend,
            artifact_type=artifact_type
        ).inc(bytes_written)
    
    def record_external_service_call(self, service: str, operation: str, status: str, duration: float):
        """Record external service call."""
        self.external_service_requests.labels(
            service=service,
            operation=operation,
            status=status
        ).inc()
        
        self.external_service_duration.labels(
            service=service,
            operation=operation
        ).observe(duration)
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            process = psutil.Process()
            self.memory_usage_bytes.set(process.memory_info().rss)
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.cpu_usage_percent.set(cpu_percent)
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.disk_usage_bytes.labels(
                        mountpoint=partition.mountpoint,
                        fstype=partition.fstype
                    ).set(usage.used)
                except (PermissionError, FileNotFoundError):
                    continue
        except ImportError:
            self.logger.warning("psutil not available - cannot update system metrics")
        except Exception as e:
            self.logger.error(f"Failed to update system metrics: {e}")
    
    def update_db_connection_metrics(self, db_name: str, active: int, idle: int):
        """Update database connection pool metrics."""
        self.db_connections_active.labels(database=db_name).set(active)
        self.db_connections_idle.labels(database=db_name).set(idle)
    
    def record_error(self, error_type: str, component: str, severity: str = 'error'):
        """Record an error occurrence."""
        self.errors_total.labels(
            error_type=error_type,
            component=component,
            severity=severity
        ).inc()
    
    def update_user_quota(self, user_id: str, quota_type: str, usage_ratio: float):
        """Update user quota usage."""
        self.user_quota_usage.labels(
            user_id=user_id,
            quota_type=quota_type
        ).set(usage_ratio)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get the content type for metrics endpoint."""
        return CONTENT_TYPE_LATEST
    
    @asynccontextmanager
    async def time_operation(self, backend: str, operation: str):
        """Context manager to time persistence operations."""
        start_time = time.time()
        success = False
        try:
            yield
            success = True
        except Exception as e:
            self.record_error(
                error_type=type(e).__name__,
                component=f"{backend}_{operation}",
                severity='error'
            )
            raise
        finally:
            duration = time.time() - start_time
            status = 'success' if success else 'error'
            self.record_persistence_operation(backend, operation, status, duration)


def http_metrics(metrics: MetricsCollector):
    """Decorator to automatically record HTTP request metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request info from FastAPI request
            request = None
            for arg in args:
                if hasattr(arg, 'method') and hasattr(arg, 'url'):
                    request = arg
                    break
            
            if not request:
                return await func(*args, **kwargs)
            
            method = request.method
            endpoint = str(request.url.path)
            start_time = time.time()
            
            try:
                response = await func(*args, **kwargs)
                status_code = getattr(response, 'status_code', 200)
                duration = time.time() - start_time
                
                metrics.record_http_request(method, endpoint, status_code, duration)
                return response
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_http_request(method, endpoint, 500, duration)
                metrics.record_error(
                    error_type=type(e).__name__,
                    component='http_handler',
                    severity='error'
                )
                raise
        
        return wrapper
    return decorator


# Global metrics collector instance
metrics = MetricsCollector()