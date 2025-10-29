"""
Structured logging with correlation IDs and context management.
"""
import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Dict, Any, Optional, Union
from functools import wraps
from contextvars import ContextVar

# Context variables for request correlation
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
job_id: ContextVar[Optional[str]] = ContextVar('job_id', default=None)


class CorrelationFormatter(logging.Formatter):
    """Logging formatter that includes correlation ID and context."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add correlation context to log record
        record.correlation_id = correlation_id.get()
        record.user_id = user_id.get()
        record.job_id = job_id.get()
        
        # Create structured log entry
        log_entry = {
            'timestamp': time.time(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': record.correlation_id,
            'user_id': record.user_id,
            'job_id': record.job_id,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from log record
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                        'filename', 'module', 'lineno', 'funcName', 'created',
                        'msecs', 'relativeCreated', 'thread', 'threadName',
                        'processName', 'process', 'correlation_id', 'user_id',
                        'job_id', 'exc_info', 'exc_text', 'stack_info')
        }
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str, separators=(',', ':'))


class StructuredLogger:
    """Enhanced logger with structured output and context management."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup structured logging format."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(CorrelationFormatter())
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)
    
    def log_operation_start(self, operation: str, **context):
        """Log the start of an operation."""
        self.info(f"Starting operation: {operation}", 
                 operation=operation, 
                 operation_status='started',
                 **context)
    
    def log_operation_success(self, operation: str, duration: float = None, **context):
        """Log successful completion of an operation."""
        extra = {
            'operation': operation,
            'operation_status': 'success',
            **context
        }
        if duration is not None:
            extra['duration_seconds'] = duration
        
        self.info(f"Operation completed successfully: {operation}", **extra)
    
    def log_operation_error(self, operation: str, error: Exception, duration: float = None, **context):
        """Log operation failure."""
        extra = {
            'operation': operation,
            'operation_status': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            **context
        }
        if duration is not None:
            extra['duration_seconds'] = duration
        
        self.error(f"Operation failed: {operation}", **extra)
    
    def log_metrics(self, metrics: Dict[str, Union[int, float, str]], **context):
        """Log metrics data."""
        self.info("Metrics recorded", metrics=metrics, **context)
    
    def log_api_request(self, method: str, path: str, status_code: int, 
                       duration: float, **context):
        """Log API request details."""
        self.info(f"API request: {method} {path}", 
                 api_method=method,
                 api_path=path,
                 api_status_code=status_code,
                 api_duration_seconds=duration,
                 **context)
    
    def log_job_status(self, status: str, stage: str = None, progress: int = None, **context):
        """Log job status update."""
        extra = {
            'job_status': status,
            **context
        }
        if stage:
            extra['job_stage'] = stage
        if progress is not None:
            extra['job_progress'] = progress
        
        self.info(f"Job status: {status}", **extra)
    
    def log_user_action(self, action: str, **context):
        """Log user action."""
        self.info(f"User action: {action}", 
                 user_action=action,
                 **context)
    
    def log_external_service(self, service: str, operation: str, status: str,
                           duration: float = None, **context):
        """Log external service interaction."""
        extra = {
            'external_service': service,
            'external_operation': operation,
            'external_status': status,
            **context
        }
        if duration is not None:
            extra['external_duration_seconds'] = duration
        
        self.info(f"External service call: {service}.{operation} -> {status}", **extra)
    
    def log_security_event(self, event_type: str, severity: str = 'warning', **context):
        """Log security-related events."""
        self.warning(f"Security event: {event_type}",
                    security_event_type=event_type,
                    security_severity=severity,
                    **context)


@contextmanager
def log_context(**context_vars):
    """Context manager to set logging context variables."""
    tokens = []
    
    try:
        # Set context variables
        for key, value in context_vars.items():
            if key == 'correlation_id':
                token = correlation_id.set(value)
                tokens.append((correlation_id, token))
            elif key == 'user_id':
                token = user_id.set(value)
                tokens.append((user_id, token))
            elif key == 'job_id':
                token = job_id.set(value)
                tokens.append((job_id, token))
        
        yield
    finally:
        # Reset context variables
        for var, token in reversed(tokens):
            var.reset(token)


@contextmanager
def log_operation(logger: StructuredLogger, operation: str, **context):
    """Context manager to log operation start/end with timing."""
    start_time = time.time()
    logger.log_operation_start(operation, **context)
    
    try:
        yield
        duration = time.time() - start_time
        logger.log_operation_success(operation, duration, **context)
    except Exception as e:
        duration = time.time() - start_time
        logger.log_operation_error(operation, e, duration, **context)
        raise


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def logged_function(operation_name: str = None):
    """Decorator to automatically log function execution."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            op_name = operation_name or f"{func.__name__}"
            
            with log_operation(logger, op_name):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            op_name = operation_name or f"{func.__name__}"
            
            with log_operation(logger, op_name):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()


def get_user_id() -> Optional[str]:
    """Get current user ID."""
    return user_id.get()


def get_job_id() -> Optional[str]:
    """Get current job ID."""
    return job_id.get()


def setup_logging(level: str = 'INFO', enable_structured: bool = True):
    """Setup global logging configuration."""
    logging_level = getattr(logging, level.upper(), logging.INFO)
    
    if enable_structured:
        # Setup structured logging for all loggers
        formatter = CorrelationFormatter()
        
        # Get root logger and configure
        root_logger = logging.getLogger()
        root_logger.setLevel(logging_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add structured handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        # Use basic logging
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


# Module-level logger for this file
logger = StructuredLogger(__name__)