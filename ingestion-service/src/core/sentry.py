"""
Sentry integration for error tracking and performance monitoring.
"""
import os
import logging
from typing import Dict, Any, Optional, Union
from functools import wraps

try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.redis import RedisIntegration
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.asyncio import AsyncioIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

from .logging import get_correlation_id, get_user_id, get_job_id


class SentryManager:
    """Manages Sentry integration for error tracking and monitoring."""
    
    def __init__(self):
        self.initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self, 
                   dsn: Optional[str] = None,
                   environment: str = 'development',
                   release: Optional[str] = None,
                   sample_rate: float = 1.0,
                   traces_sample_rate: float = 0.1,
                   enable_profiling: bool = False,
                   profiles_sample_rate: float = 0.1):
        """Initialize Sentry with configuration."""
        if not SENTRY_AVAILABLE:
            self.logger.warning("Sentry SDK not available - error tracking disabled")
            return
        
        if not dsn:
            dsn = os.getenv('SENTRY_DSN')
        
        if not dsn:
            self.logger.info("No Sentry DSN provided - error tracking disabled")
            return
        
        try:
            # Configure integrations
            integrations = [
                LoggingIntegration(
                    level=logging.INFO,        # Capture info and above as breadcrumbs
                    event_level=logging.ERROR  # Send errors and above as events
                ),
                FastApiIntegration(auto_enabling_integrations=False),
                AsyncioIntegration(),
                RedisIntegration(),
            ]
            
            # Try to add SQLAlchemy integration if available
            try:
                from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
                integrations.append(SqlalchemyIntegration())
            except ImportError:
                pass  # SQLAlchemy not available, skip integration
            
            # Initialize Sentry
            sentry_sdk.init(
                dsn=dsn,
                environment=environment,
                release=release,
                sample_rate=sample_rate,
                traces_sample_rate=traces_sample_rate,
                profiles_sample_rate=profiles_sample_rate if enable_profiling else 0.0,
                integrations=integrations,
                before_send=self._before_send,
                before_send_transaction=self._before_send_transaction,
            )
            
            # Set global tags
            sentry_sdk.set_tag("service", "ingestion-service")
            
            self.initialized = True
            self.logger.info(f"Sentry initialized for environment: {environment}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Sentry: {e}")
    
    def _before_send(self, event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter and modify events before sending to Sentry."""
        try:
            # Add correlation context
            correlation_id = get_correlation_id()
            user_id = get_user_id()
            job_id = get_job_id()
            
            if correlation_id:
                event.setdefault('tags', {})['correlation_id'] = correlation_id
            
            if user_id:
                event.setdefault('user', {})['id'] = user_id
            
            if job_id:
                event.setdefault('tags', {})['job_id'] = job_id
            
            # Sanitize sensitive data
            event = self._sanitize_event(event)
            
            # Filter out certain errors in development
            if os.getenv('APP_ENV') == 'development':
                # Skip certain development-only errors
                if self._is_development_error(event):
                    return None
            
            return event
        except Exception as e:
            self.logger.error(f"Error in Sentry before_send: {e}")
            return event
    
    def _before_send_transaction(self, event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter transactions before sending to Sentry."""
        try:
            # Skip health check transactions
            if 'transaction' in event and event['transaction'] in ['/healthz', '/readyz', '/metrics']:
                return None
            
            # Add correlation context
            correlation_id = get_correlation_id()
            if correlation_id:
                event.setdefault('tags', {})['correlation_id'] = correlation_id
            
            return event
        except Exception as e:
            self.logger.error(f"Error in Sentry before_send_transaction: {e}")
            return event
    
    def _sanitize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from Sentry events."""
        # List of keys that might contain sensitive data
        sensitive_keys = {
            'password', 'token', 'secret', 'key', 'authorization',
            'api_key', 'auth', 'credential', 'private'
        }
        
        def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively sanitize dictionary data."""
            sanitized = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    sanitized[key] = '[REDACTED]'
                elif isinstance(value, dict):
                    sanitized[key] = sanitize_dict(value)
                elif isinstance(value, list):
                    sanitized[key] = [
                        sanitize_dict(item) if isinstance(item, dict) else item 
                        for item in value
                    ]
                else:
                    sanitized[key] = value
            return sanitized
        
        # Sanitize different parts of the event
        if 'request' in event and 'data' in event['request']:
            if isinstance(event['request']['data'], dict):
                event['request']['data'] = sanitize_dict(event['request']['data'])
        
        if 'extra' in event and isinstance(event['extra'], dict):
            event['extra'] = sanitize_dict(event['extra'])
        
        return event
    
    def _is_development_error(self, event: Dict[str, Any]) -> bool:
        """Check if error should be filtered in development."""
        # Skip common development errors
        if 'exception' in event:
            for exception in event['exception'].get('values', []):
                error_type = exception.get('type', '')
                error_value = exception.get('value', '')
                
                # Skip connection errors to external services in development
                if 'ConnectionError' in error_type or 'timeout' in error_value.lower():
                    return True
                
                # Skip file not found errors for optional resources
                if 'FileNotFoundError' in error_type:
                    return True
        
        return False
    
    def capture_exception(self, 
                         exception: Exception,
                         tags: Optional[Dict[str, str]] = None,
                         extra: Optional[Dict[str, Any]] = None,
                         user: Optional[Dict[str, str]] = None,
                         level: str = 'error'):
        """Capture an exception with additional context."""
        if not self.initialized:
            return
        
        try:
            with sentry_sdk.push_scope() as scope:
                # Set level
                scope.level = level
                
                # Add tags
                if tags:
                    for key, value in tags.items():
                        scope.set_tag(key, str(value))
                
                # Add extra context
                if extra:
                    for key, value in extra.items():
                        scope.set_extra(key, value)
                
                # Set user context
                if user:
                    scope.user = user
                elif get_user_id():
                    scope.user = {"id": get_user_id()}
                
                # Add correlation context
                correlation_id = get_correlation_id()
                if correlation_id:
                    scope.set_tag("correlation_id", correlation_id)
                
                job_id = get_job_id()
                if job_id:
                    scope.set_tag("job_id", job_id)
                
                sentry_sdk.capture_exception(exception)
        except Exception as e:
            self.logger.error(f"Failed to capture exception to Sentry: {e}")
    
    def capture_message(self,
                       message: str,
                       level: str = 'info',
                       tags: Optional[Dict[str, str]] = None,
                       extra: Optional[Dict[str, Any]] = None):
        """Capture a message with context."""
        if not self.initialized:
            return
        
        try:
            with sentry_sdk.push_scope() as scope:
                scope.level = level
                
                if tags:
                    for key, value in tags.items():
                        scope.set_tag(key, str(value))
                
                if extra:
                    for key, value in extra.items():
                        scope.set_extra(key, value)
                
                # Add correlation context
                correlation_id = get_correlation_id()
                if correlation_id:
                    scope.set_tag("correlation_id", correlation_id)
                
                sentry_sdk.capture_message(message, level)
        except Exception as e:
            self.logger.error(f"Failed to capture message to Sentry: {e}")
    
    def add_breadcrumb(self,
                      message: str,
                      category: str = 'custom',
                      level: str = 'info',
                      data: Optional[Dict[str, Any]] = None):
        """Add a breadcrumb for debugging context."""
        if not self.initialized:
            return
        
        try:
            sentry_sdk.add_breadcrumb(
                message=message,
                category=category,
                level=level,
                data=data or {}
            )
        except Exception as e:
            self.logger.error(f"Failed to add breadcrumb to Sentry: {e}")
    
    def set_user(self, user_id: str, email: Optional[str] = None, username: Optional[str] = None):
        """Set user context for current scope."""
        if not self.initialized:
            return
        
        try:
            user_data = {"id": user_id}
            if email:
                user_data["email"] = email
            if username:
                user_data["username"] = username
            
            sentry_sdk.set_user(user_data)
        except Exception as e:
            self.logger.error(f"Failed to set user in Sentry: {e}")
    
    def start_transaction(self, name: str, op: str = "task") -> Optional[Any]:
        """Start a performance transaction."""
        if not self.initialized:
            return None
        
        try:
            return sentry_sdk.start_transaction(name=name, op=op)
        except Exception as e:
            self.logger.error(f"Failed to start Sentry transaction: {e}")
            return None
    
    def is_initialized(self) -> bool:
        """Check if Sentry is initialized."""
        return self.initialized


def sentry_trace(operation_name: str = None, op: str = "function"):
    """Decorator to trace function execution with Sentry."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not sentry_manager.is_initialized():
                return await func(*args, **kwargs)
            
            name = operation_name or f"{func.__module__}.{func.__name__}"
            transaction = sentry_manager.start_transaction(name=name, op=op)
            
            try:
                if transaction:
                    with transaction:
                        return await func(*args, **kwargs)
                else:
                    return await func(*args, **kwargs)
            except Exception as e:
                sentry_manager.capture_exception(
                    e,
                    tags={"function": func.__name__},
                    extra={"operation": name}
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not sentry_manager.is_initialized():
                return func(*args, **kwargs)
            
            name = operation_name or f"{func.__module__}.{func.__name__}"
            transaction = sentry_manager.start_transaction(name=name, op=op)
            
            try:
                if transaction:
                    with transaction:
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                sentry_manager.capture_exception(
                    e,
                    tags={"function": func.__name__},
                    extra={"operation": name}
                )
                raise
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global Sentry manager instance
sentry_manager = SentryManager()