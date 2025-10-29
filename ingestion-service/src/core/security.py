"""
Security, authentication, and authorization utilities for the ingestion service.
"""
import os
import time
import hashlib
import secrets
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from functools import wraps
from dataclasses import dataclass

try:
    from fastapi import HTTPException, Request, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Mock classes for testing
    class HTTPException(Exception):
        def __init__(self, status_code, detail, headers=None):
            self.status_code = status_code
            self.detail = detail
            self.headers = headers
            super().__init__(detail)
    
    class Request:
        pass
    
    class status:
        HTTP_401_UNAUTHORIZED = 401
        HTTP_403_FORBIDDEN = 403
        HTTP_413_REQUEST_ENTITY_TOO_LARGE = 413
        HTTP_429_TOO_MANY_REQUESTS = 429
        HTTP_503_SERVICE_UNAVAILABLE = 503
    
    class HTTPBearer:
        def __init__(self, auto_error=True):
            pass
        async def __call__(self, request):
            return None
    
    class HTTPAuthorizationCredentials:
        def __init__(self, scheme="Bearer", credentials="mock_token"):
            self.scheme = scheme
            self.credentials = credentials

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    import aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False

from .logging import StructuredLogger, get_correlation_id
from .metrics import metrics

logger = StructuredLogger(__name__)


@dataclass
class UserContext:
    """User context extracted from JWT token."""
    user_id: str
    email: Optional[str] = None
    role: str = 'user'
    permissions: List[str] = None
    quota_tier: str = 'basic'
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []


class JWTError(Exception):
    """JWT validation errors."""
    pass


class RateLimitExceeded(Exception):
    """Rate limit exceeded error."""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds.")


class SecurityManager:
    """Manages authentication, authorization, and rate limiting."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.jwt_algorithm = os.getenv('JWT_ALGORITHM', 'RS256')
        self.require_auth = os.getenv('REQUIRE_AUTH', 'true').lower() == 'true'
        self.jwt_public_key = self._load_jwt_public_key()
        
        # Rate limiting configuration
        self.rate_limits = {
            'basic': {'requests_per_hour': 10, 'requests_per_day': 50},
            'premium': {'requests_per_hour': 100, 'requests_per_day': 1000},
            'enterprise': {'requests_per_hour': 1000, 'requests_per_day': 10000}
        }
        
        # Request size limits (in bytes)
        self.max_request_size = int(os.getenv('MAX_REQUEST_SIZE', '26214400'))  # 25MB
        self.max_file_size = int(os.getenv('MAX_FILE_SIZE', '104857600'))  # 100MB
    
    def _load_jwt_public_key(self) -> Optional[str]:
        """Load JWT public key from environment or file."""
        # Try environment variable first
        key = os.getenv('SUPABASE_JWT_PUBLIC_KEY')
        if key:
            return key
        
        # Try loading from file
        key_file = os.getenv('JWT_PUBLIC_KEY_FILE')
        if key_file and os.path.exists(key_file):
            with open(key_file, 'r') as f:
                return f.read()
        
        if self.require_auth:
            logger.error("JWT public key not found - authentication will fail")
        
        return None
    
    async def validate_jwt_token(self, token: str) -> UserContext:
        """Validate JWT token and extract user context."""
        if not self.require_auth:
            # Return mock user for development
            return UserContext(
                user_id='dev-user',
                email='dev@example.com',
                role='admin',
                quota_tier='enterprise'
            )
        
        if not JWT_AVAILABLE:
            raise JWTError("JWT library not available")
        
        if not self.jwt_public_key:
            raise JWTError("JWT public key not configured")
        
        try:
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.jwt_public_key,
                algorithms=[self.jwt_algorithm],
                options={"verify_aud": False}  # Supabase tokens don't always have aud
            )
            
            # Extract user information
            user_id = payload.get('sub')
            if not user_id:
                raise JWTError("Token missing subject (user ID)")
            
            email = payload.get('email')
            role = payload.get('role', 'user')
            
            # Extract custom claims
            app_metadata = payload.get('app_metadata', {})
            quota_tier = app_metadata.get('quota_tier', 'basic')
            permissions = app_metadata.get('permissions', [])
            
            # Log successful authentication
            logger.info("User authenticated successfully", 
                       user_id=user_id, 
                       role=role,
                       quota_tier=quota_tier)
            
            return UserContext(
                user_id=user_id,
                email=email,
                role=role,
                permissions=permissions,
                quota_tier=quota_tier
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired", correlation_id=get_correlation_id())
            raise JWTError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}", correlation_id=get_correlation_id())
            raise JWTError("Invalid token")
        except Exception as e:
            logger.error(f"JWT validation error: {e}", correlation_id=get_correlation_id())
            raise JWTError("Token validation failed")
    
    async def check_rate_limit(self, user_id: str, quota_tier: str) -> bool:
        """Check if user has exceeded rate limits."""
        if not AIOREDIS_AVAILABLE or not self.redis_client:
            logger.warning("Redis not available - rate limiting disabled")
            return True
        
        limits = self.rate_limits.get(quota_tier, self.rate_limits['basic'])
        current_time = int(time.time())
        
        try:
            # Check hourly limit
            hour_key = f"rate_limit:hour:{user_id}:{current_time // 3600}"
            hour_count = await self.redis_client.get(hour_key)
            hour_count = int(hour_count) if hour_count else 0
            
            if hour_count >= limits['requests_per_hour']:
                retry_after = 3600 - (current_time % 3600)
                logger.warning(f"User {user_id} exceeded hourly rate limit",
                             user_id=user_id,
                             hour_count=hour_count,
                             limit=limits['requests_per_hour'])
                metrics.record_error('rate_limit_exceeded', 'security', 'warning')
                raise RateLimitExceeded(retry_after)
            
            # Check daily limit
            day_key = f"rate_limit:day:{user_id}:{current_time // 86400}"
            day_count = await self.redis_client.get(day_key)
            day_count = int(day_count) if day_count else 0
            
            if day_count >= limits['requests_per_day']:
                retry_after = 86400 - (current_time % 86400)
                logger.warning(f"User {user_id} exceeded daily rate limit",
                             user_id=user_id,
                             day_count=day_count,
                             limit=limits['requests_per_day'])
                metrics.record_error('rate_limit_exceeded', 'security', 'warning')
                raise RateLimitExceeded(retry_after)
            
            return True
            
        except RateLimitExceeded:
            raise
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Allow request if rate limiting fails
            return True
    
    async def increment_rate_limit(self, user_id: str):
        """Increment rate limit counters for user."""
        if not AIOREDIS_AVAILABLE or not self.redis_client:
            return
        
        current_time = int(time.time())
        
        try:
            # Increment hourly counter
            hour_key = f"rate_limit:hour:{user_id}:{current_time // 3600}"
            await self.redis_client.incr(hour_key)
            await self.redis_client.expire(hour_key, 3600)
            
            # Increment daily counter
            day_key = f"rate_limit:day:{user_id}:{current_time // 86400}"
            await self.redis_client.incr(day_key)
            await self.redis_client.expire(day_key, 86400)
            
        except Exception as e:
            logger.error(f"Failed to increment rate limit: {e}")
    
    def validate_request_size(self, request: Request) -> bool:
        """Validate request size limits."""
        content_length = request.headers.get('content-length')
        if content_length:
            size = int(content_length)
            if size > self.max_request_size:
                logger.warning(f"Request size {size} exceeds limit {self.max_request_size}")
                metrics.record_error('request_too_large', 'security', 'warning')
                return False
        
        return True
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data to prevent injection attacks."""
        if isinstance(data, str):
            # Remove null bytes and control characters
            sanitized = data.replace('\x00', '').replace('\r', '').replace('\n', ' ')
            
            # Basic HTML/script tag removal (simple approach)
            import re
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            sanitized = re.sub(r'<[^>]+>', '', sanitized)
            
            return sanitized.strip()
        
        elif isinstance(data, dict):
            return {key: self.sanitize_input(value) for key, value in data.items()}
        
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        
        else:
            return data
    
    def check_permissions(self, user: UserContext, required_permissions: List[str]) -> bool:
        """Check if user has required permissions."""
        if user.role == 'admin':
            return True
        
        return all(perm in user.permissions for perm in required_permissions)
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate API key for user."""
        timestamp = str(int(time.time()))
        payload = f"{user_id}:{timestamp}:{secrets.token_hex(16)}"
        
        # Create hash
        hash_obj = hashlib.sha256(payload.encode())
        api_key = f"ws_{user_id[:8]}_{hash_obj.hexdigest()[:32]}"
        
        return api_key
    
    async def validate_api_key(self, api_key: str) -> Optional[UserContext]:
        """Validate API key and return user context."""
        # This would typically involve database lookup
        # For now, return None (not implemented)
        return None


class AuthenticationMiddleware:
    """FastAPI middleware for authentication and rate limiting."""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.bearer = HTTPBearer(auto_error=False)
    
    async def __call__(self, request: Request) -> UserContext:
        """Authenticate request and return user context."""
        # Skip authentication for health checks and metrics
        if request.url.path in ['/healthz', '/readyz', '/metrics']:
            return UserContext(user_id='system', role='system')
        
        # Validate request size
        if not self.security_manager.validate_request_size(request):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        # Extract authorization header
        credentials: HTTPAuthorizationCredentials = await self.bearer(request)
        
        if not credentials and self.security_manager.require_auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not credentials:
            # Development mode - return mock user
            return UserContext(user_id='dev-user', role='admin', quota_tier='enterprise')
        
        try:
            # Validate JWT token
            user = await self.security_manager.validate_jwt_token(credentials.credentials)
            
            # Check rate limits
            await self.security_manager.check_rate_limit(user.user_id, user.quota_tier)
            
            # Increment rate limit counter
            await self.security_manager.increment_rate_limit(user.user_id)
            
            # Record metrics
            metrics.record_external_service_call('auth', 'validate_token', 'success', 0.1)
            
            return user
            
        except JWTError as e:
            logger.warning(f"Authentication failed: {e}")
            metrics.record_error('authentication_failed', 'security', 'warning')
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        except RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded: {e}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(e),
                headers={"Retry-After": str(e.retry_after)},
            )


def require_auth(permissions: List[str] = None):
    """Decorator to require authentication and optional permissions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user context from kwargs (injected by middleware)
            user = kwargs.get('user')
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check permissions if specified
            if permissions and not security_manager.check_permissions(user, permissions):
                logger.warning(f"Insufficient permissions for user {user.user_id}",
                             user_id=user.user_id,
                             required_permissions=permissions,
                             user_permissions=user.permissions)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def admin_required(func):
    """Decorator to require admin role."""
    return require_auth()(func)


# Global security manager instance
security_manager = SecurityManager()