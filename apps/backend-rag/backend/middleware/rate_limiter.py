"""
Rate Limiting Middleware for ZANTARA
Prevents API abuse and ensures fair usage

Features:
- IP-based rate limiting
- User-based rate limiting (auto-extracted from JWT)
- Configurable limits per endpoint
- Redis-backed for distributed systems
- Token bucket algorithm for burst handling
- Different limits for authenticated vs anonymous users
"""

import logging
import time
from typing import Optional

import jwt
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

logger = logging.getLogger(__name__)

# In-memory rate limit storage (fallback)
_rate_limit_storage = {}


class RateLimiter:
    """
    Rate limiter with sliding window algorithm
    """

    def __init__(self):
        self.redis_available = False
        self.redis_client = None

        # Try to connect to Redis
        from app.core.config import settings

        redis_url = settings.redis_url
        if redis_url:
            try:
                import redis

                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                self.redis_available = True
                logger.info("✅ Rate limiter using Redis")
            except Exception as e:
                logger.warning(f"⚠️ Rate limiter using memory: {e}")
        else:
            logger.info("ℹ️ Rate limiter using in-memory storage")

    def is_allowed(self, key: str, limit: int, window: int) -> tuple[bool, dict]:
        """
        Check if request is allowed under rate limit

        Args:
            key: Unique identifier (IP or user)
            limit: Max requests allowed
            window: Time window in seconds

        Returns:
            (allowed, info_dict)
        """
        current_time = int(time.time())
        window_start = current_time - window

        try:
            if self.redis_available and self.redis_client:
                # Redis-backed sliding window
                pipe = self.redis_client.pipeline()

                # Remove old entries
                pipe.zremrangebyscore(key, 0, window_start)

                # Count current requests
                pipe.zcard(key)

                # Add current request
                pipe.zadd(key, {str(current_time): current_time})

                # Set expiration
                pipe.expire(key, window)

                results = pipe.execute()
                count = results[1]

                allowed = count < limit
                remaining = max(0, limit - count - 1)

                return allowed, {
                    "limit": limit,
                    "remaining": remaining,
                    "reset": current_time + window,
                }
            else:
                # In-memory fallback
                if key not in _rate_limit_storage:
                    _rate_limit_storage[key] = []

                # Remove old entries
                _rate_limit_storage[key] = [t for t in _rate_limit_storage[key] if t > window_start]

                count = len(_rate_limit_storage[key])
                allowed = count < limit

                if allowed:
                    _rate_limit_storage[key].append(current_time)

                remaining = max(0, limit - count - 1)

                return allowed, {
                    "limit": limit,
                    "remaining": remaining,
                    "reset": current_time + window,
                }

        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # On error, allow request (fail open)
            return True, {"limit": limit, "remaining": limit, "reset": current_time + window}


# Global rate limiter instance
rate_limiter = RateLimiter()


def extract_user_from_request(request: Request) -> tuple[Optional[str], bool]:
    """
    Extract user identifier from request (JWT token, cookie, or header).

    Returns:
        (user_id, is_authenticated) - user_id is email or IP, is_authenticated indicates JWT validity
    """
    client_ip = request.client.host if request.client else "unknown"

    # 1. Try X-User-ID header (explicit override)
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return user_id, True

    # 2. Try JWT from Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key or "default-secret",
                algorithms=["HS256"],
                options={"verify_exp": False}  # Don't fail on expired for rate limiting
            )
            user_email = payload.get("sub") or payload.get("email")
            if user_email:
                return user_email, True
        except jwt.InvalidTokenError:
            pass

    # 3. Try JWT from cookie
    token = request.cookies.get("access_token")
    if token:
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key or "default-secret",
                algorithms=["HS256"],
                options={"verify_exp": False}
            )
            user_email = payload.get("sub") or payload.get("email")
            if user_email:
                return user_email, True
        except jwt.InvalidTokenError:
            pass

    # 4. Fallback to IP address
    return client_ip, False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce rate limits on API endpoints
    """

    # Rate limit configuration per endpoint pattern
    # Format: (anonymous_limit, authenticated_limit, window_seconds)
    # If only 2 values, same limit for both
    RATE_LIMITS = {
        # Strict limits for expensive operations
        "/api/agents/journey/create": (5, 20, 3600),  # 5/20 per hour
        "/api/agents/compliance/track": (10, 40, 3600),  # 10/40 per hour
        "/api/agents/ingestion/run": (2, 10, 3600),  # 2/10 per hour
        # CRM endpoints - moderate limits
        "/api/crm/clients": (30, 200, 60),  # 30/200 per minute
        "/api/crm/practices": (30, 200, 60),  # 30/200 per minute
        "/api/crm/interactions": (50, 300, 60),  # 50/300 per minute
        "/api/crm/shared-memory": (20, 120, 60),  # 20/120 per minute
        # CRM stats endpoints - stricter limits
        "/api/crm/clients/stats": (10, 60, 60),  # 10/60 per minute
        "/api/crm/practices/stats": (10, 60, 60),  # 10/60 per minute
        "/api/crm/interactions/stats": (10, 60, 60),  # 10/60 per minute
        "/api/crm/shared-memory/team-overview": (10, 40, 60),  # 10/40 per minute
        "/api/crm/shared-memory/upcoming-renewals": (10, 40, 60),  # 10/40 per minute
        # Moderate limits for read operations
        "/api/agents/journey/": (20, 120, 60),  # 20/120 per minute
        "/api/agents/compliance/": (20, 120, 60),  # 20/120 per minute
        "/api/agents/": (30, 200, 60),  # 30/200 per minute
        # Chat endpoints - different limits for anonymous vs authenticated
        "/bali-zero/chat": (10, 60, 60),  # 10/60 per minute
        "/api/v1/agentic-rag/stream": (10, 60, 60),  # 10/60 per minute
        "/api/v1/agentic-rag/query": (10, 60, 60),  # 10/60 per minute
        # Search endpoints
        "/search": (20, 120, 60),  # 20/120 per minute
        "/api/": (40, 200, 60),  # 40/200 per minute
        # Reranker-specific endpoints
        "/rerank": (30, 200, 60),  # 30/200 per minute
        # Default for all other endpoints
        "*": (60, 300, 60),  # 60/300 per minute
    }

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/docs", "/openapi.json", "/metrics"]:
            return await call_next(request)

        # Extract user identifier from JWT, cookie, or IP
        user_id, is_authenticated = extract_user_from_request(request)

        # Find matching rate limit (different for authenticated vs anonymous)
        limit, window = self._get_rate_limit(request.url.path, is_authenticated)

        # Check rate limit - use user-specific key
        rate_limit_key = f"ratelimit:{user_id}:{request.url.path}"
        allowed, info = rate_limiter.is_allowed(rate_limit_key, limit, window)

        if not allowed:
            logger.warning(f"⚠️ Rate limit exceeded: {user_id} on {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {limit} per {window}s",
                    "limit": info["limit"],
                    "remaining": info["remaining"],
                    "reset": info["reset"],
                },
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": str(window),
                },
            )

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])

        return response

    def _get_rate_limit(self, path: str, is_authenticated: bool = False) -> tuple[int, int]:
        """
        Find matching rate limit for path.

        Args:
            path: Request path
            is_authenticated: Whether user is authenticated (gets higher limits)

        Returns:
            (limit, window) tuple
        """
        config = None

        # Try exact match first
        if path in self.RATE_LIMITS:
            config = self.RATE_LIMITS[path]
        else:
            # Try prefix match
            for pattern, limit_config in self.RATE_LIMITS.items():
                if pattern != "*" and path.startswith(pattern):
                    config = limit_config
                    break

        # Default rate limit
        if config is None:
            config = self.RATE_LIMITS["*"]

        # Extract limit based on authentication status
        # Format: (anon_limit, auth_limit, window) or (limit, window)
        if len(config) == 3:
            anon_limit, auth_limit, window = config
            limit = auth_limit if is_authenticated else anon_limit
        else:
            limit, window = config

        return limit, window


def get_rate_limit_stats() -> dict:
    """Get rate limiting statistics"""
    return {
        "backend": "redis" if rate_limiter.redis_available else "memory",
        "connected": rate_limiter.redis_available,
        "rate_limits_configured": len(RateLimitMiddleware.RATE_LIMITS),
    }
