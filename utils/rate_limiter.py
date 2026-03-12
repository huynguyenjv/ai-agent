"""
Redis-based rate limiting service.

Provides thread-safe, distributed rate limiting using Redis.
Falls back to in-memory implementation if Redis is unavailable.
"""

from __future__ import annotations

import time
from typing import Optional, Any
from collections import defaultdict
import threading

import structlog

logger = structlog.get_logger()

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore[assignment]


class RateLimiter:
    """Thread-safe rate limiter with Redis backend."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        redis_db: int = 0,
        requests_per_window: int = 10,
        window_seconds: int = 60,
    ):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.redis_client: Optional[Any] = None  # redis.Redis or None
        
        # Fallback in-memory store with thread safety
        self._memory_store: dict[str, list[float]] = defaultdict(list)
        self._memory_lock = threading.Lock()
        
        # Try to initialize Redis
        if REDIS_AVAILABLE and redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    db=redis_db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Rate limiter initialized with Redis backend", 
                           host=redis_host, port=redis_port)
            except Exception as e:
                logger.warning("Failed to connect to Redis, using in-memory fallback", 
                             error=str(e))
                self.redis_client = None
        else:
            logger.info("Redis not available, using in-memory rate limiting")

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit. Returns True if allowed."""
        if self.redis_client:
            return self._check_redis_rate_limit(client_id)
        else:
            return self._check_memory_rate_limit(client_id)

    def _check_redis_rate_limit(self, client_id: str) -> bool:
        """Redis-based rate limiting using sliding window."""
        try:
            now = time.time()
            window_start = now - self.window_seconds
            key = f"rate_limit:{client_id}"
            
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            pipe.zcard(key)
            
            # Execute pipeline
            results = pipe.execute()
            current_count = results[1]
            
            # Check limit
            if current_count >= self.requests_per_window:
                return False
            
            # Add current request
            self.redis_client.zadd(key, {str(now): now})
            
            # Set expiry (cleanup)
            self.redis_client.expire(key, self.window_seconds + 1)
            
            return True
            
        except Exception as e:
            logger.error("Redis rate limit check failed, allowing request", error=str(e))
            # Fail open - allow request if Redis is down
            return True

    def _check_memory_rate_limit(self, client_id: str) -> bool:
        """Thread-safe in-memory rate limiting."""
        now = time.time()
        window_start = now - self.window_seconds
        
        with self._memory_lock:
            # Clean old entries
            self._memory_store[client_id] = [
                ts for ts in self._memory_store[client_id] if ts > window_start
            ]
            
            # Check limit
            if len(self._memory_store[client_id]) >= self.requests_per_window:
                return False
            
            # Record this request
            self._memory_store[client_id].append(now)
            return True

    def get_remaining_requests(self, client_id: str) -> int:
        """Get number of remaining requests for client."""
        if self.redis_client:
            try:
                now = time.time()
                window_start = now - self.window_seconds
                key = f"rate_limit:{client_id}"
                
                # Count current entries
                current_count = self.redis_client.zcount(key, window_start, now)
                return max(0, self.requests_per_window - current_count)
                
            except Exception:
                return self.requests_per_window  # Fail open
        else:
            now = time.time()
            window_start = now - self.window_seconds
            
            with self._memory_lock:
                # Clean old entries
                self._memory_store[client_id] = [
                    ts for ts in self._memory_store[client_id] if ts > window_start
                ]
                current_count = len(self._memory_store[client_id])
                return max(0, self.requests_per_window - current_count)

    def reset_client_limit(self, client_id: str) -> bool:
        """Reset rate limit for a specific client (admin function)."""
        try:
            if self.redis_client:
                key = f"rate_limit:{client_id}"
                self.redis_client.delete(key)
            else:
                with self._memory_lock:
                    if client_id in self._memory_store:
                        del self._memory_store[client_id]
            return True
        except Exception as e:
            logger.error("Failed to reset client rate limit", client_id=client_id, error=str(e))
            return False