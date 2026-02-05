"""
Extraction Cache Layer

Caches LLM extraction results for common queries to reduce
latency and API costs.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from collections import OrderedDict
import hashlib
import re
import time
import json


@dataclass
class CacheEntry:
    """A single cache entry."""
    result: Any
    timestamp: float
    hits: int = 0


class ExtractionCache:
    """
    LRU cache for extraction results.

    Features:
    - Query normalization for better cache hits
    - TTL-based expiration
    - Hit tracking for analytics
    - Memory-bounded storage
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,  # 1 hour default
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for better cache matching.

        - Lowercase
        - Collapse whitespace
        - Remove punctuation variations
        """
        normalized = query.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s$@]', '', normalized)
        return normalized

    def _hash_query(self, query: str) -> str:
        """Generate hash key for query."""
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        """
        Get cached result for query.

        Returns None if not found or expired.
        """
        key = self._hash_query(query)

        if key not in self.cache:
            self.stats["misses"] += 1
            return None

        entry = self.cache[key]

        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self.cache[key]
            self.stats["misses"] += 1
            return None

        # Update access order for LRU
        self.cache.move_to_end(key)
        entry.hits += 1
        self.stats["hits"] += 1

        return entry.result

    def set(self, query: str, result: Any) -> None:
        """Store result in cache."""
        key = self._hash_query(query)

        # Evict if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1

        self.cache[key] = CacheEntry(
            result=result,
            timestamp=time.time(),
        )

    def invalidate(self, query: str) -> bool:
        """Remove specific query from cache."""
        key = self._hash_query(query)
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "hit_rate": round(hit_rate, 3),
        }

    def get_top_queries(self, limit: int = 10) -> list:
        """Get most frequently accessed queries."""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].hits,
            reverse=True,
        )
        return [
            {"key": k, "hits": v.hits, "age_seconds": time.time() - v.timestamp}
            for k, v in sorted_entries[:limit]
        ]


class SemanticCache:
    """
    Advanced cache with semantic similarity matching.

    Uses embeddings to find similar cached queries,
    enabling cache hits for paraphrased queries.
    """

    def __init__(
        self,
        max_size: int = 500,
        similarity_threshold: float = 0.9,
    ):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.entries: list = []
        self.base_cache = ExtractionCache(max_size=max_size)

    def get(self, query: str) -> Optional[Any]:
        """Get from cache, checking semantic similarity."""
        # Try exact match first
        exact = self.base_cache.get(query)
        if exact:
            return exact

        # TODO: Implement semantic similarity check
        # Would require embedding model integration
        return None

    def set(self, query: str, result: Any) -> None:
        """Store in cache."""
        self.base_cache.set(query, result)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.base_cache.get_stats()


class DistributedCache:
    """
    Redis-backed distributed cache for production deployments.

    Placeholder implementation - would integrate with Redis
    in production.
    """

    def __init__(
        self,
        redis_url: str = None,
        prefix: str = "filter_cache:",
        ttl_seconds: int = 3600,
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.redis_client = None

        # Fall back to in-memory if no Redis
        self._fallback = ExtractionCache(max_size=1000, ttl_seconds=ttl_seconds)

        if redis_url:
            self._connect()

    def _connect(self):
        """Connect to Redis."""
        try:
            import redis
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}, using in-memory fallback")
            self.redis_client = None

    def _make_key(self, query: str) -> str:
        """Generate Redis key."""
        normalized = query.lower().strip()
        hash_val = hashlib.md5(normalized.encode()).hexdigest()
        return f"{self.prefix}{hash_val}"

    def get(self, query: str) -> Optional[Any]:
        """Get from Redis cache."""
        if not self.redis_client:
            return self._fallback.get(query)

        try:
            key = self._make_key(query)
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception:
            return self._fallback.get(query)

    def set(self, query: str, result: Any) -> None:
        """Store in Redis cache."""
        if not self.redis_client:
            self._fallback.set(query, result)
            return

        try:
            key = self._make_key(query)
            # Need to serialize the result
            data = json.dumps(result.to_dict() if hasattr(result, 'to_dict') else result)
            self.redis_client.setex(key, self.ttl_seconds, data)
        except Exception:
            self._fallback.set(query, result)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.redis_client:
            return self._fallback.get_stats()

        try:
            info = self.redis_client.info("stats")
            return {
                "backend": "redis",
                "connected": True,
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception:
            return {"backend": "fallback", **self._fallback.get_stats()}
