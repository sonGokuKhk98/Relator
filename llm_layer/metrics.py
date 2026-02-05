"""
Metrics and Observability

Track extraction performance, strategy usage, and quality metrics
for monitoring and improvement.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import Counter, deque
from datetime import datetime
import time
import json


@dataclass
class ExtractionMetrics:
    """Aggregated extraction metrics."""
    total_queries: int = 0
    strategy_counts: Counter = field(default_factory=Counter)
    confidence_sum: float = 0.0
    latency_sum_ms: float = 0.0
    validation_failures: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Per-entity metrics
    entity_counts: Counter = field(default_factory=Counter)
    column_usage: Counter = field(default_factory=Counter)

    @property
    def avg_confidence(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.confidence_sum / self.total_queries

    @property
    def avg_latency_ms(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.latency_sum_ms / self.total_queries

    @property
    def validation_failure_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.validation_failures / self.total_queries

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


@dataclass
class QueryLogEntry:
    """Single query log entry."""
    timestamp: str
    query: str
    strategy: str
    confidence: float
    latency_ms: float
    filter_count: int
    validation_valid: bool
    entities: List[str] = field(default_factory=list)
    interpretation: Optional[str] = None


class MetricsCollector:
    """
    Collect and aggregate extraction metrics.

    Features:
    - Real-time metric tracking
    - Query logging with rotation
    - Performance analytics
    - Strategy effectiveness analysis
    """

    def __init__(self, max_log_size: int = 10000):
        self.metrics = ExtractionMetrics()
        self.query_log: deque = deque(maxlen=max_log_size)
        self.hourly_stats: Dict[str, Dict] = {}
        self.start_time = time.time()

    def record_extraction(
        self,
        query: str,
        result: Any,  # ExtractionResult
        latency_ms: float,
    ) -> None:
        """Record an extraction event."""
        self.metrics.total_queries += 1
        self.metrics.strategy_counts[result.strategy_used.value] += 1
        self.metrics.confidence_sum += result.confidence
        self.metrics.latency_sum_ms += latency_ms

        # Track validation
        if result.validation_result and not result.validation_result.valid:
            self.metrics.validation_failures += 1

        # Track entities
        for entity in result.entities_resolved:
            self.metrics.entity_counts[entity.entity_type.value] += 1

        # Track column usage
        for node in result.ast.filters.flatten():
            self.metrics.column_usage[f"{node.table}.{node.column}"] += 1

        # Log entry
        entry = QueryLogEntry(
            timestamp=datetime.now().isoformat(),
            query=query,
            strategy=result.strategy_used.value,
            confidence=result.confidence,
            latency_ms=latency_ms,
            filter_count=len(result.ast.filters.flatten()),
            validation_valid=result.validation_result.valid if result.validation_result else True,
            entities=[e.canonical_value for e in result.entities_resolved],
            interpretation=result.llm_interpretation,
        )
        self.query_log.append(entry)

        # Update hourly stats
        self._update_hourly_stats(entry)

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.metrics.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.metrics.cache_misses += 1

    def _update_hourly_stats(self, entry: QueryLogEntry) -> None:
        """Update hourly aggregation."""
        hour_key = entry.timestamp[:13]  # YYYY-MM-DDTHH

        if hour_key not in self.hourly_stats:
            self.hourly_stats[hour_key] = {
                "count": 0,
                "confidence_sum": 0.0,
                "latency_sum": 0.0,
                "strategies": Counter(),
            }

        stats = self.hourly_stats[hour_key]
        stats["count"] += 1
        stats["confidence_sum"] += entry.confidence
        stats["latency_sum"] += entry.latency_ms
        stats["strategies"][entry.strategy] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        uptime_hours = (time.time() - self.start_time) / 3600

        return {
            "uptime_hours": round(uptime_hours, 2),
            "total_queries": self.metrics.total_queries,
            "queries_per_hour": round(self.metrics.total_queries / max(uptime_hours, 0.01), 1),
            "avg_confidence": round(self.metrics.avg_confidence, 3),
            "avg_latency_ms": round(self.metrics.avg_latency_ms, 1),
            "validation_failure_rate": round(self.metrics.validation_failure_rate, 3),
            "cache_hit_rate": round(self.metrics.cache_hit_rate, 3),
            "strategy_distribution": dict(self.metrics.strategy_counts),
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        recent_queries = list(self.query_log)[-20:]

        return {
            "summary": self.get_summary(),
            "recent_queries": [
                {
                    "timestamp": q.timestamp,
                    "query": q.query[:100],  # Truncate for display
                    "strategy": q.strategy,
                    "confidence": q.confidence,
                    "latency_ms": round(q.latency_ms, 1),
                    "filter_count": q.filter_count,
                    "valid": q.validation_valid,
                }
                for q in recent_queries
            ],
            "top_entities": dict(self.metrics.entity_counts.most_common(10)),
            "top_columns": dict(self.metrics.column_usage.most_common(10)),
            "hourly_trend": self._get_hourly_trend(),
        }

    def _get_hourly_trend(self, hours: int = 24) -> List[Dict]:
        """Get hourly trend data."""
        sorted_hours = sorted(self.hourly_stats.keys())[-hours:]

        return [
            {
                "hour": hour,
                "count": self.hourly_stats[hour]["count"],
                "avg_confidence": round(
                    self.hourly_stats[hour]["confidence_sum"] /
                    max(self.hourly_stats[hour]["count"], 1),
                    3
                ),
                "avg_latency": round(
                    self.hourly_stats[hour]["latency_sum"] /
                    max(self.hourly_stats[hour]["count"], 1),
                    1
                ),
            }
            for hour in sorted_hours
        ]

    def get_strategy_analysis(self) -> Dict[str, Any]:
        """Analyze strategy effectiveness."""
        strategy_metrics = {}

        for entry in self.query_log:
            strategy = entry.strategy
            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    "count": 0,
                    "confidence_sum": 0.0,
                    "latency_sum": 0.0,
                    "validation_failures": 0,
                }

            stats = strategy_metrics[strategy]
            stats["count"] += 1
            stats["confidence_sum"] += entry.confidence
            stats["latency_sum"] += entry.latency_ms
            if not entry.validation_valid:
                stats["validation_failures"] += 1

        # Calculate averages
        analysis = {}
        for strategy, stats in strategy_metrics.items():
            count = stats["count"]
            analysis[strategy] = {
                "count": count,
                "avg_confidence": round(stats["confidence_sum"] / max(count, 1), 3),
                "avg_latency_ms": round(stats["latency_sum"] / max(count, 1), 1),
                "validation_failure_rate": round(stats["validation_failures"] / max(count, 1), 3),
            }

        return analysis

    def get_query_patterns(self, limit: int = 20) -> List[Dict]:
        """Analyze common query patterns."""
        # Simple word frequency analysis
        word_counts: Counter = Counter()

        for entry in self.query_log:
            words = entry.query.lower().split()
            word_counts.update(words)

        # Filter out common words
        stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'to', 'of'}
        filtered = [(word, count) for word, count in word_counts.most_common(limit * 2)
                   if word not in stop_words]

        return [{"word": word, "count": count} for word, count in filtered[:limit]]

    def export_logs(self, format: str = "json") -> str:
        """Export query logs."""
        if format == "json":
            return json.dumps([
                {
                    "timestamp": e.timestamp,
                    "query": e.query,
                    "strategy": e.strategy,
                    "confidence": e.confidence,
                    "latency_ms": e.latency_ms,
                    "filter_count": e.filter_count,
                    "validation_valid": e.validation_valid,
                    "entities": e.entities,
                    "interpretation": e.interpretation,
                }
                for e in self.query_log
            ], indent=2)
        elif format == "csv":
            lines = ["timestamp,query,strategy,confidence,latency_ms,filter_count,valid"]
            for e in self.query_log:
                query_escaped = e.query.replace('"', '""')
                lines.append(
                    f'{e.timestamp},"{query_escaped}",{e.strategy},{e.confidence},'
                    f'{e.latency_ms},{e.filter_count},{e.validation_valid}'
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = ExtractionMetrics()
        self.query_log.clear()
        self.hourly_stats.clear()
        self.start_time = time.time()


# Global metrics instance
_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def record_extraction(query: str, result: Any, latency_ms: float) -> None:
    """Convenience function to record extraction."""
    get_metrics().record_extraction(query, result, latency_ms)
