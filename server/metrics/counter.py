"""Metrics counter service with PostgreSQL/SQLite storage."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import statistics
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Any

from server.metrics.models import RequestMetrics, AggregatedMetrics

logger = logging.getLogger("server.metrics")


class MetricsStorage(ABC):
    """Abstract base class for metrics storage backends."""

    @abstractmethod
    def record(self, metrics: RequestMetrics) -> None:
        """Record a request's metrics."""
        pass

    @abstractmethod
    def get_recent(self, limit: int = 100, model: str | None = None) -> list[dict]:
        """Get recent request metrics."""
        pass

    @abstractmethod
    def get_rows_for_aggregation(
        self, model: str | None, period_start: datetime
    ) -> list[dict]:
        """Get rows for aggregation."""
        pass

    @abstractmethod
    def get_distinct_models(self, period_start: datetime) -> list[str]:
        """Get distinct models since period_start."""
        pass

    @abstractmethod
    def clear_old(self, days: int = 30) -> int:
        """Clear metrics older than specified days."""
        pass


class SQLiteStorage(MetricsStorage):
    """SQLite storage backend."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    correlation_id TEXT,
                    timestamp TEXT NOT NULL,
                    model TEXT,
                    intent TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    time_to_first_token_ms REAL DEFAULT 0,
                    total_time_ms REAL DEFAULT 0,
                    tokens_per_second REAL DEFAULT 0,
                    tool_calls_count INTEGER DEFAULT 0,
                    tool_names TEXT,
                    success INTEGER DEFAULT 1,
                    error_message TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp "
                "ON request_metrics(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_model "
                "ON request_metrics(model)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_intent "
                "ON request_metrics(intent)"
            )
            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def record(self, metrics: RequestMetrics) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO request_metrics (
                        request_id, correlation_id, timestamp, model, intent,
                        input_tokens, output_tokens, total_tokens,
                        time_to_first_token_ms, total_time_ms, tokens_per_second,
                        tool_calls_count, tool_names, success, error_message, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metrics.request_id,
                        metrics.correlation_id,
                        metrics.timestamp.isoformat(),
                        metrics.model,
                        metrics.intent,
                        metrics.input_tokens,
                        metrics.output_tokens,
                        metrics.total_tokens,
                        metrics.time_to_first_token_ms,
                        metrics.total_time_ms,
                        metrics.tokens_per_second,
                        metrics.tool_calls_count,
                        json.dumps(metrics.tool_names),
                        1 if metrics.success else 0,
                        metrics.error_message,
                        json.dumps(metrics.metadata),
                    ),
                )
                conn.commit()

    def get_recent(self, limit: int = 100, model: str | None = None) -> list[dict]:
        with self._get_connection() as conn:
            if model:
                cursor = conn.execute(
                    "SELECT * FROM request_metrics WHERE model = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (model, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM request_metrics ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_rows_for_aggregation(
        self, model: str | None, period_start: datetime
    ) -> list[dict]:
        with self._get_connection() as conn:
            if model:
                cursor = conn.execute(
                    "SELECT * FROM request_metrics WHERE model = ? AND timestamp >= ? "
                    "ORDER BY timestamp",
                    (model, period_start.isoformat()),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM request_metrics WHERE timestamp >= ? "
                    "ORDER BY timestamp",
                    (period_start.isoformat(),),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_distinct_models(self, period_start: datetime) -> list[str]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT model FROM request_metrics WHERE timestamp >= ?",
                (period_start.isoformat(),),
            )
            return [row["model"] for row in cursor.fetchall()]

    def clear_old(self, days: int = 30) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM request_metrics WHERE timestamp < ?",
                    (cutoff.isoformat(),),
                )
                deleted = cursor.rowcount
                conn.commit()
        return deleted

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        return {
            "request_id": row["request_id"],
            "correlation_id": row["correlation_id"],
            "timestamp": row["timestamp"],
            "model": row["model"],
            "intent": row["intent"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["total_tokens"],
            "time_to_first_token_ms": row["time_to_first_token_ms"],
            "total_time_ms": row["total_time_ms"],
            "tokens_per_second": row["tokens_per_second"],
            "tool_calls_count": row["tool_calls_count"],
            "tool_names": json.loads(row["tool_names"] or "[]"),
            "success": bool(row["success"]),
            "error_message": row["error_message"],
        }


class PostgresStorage(MetricsStorage):
    """PostgreSQL storage backend with connection pooling."""

    def __init__(self, database_url: str):
        self._database_url = database_url
        self._pool: Any = None
        self._init_pool()

    def _init_pool(self) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(
            self._database_url,
            min_size=2,
            max_size=10,
            timeout=30,
        )
        logger.info("PostgreSQL connection pool initialized")

    def record(self, metrics: RequestMetrics) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO request_metrics (
                        request_id, correlation_id, timestamp, model, intent,
                        input_tokens, output_tokens, total_tokens,
                        time_to_first_token_ms, total_time_ms, tokens_per_second,
                        tool_calls_count, tool_names, success, error_message, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        metrics.request_id,
                        metrics.correlation_id,
                        metrics.timestamp,
                        metrics.model,
                        metrics.intent,
                        metrics.input_tokens,
                        metrics.output_tokens,
                        metrics.total_tokens,
                        metrics.time_to_first_token_ms,
                        metrics.total_time_ms,
                        metrics.tokens_per_second,
                        metrics.tool_calls_count,
                        json.dumps(metrics.tool_names),
                        metrics.success,
                        metrics.error_message,
                        json.dumps(metrics.metadata),
                    ),
                )
            conn.commit()

    def get_recent(self, limit: int = 100, model: str | None = None) -> list[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if model:
                    cur.execute(
                        "SELECT * FROM request_metrics WHERE model = %s "
                        "ORDER BY timestamp DESC LIMIT %s",
                        (model, limit),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM request_metrics ORDER BY timestamp DESC LIMIT %s",
                        (limit,),
                    )
                columns = [desc[0] for desc in cur.description]
                return [self._row_to_dict(dict(zip(columns, row))) for row in cur.fetchall()]

    def get_rows_for_aggregation(
        self, model: str | None, period_start: datetime
    ) -> list[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if model:
                    cur.execute(
                        "SELECT * FROM request_metrics WHERE model = %s AND timestamp >= %s "
                        "ORDER BY timestamp",
                        (model, period_start),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM request_metrics WHERE timestamp >= %s ORDER BY timestamp",
                        (period_start,),
                    )
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_distinct_models(self, period_start: datetime) -> list[str]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT model FROM request_metrics WHERE timestamp >= %s",
                    (period_start,),
                )
                return [row[0] for row in cur.fetchall()]

    def clear_old(self, days: int = 30) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM request_metrics WHERE timestamp < %s", (cutoff,)
                )
                deleted = cur.rowcount
            conn.commit()
        return deleted

    def _row_to_dict(self, row: dict) -> dict:
        tool_names = row.get("tool_names", [])
        if isinstance(tool_names, str):
            tool_names = json.loads(tool_names)

        timestamp = row.get("timestamp")
        if hasattr(timestamp, "isoformat"):
            timestamp = timestamp.isoformat()

        return {
            "request_id": row["request_id"],
            "correlation_id": row["correlation_id"],
            "timestamp": timestamp,
            "model": row["model"],
            "intent": row["intent"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["total_tokens"],
            "time_to_first_token_ms": row["time_to_first_token_ms"],
            "total_time_ms": row["total_time_ms"],
            "tokens_per_second": row["tokens_per_second"],
            "tool_calls_count": row["tool_calls_count"],
            "tool_names": tool_names,
            "success": row["success"],
            "error_message": row["error_message"],
        }


class MetricsCounter:
    """Thread-safe metrics counter with pluggable storage."""

    def __init__(self, db_path: str | None = None, database_url: str | None = None):
        """Initialize the metrics counter.

        Args:
            db_path: Path to SQLite database (fallback if DATABASE_URL not set)
            database_url: PostgreSQL connection URL (takes precedence)
        """
        database_url = database_url or os.getenv("DATABASE_URL")

        if database_url:
            self._storage = PostgresStorage(database_url)
            logger.info("MetricsCounter initialized with PostgreSQL")
        else:
            if db_path is None:
                data_dir = Path(__file__).parent.parent.parent / "data"
                data_dir.mkdir(exist_ok=True)
                db_path = str(data_dir / "metrics.db")
            self._storage = SQLiteStorage(db_path)
            logger.info("MetricsCounter initialized with SQLite: %s", db_path)

    def record(self, metrics: RequestMetrics) -> None:
        """Record a request's metrics."""
        metrics.calculate_derived_metrics()
        self._storage.record(metrics)
        logger.debug(
            "Recorded metrics: req=%s model=%s tokens=%d/%d time=%.0fms",
            metrics.request_id[:8],
            metrics.model,
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.total_time_ms,
        )

    def get_recent(self, limit: int = 100, model: str | None = None) -> list[dict]:
        """Get recent request metrics."""
        return self._storage.get_recent(limit, model)

    def get_aggregated(
        self,
        model: str | None = None,
        hours: int = 24,
    ) -> AggregatedMetrics | dict[str, AggregatedMetrics]:
        """Get aggregated metrics for the specified period."""
        period_end = datetime.now()
        period_start = period_end - timedelta(hours=hours)

        if model:
            rows = self._storage.get_rows_for_aggregation(model, period_start)
            return self._aggregate_rows(rows, model, period_start, period_end)
        else:
            models = self._storage.get_distinct_models(period_start)
            result = {}
            for m in models:
                rows = self._storage.get_rows_for_aggregation(m, period_start)
                result[m] = self._aggregate_rows(rows, m, period_start, period_end)
            return result

    def compare_models(self, hours: int = 24) -> dict:
        """Compare metrics across all models."""
        aggregated = self.get_aggregated(hours=hours)
        if isinstance(aggregated, AggregatedMetrics):
            return {aggregated.model: aggregated.to_dict()}
        return {model: metrics.to_dict() for model, metrics in aggregated.items()}

    def _aggregate_rows(
        self,
        rows: list[dict],
        model: str,
        period_start: datetime,
        period_end: datetime,
    ) -> AggregatedMetrics:
        """Aggregate metrics from database rows."""
        agg = AggregatedMetrics(
            model=model,
            period_start=period_start,
            period_end=period_end,
        )

        if not rows:
            return agg

        total_times = []
        ttfts = []
        tps_values = []
        intent_counts: dict[str, int] = {}

        for row in rows:
            agg.total_requests += 1
            if row["success"]:
                agg.successful_requests += 1
            else:
                agg.failed_requests += 1

            agg.total_input_tokens += row["input_tokens"] or 0
            agg.total_output_tokens += row["output_tokens"] or 0
            agg.total_tool_calls += row["tool_calls_count"] or 0

            if row["total_time_ms"]:
                total_times.append(row["total_time_ms"])
            if row["time_to_first_token_ms"]:
                ttfts.append(row["time_to_first_token_ms"])
            if row["tokens_per_second"]:
                tps_values.append(row["tokens_per_second"])

            intent = row["intent"] or "unknown"
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        n = agg.total_requests
        if n > 0:
            agg.avg_input_tokens = agg.total_input_tokens / n
            agg.avg_output_tokens = agg.total_output_tokens / n
            agg.avg_tool_calls_per_request = agg.total_tool_calls / n

        if total_times:
            agg.avg_total_time_ms = statistics.mean(total_times)
            sorted_times = sorted(total_times)
            agg.p50_total_time_ms = self._percentile(sorted_times, 50)
            agg.p95_total_time_ms = self._percentile(sorted_times, 95)
            agg.p99_total_time_ms = self._percentile(sorted_times, 99)

        if ttfts:
            agg.avg_ttft_ms = statistics.mean(ttfts)

        if tps_values:
            agg.avg_tokens_per_second = statistics.mean(tps_values)

        agg.intent_counts = intent_counts
        return agg

    def _percentile(self, sorted_data: list[float], p: int) -> float:
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)

    def clear_old(self, days: int = 30) -> int:
        """Clear metrics older than specified days. Returns count deleted."""
        deleted = self._storage.clear_old(days)
        logger.info("Cleared %d metrics older than %d days", deleted, days)
        return deleted


# =============================================================================
# Singleton
# =============================================================================

_metrics_counter: MetricsCounter | None = None
_counter_lock = threading.Lock()


def get_metrics_counter() -> MetricsCounter:
    """Get the singleton MetricsCounter instance."""
    global _metrics_counter

    if _metrics_counter is None:
        with _counter_lock:
            if _metrics_counter is None:
                _metrics_counter = MetricsCounter()

    return _metrics_counter


# =============================================================================
# Context Manager for Request Timing
# =============================================================================


class RequestTimer:
    """Context manager for timing requests and recording metrics."""

    def __init__(
        self,
        request_id: str,
        model: str = "",
        correlation_id: str = "",
        intent: str = "",
    ):
        self.metrics = RequestMetrics(
            request_id=request_id,
            correlation_id=correlation_id,
            model=model,
            intent=intent,
        )
        self._start_time: float = time.perf_counter()
        self._first_token_time: float | None = None

    def __enter__(self) -> "RequestTimer":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.metrics.total_time_ms = (time.perf_counter() - self._start_time) * 1000

        if exc_type is not None:
            self.metrics.success = False
            self.metrics.error_message = str(exc_val)

        get_metrics_counter().record(self.metrics)

    def mark_first_token(self) -> None:
        """Mark when the first token was received."""
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()
            self.metrics.time_to_first_token_ms = (
                self._first_token_time - self._start_time
            ) * 1000

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Set token counts."""
        self.metrics.input_tokens = input_tokens
        self.metrics.output_tokens = output_tokens

    def set_tool_calls(self, tool_calls: list[dict]) -> None:
        """Set tool call information."""
        self.metrics.tool_calls_count = len(tool_calls)
        self.metrics.tool_names = [
            tc.get("function", {}).get("name", "") for tc in tool_calls
        ]

    def set_intent(self, intent: str) -> None:
        """Set the detected intent."""
        self.metrics.intent = intent

    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds since timer started."""
        return (time.perf_counter() - self._start_time) * 1000
