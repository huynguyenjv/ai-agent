-- =============================================================================
-- Metrics Counter Schema with Monthly Partitioning
-- =============================================================================

-- Create partitioned table
CREATE TABLE IF NOT EXISTS request_metrics (
    id              BIGSERIAL,
    request_id      TEXT NOT NULL,
    correlation_id  TEXT,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model           TEXT,
    intent          TEXT,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    time_to_first_token_ms  DOUBLE PRECISION DEFAULT 0,
    total_time_ms   DOUBLE PRECISION DEFAULT 0,
    tokens_per_second DOUBLE PRECISION DEFAULT 0,
    tool_calls_count INTEGER DEFAULT 0,
    tool_names      JSONB DEFAULT '[]',
    success         BOOLEAN DEFAULT TRUE,
    error_message   TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create indexes on parent table (inherited by partitions)
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON request_metrics (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_model ON request_metrics (model);
CREATE INDEX IF NOT EXISTS idx_metrics_intent ON request_metrics (intent);
CREATE INDEX IF NOT EXISTS idx_metrics_request_id ON request_metrics (request_id);
CREATE INDEX IF NOT EXISTS idx_metrics_success ON request_metrics (success) WHERE NOT success;

-- Function to auto-create monthly partitions
CREATE OR REPLACE FUNCTION create_metrics_partition()
RETURNS TRIGGER AS $$
DECLARE
    partition_name TEXT;
    partition_start DATE;
    partition_end DATE;
BEGIN
    partition_start := DATE_TRUNC('month', NEW.timestamp);
    partition_end := partition_start + INTERVAL '1 month';
    partition_name := 'request_metrics_' || TO_CHAR(partition_start, 'YYYY_MM');

    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = partition_name AND n.nspname = 'public'
    ) THEN
        EXECUTE FORMAT(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF request_metrics
             FOR VALUES FROM (%L) TO (%L)',
            partition_name, partition_start, partition_end
        );
        RAISE NOTICE 'Created partition: %', partition_name;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-create partitions before insert
DROP TRIGGER IF EXISTS trg_create_metrics_partition ON request_metrics;
CREATE TRIGGER trg_create_metrics_partition
    BEFORE INSERT ON request_metrics
    FOR EACH ROW
    EXECUTE FUNCTION create_metrics_partition();

-- Create initial partitions for current and next 2 months
DO $$
DECLARE
    i INTEGER;
    partition_name TEXT;
    partition_start DATE;
    partition_end DATE;
BEGIN
    FOR i IN 0..2 LOOP
        partition_start := DATE_TRUNC('month', CURRENT_DATE + (i || ' month')::INTERVAL);
        partition_end := partition_start + INTERVAL '1 month';
        partition_name := 'request_metrics_' || TO_CHAR(partition_start, 'YYYY_MM');

        IF NOT EXISTS (
            SELECT 1 FROM pg_class WHERE relname = partition_name
        ) THEN
            EXECUTE FORMAT(
                'CREATE TABLE IF NOT EXISTS %I PARTITION OF request_metrics
                 FOR VALUES FROM (%L) TO (%L)',
                partition_name, partition_start, partition_end
            );
            RAISE NOTICE 'Created initial partition: %', partition_name;
        END IF;
    END LOOP;
END $$;

-- View for aggregated daily stats
CREATE OR REPLACE VIEW daily_metrics_summary AS
SELECT
    DATE_TRUNC('day', timestamp) AS day,
    model,
    COUNT(*) AS total_requests,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) AS successful_requests,
    SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS failed_requests,
    SUM(input_tokens) AS total_input_tokens,
    SUM(output_tokens) AS total_output_tokens,
    AVG(total_time_ms) AS avg_latency_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_time_ms) AS p50_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time_ms) AS p95_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_time_ms) AS p99_latency_ms,
    AVG(tokens_per_second) AS avg_tokens_per_second
FROM request_metrics
GROUP BY DATE_TRUNC('day', timestamp), model;

-- Function to drop old partitions (retention policy)
CREATE OR REPLACE FUNCTION drop_old_partitions(retention_months INTEGER DEFAULT 6)
RETURNS INTEGER AS $$
DECLARE
    partition_record RECORD;
    dropped_count INTEGER := 0;
    cutoff_date DATE;
BEGIN
    cutoff_date := DATE_TRUNC('month', CURRENT_DATE - (retention_months || ' months')::INTERVAL);

    FOR partition_record IN
        SELECT c.relname AS partition_name
        FROM pg_class c
        JOIN pg_inherits i ON c.oid = i.inhrelid
        JOIN pg_class p ON i.inhparent = p.oid
        WHERE p.relname = 'request_metrics'
          AND c.relname ~ '^request_metrics_\d{4}_\d{2}$'
    LOOP
        IF TO_DATE(RIGHT(partition_record.partition_name, 7), 'YYYY_MM') < cutoff_date THEN
            EXECUTE FORMAT('DROP TABLE IF EXISTS %I', partition_record.partition_name);
            dropped_count := dropped_count + 1;
            RAISE NOTICE 'Dropped partition: %', partition_record.partition_name;
        END IF;
    END LOOP;

    RETURN dropped_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION drop_old_partitions IS 'Drop partitions older than retention_months. Default 6 months.';
