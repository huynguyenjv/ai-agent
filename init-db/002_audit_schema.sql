-- =============================================================================
-- Audit Log Schema (Phase 10.5)
-- =============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    event_type      TEXT NOT NULL,           -- tool_execution | auth | security_violation | admin
    action          TEXT NOT NULL,
    actor           TEXT,                     -- api-key id or client IP
    outcome         TEXT,                     -- ok | blocked | error
    detail          JSONB DEFAULT '{}',
    correlation_id  TEXT,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log (event_type);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log (actor);
CREATE INDEX IF NOT EXISTS idx_audit_outcome ON audit_log (outcome) WHERE outcome <> 'ok';

-- Retention: delete entries older than 90 days. Run from a scheduled job:
--   DELETE FROM audit_log WHERE timestamp < NOW() - INTERVAL '90 days';
