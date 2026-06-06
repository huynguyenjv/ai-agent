-- =============================================================================
-- Cross-session Agent Memory (Phase R10 / 15.4)
-- =============================================================================

CREATE TABLE IF NOT EXISTS agent_memory (
    id          BIGSERIAL PRIMARY KEY,
    scope       TEXT NOT NULL,            -- conversation_id / api-key / project
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_mem_scope ON agent_memory (scope);
CREATE INDEX IF NOT EXISTS idx_mem_created ON agent_memory (created_at DESC);
