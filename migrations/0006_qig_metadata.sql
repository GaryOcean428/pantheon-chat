-- QIG metadata key/value store
-- Stores small operational metadata (e.g., schema versions, feature flags)

CREATE TABLE IF NOT EXISTS qig_metadata (
    config_key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);
