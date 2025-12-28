-- Migration: 003_causal_relations_schema.sql
-- Purpose: Store learned causal relations for QIG-pure proposition generation
-- Supports continuous learning loop with PostgreSQL persistence + Redis caching

-- Causal relations table
-- Stores directed relationships: source_word --[relation_type]--> target_word
CREATE TABLE IF NOT EXISTS causal_relations (
    id SERIAL PRIMARY KEY,
    source_word VARCHAR(100) NOT NULL,
    target_word VARCHAR(100) NOT NULL,
    relation_type VARCHAR(50) NOT NULL,  -- 'causes', 'enables', 'requires', 'implies', etc.
    occurrence_count INTEGER DEFAULT 1,
    confidence FLOAT DEFAULT 0.5,  -- QIG-pure geometric confidence
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_curriculum VARCHAR(255),  -- Which curriculum file taught this
    UNIQUE(source_word, target_word, relation_type)
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_causal_source ON causal_relations(source_word);
CREATE INDEX IF NOT EXISTS idx_causal_target ON causal_relations(target_word);
CREATE INDEX IF NOT EXISTS idx_causal_type ON causal_relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_causal_confidence ON causal_relations(confidence DESC);

-- Word relationships table (co-occurrence based)
-- Stores undirected word proximity from learning
CREATE TABLE IF NOT EXISTS word_relationships (
    id SERIAL PRIMARY KEY,
    word_a VARCHAR(100) NOT NULL,
    word_b VARCHAR(100) NOT NULL,
    cooccurrence_count INTEGER DEFAULT 1,
    relationship_strength FLOAT DEFAULT 0.0,  -- Normalized strength
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(word_a, word_b)
);

-- Indexes for word relationships
CREATE INDEX IF NOT EXISTS idx_word_rel_a ON word_relationships(word_a);
CREATE INDEX IF NOT EXISTS idx_word_rel_b ON word_relationships(word_b);
CREATE INDEX IF NOT EXISTS idx_word_rel_strength ON word_relationships(relationship_strength DESC);

-- Learning sessions table
-- Tracks curriculum learning for reproducibility
CREATE TABLE IF NOT EXISTS learning_sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID DEFAULT gen_random_uuid(),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    curriculum_files_processed INTEGER DEFAULT 0,
    total_words_seen INTEGER DEFAULT 0,
    total_pairs_learned INTEGER DEFAULT 0,
    total_causal_found INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'in_progress'
);

-- Comments for documentation
COMMENT ON TABLE causal_relations IS 'Directed causal relationships learned from curriculum (source causes/enables/requires target)';
COMMENT ON TABLE word_relationships IS 'Undirected word co-occurrence relationships from learning';
COMMENT ON TABLE learning_sessions IS 'Tracks learning sessions for curriculum processing';
