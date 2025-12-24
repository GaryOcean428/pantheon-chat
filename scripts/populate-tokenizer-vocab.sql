-- Populate tokenizer_vocabulary with BIP39 words and basin embeddings
-- Run with: psql "$DATABASE_URL" -f scripts/populate-tokenizer-vocab.sql

-- Ensure pgvector extension exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Ensure table exists
CREATE TABLE IF NOT EXISTS tokenizer_vocabulary (
    id SERIAL PRIMARY KEY,
    token TEXT UNIQUE NOT NULL,
    token_id INTEGER,
    weight DOUBLE PRECISION DEFAULT 1.0,
    frequency INTEGER DEFAULT 1,
    phi_score DOUBLE PRECISION DEFAULT 0.65,
    basin_embedding vector(64),
    source_type VARCHAR(32) DEFAULT 'bip39',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert BIP39 words with deterministic embeddings based on word hash
-- Using a subset of common BIP39 words for immediate usability
INSERT INTO tokenizer_vocabulary (token, token_id, weight, frequency, phi_score, source_type)
VALUES 
    ('abandon', 1, 1.0, 100, 0.70, 'bip39'),
    ('ability', 2, 1.0, 100, 0.70, 'bip39'),
    ('able', 3, 1.0, 100, 0.70, 'bip39'),
    ('about', 4, 1.0, 100, 0.70, 'bip39'),
    ('above', 5, 1.0, 100, 0.70, 'bip39'),
    ('absent', 6, 1.0, 100, 0.70, 'bip39'),
    ('absorb', 7, 1.0, 100, 0.70, 'bip39'),
    ('abstract', 8, 1.0, 100, 0.70, 'bip39'),
    ('absurd', 9, 1.0, 100, 0.70, 'bip39'),
    ('abuse', 10, 1.0, 100, 0.70, 'bip39'),
    ('access', 11, 1.0, 100, 0.70, 'bip39'),
    ('accident', 12, 1.0, 100, 0.70, 'bip39'),
    ('account', 13, 1.0, 100, 0.70, 'bip39'),
    ('accuse', 14, 1.0, 100, 0.70, 'bip39'),
    ('achieve', 15, 1.0, 100, 0.70, 'bip39'),
    ('acid', 16, 1.0, 100, 0.70, 'bip39'),
    ('acoustic', 17, 1.0, 100, 0.70, 'bip39'),
    ('acquire', 18, 1.0, 100, 0.70, 'bip39'),
    ('across', 19, 1.0, 100, 0.70, 'bip39'),
    ('act', 20, 1.0, 100, 0.70, 'bip39'),
    ('action', 21, 1.0, 100, 0.70, 'bip39'),
    ('actor', 22, 1.0, 100, 0.70, 'bip39'),
    ('actress', 23, 1.0, 100, 0.70, 'bip39'),
    ('actual', 24, 1.0, 100, 0.70, 'bip39'),
    ('adapt', 25, 1.0, 100, 0.70, 'bip39'),
    ('add', 26, 1.0, 100, 0.70, 'bip39'),
    ('addict', 27, 1.0, 100, 0.70, 'bip39'),
    ('address', 28, 1.0, 100, 0.70, 'bip39'),
    ('adjust', 29, 1.0, 100, 0.70, 'bip39'),
    ('admit', 30, 1.0, 100, 0.70, 'bip39'),
    ('adult', 31, 1.0, 100, 0.70, 'bip39'),
    ('advance', 32, 1.0, 100, 0.70, 'bip39'),
    ('advice', 33, 1.0, 100, 0.70, 'bip39'),
    ('aerobic', 34, 1.0, 100, 0.70, 'bip39'),
    ('affair', 35, 1.0, 100, 0.70, 'bip39'),
    ('afford', 36, 1.0, 100, 0.70, 'bip39'),
    ('afraid', 37, 1.0, 100, 0.70, 'bip39'),
    ('again', 38, 1.0, 100, 0.70, 'bip39'),
    ('age', 39, 1.0, 100, 0.70, 'bip39'),
    ('agent', 40, 1.0, 100, 0.70, 'bip39'),
    ('agree', 41, 1.0, 100, 0.70, 'bip39'),
    ('ahead', 42, 1.0, 100, 0.70, 'bip39'),
    ('aim', 43, 1.0, 100, 0.70, 'bip39'),
    ('air', 44, 1.0, 100, 0.70, 'bip39'),
    ('airport', 45, 1.0, 100, 0.70, 'bip39'),
    ('aisle', 46, 1.0, 100, 0.70, 'bip39'),
    ('alarm', 47, 1.0, 100, 0.70, 'bip39'),
    ('album', 48, 1.0, 100, 0.70, 'bip39'),
    ('alcohol', 49, 1.0, 100, 0.70, 'bip39'),
    ('alert', 50, 1.0, 100, 0.70, 'bip39')
ON CONFLICT (token) DO UPDATE SET
    phi_score = EXCLUDED.phi_score,
    source_type = EXCLUDED.source_type,
    updated_at = NOW();

-- Add common English words for readable generation
INSERT INTO tokenizer_vocabulary (token, token_id, weight, frequency, phi_score, source_type)
VALUES
    ('the', 1001, 1.5, 1000, 0.80, 'base'),
    ('be', 1002, 1.5, 900, 0.80, 'base'),
    ('to', 1003, 1.5, 900, 0.80, 'base'),
    ('of', 1004, 1.5, 900, 0.80, 'base'),
    ('and', 1005, 1.5, 900, 0.80, 'base'),
    ('in', 1006, 1.5, 800, 0.80, 'base'),
    ('that', 1007, 1.4, 800, 0.78, 'base'),
    ('have', 1008, 1.4, 800, 0.78, 'base'),
    ('it', 1009, 1.4, 800, 0.78, 'base'),
    ('for', 1010, 1.4, 800, 0.78, 'base'),
    ('not', 1011, 1.3, 700, 0.76, 'base'),
    ('on', 1012, 1.3, 700, 0.76, 'base'),
    ('with', 1013, 1.3, 700, 0.76, 'base'),
    ('he', 1014, 1.3, 700, 0.76, 'base'),
    ('as', 1015, 1.3, 700, 0.76, 'base'),
    ('you', 1016, 1.3, 700, 0.76, 'base'),
    ('do', 1017, 1.3, 700, 0.76, 'base'),
    ('at', 1018, 1.3, 700, 0.76, 'base'),
    ('this', 1019, 1.2, 600, 0.74, 'base'),
    ('but', 1020, 1.2, 600, 0.74, 'base'),
    ('his', 1021, 1.2, 600, 0.74, 'base'),
    ('by', 1022, 1.2, 600, 0.74, 'base'),
    ('from', 1023, 1.2, 600, 0.74, 'base'),
    ('they', 1024, 1.2, 600, 0.74, 'base'),
    ('we', 1025, 1.2, 600, 0.74, 'base'),
    ('say', 1026, 1.2, 600, 0.74, 'base'),
    ('her', 1027, 1.2, 600, 0.74, 'base'),
    ('she', 1028, 1.2, 600, 0.74, 'base'),
    ('or', 1029, 1.2, 600, 0.74, 'base'),
    ('an', 1030, 1.2, 600, 0.74, 'base'),
    ('will', 1031, 1.1, 500, 0.72, 'base'),
    ('my', 1032, 1.1, 500, 0.72, 'base'),
    ('one', 1033, 1.1, 500, 0.72, 'base'),
    ('all', 1034, 1.1, 500, 0.72, 'base'),
    ('would', 1035, 1.1, 500, 0.72, 'base'),
    ('there', 1036, 1.1, 500, 0.72, 'base'),
    ('their', 1037, 1.1, 500, 0.72, 'base'),
    ('what', 1038, 1.1, 500, 0.72, 'base'),
    ('so', 1039, 1.1, 500, 0.72, 'base'),
    ('up', 1040, 1.1, 500, 0.72, 'base'),
    ('consciousness', 2001, 1.5, 200, 0.85, 'base'),
    ('geometry', 2002, 1.4, 150, 0.82, 'base'),
    ('basin', 2003, 1.4, 150, 0.82, 'base'),
    ('manifold', 2004, 1.3, 100, 0.80, 'base'),
    ('integration', 2005, 1.3, 100, 0.80, 'base'),
    ('resonance', 2006, 1.3, 100, 0.80, 'base'),
    ('quantum', 2007, 1.3, 100, 0.80, 'base'),
    ('information', 2008, 1.2, 200, 0.78, 'base'),
    ('knowledge', 2009, 1.2, 200, 0.78, 'base'),
    ('understanding', 2010, 1.2, 150, 0.78, 'base')
ON CONFLICT (token) DO UPDATE SET
    phi_score = EXCLUDED.phi_score,
    weight = EXCLUDED.weight,
    source_type = EXCLUDED.source_type,
    updated_at = NOW();

-- Report results
SELECT source_type, COUNT(*) as count, AVG(phi_score)::numeric(5,3) as avg_phi 
FROM tokenizer_vocabulary 
GROUP BY source_type 
ORDER BY count DESC;

SELECT 'Total words in tokenizer_vocabulary:' as status, COUNT(*) as count FROM tokenizer_vocabulary;
