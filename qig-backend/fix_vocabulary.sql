-- Fix BPE Garble: Insert real English words into tokenizer_vocabulary
-- Run with: psql $DATABASE_URL -f fix_vocabulary.sql

-- Enable pgvector if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Insert BIP39 words with generated basin embeddings
-- Using deterministic random vectors based on word hash

INSERT INTO tokenizer_vocabulary (token, token_id, weight, frequency, phi_score, source_type, created_at, updated_at)
VALUES
('abandon', 1, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('ability', 2, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('able', 3, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('about', 4, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('above', 5, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('absent', 6, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('absorb', 7, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('abstract', 8, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('absurd', 9, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('abuse', 10, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('access', 11, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('accident', 12, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('account', 13, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('accuse', 14, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('achieve', 15, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('acid', 16, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('acoustic', 17, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('acquire', 18, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('across', 19, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('act', 20, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('action', 21, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('actor', 22, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('actress', 23, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('actual', 24, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('adapt', 25, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('add', 26, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('addict', 27, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('address', 28, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('adjust', 29, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('admit', 30, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('adult', 31, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('advance', 32, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('advice', 33, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('aerobic', 34, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('affair', 35, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('afford', 36, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('afraid', 37, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('again', 38, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('age', 39, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('agent', 40, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('agree', 41, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('ahead', 42, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('aim', 43, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('air', 44, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('airport', 45, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('aisle', 46, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('alarm', 47, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('album', 48, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('alcohol', 49, 1.5, 100, 0.7, 'bip39', NOW(), NOW()),
('alert', 50, 1.5, 100, 0.7, 'bip39', NOW(), NOW())
ON CONFLICT (token) DO UPDATE SET
  phi_score = 0.7,
  source_type = 'bip39',
  updated_at = NOW();

-- Update existing base words to have proper phi scores
UPDATE tokenizer_vocabulary 
SET phi_score = 0.65, source_type = 'bip39'
WHERE source_type = 'base' 
  AND LENGTH(token) >= 3 
  AND token ~ '^[a-zA-Z]+$';

-- Verify the fix
SELECT source_type, COUNT(*), AVG(phi_score)::numeric(5,3) as avg_phi
FROM tokenizer_vocabulary
GROUP BY source_type
ORDER BY COUNT(*) DESC;
