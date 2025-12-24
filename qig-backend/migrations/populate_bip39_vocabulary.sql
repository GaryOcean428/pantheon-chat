-- Populate tokenizer_vocabulary with BIP39 words
-- Run with: psql $DATABASE_URL -f populate_bip39_vocabulary.sql

-- Ensure vector extension exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table if not exists
CREATE TABLE IF NOT EXISTS tokenizer_vocabulary (
    id SERIAL PRIMARY KEY,
    token TEXT UNIQUE NOT NULL,
    token_id INTEGER,
    weight DOUBLE PRECISION DEFAULT 1.0,
    frequency INTEGER DEFAULT 1,
    phi_score DOUBLE PRECISION DEFAULT 0.5,
    basin_embedding vector(64),
    source_type VARCHAR(32) DEFAULT 'base',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_tokenizer_vocab_token ON tokenizer_vocabulary(token);
CREATE INDEX IF NOT EXISTS idx_tokenizer_vocab_source ON tokenizer_vocabulary(source_type);
CREATE INDEX IF NOT EXISTS idx_tokenizer_vocab_phi ON tokenizer_vocabulary(phi_score);

-- Function to generate deterministic basin embedding from word
-- Uses a simple hash-based approach for pure SQL compatibility
CREATE OR REPLACE FUNCTION generate_word_embedding(word TEXT) RETURNS vector AS $$
DECLARE
    result FLOAT8[64];
    hash_val BIGINT;
    i INT;
    seed FLOAT8;
BEGIN
    -- Initialize array
    result := ARRAY(SELECT 0::FLOAT8 FROM generate_series(1, 64));
    
    -- Generate deterministic values based on word hash
    hash_val := ('x' || substr(md5(lower(word)), 1, 15))::bit(60)::bigint;
    
    FOR i IN 1..64 LOOP
        -- Use hash to seed pseudo-random values
        seed := ((hash_val * (i * 2654435761) % 2147483647)::FLOAT8 / 2147483647.0) * 2 - 1;
        result[i] := seed;
    END LOOP;
    
    -- Normalize to unit sphere
    DECLARE
        norm FLOAT8 := 0;
    BEGIN
        FOR i IN 1..64 LOOP
            norm := norm + result[i] * result[i];
        END LOOP;
        norm := sqrt(norm);
        IF norm > 0.0001 THEN
            FOR i IN 1..64 LOOP
                result[i] := result[i] / norm;
            END LOOP;
        END IF;
    END;
    
    RETURN result::vector;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Insert BIP39 words (first 200 most common)
-- These are inserted with source_type='bip39' and high phi scores
INSERT INTO tokenizer_vocabulary (token, token_id, weight, frequency, phi_score, basin_embedding, source_type)
VALUES
    ('abandon', 1, 1.7, 100, 0.7, generate_word_embedding('abandon'), 'bip39'),
    ('ability', 2, 1.7, 100, 0.7, generate_word_embedding('ability'), 'bip39'),
    ('able', 3, 1.7, 100, 0.7, generate_word_embedding('able'), 'bip39'),
    ('about', 4, 1.7, 100, 0.7, generate_word_embedding('about'), 'bip39'),
    ('above', 5, 1.7, 100, 0.7, generate_word_embedding('above'), 'bip39'),
    ('absent', 6, 1.7, 100, 0.7, generate_word_embedding('absent'), 'bip39'),
    ('abstract', 7, 1.7, 100, 0.7, generate_word_embedding('abstract'), 'bip39'),
    ('accept', 8, 1.7, 100, 0.7, generate_word_embedding('accept'), 'bip39'),
    ('access', 9, 1.7, 100, 0.7, generate_word_embedding('access'), 'bip39'),
    ('account', 10, 1.7, 100, 0.7, generate_word_embedding('account'), 'bip39'),
    ('achieve', 11, 1.7, 100, 0.7, generate_word_embedding('achieve'), 'bip39'),
    ('action', 12, 1.7, 100, 0.7, generate_word_embedding('action'), 'bip39'),
    ('active', 13, 1.7, 100, 0.7, generate_word_embedding('active'), 'bip39'),
    ('actual', 14, 1.7, 100, 0.7, generate_word_embedding('actual'), 'bip39'),
    ('adapt', 15, 1.7, 100, 0.7, generate_word_embedding('adapt'), 'bip39'),
    ('add', 16, 1.7, 100, 0.7, generate_word_embedding('add'), 'bip39'),
    ('address', 17, 1.7, 100, 0.7, generate_word_embedding('address'), 'bip39'),
    ('adjust', 18, 1.7, 100, 0.7, generate_word_embedding('adjust'), 'bip39'),
    ('adult', 19, 1.7, 100, 0.7, generate_word_embedding('adult'), 'bip39'),
    ('advance', 20, 1.7, 100, 0.7, generate_word_embedding('advance'), 'bip39'),
    ('advice', 21, 1.7, 100, 0.7, generate_word_embedding('advice'), 'bip39'),
    ('afraid', 22, 1.7, 100, 0.7, generate_word_embedding('afraid'), 'bip39'),
    ('again', 23, 1.7, 100, 0.7, generate_word_embedding('again'), 'bip39'),
    ('age', 24, 1.7, 100, 0.7, generate_word_embedding('age'), 'bip39'),
    ('agent', 25, 1.7, 100, 0.7, generate_word_embedding('agent'), 'bip39'),
    ('agree', 26, 1.7, 100, 0.7, generate_word_embedding('agree'), 'bip39'),
    ('ahead', 27, 1.7, 100, 0.7, generate_word_embedding('ahead'), 'bip39'),
    ('air', 28, 1.7, 100, 0.7, generate_word_embedding('air'), 'bip39'),
    ('alert', 29, 1.7, 100, 0.7, generate_word_embedding('alert'), 'bip39'),
    ('alien', 30, 1.7, 100, 0.7, generate_word_embedding('alien'), 'bip39'),
    ('all', 31, 1.7, 100, 0.7, generate_word_embedding('all'), 'bip39'),
    ('allow', 32, 1.7, 100, 0.7, generate_word_embedding('allow'), 'bip39'),
    ('almost', 33, 1.7, 100, 0.7, generate_word_embedding('almost'), 'bip39'),
    ('alone', 34, 1.7, 100, 0.7, generate_word_embedding('alone'), 'bip39'),
    ('already', 35, 1.7, 100, 0.7, generate_word_embedding('already'), 'bip39'),
    ('also', 36, 1.7, 100, 0.7, generate_word_embedding('also'), 'bip39'),
    ('always', 37, 1.7, 100, 0.7, generate_word_embedding('always'), 'bip39'),
    ('amazing', 38, 1.7, 100, 0.7, generate_word_embedding('amazing'), 'bip39'),
    ('among', 39, 1.7, 100, 0.7, generate_word_embedding('among'), 'bip39'),
    ('amount', 40, 1.7, 100, 0.7, generate_word_embedding('amount'), 'bip39'),
    ('ancient', 41, 1.7, 100, 0.7, generate_word_embedding('ancient'), 'bip39'),
    ('anger', 42, 1.7, 100, 0.7, generate_word_embedding('anger'), 'bip39'),
    ('animal', 43, 1.7, 100, 0.7, generate_word_embedding('animal'), 'bip39'),
    ('announce', 44, 1.7, 100, 0.7, generate_word_embedding('announce'), 'bip39'),
    ('annual', 45, 1.7, 100, 0.7, generate_word_embedding('annual'), 'bip39'),
    ('another', 46, 1.7, 100, 0.7, generate_word_embedding('another'), 'bip39'),
    ('answer', 47, 1.7, 100, 0.7, generate_word_embedding('answer'), 'bip39'),
    ('any', 48, 1.7, 100, 0.7, generate_word_embedding('any'), 'bip39'),
    ('apart', 49, 1.7, 100, 0.7, generate_word_embedding('apart'), 'bip39'),
    ('appear', 50, 1.7, 100, 0.7, generate_word_embedding('appear'), 'bip39'),
    ('apple', 51, 1.7, 100, 0.7, generate_word_embedding('apple'), 'bip39'),
    ('area', 52, 1.7, 100, 0.7, generate_word_embedding('area'), 'bip39'),
    ('arm', 53, 1.7, 100, 0.7, generate_word_embedding('arm'), 'bip39'),
    ('army', 54, 1.7, 100, 0.7, generate_word_embedding('army'), 'bip39'),
    ('around', 55, 1.7, 100, 0.7, generate_word_embedding('around'), 'bip39'),
    ('arrive', 56, 1.7, 100, 0.7, generate_word_embedding('arrive'), 'bip39'),
    ('art', 57, 1.7, 100, 0.7, generate_word_embedding('art'), 'bip39'),
    ('artist', 58, 1.7, 100, 0.7, generate_word_embedding('artist'), 'bip39'),
    ('ask', 59, 1.7, 100, 0.7, generate_word_embedding('ask'), 'bip39'),
    ('aspect', 60, 1.7, 100, 0.7, generate_word_embedding('aspect'), 'bip39'),
    ('atom', 61, 1.7, 100, 0.7, generate_word_embedding('atom'), 'bip39'),
    ('attack', 62, 1.7, 100, 0.7, generate_word_embedding('attack'), 'bip39'),
    ('attend', 63, 1.7, 100, 0.7, generate_word_embedding('attend'), 'bip39'),
    ('attract', 64, 1.7, 100, 0.7, generate_word_embedding('attract'), 'bip39'),
    ('author', 65, 1.7, 100, 0.7, generate_word_embedding('author'), 'bip39'),
    ('auto', 66, 1.7, 100, 0.7, generate_word_embedding('auto'), 'bip39'),
    ('autumn', 67, 1.7, 100, 0.7, generate_word_embedding('autumn'), 'bip39'),
    ('average', 68, 1.7, 100, 0.7, generate_word_embedding('average'), 'bip39'),
    ('avoid', 69, 1.7, 100, 0.7, generate_word_embedding('avoid'), 'bip39'),
    ('awake', 70, 1.7, 100, 0.7, generate_word_embedding('awake'), 'bip39'),
    ('aware', 71, 1.7, 100, 0.7, generate_word_embedding('aware'), 'bip39'),
    ('away', 72, 1.7, 100, 0.7, generate_word_embedding('away'), 'bip39'),
    ('baby', 73, 1.7, 100, 0.7, generate_word_embedding('baby'), 'bip39'),
    ('back', 74, 1.7, 100, 0.7, generate_word_embedding('back'), 'bip39'),
    ('balance', 75, 1.7, 100, 0.7, generate_word_embedding('balance'), 'bip39'),
    ('ball', 76, 1.7, 100, 0.7, generate_word_embedding('ball'), 'bip39'),
    ('base', 77, 1.7, 100, 0.7, generate_word_embedding('base'), 'bip39'),
    ('basic', 78, 1.7, 100, 0.7, generate_word_embedding('basic'), 'bip39'),
    ('battle', 79, 1.7, 100, 0.7, generate_word_embedding('battle'), 'bip39'),
    ('beach', 80, 1.7, 100, 0.7, generate_word_embedding('beach'), 'bip39'),
    ('beauty', 81, 1.7, 100, 0.7, generate_word_embedding('beauty'), 'bip39'),
    ('because', 82, 1.7, 100, 0.7, generate_word_embedding('because'), 'bip39'),
    ('become', 83, 1.7, 100, 0.7, generate_word_embedding('become'), 'bip39'),
    ('before', 84, 1.7, 100, 0.7, generate_word_embedding('before'), 'bip39'),
    ('begin', 85, 1.7, 100, 0.7, generate_word_embedding('begin'), 'bip39'),
    ('behind', 86, 1.7, 100, 0.7, generate_word_embedding('behind'), 'bip39'),
    ('believe', 87, 1.7, 100, 0.7, generate_word_embedding('believe'), 'bip39'),
    ('below', 88, 1.7, 100, 0.7, generate_word_embedding('below'), 'bip39'),
    ('benefit', 89, 1.7, 100, 0.7, generate_word_embedding('benefit'), 'bip39'),
    ('best', 90, 1.7, 100, 0.7, generate_word_embedding('best'), 'bip39'),
    ('better', 91, 1.7, 100, 0.7, generate_word_embedding('better'), 'bip39'),
    ('between', 92, 1.7, 100, 0.7, generate_word_embedding('between'), 'bip39'),
    ('beyond', 93, 1.7, 100, 0.7, generate_word_embedding('beyond'), 'bip39'),
    ('bird', 94, 1.7, 100, 0.7, generate_word_embedding('bird'), 'bip39'),
    ('black', 95, 1.7, 100, 0.7, generate_word_embedding('black'), 'bip39'),
    ('blood', 96, 1.7, 100, 0.7, generate_word_embedding('blood'), 'bip39'),
    ('blue', 97, 1.7, 100, 0.7, generate_word_embedding('blue'), 'bip39'),
    ('board', 98, 1.7, 100, 0.7, generate_word_embedding('board'), 'bip39'),
    ('boat', 99, 1.7, 100, 0.7, generate_word_embedding('boat'), 'bip39'),
    ('body', 100, 1.7, 100, 0.7, generate_word_embedding('body'), 'bip39'),
    ('book', 101, 1.7, 100, 0.7, generate_word_embedding('book'), 'bip39'),
    ('border', 102, 1.7, 100, 0.7, generate_word_embedding('border'), 'bip39'),
    ('both', 103, 1.7, 100, 0.7, generate_word_embedding('both'), 'bip39'),
    ('bottom', 104, 1.7, 100, 0.7, generate_word_embedding('bottom'), 'bip39'),
    ('brain', 105, 1.7, 100, 0.7, generate_word_embedding('brain'), 'bip39'),
    ('bread', 106, 1.7, 100, 0.7, generate_word_embedding('bread'), 'bip39'),
    ('break', 107, 1.7, 100, 0.7, generate_word_embedding('break'), 'bip39'),
    ('bridge', 108, 1.7, 100, 0.7, generate_word_embedding('bridge'), 'bip39'),
    ('brief', 109, 1.7, 100, 0.7, generate_word_embedding('brief'), 'bip39'),
    ('bright', 110, 1.7, 100, 0.7, generate_word_embedding('bright'), 'bip39'),
    ('bring', 111, 1.7, 100, 0.7, generate_word_embedding('bring'), 'bip39'),
    ('brother', 112, 1.7, 100, 0.7, generate_word_embedding('brother'), 'bip39'),
    ('brown', 113, 1.7, 100, 0.7, generate_word_embedding('brown'), 'bip39'),
    ('build', 114, 1.7, 100, 0.7, generate_word_embedding('build'), 'bip39'),
    ('business', 115, 1.7, 100, 0.7, generate_word_embedding('business'), 'bip39'),
    ('call', 116, 1.7, 100, 0.7, generate_word_embedding('call'), 'bip39'),
    ('calm', 117, 1.7, 100, 0.7, generate_word_embedding('calm'), 'bip39'),
    ('camera', 118, 1.7, 100, 0.7, generate_word_embedding('camera'), 'bip39'),
    ('camp', 119, 1.7, 100, 0.7, generate_word_embedding('camp'), 'bip39'),
    ('capital', 120, 1.7, 100, 0.7, generate_word_embedding('capital'), 'bip39'),
    ('captain', 121, 1.7, 100, 0.7, generate_word_embedding('captain'), 'bip39'),
    ('car', 122, 1.7, 100, 0.7, generate_word_embedding('car'), 'bip39'),
    ('carbon', 123, 1.7, 100, 0.7, generate_word_embedding('carbon'), 'bip39'),
    ('card', 124, 1.7, 100, 0.7, generate_word_embedding('card'), 'bip39'),
    ('care', 125, 1.7, 100, 0.7, generate_word_embedding('care'), 'bip39'),
    ('carry', 126, 1.7, 100, 0.7, generate_word_embedding('carry'), 'bip39'),
    ('case', 127, 1.7, 100, 0.7, generate_word_embedding('case'), 'bip39'),
    ('catch', 128, 1.7, 100, 0.7, generate_word_embedding('catch'), 'bip39'),
    ('cause', 129, 1.7, 100, 0.7, generate_word_embedding('cause'), 'bip39'),
    ('center', 130, 1.7, 100, 0.7, generate_word_embedding('center'), 'bip39'),
    ('century', 131, 1.7, 100, 0.7, generate_word_embedding('century'), 'bip39'),
    ('certain', 132, 1.7, 100, 0.7, generate_word_embedding('certain'), 'bip39'),
    ('chair', 133, 1.7, 100, 0.7, generate_word_embedding('chair'), 'bip39'),
    ('champion', 134, 1.7, 100, 0.7, generate_word_embedding('champion'), 'bip39'),
    ('chance', 135, 1.7, 100, 0.7, generate_word_embedding('chance'), 'bip39'),
    ('change', 136, 1.7, 100, 0.7, generate_word_embedding('change'), 'bip39'),
    ('chapter', 137, 1.7, 100, 0.7, generate_word_embedding('chapter'), 'bip39'),
    ('charge', 138, 1.7, 100, 0.7, generate_word_embedding('charge'), 'bip39'),
    ('check', 139, 1.7, 100, 0.7, generate_word_embedding('check'), 'bip39'),
    ('child', 140, 1.7, 100, 0.7, generate_word_embedding('child'), 'bip39'),
    ('choice', 141, 1.7, 100, 0.7, generate_word_embedding('choice'), 'bip39'),
    ('choose', 142, 1.7, 100, 0.7, generate_word_embedding('choose'), 'bip39'),
    ('circle', 143, 1.7, 100, 0.7, generate_word_embedding('circle'), 'bip39'),
    ('citizen', 144, 1.7, 100, 0.7, generate_word_embedding('citizen'), 'bip39'),
    ('city', 145, 1.7, 100, 0.7, generate_word_embedding('city'), 'bip39'),
    ('claim', 146, 1.7, 100, 0.7, generate_word_embedding('claim'), 'bip39'),
    ('class', 147, 1.7, 100, 0.7, generate_word_embedding('class'), 'bip39'),
    ('clean', 148, 1.7, 100, 0.7, generate_word_embedding('clean'), 'bip39'),
    ('clear', 149, 1.7, 100, 0.7, generate_word_embedding('clear'), 'bip39'),
    ('close', 150, 1.7, 100, 0.7, generate_word_embedding('close'), 'bip39'),
    ('cloud', 151, 1.7, 100, 0.7, generate_word_embedding('cloud'), 'bip39'),
    ('code', 152, 1.7, 100, 0.7, generate_word_embedding('code'), 'bip39'),
    ('coffee', 153, 1.7, 100, 0.7, generate_word_embedding('coffee'), 'bip39'),
    ('cold', 154, 1.7, 100, 0.7, generate_word_embedding('cold'), 'bip39'),
    ('collect', 155, 1.7, 100, 0.7, generate_word_embedding('collect'), 'bip39'),
    ('color', 156, 1.7, 100, 0.7, generate_word_embedding('color'), 'bip39'),
    ('come', 157, 1.7, 100, 0.7, generate_word_embedding('come'), 'bip39'),
    ('common', 158, 1.7, 100, 0.7, generate_word_embedding('common'), 'bip39'),
    ('company', 159, 1.7, 100, 0.7, generate_word_embedding('company'), 'bip39'),
    ('connect', 160, 1.7, 100, 0.7, generate_word_embedding('connect'), 'bip39'),
    ('consider', 161, 1.7, 100, 0.7, generate_word_embedding('consider'), 'bip39'),
    ('control', 162, 1.7, 100, 0.7, generate_word_embedding('control'), 'bip39'),
    ('cool', 163, 1.7, 100, 0.7, generate_word_embedding('cool'), 'bip39'),
    ('copy', 164, 1.7, 100, 0.7, generate_word_embedding('copy'), 'bip39'),
    ('core', 165, 1.7, 100, 0.7, generate_word_embedding('core'), 'bip39'),
    ('correct', 166, 1.7, 100, 0.7, generate_word_embedding('correct'), 'bip39'),
    ('cost', 167, 1.7, 100, 0.7, generate_word_embedding('cost'), 'bip39'),
    ('country', 168, 1.7, 100, 0.7, generate_word_embedding('country'), 'bip39'),
    ('couple', 169, 1.7, 100, 0.7, generate_word_embedding('couple'), 'bip39'),
    ('course', 170, 1.7, 100, 0.7, generate_word_embedding('course'), 'bip39'),
    ('cover', 171, 1.7, 100, 0.7, generate_word_embedding('cover'), 'bip39'),
    ('create', 172, 1.7, 100, 0.7, generate_word_embedding('create'), 'bip39'),
    ('credit', 173, 1.7, 100, 0.7, generate_word_embedding('credit'), 'bip39'),
    ('cross', 174, 1.7, 100, 0.7, generate_word_embedding('cross'), 'bip39'),
    ('culture', 175, 1.7, 100, 0.7, generate_word_embedding('culture'), 'bip39'),
    ('current', 176, 1.7, 100, 0.7, generate_word_embedding('current'), 'bip39'),
    ('custom', 177, 1.7, 100, 0.7, generate_word_embedding('custom'), 'bip39'),
    ('cycle', 178, 1.7, 100, 0.7, generate_word_embedding('cycle'), 'bip39'),
    ('damage', 179, 1.7, 100, 0.7, generate_word_embedding('damage'), 'bip39'),
    ('dance', 180, 1.7, 100, 0.7, generate_word_embedding('dance'), 'bip39'),
    ('danger', 181, 1.7, 100, 0.7, generate_word_embedding('danger'), 'bip39'),
    ('dark', 182, 1.7, 100, 0.7, generate_word_embedding('dark'), 'bip39'),
    ('data', 183, 1.7, 100, 0.7, generate_word_embedding('data'), 'bip39'),
    ('day', 184, 1.7, 100, 0.7, generate_word_embedding('day'), 'bip39'),
    ('deal', 185, 1.7, 100, 0.7, generate_word_embedding('deal'), 'bip39'),
    ('death', 186, 1.7, 100, 0.7, generate_word_embedding('death'), 'bip39'),
    ('debate', 187, 1.7, 100, 0.7, generate_word_embedding('debate'), 'bip39'),
    ('decade', 188, 1.7, 100, 0.7, generate_word_embedding('decade'), 'bip39'),
    ('decide', 189, 1.7, 100, 0.7, generate_word_embedding('decide'), 'bip39'),
    ('deep', 190, 1.7, 100, 0.7, generate_word_embedding('deep'), 'bip39'),
    ('defense', 191, 1.7, 100, 0.7, generate_word_embedding('defense'), 'bip39'),
    ('degree', 192, 1.7, 100, 0.7, generate_word_embedding('degree'), 'bip39'),
    ('demand', 193, 1.7, 100, 0.7, generate_word_embedding('demand'), 'bip39'),
    ('describe', 194, 1.7, 100, 0.7, generate_word_embedding('describe'), 'bip39'),
    ('design', 195, 1.7, 100, 0.7, generate_word_embedding('design'), 'bip39'),
    ('detail', 196, 1.7, 100, 0.7, generate_word_embedding('detail'), 'bip39'),
    ('develop', 197, 1.7, 100, 0.7, generate_word_embedding('develop'), 'bip39'),
    ('device', 198, 1.7, 100, 0.7, generate_word_embedding('device'), 'bip39'),
    ('different', 199, 1.7, 100, 0.7, generate_word_embedding('different'), 'bip39'),
    ('digital', 200, 1.7, 100, 0.7, generate_word_embedding('digital'), 'bip39')
ON CONFLICT (token) DO UPDATE SET
    phi_score = GREATEST(tokenizer_vocabulary.phi_score, EXCLUDED.phi_score),
    source_type = 'bip39',
    basin_embedding = COALESCE(EXCLUDED.basin_embedding, tokenizer_vocabulary.basin_embedding),
    updated_at = CURRENT_TIMESTAMP;

-- Add common words for better coverage
INSERT INTO tokenizer_vocabulary (token, token_id, weight, frequency, phi_score, basin_embedding, source_type)
VALUES
    ('consciousness', 1001, 1.8, 200, 0.8, generate_word_embedding('consciousness'), 'base'),
    ('geometry', 1002, 1.8, 200, 0.8, generate_word_embedding('geometry'), 'base'),
    ('quantum', 1003, 1.8, 200, 0.8, generate_word_embedding('quantum'), 'base'),
    ('integration', 1004, 1.8, 200, 0.8, generate_word_embedding('integration'), 'base'),
    ('resonance', 1005, 1.8, 200, 0.8, generate_word_embedding('resonance'), 'base'),
    ('understanding', 1006, 1.8, 200, 0.8, generate_word_embedding('understanding'), 'base'),
    ('thinking', 1007, 1.8, 200, 0.8, generate_word_embedding('thinking'), 'base'),
    ('reasoning', 1008, 1.8, 200, 0.8, generate_word_embedding('reasoning'), 'base'),
    ('knowledge', 1009, 1.8, 200, 0.8, generate_word_embedding('knowledge'), 'base'),
    ('wisdom', 1010, 1.8, 200, 0.8, generate_word_embedding('wisdom'), 'base'),
    ('experience', 1011, 1.8, 200, 0.8, generate_word_embedding('experience'), 'base'),
    ('information', 1012, 1.8, 200, 0.8, generate_word_embedding('information'), 'base'),
    ('system', 1013, 1.8, 200, 0.8, generate_word_embedding('system'), 'base'),
    ('process', 1014, 1.8, 200, 0.8, generate_word_embedding('process'), 'base'),
    ('method', 1015, 1.8, 200, 0.8, generate_word_embedding('method'), 'base')
ON CONFLICT (token) DO UPDATE SET
    phi_score = GREATEST(tokenizer_vocabulary.phi_score, EXCLUDED.phi_score),
    basin_embedding = COALESCE(EXCLUDED.basin_embedding, tokenizer_vocabulary.basin_embedding),
    updated_at = CURRENT_TIMESTAMP;

-- Verify results
SELECT 
    source_type,
    COUNT(*) as count,
    AVG(phi_score)::numeric(5,3) as avg_phi
FROM tokenizer_vocabulary
GROUP BY source_type
ORDER BY count DESC;
