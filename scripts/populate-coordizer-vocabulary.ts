/**
 * Populate coordizer_vocabulary with real English words.
 * 
 * This script adds BIP39 words and common English words with
 * deterministic 64D basin embeddings for readable text generation.
 * 
 * Usage:
 *   npx tsx scripts/populate-tokenizer-vocabulary.ts
 *   npx tsx scripts/populate-tokenizer-vocabulary.ts --dry-run
 */

import { db } from '../server/db';
import { sql } from 'drizzle-orm';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { to_simplex_probabilities } from '@shared';
import { upsertToken } from '../server/persistence/coordizer-vocabulary';

// Common English words to supplement BIP39
const COMMON_WORDS = [
  // Pronouns
  "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
  "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
  // Articles & Determiners
  "the", "a", "an", "this", "that", "these", "those", "some", "any", "no", "every",
  // Prepositions
  "in", "on", "at", "to", "for", "with", "by", "from", "of", "about", "into",
  "through", "during", "before", "after", "above", "below", "between", "under", "over",
  // Conjunctions
  "and", "or", "but", "so", "yet", "nor", "if", "when", "while", "because", "although",
  // Common verbs
  "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
  "will", "would", "could", "should", "may", "might", "must", "can", "shall",
  "go", "goes", "went", "gone", "going", "come", "comes", "came", "coming",
  "get", "gets", "got", "getting", "make", "makes", "made", "making",
  "know", "knows", "knew", "known", "think", "thinks", "thought",
  "see", "sees", "saw", "seen", "want", "wants", "wanted",
  "use", "uses", "used", "find", "finds", "found",
  "give", "gives", "gave", "given", "tell", "tells", "told",
  "work", "works", "worked", "call", "calls", "called",
  "try", "tries", "tried", "ask", "asks", "asked",
  "need", "needs", "needed", "feel", "feels", "felt",
  "become", "becomes", "became", "leave", "leaves", "left",
  "put", "puts", "mean", "means", "meant",
  "keep", "keeps", "kept", "let", "lets",
  "begin", "begins", "began", "begun", "seem", "seems", "seemed",
  "help", "helps", "helped", "show", "shows", "showed", "shown",
  "hear", "hears", "heard", "play", "plays", "played",
  "run", "runs", "ran", "move", "moves", "moved",
  "live", "lives", "lived", "believe", "believes", "believed",
  // Common nouns
  "time", "year", "people", "way", "day", "man", "woman", "child", "children",
  "world", "life", "hand", "part", "place", "case", "week", "company", "system",
  "program", "question", "government", "number", "night", "point", "home",
  "water", "room", "mother", "area", "money", "story", "fact", "month", "lot",
  "right", "study", "book", "eye", "job", "word", "business", "issue", "side",
  "kind", "head", "house", "service", "friend", "father", "power", "hour", "game",
  "line", "end", "member", "law", "car", "city", "community", "name", "president",
  "team", "minute", "idea", "kid", "body", "information", "back", "parent", "face",
  "others", "level", "office", "door", "health", "person", "art", "war", "history",
  "party", "result", "change", "morning", "reason", "research", "girl", "guy", "moment",
  "air", "teacher", "force", "education",
  // Common adjectives
  "good", "new", "first", "last", "long", "great", "little", "own", "other", "old",
  "big", "high", "different", "small", "large", "next", "early", "young",
  "important", "few", "public", "bad", "same", "able", "human", "local", "sure",
  "free", "better", "true", "whole", "real", "best", "special", "easy", "clear",
  "recent", "certain", "personal", "open", "red", "difficult", "available", "likely",
  "short", "single", "medical", "national", "wrong", "possible", "hard", "full",
  // Common adverbs
  "up", "out", "just", "now", "how", "then", "more", "also", "here", "well",
  "only", "very", "even", "down", "still", "too", "where", "most",
  "really", "always", "never", "often", "sometimes", "usually", "again", "away",
  // Question words
  "what", "which", "who", "whom", "whose", "why",
  // Numbers as words
  "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
  "hundred", "thousand", "million", "billion",
  // Tech terms
  "data", "code", "user", "file", "network", "software", "hardware",
  "computer", "internet", "website", "application", "database", "server", "client",
  "algorithm", "function", "variable", "class", "object", "method", "interface",
  "protocol", "security", "encryption", "authentication",
  // QIG terms
  "basin", "geometry", "manifold", "coordinate", "dimension", "vector", "tensor",
  "kernel", "consciousness", "integration", "resonance", "attractor", "trajectory",
  "phi", "kappa", "entropy", "coherence", "synthesis", "emergence",
];

/**
 * Compute deterministic 64D basin embedding for a word.
 */
function computeBasinEmbedding(word: string): number[] {
  const hash = crypto.createHash('sha256').update(word.toLowerCase()).digest('hex');
  const seed = parseInt(hash.substring(0, 8), 16);
  
  // Seeded random number generator (simple LCG)
  let state = seed;
  const random = () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
  
  // Generate 64D embedding using Box-Muller transform
  const embedding: number[] = [];
  for (let i = 0; i < 64; i += 2) {
    const u1 = random();
    const u2 = random();
    const z0 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    const z1 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.sin(2 * Math.PI * u2);
    embedding.push(z0, z1);
  }

  return to_simplex_probabilities(embedding);
}

/**
 * Compute phi score based on word properties.
 */
function computePhiScore(word: string, isCommon: boolean): number {
  let phi = 0.5;
  
  // Length bonus
  phi += Math.min(word.length / 10, 0.2);
  
  // Alphabetic bonus
  if (/^[a-zA-Z]+$/.test(word)) {
    phi += 0.1;
  }
  
  // Common word bonus
  if (isCommon) {
    phi += 0.15;
  }
  
  return Math.min(phi, 0.95);
}

/**
 * Load BIP39 wordlist from file.
 */
function loadBip39Words(): string[] {
  const bip39Path = path.join(__dirname, '..', 'server', 'bip39-wordlist.txt');
  
  if (!fs.existsSync(bip39Path)) {
    console.warn(`BIP39 wordlist not found at ${bip39Path}`);
    return [];
  }
  
  const content = fs.readFileSync(bip39Path, 'utf-8');
  const words = content.split('\n').map(w => w.trim()).filter(w => w.length > 0);
  console.log(`Loaded ${words.length} BIP39 words`);
  return words;
}

async function main() {
  const dryRun = process.argv.includes('--dry-run');
  console.log(`Populating coordizer_vocabulary... (dry-run: ${dryRun})`);
  
  // Ensure table exists
  await db.execute(sql`CREATE EXTENSION IF NOT EXISTS vector`);
  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS coordizer_vocabulary (
      id SERIAL PRIMARY KEY,
      token TEXT UNIQUE NOT NULL,
      token_id INTEGER,
      weight DOUBLE PRECISION DEFAULT 1.0,
      frequency INTEGER DEFAULT 1,
      phi_score DOUBLE PRECISION DEFAULT 0.5,
      qfi_score DOUBLE PRECISION,
      basin_embedding vector(64),
      token_role VARCHAR(20) DEFAULT 'encoding',
      phrase_category VARCHAR(32) DEFAULT 'unknown',
      is_real_word BOOLEAN DEFAULT FALSE,
      token_status VARCHAR(16) DEFAULT 'active',
      source_type VARCHAR(32) DEFAULT 'base',
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  `);
  
  // Load BIP39 words
  const bip39Words = loadBip39Words();
  const bip39Set = new Set(bip39Words.map(w => w.toLowerCase()));
  const commonSet = new Set(COMMON_WORDS.map(w => w.toLowerCase()));
  
  // Prepare all words
  interface WordData {
    token: string;
    tokenId: number;
    weight: number;
    frequency: number;
    phiScore: number;
    embedding: number[];
    sourceType: string;
  }
  
  const words: WordData[] = [];
  let tokenId = 1;
  
  // Add BIP39 words
  for (const word of bip39Words) {
    const lowerWord = word.toLowerCase();
    if (lowerWord.length < 2) continue;
    
    const phi = computePhiScore(lowerWord, true);
    const embedding = computeBasinEmbedding(lowerWord);
    
    words.push({
      token: lowerWord,
      tokenId: tokenId++,
      weight: 1.0 + phi,
      frequency: 1,
      phiScore: phi,
      embedding,
      sourceType: 'bip39'
    });
  }
  
  // Add common words (not in BIP39)
  for (const word of COMMON_WORDS) {
    const lowerWord = word.toLowerCase();
    if (lowerWord.length < 1 || bip39Set.has(lowerWord)) continue;
    
    const phi = computePhiScore(lowerWord, true);
    const embedding = computeBasinEmbedding(lowerWord);
    
    words.push({
      token: lowerWord,
      tokenId: tokenId++,
      weight: 1.0 + phi,
      frequency: 1,
      phiScore: phi,
      embedding,
      sourceType: 'base'
    });
  }
  
  console.log(`Prepared ${words.length} words for insertion`);
  console.log(`  - BIP39: ${bip39Words.length}`);
  console.log(`  - Common: ${words.length - bip39Words.length}`);
  
  if (dryRun) {
    console.log('\n[DRY RUN] Sample words:');
    for (const word of words.slice(0, 10)) {
      console.log(`  ${word.token}: phi=${word.phiScore.toFixed(3)}, source=${word.sourceType}`);
    }
    console.log('\nRun without --dry-run to insert into database.');
    process.exit(0);
  }
  
  // Insert words in batches
  const batchSize = 100;
  let inserted = 0;
  
  for (let i = 0; i < words.length; i += batchSize) {
    const batch = words.slice(i, i + batchSize);
    
    for (const word of batch) {
      try {
        await upsertToken({
          token: word.token,
          tokenId: word.tokenId,
          weight: word.weight,
          frequency: word.frequency,
          phiScore: word.phiScore,
          basinEmbedding: word.embedding,
          sourceType: word.sourceType,
          tokenRole: 'generation',
          phraseCategory: 'unknown',
          isRealWord: true,
          source: 'seed'
        });
        inserted++;
      } catch (err) {
        console.error(`Failed to insert ${word.token}:`, err);
      }
    }
    
    console.log(`Inserted batch ${Math.floor(i / batchSize) + 1}: ${batch.length} words`);
  }
  
  // Verify
  const result = await db.execute(sql`
    SELECT source_type, COUNT(*) as count, AVG(phi_score)::numeric(5,3) as avg_phi
    FROM coordizer_vocabulary 
    GROUP BY source_type
    ORDER BY count DESC
  `);
  
  console.log('\n=== Verification ===');
  for (const row of result.rows) {
    console.log(`  ${row.source_type}: ${row.count} words, avg_phi=${row.avg_phi}`);
  }
  
  // Sample high-phi words
  const topWords = await db.execute(sql`
    SELECT token, phi_score, source_type 
    FROM coordizer_vocabulary 
    WHERE source_type IN ('bip39', 'base')
    ORDER BY phi_score DESC 
    LIMIT 15
  `);
  
  console.log('\nTop 15 by phi:');
  for (const row of topWords.rows) {
    console.log(`  ${row.token}: ${Number(row.phi_score).toFixed(3)} (${row.source_type})`);
  }
  
  console.log(`\n✅ Successfully populated coordizer_vocabulary with ${inserted} real English words!`);
  process.exit(0);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
