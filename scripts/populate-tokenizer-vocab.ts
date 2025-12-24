#!/usr/bin/env npx tsx
/**
 * Populate tokenizer_vocabulary with real English words.
 * 
 * This adds BIP39 words and common English words with 64D basin embeddings
 * to fix the BPE garble issue in kernel generation.
 * 
 * Usage: npx tsx scripts/populate-tokenizer-vocab.ts
 */

import { db } from '../server/db';
import { sql } from 'drizzle-orm';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Common English words to add
const COMMON_WORDS = [
  // Pronouns
  "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
  "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
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
  "program", "question", "work", "government", "number", "night", "point", "home",
  "water", "room", "mother", "area", "money", "story", "fact", "month", "lot",
  "right", "study", "book", "eye", "job", "word", "business", "issue", "side",
  "kind", "head", "house", "service", "friend", "father", "power", "hour", "game",
  "line", "end", "member", "law", "car", "city", "community", "name", "president",
  "team", "minute", "idea", "kid", "body", "information", "back", "parent", "face",
  "level", "office", "door", "health", "person", "art", "war", "history",
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
  // Numbers
  "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
  "hundred", "thousand", "million",
  // Tech terms
  "data", "code", "user", "file", "network", "software", "hardware",
  "computer", "internet", "website", "application", "database", "server", "client",
  "algorithm", "function", "variable", "class", "object", "method", "interface",
  // QIG terms
  "basin", "geometry", "manifold", "coordinate", "dimension", "vector", "tensor",
  "kernel", "consciousness", "integration", "resonance", "attractor", "trajectory",
  "phi", "kappa", "entropy", "coherence", "synthesis", "emergence",
];

function computeBasinEmbedding(word: string): number[] {
  // Create deterministic 64D embedding from word hash
  const hash = crypto.createHash('sha256').update(word.toLowerCase()).digest('hex');
  const seed = parseInt(hash.substring(0, 8), 16);
  
  // Simple seeded random for reproducibility
  let state = seed;
  const random = () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return (state / 0x7fffffff) * 2 - 1; // Range [-1, 1]
  };
  
  // Generate 64D vector
  const embedding: number[] = [];
  for (let i = 0; i < 64; i++) {
    embedding.push(random());
  }
  
  // Normalize to unit sphere
  const norm = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0));
  return embedding.map(x => x / norm);
}

function computePhiScore(word: string): number {
  let phi = 0.5;
  
  // Length bonus
  phi += Math.min(word.length / 10, 0.2);
  
  // Alphabetic bonus
  if (/^[a-zA-Z]+$/.test(word)) {
    phi += 0.1;
  }
  
  // Common word bonus
  if (COMMON_WORDS.slice(0, 100).includes(word.toLowerCase())) {
    phi += 0.15;
  }
  
  return Math.min(phi, 0.95);
}

async function loadBip39Words(): Promise<string[]> {
  const bip39Path = path.join(__dirname, '..', 'server', 'bip39-wordlist.txt');
  try {
    const content = fs.readFileSync(bip39Path, 'utf-8');
    return content.split('\n').map(w => w.trim()).filter(w => w.length > 0);
  } catch {
    console.log('BIP39 wordlist not found, using common words only');
    return [];
  }
}

async function main() {
  console.log('Populating tokenizer_vocabulary with real English words...\n');
  
  // Load BIP39 words
  const bip39Words = await loadBip39Words();
  console.log(`Loaded ${bip39Words.length} BIP39 words`);
  
  // Combine all words (deduplicated)
  const allWords = new Set<string>();
  bip39Words.forEach(w => allWords.add(w.toLowerCase()));
  COMMON_WORDS.forEach(w => allWords.add(w.toLowerCase()));
  
  console.log(`Total unique words: ${allWords.size}`);
  
  // Prepare word data
  const wordData: Array<{
    token: string;
    tokenId: number;
    weight: number;
    frequency: number;
    phiScore: number;
    basinEmbedding: string;
    sourceType: string;
  }> = [];
  
  let tokenId = 10000; // Start after any existing token IDs
  
  for (const word of allWords) {
    if (word.length < 2) continue;
    
    const phi = computePhiScore(word);
    const embedding = computeBasinEmbedding(word);
    const embeddingStr = '[' + embedding.map(x => x.toFixed(6)).join(',') + ']';
    const sourceType = bip39Words.includes(word) ? 'bip39' : 'base';
    
    wordData.push({
      token: word,
      tokenId: tokenId++,
      weight: 1.0 + phi,
      frequency: 1,
      phiScore: phi,
      basinEmbedding: embeddingStr,
      sourceType,
    });
  }
  
  console.log(`Prepared ${wordData.length} words for insertion`);
  
  // Insert in batches
  const batchSize = 100;
  let inserted = 0;
  
  for (let i = 0; i < wordData.length; i += batchSize) {
    const batch = wordData.slice(i, i + batchSize);
    
    for (const word of batch) {
      try {
        await db.execute(sql`
          INSERT INTO tokenizer_vocabulary 
            (token, token_id, weight, frequency, phi_score, basin_embedding, source_type)
          VALUES 
            (${word.token}, ${word.tokenId}, ${word.weight}, ${word.frequency}, 
             ${word.phiScore}, ${word.basinEmbedding}::vector, ${word.sourceType})
          ON CONFLICT (token) DO UPDATE SET
            weight = EXCLUDED.weight,
            phi_score = EXCLUDED.phi_score,
            basin_embedding = EXCLUDED.basin_embedding,
            source_type = EXCLUDED.source_type,
            updated_at = CURRENT_TIMESTAMP
        `);
        inserted++;
      } catch (err) {
        // Skip duplicates or errors
      }
    }
    
    console.log(`Inserted batch ${Math.floor(i / batchSize) + 1}: ${inserted} words so far`);
  }
  
  console.log(`\n=== Summary ===`);
  console.log(`Total words inserted: ${inserted}`);
  
  // Verify
  const result = await db.execute(sql`
    SELECT source_type, COUNT(*) as count, AVG(phi_score)::numeric(5,3) as avg_phi 
    FROM tokenizer_vocabulary 
    GROUP BY source_type
  `);
  
  console.log('\nVerification:');
  for (const row of result.rows) {
    console.log(`  ${row.source_type}: ${row.count} words, avg_phi=${row.avg_phi}`);
  }
  
  // Sample high-phi words
  const sampleResult = await db.execute(sql`
    SELECT token, phi_score, source_type 
    FROM tokenizer_vocabulary 
    WHERE source_type IN ('bip39', 'base')
    ORDER BY phi_score DESC 
    LIMIT 10
  `);
  
  console.log('\nTop 10 words by phi:');
  for (const row of sampleResult.rows) {
    console.log(`  ${row.token}: ${row.phi_score} (${row.source_type})`);
  }
  
  console.log('\nDone!');
  process.exit(0);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
