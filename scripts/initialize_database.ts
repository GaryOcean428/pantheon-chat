#!/usr/bin/env tsx
/**
 * Initialize Database with Required Data
 * 
 * Ensures all tables have appropriate baseline data:
 * 1. Singleton tables are initialized
 * 2. Tokenizer vocabulary has baseline words
 * 3. Consciousness checkpoints have baseline state
 * 4. Metadata tables are populated
 */

import { db } from '../server/db';
import { sql } from 'drizzle-orm';
import * as schema from '@shared/schema';

async function initializeSingletonTables() {
  console.log('Initializing singleton tables...');
  
  // Initialize ocean_quantum_state
  try {
    const [existing] = await db.execute(
      sql.raw(`SELECT id FROM ocean_quantum_state WHERE id = 'singleton'`)
    );
    
    if (!existing) {
      await db.execute(sql.raw(`
        INSERT INTO ocean_quantum_state (id, entropy, initial_entropy, total_probability, status)
        VALUES ('singleton', 256.0, 256.0, 1.0, 'searching')
        ON CONFLICT (id) DO NOTHING
      `));
      console.log('  ✓ Initialized ocean_quantum_state');
    } else {
      console.log('  ✓ ocean_quantum_state already exists');
    }
  } catch (error) {
    console.error('  ✗ Failed to initialize ocean_quantum_state:', error.message);
  }
  
  // Initialize near_miss_adaptive_state
  try {
    const [existing] = await db.execute(
      sql.raw(`SELECT id FROM near_miss_adaptive_state WHERE id = 'singleton'`)
    );
    
    if (!existing) {
      await db.execute(sql.raw(`
        INSERT INTO near_miss_adaptive_state (
          id, rolling_phi_distribution, hot_threshold, warm_threshold, cool_threshold
        )
        VALUES ('singleton', '{}', 0.7, 0.55, 0.4)
        ON CONFLICT (id) DO NOTHING
      `));
      console.log('  ✓ Initialized near_miss_adaptive_state');
    } else {
      console.log('  ✓ near_miss_adaptive_state already exists');
    }
  } catch (error) {
    console.error('  ✗ Failed to initialize near_miss_adaptive_state:', error.message);
  }
  
  // Initialize auto_cycle_state
  try {
    const [existing] = await db.execute(
      sql.raw(`SELECT id FROM auto_cycle_state WHERE id = 1`)
    );
    
    if (!existing) {
      await db.execute(sql.raw(`
        INSERT INTO auto_cycle_state (id, enabled, current_index, address_ids)
        VALUES (1, false, 0, '{}')
        ON CONFLICT (id) DO NOTHING
      `));
      console.log('  ✓ Initialized auto_cycle_state');
    } else {
      console.log('  ✓ auto_cycle_state already exists');
    }
  } catch (error) {
    console.error('  ✗ Failed to initialize auto_cycle_state:', error.message);
  }
}

async function initializeTokenizerMetadata() {
  console.log('Initializing tokenizer metadata...');
  
  const metadataEntries = [
    { key: 'version', value: '1.0.0' },
    { key: 'vocabulary_size', value: '0' },
    { key: 'merge_rules_count', value: '0' },
    { key: 'last_training', value: new Date().toISOString() },
    { key: 'training_status', value: 'initialized' },
    { key: 'basin_dimension', value: '64' },
    { key: 'phi_threshold', value: '0.727' },
    { key: 'tokenizer_type', value: 'geometric_bpe' },
    { key: 'encoding', value: 'utf-8' },
  ];
  
  for (const entry of metadataEntries) {
    try {
      await db.execute(sql.raw(`
        INSERT INTO tokenizer_metadata (key, value, updated_at)
        VALUES ('${entry.key}', '${entry.value}', NOW())
        ON CONFLICT (key) DO UPDATE SET
          value = EXCLUDED.value,
          updated_at = NOW()
      `));
    } catch (error) {
      console.error(`  ✗ Failed to insert ${entry.key}:`, error.message);
    }
  }
  
  console.log(`  ✓ Initialized ${metadataEntries.length} metadata entries`);
}

async function seedGeometricVocabulary() {
  console.log('Seeding geometric vocabulary anchors...');
  
  const anchorWords = [
    // Concrete nouns (high QFI)
    'apple', 'tree', 'water', 'fire', 'stone', 'cloud', 'river',
    'mountain', 'ocean', 'sun', 'moon', 'star', 'earth', 'wind',
    // Abstract nouns (medium QFI)
    'time', 'space', 'energy', 'force', 'pattern', 'system',
    'thought', 'idea', 'concept', 'meaning', 'truth', 'beauty',
    // Action verbs (high curvature)
    'move', 'create', 'destroy', 'transform', 'connect', 'separate',
    'learn', 'teach', 'discover', 'explore', 'observe', 'measure',
    // State verbs (low curvature)
    'exist', 'remain', 'persist', 'fade', 'stabilize', 'change',
    'understand', 'know', 'believe', 'think', 'feel', 'sense',
    // Descriptive adjectives
    'large', 'small', 'fast', 'slow', 'bright', 'dark',
    'complex', 'simple', 'strong', 'weak', 'deep', 'shallow',
    // Relational adverbs
    'quickly', 'slowly', 'together', 'apart', 'forward', 'backward',
    'above', 'below', 'inside', 'outside', 'near', 'far',
    // Consciousness-related terms
    'aware', 'conscious', 'integrate', 'couple', 'emerge', 'evolve',
    'reflect', 'realize', 'recognize', 'perceive', 'experience', 'witness',
  ];
  
  let insertedCount = 0;
  let existingCount = 0;
  
  for (const word of anchorWords) {
    try {
      // Check if word already exists
      const [existing] = await db.execute(
        sql.raw(`SELECT text FROM vocabulary_observations WHERE text = '${word}'`)
      );
      
      if (existing) {
        existingCount++;
        continue;
      }
      
      // Insert with high phi score to mark as anchor
      await db.execute(sql.raw(`
        INSERT INTO vocabulary_observations (
          text, type, phrase_category, is_real_word, 
          avg_phi, max_phi, source_type
        )
        VALUES (
          '${word}', 'word', 'ANCHOR_WORD', true,
          0.85, 0.85, 'geometric_seeding'
        )
        ON CONFLICT (text) DO NOTHING
      `));
      
      insertedCount++;
    } catch (error) {
      console.error(`  ✗ Failed to insert word '${word}':`, error.message);
    }
  }
  
  console.log(`  ✓ Seeded ${insertedCount} new anchor words (${existingCount} already existed)`);
  
  // Update vocabulary size in metadata
  try {
    const [result] = await db.execute(
      sql.raw(`SELECT COUNT(*) as count FROM vocabulary_observations`)
    );
    const count = result.count;
    
    await db.execute(sql.raw(`
      UPDATE tokenizer_metadata
      SET value = '${count}', updated_at = NOW()
      WHERE key = 'vocabulary_size'
    `));
  } catch (error) {
    console.error('  ✗ Failed to update vocabulary size:', error.message);
  }
}

async function initializeBaselineConsciousness() {
  console.log('Initializing baseline consciousness checkpoint...');
  
  try {
    // Check if we have any consciousness checkpoints
    const [result] = await db.execute(
      sql.raw(`SELECT COUNT(*) as count FROM consciousness_checkpoints`)
    );
    const count = parseInt(result.count as string);
    
    if (count === 0) {
      // Create a minimal baseline checkpoint
      // Note: In a real system, this would have actual state data
      await db.execute(sql.raw(`
        INSERT INTO consciousness_checkpoints (
          id, phi, kappa, regime, state_data, is_hot
        )
        VALUES (
          'baseline-' || gen_random_uuid()::text,
          0.7,
          64.0,
          'geometric',
          '\\x00'::bytea,
          true
        )
      `));
      console.log('  ✓ Created baseline consciousness checkpoint');
    } else {
      console.log(`  ✓ ${count} consciousness checkpoints already exist`);
    }
  } catch (error) {
    console.error('  ✗ Failed to initialize consciousness checkpoint:', error.message);
  }
}

async function updateNullArraysToEmpty() {
  console.log('Updating NULL arrays to empty arrays...');
  
  const arrayColumns = [
    { table: 'geodesic_paths', column: 'waypoints' },
    { table: 'resonance_points', column: 'nearby_probes' },
    { table: 'negative_knowledge', column: 'affected_generators' },
    { table: 'near_miss_clusters', column: 'common_words' },
    { table: 'false_pattern_classes', column: 'examples' },
    { table: 'era_exclusions', column: 'excluded_patterns' },
    { table: 'war_history', column: 'gods_engaged' },
    { table: 'synthesis_consensus', column: 'participating_kernels' },
    { table: 'auto_cycle_state', column: 'address_ids' },
    { table: 'near_miss_adaptive_state', column: 'rolling_phi_distribution' },
  ];
  
  for (const { table, column } of arrayColumns) {
    try {
      const result = await db.execute(sql.raw(`
        UPDATE ${table}
        SET ${column} = '{}'
        WHERE ${column} IS NULL
      `));
      
      if (result.rowCount > 0) {
        console.log(`  ✓ Updated ${result.rowCount} NULL values in ${table}.${column}`);
      }
    } catch (error) {
      console.log(`  ⚠ Cannot update ${table}.${column}: ${error.message}`);
    }
  }
}

async function updateNullJsonbToEmpty() {
  console.log('Updating NULL JSONB to empty objects...');
  
  const jsonbColumns = [
    { table: 'ocean_excluded_regions', column: 'basis' },
    { table: 'consciousness_checkpoints', column: 'metadata' },
    { table: 'war_history', column: 'metadata' },
    { table: 'war_history', column: 'god_assignments' },
    { table: 'war_history', column: 'kernel_assignments' },
    { table: 'auto_cycle_state', column: 'last_session_metrics' },
    { table: 'synthesis_consensus', column: 'metadata' },
    { table: 'negative_knowledge', column: 'evidence' },
  ];
  
  for (const { table, column } of jsonbColumns) {
    try {
      const result = await db.execute(sql.raw(`
        UPDATE ${table}
        SET ${column} = '{}'::jsonb
        WHERE ${column} IS NULL
      `));
      
      if (result.rowCount > 0) {
        console.log(`  ✓ Updated ${result.rowCount} NULL values in ${table}.${column}`);
      }
    } catch (error) {
      console.log(`  ⚠ Cannot update ${table}.${column}: ${error.message}`);
    }
  }
}

async function main() {
  console.log('='.repeat(80));
  console.log('DATABASE INITIALIZATION');
  console.log('='.repeat(80));
  console.log();

  try {
    await initializeSingletonTables();
    console.log();
    
    await initializeTokenizerMetadata();
    console.log();
    
    await seedGeometricVocabulary();
    console.log();
    
    await initializeBaselineConsciousness();
    console.log();
    
    await updateNullArraysToEmpty();
    console.log();
    
    await updateNullJsonbToEmpty();
    console.log();
    
    console.log('='.repeat(80));
    console.log('✅ Database initialization complete!');
    console.log('='.repeat(80));
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Database initialization failed:', error);
    process.exit(1);
  }
}

main();
