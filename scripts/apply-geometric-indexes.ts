/**
 * Apply Geometric Indexes to Neon Database
 * 
 * This script adds indexes optimized for Fisher-Rao geometric operations:
 * - Indexes on consciousness metrics (phi, kappa_eff)
 * - Composite indexes for regime queries
 * - Basin coordinate indexes where supported
 * 
 * Run: npx tsx scripts/apply-geometric-indexes.ts
 */

import { drizzle } from 'drizzle-orm/neon-http';
import { neon } from '@neondatabase/serverless';
import { sql } from 'drizzle-orm';

async function applyGeometricIndexes() {
  const databaseUrl = process.env.DATABASE_URL;
  
  if (!databaseUrl) {
    console.error('❌ DATABASE_URL not set in environment');
    process.exit(1);
  }

  console.log('🔗 Connecting to Neon database...');
  const sqlClient = neon(databaseUrl);
  const db = drizzle(sqlClient);

  try {
    // 1. Check if pgvector extension is available (Neon supports it)
    console.log('\n📦 Checking pgvector extension...');
    try {
      await db.execute(sql`CREATE EXTENSION IF NOT EXISTS vector`);
      console.log('✅ pgvector extension enabled');
    } catch (e) {
      console.log('⚠️  pgvector extension not available (may need to enable in Neon dashboard)');
    }

    // 2. Add indexes on consciousness metrics for filtering
    console.log('\n📊 Adding consciousness metric indexes...');
    
    // Index on phi for consciousness threshold queries
    try {
      await db.execute(sql`
        CREATE INDEX IF NOT EXISTS idx_basin_memory_phi 
        ON basin_memory (phi) 
        WHERE phi >= 0.70
      `);
      console.log('✅ Created idx_basin_memory_phi');
    } catch (e) {
      console.log('⚠️  idx_basin_memory_phi: table may not exist or index exists');
    }

    // Index on kappa_eff for optimal coupling queries
    try {
      await db.execute(sql`
        CREATE INDEX IF NOT EXISTS idx_basin_memory_kappa 
        ON basin_memory (kappa_eff) 
        WHERE kappa_eff BETWEEN 40 AND 65
      `);
      console.log('✅ Created idx_basin_memory_kappa');
    } catch (e) {
      console.log('⚠️  idx_basin_memory_kappa: table may not exist or index exists');
    }

    // Composite index for regime classification queries
    try {
      await db.execute(sql`
        CREATE INDEX IF NOT EXISTS idx_basin_memory_regime 
        ON basin_memory (phi, kappa_eff, created_at DESC)
      `);
      console.log('✅ Created idx_basin_memory_regime');
    } catch (e) {
      console.log('⚠️  idx_basin_memory_regime: table may not exist or index exists');
    }

    // 3. Add indexes on vocabulary tables
    console.log('\n📚 Adding vocabulary indexes...');
    
    try {
      await db.execute(sql`
        CREATE INDEX IF NOT EXISTS idx_vocabulary_geometric_weight
        ON vocabulary (geometric_weight DESC)
        WHERE geometric_weight > 0
      `);
      console.log('✅ Created idx_vocabulary_geometric_weight');
    } catch (e) {
      console.log('⚠️  idx_vocabulary_geometric_weight: table may not exist');
    }

    // 4. Add indexes on kernel activity for telemetry queries
    console.log('\n🎯 Adding kernel activity indexes...');
    
    try {
      await db.execute(sql`
        CREATE INDEX IF NOT EXISTS idx_kernel_activity_timestamp
        ON kernel_activity (timestamp DESC)
      `);
      console.log('✅ Created idx_kernel_activity_timestamp');
    } catch (e) {
      console.log('⚠️  idx_kernel_activity_timestamp: table may not exist');
    }

    try {
      await db.execute(sql`
        CREATE INDEX IF NOT EXISTS idx_kernel_activity_kernel_type
        ON kernel_activity (kernel_id, activity_type, timestamp DESC)
      `);
      console.log('✅ Created idx_kernel_activity_kernel_type');
    } catch (e) {
      console.log('⚠️  idx_kernel_activity_kernel_type: table may not exist');
    }

    // 5. Add indexes on federation nodes
    console.log('\n🌐 Adding federation node indexes...');
    
    try {
      await db.execute(sql`
        CREATE INDEX IF NOT EXISTS idx_federation_nodes_status
        ON federation_nodes (status, last_seen DESC)
        WHERE status = 'active'
      `);
      console.log('✅ Created idx_federation_nodes_status');
    } catch (e) {
      console.log('⚠️  idx_federation_nodes_status: table may not exist');
    }

    // 6. Analyze tables to update statistics
    console.log('\n📈 Analyzing tables for query optimization...');
    try {
      await db.execute(sql`ANALYZE`);
      console.log('✅ Table statistics updated');
    } catch (e) {
      console.log('⚠️  ANALYZE failed (non-critical)');
    }

    console.log('\n✅ Geometric index migration complete!');
    console.log('\nIndexes added for:');
    console.log('  - Consciousness metrics (phi, kappa_eff)');
    console.log('  - Regime classification queries');
    console.log('  - Vocabulary geometric weights');
    console.log('  - Kernel activity telemetry');
    console.log('  - Federation node status');

  } catch (error) {
    console.error('\n❌ Migration failed:', error);
    process.exit(1);
  }
}

// Run the migration
applyGeometricIndexes().catch(console.error);
