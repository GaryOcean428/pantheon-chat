#!/usr/bin/env tsx
/**
 * Introspect missing tables from Neon database to reconstruct schema definitions
 */
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';

const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) {
  throw new Error('DATABASE_URL not set');
}

const client = postgres(DATABASE_URL);
const db = drizzle(client);

async function introspectTable(tableName: string) {
  console.log(`\n=== ${tableName} ===`);
  
  // Get column definitions
  const columns = await client`
    SELECT 
      column_name,
      data_type,
      character_maximum_length,
      is_nullable,
      column_default
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = ${tableName}
    ORDER BY ordinal_position;
  `;
  
  console.log('Columns:');
  columns.forEach(col => {
    const nullable = col.is_nullable === 'YES' ? '' : 'NOT NULL';
    const length = col.character_maximum_length ? `(${col.character_maximum_length})` : '';
    const defaultVal = col.column_default ? `DEFAULT ${col.column_default}` : '';
    console.log(`  ${col.column_name}: ${col.data_type}${length} ${nullable} ${defaultVal}`.trim());
  });
  
  // Get indexes
  const indexes = await client`
    SELECT 
      indexname,
      indexdef
    FROM pg_indexes
    WHERE schemaname = 'public' AND tablename = ${tableName};
  `;
  
  if (indexes.length > 0) {
    console.log('Indexes:');
    indexes.forEach(idx => console.log(`  ${idx.indexname}: ${idx.indexdef}`));
  }
  
  // Get row count
  const count = await client`SELECT COUNT(*) as count FROM ${client(tableName)}`;
  console.log(`Row count: ${count[0].count}`);
}

async function main() {
  const missingTables = [
    'qig_metadata',
    'governance_proposals', 
    'tool_requests',
    'pattern_discoveries',
    'vocabulary_stats',
    'federation_peers',
    'passphrase_vocabulary'
  ];
  
  for (const table of missingTables) {
    try {
      await introspectTable(table);
    } catch (err) {
      console.error(`Error introspecting ${table}:`, err);
    }
  }
  
  await client.end();
}

main().catch(console.error);
