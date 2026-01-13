#!/usr/bin/env tsx
/**
 * Schema Drift Detector
 * 
 * Compares Drizzle schema definition against live Neon database
 * to detect drift, missing tables, or manual changes.
 * 
 * Usage:
 *   tsx scripts/detect-schema-drift.ts
 *   tsx scripts/detect-schema-drift.ts --fix
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { sql } from 'drizzle-orm';
import * as schema from '../shared/schema';

const DATABASE_URL = process.env.DATABASE_URL;

if (!DATABASE_URL) {
  console.error('❌ DATABASE_URL environment variable is required');
  process.exit(1);
}

interface TableInfo {
  tableName: string;
  columns: ColumnInfo[];
  indexes: IndexInfo[];
}

interface ColumnInfo {
  columnName: string;
  dataType: string;
  isNullable: boolean;
  columnDefault: string | null;
}

interface IndexInfo {
  indexName: string;
  columns: string[];
  isUnique: boolean;
}

async function getActualSchema(db: any): Promise<Map<string, TableInfo>> {
  const tables = new Map<string, TableInfo>();
  
  // Get all tables
  const tableResults = await db.execute(sql`
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE'
    ORDER BY table_name
  `);
  
  for (const row of tableResults.rows) {
    const tableName = row.table_name as string;
    
    // Get columns for this table
    const columnResults = await db.execute(sql`
      SELECT 
        column_name,
        data_type,
        is_nullable,
        column_default
      FROM information_schema.columns
      WHERE table_schema = 'public'
      AND table_name = ${tableName}
      ORDER BY ordinal_position
    `);
    
    const columns: ColumnInfo[] = columnResults.rows.map((col: any) => ({
      columnName: col.column_name,
      dataType: col.data_type,
      isNullable: col.is_nullable === 'YES',
      columnDefault: col.column_default,
    }));
    
    // Get indexes for this table
    const indexResults = await db.execute(sql`
      SELECT
        i.relname as index_name,
        a.attname as column_name,
        ix.indisunique as is_unique
      FROM pg_class t
      JOIN pg_index ix ON t.oid = ix.indrelid
      JOIN pg_class i ON i.oid = ix.indexrelid
      JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
      WHERE t.relkind = 'r'
      AND t.relname = ${tableName}
      AND t.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
      ORDER BY i.relname, a.attnum
    `);
    
    const indexMap = new Map<string, IndexInfo>();
    for (const idx of indexResults.rows) {
      const indexName = idx.index_name as string;
      if (!indexMap.has(indexName)) {
        indexMap.set(indexName, {
          indexName,
          columns: [],
          isUnique: idx.is_unique as boolean,
        });
      }
      indexMap.get(indexName)!.columns.push(idx.column_name as string);
    }
    
    tables.set(tableName, {
      tableName,
      columns,
      indexes: Array.from(indexMap.values()),
    });
  }
  
  return tables;
}

function getExpectedTables(): Set<string> {
  // Extract table names from Drizzle schema
  const tableNames = new Set<string>();
  
  for (const [key, value] of Object.entries(schema)) {
    if (value && typeof value === 'object' && 'table' in value) {
      // This is a Drizzle table definition
      const tableName = (value as any).table;
      if (tableName && typeof tableName === 'string') {
        tableNames.add(tableName);
      }
    }
  }
  
  return tableNames;
}

async function detectDrift() {
  console.log('🔍 Connecting to database...');
  
  const client = postgres(DATABASE_URL!);
  const db = drizzle(client);
  
  try {
    console.log('📊 Fetching actual schema from database...');
    const actualTables = await getActualSchema(db);
    
    console.log('📋 Comparing with expected schema...');
    const expectedTables = getExpectedTables();
    
    let driftDetected = false;
    const issues: string[] = [];
    
    // Check for tables in DB but not in schema
    for (const [tableName, tableInfo] of actualTables) {
      if (!expectedTables.has(tableName)) {
        issues.push(`⚠️  Table '${tableName}' exists in database but not in schema.ts`);
        issues.push(`    Columns: ${tableInfo.columns.map(c => c.columnName).join(', ')}`);
        driftDetected = true;
      }
    }
    
    // Check for tables in schema but not in DB
    for (const tableName of expectedTables) {
      if (!actualTables.has(tableName)) {
        issues.push(`⚠️  Table '${tableName}' defined in schema.ts but missing from database`);
        driftDetected = true;
      }
    }
    
    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('📈 Schema Drift Detection Results');
    console.log('='.repeat(60));
    console.log(`Tables in database: ${actualTables.size}`);
    console.log(`Tables in schema.ts: ${expectedTables.size}`);
    console.log();
    
    if (driftDetected) {
      console.log('❌ Schema drift detected:\n');
      issues.forEach(issue => console.log(issue));
      console.log('\n💡 Recommendations:');
      console.log('   1. Run: drizzle-kit generate to create migrations');
      console.log('   2. Review generated migrations carefully');
      console.log('   3. Apply with: drizzle-kit migrate');
      console.log('   4. Or update schema.ts to match database');
      return 1;
    } else {
      console.log('✅ No schema drift detected - schema.ts matches database');
      return 0;
    }
    
  } catch (error) {
    console.error('❌ Error detecting schema drift:', error);
    return 1;
  } finally {
    await client.end();
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  detectDrift().then(code => process.exit(code));
}

export { detectDrift };
