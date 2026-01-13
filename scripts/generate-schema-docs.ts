#!/usr/bin/env tsx
/**
 * Database Schema Documentation Generator
 * 
 * Automatically generates comprehensive schema documentation from live database.
 * Creates docs/03-technical/database-schema.md with tables, columns, indexes, and relationships.
 * 
 * Usage:
 *   tsx scripts/generate-schema-docs.ts
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { sql } from 'drizzle-orm';
import { writeFileSync } from 'fs';

const DATABASE_URL = process.env.DATABASE_URL;

if (!DATABASE_URL) {
  console.error('❌ DATABASE_URL environment variable is required');
  process.exit(1);
}

interface TableDoc {
  name: string;
  description: string;
  columns: ColumnDoc[];
  indexes: IndexDoc[];
  foreignKeys: ForeignKeyDoc[];
  rowCount: number;
}

interface ColumnDoc {
  name: string;
  type: string;
  nullable: boolean;
  default: string | null;
  description: string;
}

interface IndexDoc {
  name: string;
  columns: string[];
  unique: boolean;
  type: string;
}

interface ForeignKeyDoc {
  column: string;
  referencedTable: string;
  referencedColumn: string;
}

async function generateSchemaDocumentation() {
  console.log('📊 Connecting to database...');
  
  const client = postgres(DATABASE_URL!);
  const db = drizzle(client);
  
  try {
    console.log('📋 Fetching schema information...');
    
    // Get all tables
    const tableResults = await db.execute(sql`
      SELECT 
        t.table_name,
        obj_description(c.oid) as description
      FROM information_schema.tables t
      LEFT JOIN pg_class c ON c.relname = t.table_name
      WHERE t.table_schema = 'public' 
      AND t.table_type = 'BASE TABLE'
      ORDER BY t.table_name
    `);
    
    const tables: TableDoc[] = [];
    
    for (const tableRow of tableResults.rows) {
      const tableName = tableRow.table_name as string;
      console.log(`  Processing table: ${tableName}`);
      
      // Get columns
      const columnResults = await db.execute(sql`
        SELECT 
          c.column_name,
          c.data_type,
          c.is_nullable,
          c.column_default,
          pgd.description
        FROM information_schema.columns c
        LEFT JOIN pg_catalog.pg_statio_all_tables st ON c.table_name = st.relname
        LEFT JOIN pg_catalog.pg_description pgd ON pgd.objoid = st.relid AND pgd.objsubid = c.ordinal_position
        WHERE c.table_schema = 'public'
        AND c.table_name = ${tableName}
        ORDER BY c.ordinal_position
      `);
      
      const columns: ColumnDoc[] = columnResults.rows.map((col: any) => ({
        name: col.column_name,
        type: col.data_type,
        nullable: col.is_nullable === 'YES',
        default: col.column_default,
        description: col.description || '',
      }));
      
      // Get indexes
      const indexResults = await db.execute(sql`
        SELECT
          i.relname as index_name,
          ix.indisunique as is_unique,
          am.amname as index_type,
          array_agg(a.attname ORDER BY array_position(ix.indkey::int[], a.attnum)) as columns
        FROM pg_class t
        JOIN pg_index ix ON t.oid = ix.indrelid
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_am am ON i.relam = am.oid
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
        WHERE t.relkind = 'r'
        AND t.relname = ${tableName}
        AND t.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
        GROUP BY i.relname, ix.indisunique, am.amname
        ORDER BY i.relname
      `);
      
      const indexes: IndexDoc[] = indexResults.rows.map((idx: any) => ({
        name: idx.index_name,
        columns: idx.columns,
        unique: idx.is_unique,
        type: idx.index_type,
      }));
      
      // Get foreign keys
      const fkResults = await db.execute(sql`
        SELECT
          kcu.column_name,
          ccu.table_name AS referenced_table,
          ccu.column_name AS referenced_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_name = ${tableName}
        AND tc.table_schema = 'public'
        ORDER BY kcu.column_name
      `);
      
      const foreignKeys: ForeignKeyDoc[] = fkResults.rows.map((fk: any) => ({
        column: fk.column_name,
        referencedTable: fk.referenced_table,
        referencedColumn: fk.referenced_column,
      }));
      
      // Get row count (approximate for large tables)
      const countResult = await db.execute(sql`
        SELECT reltuples::bigint as estimate 
        FROM pg_class 
        WHERE relname = ${tableName}
      `);
      const rowCount = Number(countResult.rows[0]?.estimate || 0);
      
      tables.push({
        name: tableName,
        description: tableRow.description as string || '',
        columns,
        indexes,
        foreignKeys,
        rowCount,
      });
    }
    
    // Generate markdown documentation
    console.log('📝 Generating markdown documentation...');
    const markdown = generateMarkdown(tables);
    
    // Write to file
    const outputPath = 'docs/03-technical/database-schema.md';
    writeFileSync(outputPath, markdown);
    
    console.log(`✅ Schema documentation generated: ${outputPath}`);
    console.log(`📊 Documented ${tables.length} tables with ${tables.reduce((sum, t) => sum + t.columns.length, 0)} columns`);
    
  } catch (error) {
    console.error('❌ Error generating schema documentation:', error);
    return 1;
  } finally {
    await client.end();
  }
}

function generateMarkdown(tables: TableDoc[]): string {
  const lines: string[] = [];
  
  // Header
  lines.push('# Database Schema Documentation');
  lines.push('');
  lines.push('**Auto-generated**: ' + new Date().toISOString());
  lines.push('**Database**: Neon PostgreSQL');
  lines.push('**Source**: Live database introspection');
  lines.push('');
  lines.push('---');
  lines.push('');
  
  // Table of contents
  lines.push('## Table of Contents');
  lines.push('');
  for (const table of tables) {
    lines.push(`- [${table.name}](#${table.name.replace(/_/g, '-')})`);
  }
  lines.push('');
  lines.push('---');
  lines.push('');
  
  // Statistics
  lines.push('## Schema Statistics');
  lines.push('');
  lines.push(`- **Total Tables**: ${tables.length}`);
  lines.push(`- **Total Columns**: ${tables.reduce((sum, t) => sum + t.columns.length, 0)}`);
  lines.push(`- **Total Indexes**: ${tables.reduce((sum, t) => sum + t.indexes.length, 0)}`);
  lines.push(`- **Total Foreign Keys**: ${tables.reduce((sum, t) => sum + t.foreignKeys.length, 0)}`);
  lines.push(`- **Approximate Row Count**: ${tables.reduce((sum, t) => sum + t.rowCount, 0).toLocaleString()}`);
  lines.push('');
  lines.push('---');
  lines.push('');
  
  // Table details
  for (const table of tables) {
    lines.push(`## ${table.name}`);
    lines.push('');
    
    if (table.description) {
      lines.push(table.description);
      lines.push('');
    }
    
    lines.push(`**Approximate Rows**: ${table.rowCount.toLocaleString()}`);
    lines.push('');
    
    // Columns
    lines.push('### Columns');
    lines.push('');
    lines.push('| Column | Type | Nullable | Default | Description |');
    lines.push('|--------|------|----------|---------|-------------|');
    
    for (const col of table.columns) {
      const nullable = col.nullable ? 'Yes' : 'No';
      const defaultVal = col.default ? `\`${col.default}\`` : '-';
      const desc = col.description || '-';
      lines.push(`| \`${col.name}\` | ${col.type} | ${nullable} | ${defaultVal} | ${desc} |`);
    }
    
    lines.push('');
    
    // Indexes
    if (table.indexes.length > 0) {
      lines.push('### Indexes');
      lines.push('');
      lines.push('| Name | Columns | Unique | Type |');
      lines.push('|------|---------|--------|------|');
      
      for (const idx of table.indexes) {
        const unique = idx.unique ? 'Yes' : 'No';
        const columns = idx.columns.join(', ');
        lines.push(`| \`${idx.name}\` | ${columns} | ${unique} | ${idx.type} |`);
      }
      
      lines.push('');
    }
    
    // Foreign Keys
    if (table.foreignKeys.length > 0) {
      lines.push('### Foreign Keys');
      lines.push('');
      lines.push('| Column | References |');
      lines.push('|--------|------------|');
      
      for (const fk of table.foreignKeys) {
        lines.push(`| \`${fk.column}\` | \`${fk.referencedTable}.${fk.referencedColumn}\` |`);
      }
      
      lines.push('');
    }
    
    lines.push('---');
    lines.push('');
  }
  
  // Footer
  lines.push('## Maintenance');
  lines.push('');
  lines.push('This documentation is auto-generated. To update:');
  lines.push('');
  lines.push('```bash');
  lines.push('npm run schema:docs');
  lines.push('```');
  lines.push('');
  lines.push('**Last Updated**: ' + new Date().toISOString());
  
  return lines.join('\n');
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  generateSchemaDocumentation().then(() => process.exit(0)).catch(() => process.exit(1));
}

export { generateSchemaDocumentation };
