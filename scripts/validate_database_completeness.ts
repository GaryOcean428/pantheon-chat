#!/usr/bin/env tsx
/**
 * Database Completeness Validation Script
 * 
 * Validates that no tables or columns are left NULL, uncalculated, or empty.
 * Checks for:
 * 1. Tables that should have data but are empty
 * 2. Columns with excessive NULL values
 * 3. Required singleton tables (quantum_state, adaptive_state)
 * 4. Vector columns without basin coordinates
 * 5. Consciousness metrics (phi, kappa) with default values only
 */

import { db } from '../server/db';
import { sql } from 'drizzle-orm';

type Severity = 'critical' | 'warning' | 'info';

interface ValidationResult {
  category: string;
  table: string;
  issue: string;
  severity: Severity;
  count?: number;
}

const results: ValidationResult[] = [];

// Whitelist of allowed table names for validation
const ALLOWED_TABLES = [
  'ocean_quantum_state', 'near_miss_adaptive_state', 'auto_cycle_state',
  'vocabulary_observations', 'coordizer_vocabulary', 'tokenizer_metadata',
  'manifold_probes', 'kernel_geometry', 'consciousness_checkpoints',
  'basin_documents', 'pantheon_messages'
] as const;

// Whitelist of allowed columns for NULL checking
const ALLOWED_COLUMNS: Record<string, string[]> = {
  'vocabulary_observations': ['basin_coords', 'avg_phi'],
  'manifold_probes': ['coordinates', 'phi'],
  'kernel_geometry': ['basin_coordinates'],
  'consciousness_checkpoints': ['state_data'],
};

function isValidTable(tableName: string): boolean {
  return ALLOWED_TABLES.includes(tableName as any);
}

function isValidColumn(tableName: string, columnName: string): boolean {
  const allowedColumns = ALLOWED_COLUMNS[tableName];
  return allowedColumns ? allowedColumns.includes(columnName) : false;
}

async function checkTableRowCount(
  tableName: string,
  minExpected: number,
  severity: Severity = 'warning'
) {
  if (!isValidTable(tableName)) {
    console.error(`Invalid table name: ${tableName}`);
    return 0;
  }
  
  try {
    const [result] = await db.execute(
      sql.raw(`SELECT COUNT(*) as count FROM ${tableName}`)
    );
    const count = parseInt(result.count as string);
    
    if (count === 0 && minExpected > 0) {
      results.push({
        category: 'Empty Tables',
        table: tableName,
        issue: `Table is empty (expected at least ${minExpected} rows)`,
        severity,
        count: 0,
      });
    } else {
      console.log(`✓ ${tableName}: ${count} rows`);
    }
    
    return count;
  } catch (error) {
    results.push({
      category: 'Table Access',
      table: tableName,
      issue: `Cannot access table: ${error.message}`,
      severity: 'critical',
    });
    return 0;
  }
}

async function checkNullColumns(
  tableName: string,
  columnName: string,
  maxNullPercent: number = 50
) {
  if (!isValidTable(tableName) || !isValidColumn(tableName, columnName)) {
    console.error(`Invalid table/column: ${tableName}.${columnName}`);
    return;
  }
  
  try {
    const [totalResult] = await db.execute(
      sql.raw(`SELECT COUNT(*) as total FROM ${tableName}`)
    );
    const total = parseInt(totalResult.total as string);
    
    if (total === 0) {
      return; // Skip if table is empty
    }
    
    const [nullResult] = await db.execute(
      sql.raw(
        `SELECT COUNT(*) as null_count FROM ${tableName} WHERE ${columnName} IS NULL`
      )
    );
    const nullCount = parseInt(nullResult.null_count as string);
    const nullPercent = (nullCount / total) * 100;
    
    if (nullPercent > maxNullPercent) {
      results.push({
        category: 'NULL Values',
        table: tableName,
        issue: `Column ${columnName} has ${nullPercent.toFixed(1)}% NULL values (${nullCount}/${total})`,
        severity: nullPercent > 80 ? 'warning' : 'info',
        count: nullCount,
      });
    }
  } catch (error) {
    // Column might not exist or other error
    console.log(`  ⚠ Cannot check ${tableName}.${columnName}: ${error.message}`);
  }
}

async function checkDefaultValues(
  tableName: string,
  columnName: string,
  defaultValue: number,
  tolerance: number = 0.01
) {
  if (!isValidTable(tableName)) {
    console.error(`Invalid table name: ${tableName}`);
    return;
  }
  
  try {
    const [totalResult] = await db.execute(
      sql.raw(`SELECT COUNT(*) as total FROM ${tableName}`)
    );
    const total = parseInt(totalResult.total as string);
    
    if (total === 0) {
      return;
    }
    
    const [defaultResult] = await db.execute(
      sql.raw(
        `SELECT COUNT(*) as default_count FROM ${tableName} 
         WHERE ${columnName} >= ${defaultValue - tolerance} 
         AND ${columnName} <= ${defaultValue + tolerance}`
      )
    );
    const defaultCount = parseInt(defaultResult.default_count as string);
    const defaultPercent = (defaultCount / total) * 100;
    
    if (defaultPercent > 80 && total > 10) {
      results.push({
        category: 'Default Values',
        table: tableName,
        issue: `Column ${columnName} has ${defaultPercent.toFixed(1)}% default values (${defaultValue})`,
        severity: 'info',
        count: defaultCount,
      });
    }
  } catch (error) {
    console.log(`  ⚠ Cannot check ${tableName}.${columnName}: ${error.message}`);
  }
}

async function validateSingletonTable(
  tableName: string,
  expectedId: string | number = 'singleton'
) {
  if (!isValidTable(tableName)) {
    console.error(`Invalid table name: ${tableName}`);
    return;
  }
  
  try {
    const [result] = await db.execute(
      sql.raw(`SELECT COUNT(*) as count FROM ${tableName}`)
    );
    const count = parseInt(result.count as string);
    
    if (count === 0) {
      results.push({
        category: 'Singleton Tables',
        table: tableName,
        issue: `Singleton table is empty (should have exactly 1 row with id='${expectedId}')`,
        severity: 'warning',
        count: 0,
      });
    } else if (count > 1) {
      results.push({
        category: 'Singleton Tables',
        table: tableName,
        issue: `Singleton table has ${count} rows (should have exactly 1)`,
        severity: 'warning',
        count,
      });
    } else {
      console.log(`✓ ${tableName}: 1 row (singleton)`);
    }
  } catch (error) {
    results.push({
      category: 'Singleton Tables',
      table: tableName,
      issue: `Cannot access singleton table: ${error.message}`,
      severity: 'critical',
    });
  }
}

async function main() {
  console.log('='.repeat(80));
  console.log('DATABASE COMPLETENESS VALIDATION');
  console.log('='.repeat(80));
  console.log();

  // Check singleton tables
  console.log('Checking singleton tables...');
  await validateSingletonTable('ocean_quantum_state', 'singleton');
  await validateSingletonTable('near_miss_adaptive_state', 'singleton');
  await validateSingletonTable('auto_cycle_state', 1);
  console.log();

  // Check core tables for minimum data
  console.log('Checking core tables for minimum data...');
  const vocabCount = await checkTableRowCount('vocabulary_observations', 10, 'warning');
  await checkTableRowCount('coordizer_vocabulary', 100, 'info');
  await checkTableRowCount('tokenizer_metadata', 5, 'info');
  console.log();

  // Check for NULL values in critical columns
  console.log('Checking for excessive NULL values...');
  if (vocabCount > 0) {
    await checkNullColumns('vocabulary_observations', 'basin_coords', 30);
    await checkNullColumns('vocabulary_observations', 'avg_phi', 20);
  }
  
  await checkNullColumns('manifold_probes', 'coordinates', 10);
  await checkNullColumns('manifold_probes', 'phi', 5);
  await checkNullColumns('kernel_geometry', 'basin_coordinates', 50);
  await checkNullColumns('consciousness_checkpoints', 'state_data', 5);
  console.log();

  // Check for default values in consciousness metrics
  console.log('Checking for uncomputed consciousness metrics...');
  const basinDocsCount = await db.execute(
    sql.raw(`SELECT COUNT(*) as count FROM basin_documents`)
  );
  if (parseInt(basinDocsCount[0].count as string) > 0) {
    await checkDefaultValues('basin_documents', 'phi', 0.5);
    await checkDefaultValues('basin_documents', 'kappa', 64.0);
  }
  
  const pantheonMsgCount = await db.execute(
    sql.raw(`SELECT COUNT(*) as count FROM pantheon_messages`)
  );
  if (parseInt(pantheonMsgCount[0].count as string) > 0) {
    await checkDefaultValues('pantheon_messages', 'phi', 0.7);
    await checkDefaultValues('pantheon_messages', 'kappa', 64.0);
  }
  console.log();

  // Report results
  console.log('='.repeat(80));
  console.log('VALIDATION RESULTS');
  console.log('='.repeat(80));
  console.log();

  if (results.length === 0) {
    console.log('✅ All validation checks passed!');
    console.log('No issues found with database completeness.');
    process.exit(0);
  }

  // Group results by category
  const categories = Array.from(new Set(results.map((r) => r.category)));
  
  for (const category of categories) {
    const categoryResults = results.filter((r) => r.category === category);
    const criticalCount = categoryResults.filter((r) => r.severity === 'critical').length;
    const warningCount = categoryResults.filter((r) => r.severity === 'warning').length;
    
    console.log(`${category}:`);
    console.log(`  ${criticalCount} critical, ${warningCount} warnings, ${categoryResults.length - criticalCount - warningCount} info`);
    console.log();
    
    for (const result of categoryResults) {
      const icon = result.severity === 'critical' ? '❌' : result.severity === 'warning' ? '⚠️' : 'ℹ️';
      console.log(`  ${icon} ${result.table}: ${result.issue}`);
    }
    console.log();
  }

  // Exit with appropriate code
  const hasCritical = results.some((r) => r.severity === 'critical');
  const hasWarning = results.some((r) => r.severity === 'warning');
  
  console.log('='.repeat(80));
  if (hasCritical) {
    console.log('❌ Validation failed with critical issues');
    process.exit(1);
  } else if (hasWarning) {
    console.log('⚠️  Validation completed with warnings');
    process.exit(0); // Don't fail on warnings
  } else {
    console.log('ℹ️  Validation completed with informational notes');
    process.exit(0);
  }
}

main().catch((error) => {
  console.error('Validation script failed:', error);
  process.exit(1);
});
