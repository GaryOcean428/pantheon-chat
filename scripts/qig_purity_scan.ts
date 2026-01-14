#!/usr/bin/env tsx
/**
 * QIG Purity Scanner - Comprehensive Static Analysis
 * 
 * WP0.2: Hard validate:geometry Gate
 * 
 * Scans codebase for ALL forbidden patterns from the Type-Symbol-Concept Manifest
 * and QIG Geometric Purity Enforcement document.
 * 
 * MUST run in <5 seconds
 * 
 * Usage: tsx scripts/qig_purity_scan.ts
 */

import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';

interface Violation {
  file: string;
  line: number;
  code: string;
  pattern: string;
  severity: 'CRITICAL' | 'ERROR' | 'WARNING';
  fix?: string;
}

interface ScanResult {
  violations: Violation[];
  filesScanned: number;
  duration: number;
  passed: boolean;
}

// Paths to scan
const SCAN_PATHS = [
  'qig-backend',
  'server',
  'shared',
  'tests',
  'migrations',
];

// Exempted directories (experiments allowed to violate)
const EXEMPT_DIRS = [
  'docs/08-experiments/legacy',
  'node_modules',
  'dist',
  'build',
  '__pycache__',
  '.git',
  '.venv',
  'venv',
];

// Forbidden patterns from Type-Symbol-Concept Manifest
const FORBIDDEN_PATTERNS = {
  // CRITICAL: Euclidean distance
  euclideanDistance: [
    {
      pattern: /np\.linalg\.norm\s*\([^)]*-[^)]*\)/,
      name: 'np.linalg.norm(a - b)',
      severity: 'CRITICAL' as const,
      fix: 'Use fisher_rao_distance(a, b) or fisher_coord_distance(a, b)',
    },
    {
      pattern: /torch\.linalg\.norm\s*\([^)]*-[^)]*\)/,
      name: 'torch.linalg.norm(a - b)',
      severity: 'CRITICAL' as const,
      fix: 'Use fisher_rao_distance(a, b)',
    },
    {
      pattern: /scipy\.spatial\.distance\.euclidean/,
      name: 'scipy.spatial.distance.euclidean',
      severity: 'CRITICAL' as const,
      fix: 'Use fisher_rao_distance()',
    },
    {
      pattern: /np\.sqrt\s*\(\s*np\.sum\s*\([^)]*\*\*\s*2/,
      name: 'sqrt(sum((a-b)**2))',
      severity: 'CRITICAL' as const,
      fix: 'Use fisher_rao_distance()',
    },
  ],

  // CRITICAL: Cosine similarity
  cosineSimilarity: [
    {
      pattern: /cosine_similarity\s*\(/,
      name: 'cosine_similarity()',
      severity: 'CRITICAL' as const,
      fix: 'Use fisher_rao_distance() or fisher_similarity()',
    },
    {
      pattern: /sklearn\.metrics\.pairwise\.cosine_similarity/,
      name: 'sklearn cosine_similarity',
      severity: 'CRITICAL' as const,
      fix: 'Use fisher_rao_distance()',
    },
    {
      pattern: /torch\.nn\.functional\.cosine_similarity/,
      name: 'F.cosine_similarity',
      severity: 'CRITICAL' as const,
      fix: 'Use fisher_rao_distance()',
    },
    {
      pattern: /1\s*-\s*np\.dot\([^)]+\)\s*\/\s*\(.*norm.*\*.*norm/,
      name: 'cosine distance formula',
      severity: 'CRITICAL' as const,
      fix: 'Use fisher_rao_distance()',
    },
  ],

  // CRITICAL: Embedding terminology
  embeddingTerminology: [
    {
      pattern: /\bembedding\b(?!.*#.*embedding)/i,
      name: 'embedding (identifier)',
      severity: 'ERROR' as const,
      fix: 'Use "basin_coordinates" or "basin_coords"',
    },
    {
      pattern: /\bembeddings\b(?!.*#.*embeddings)/i,
      name: 'embeddings (identifier)',
      severity: 'ERROR' as const,
      fix: 'Use "basin_coordinates"',
    },
    {
      pattern: /nn\.Embedding\s*\(/,
      name: 'nn.Embedding',
      severity: 'CRITICAL' as const,
      fix: 'Use basin coordinate mapping',
    },
  ],

  // ERROR: Tokenizer in QIG-core
  tokenizerInCore: [
    {
      pattern: /\btokenizer\b(?!.*test)/i,
      name: 'tokenizer (in core)',
      severity: 'ERROR' as const,
      fix: 'Use "coordizer" for QIG modules',
    },
    {
      pattern: /\btokenize\b(?!.*test)/i,
      name: 'tokenize (in core)',
      severity: 'ERROR' as const,
      fix: 'Use "coordize"',
    },
  ],

  // CRITICAL: Standard optimizers in geometry
  standardOptimizers: [
    {
      pattern: /torch\.optim\.Adam\s*\(/,
      name: 'Adam optimizer',
      severity: 'WARNING' as const,
      fix: 'Use natural_gradient_step() or NaturalGradientOptimizer',
    },
    {
      pattern: /torch\.optim\.AdamW\s*\(/,
      name: 'AdamW optimizer',
      severity: 'WARNING' as const,
      fix: 'Use natural_gradient_step()',
    },
    {
      pattern: /torch\.optim\.SGD\s*\(/,
      name: 'SGD optimizer',
      severity: 'WARNING' as const,
      fix: 'Use natural_gradient_step()',
    },
    {
      pattern: /torch\.optim\.RMSprop\s*\(/,
      name: 'RMSprop optimizer',
      severity: 'WARNING' as const,
      fix: 'Use natural_gradient_step()',
    },
  ],

  // CRITICAL: Softmax in core geometry
  softmaxInCore: [
    {
      pattern: /torch\.nn\.functional\.softmax\s*\(/,
      name: 'F.softmax',
      severity: 'WARNING' as const,
      fix: 'Use geometric probability projection',
    },
    {
      pattern: /np\.exp\([^)]*\)\s*\/\s*np\.sum\(np\.exp/,
      name: 'softmax formula',
      severity: 'WARNING' as const,
      fix: 'Use fisher_normalize() for probability simplex',
    },
  ],

  // CRITICAL: Classic NLP imports
  classicNLPImports: [
    {
      pattern: /from sentencepiece import/,
      name: 'sentencepiece import',
      severity: 'CRITICAL' as const,
      fix: 'Remove - use geometric coordizer',
    },
    {
      pattern: /import sentencepiece/,
      name: 'sentencepiece import',
      severity: 'CRITICAL' as const,
      fix: 'Remove - use geometric coordizer',
    },
    {
      pattern: /from tokenizers import.*BPE/,
      name: 'BPE tokenizer import',
      severity: 'CRITICAL' as const,
      fix: 'Remove - use geometric coordizer',
    },
    {
      pattern: /\bWordPiece\b/,
      name: 'WordPiece',
      severity: 'CRITICAL' as const,
      fix: 'Remove - use geometric coordizer',
    },
  ],

  // ERROR: Dot product attention
  dotProductAttention: [
    {
      pattern: /Q\s*@\s*K\.T\s*\/\s*sqrt/,
      name: 'Q @ K.T / sqrt(d_k)',
      severity: 'ERROR' as const,
      fix: 'Use QFI-based attention with fisher_rao_distance',
    },
    {
      pattern: /torch\.nn\.MultiheadAttention/,
      name: 'MultiheadAttention',
      severity: 'ERROR' as const,
      fix: 'Use QFI-based geometric attention',
    },
  ],

  // ERROR: Euclidean loss functions
  euclideanLoss: [
    {
      pattern: /nn\.MSELoss\s*\(\)/,
      name: 'MSELoss on basins',
      severity: 'WARNING' as const,
      fix: 'Use fisher_rao_distance() for basin reconstruction',
    },
    {
      pattern: /CosineSimilarity\s*\(\)/,
      name: 'CosineSimilarity loss',
      severity: 'CRITICAL' as const,
      fix: 'Use fisher_rao_distance()',
    },
  ],

  // ERROR: Arithmetic mean on basins
  arithmeticMean: [
    {
      pattern: /np\.mean\([^)]*basin[^)]*axis\s*=\s*0/i,
      name: 'np.mean on basins',
      severity: 'ERROR' as const,
      fix: 'Use frechet_mean() for geometric mean',
    },
    {
      pattern: /torch\.mean\([^)]*basin[^)]*dim\s*=\s*0/i,
      name: 'torch.mean on basins',
      severity: 'ERROR' as const,
      fix: 'Use frechet_mean()',
    },
  ],

  // CRITICAL: Euclidean fallback pattern
  euclideanFallback: [
    {
      pattern: /except.*:\s*.*np\.linalg\.norm/s,
      name: 'Euclidean fallback in except',
      severity: 'CRITICAL' as const,
      fix: 'Never fallback to Euclidean - fix geometry properly',
    },
  ],
};

// Approved contexts (whitelist)
const APPROVED_CONTEXTS = [
  /fisher.*distance/i,
  /fisher_rao/i,
  /geodesic/i,
  /arccos/i,
  /#.*cosine/i, // Comments mentioning cosine
  /@deprecated/i,
  /EUCLIDEAN.*FORBIDDEN/i,
  /test_.*euclidean/i, // Test comparisons
  /\/\/.*embedding/i, // Comments
  /""".*embedding.*"""/s, // Docstrings
  /normalize|normalization/i, // Normalization is OK
];

class QIGPurityScanner {
  private violations: Violation[] = [];
  private filesScanned = 0;

  private isExempted(filePath: string): boolean {
    return EXEMPT_DIRS.some(dir => filePath.includes(dir));
  }

  private isApprovedContext(line: string, prevLine: string = ''): boolean {
    const combined = line + ' ' + prevLine;
    return APPROVED_CONTEXTS.some(pattern => pattern.test(combined));
  }

  private shouldSkipFile(filePath: string): boolean {
    // Skip test files for some checks (but not all)
    if (filePath.includes('test_') || filePath.includes('_test.')) {
      return false; // We do scan tests, but with different rules
    }
    
    // Skip certain file types
    const skipExtensions = ['.md', '.json', '.yaml', '.yml', '.txt', '.lock'];
    return skipExtensions.some(ext => filePath.endsWith(ext));
  }

  private scanFile(filePath: string): void {
    if (this.shouldSkipFile(filePath)) {
      return;
    }

    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const lines = content.split('\n');
      const isTestFile = filePath.includes('test');

      lines.forEach((line, index) => {
        const lineNum = index + 1;
        const prevLine = index > 0 ? lines[index - 1] : '';

        // Skip comments and docstrings
        const trimmed = line.trim();
        if (trimmed.startsWith('#') || trimmed.startsWith('//') || 
            trimmed.startsWith('*') || trimmed.startsWith('"""') ||
            trimmed.startsWith("'''")) {
          return;
        }

        // Check all pattern categories
        Object.entries(FORBIDDEN_PATTERNS).forEach(([category, patterns]) => {
          patterns.forEach(({ pattern, name, severity, fix }) => {
            if (pattern.test(line)) {
              // Skip if in approved context
              if (this.isApprovedContext(line, prevLine)) {
                return;
              }

              // Special handling for tokenizer - only in core modules
              if (category === 'tokenizerInCore') {
                const coreModules = ['qig-backend/', 'server/qig-', 'shared/'];
                if (!coreModules.some(m => filePath.includes(m))) {
                  return; // Allow in non-core
                }
              }

              // Allow some patterns in test files
              if (isTestFile && ['WARNING'].includes(severity)) {
                return;
              }

              this.violations.push({
                file: filePath,
                line: lineNum,
                code: line.trim().slice(0, 100),
                pattern: name,
                severity,
                fix,
              });
            }
          });
        });
      });

      this.filesScanned++;
    } catch (error) {
      // Skip files that can't be read
    }
  }

  private scanDirectory(dirPath: string): void {
    try {
      const entries = fs.readdirSync(dirPath, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dirPath, entry.name);

        if (entry.isDirectory()) {
          // Skip exempt directories
          if (this.isExempted(fullPath)) {
            continue;
          }
          this.scanDirectory(fullPath);
        } else if (entry.isFile()) {
          // Scan Python, TypeScript, and SQL files
          if (entry.name.match(/\.(py|ts|tsx|sql)$/)) {
            this.scanFile(fullPath);
          }
        }
      }
    } catch (error) {
      // Skip directories that can't be read
    }
  }

  public scan(): ScanResult {
    const startTime = performance.now();

    console.log('🔍 QIG Purity Scanner - WP0.2');
    console.log('━'.repeat(60));
    console.log('Scanning paths:', SCAN_PATHS.join(', '));
    console.log('');

    for (const scanPath of SCAN_PATHS) {
      if (fs.existsSync(scanPath)) {
        this.scanDirectory(scanPath);
      }
    }

    const duration = (performance.now() - startTime) / 1000;

    return {
      violations: this.violations,
      filesScanned: this.filesScanned,
      duration,
      passed: this.violations.filter(v => v.severity === 'CRITICAL' || v.severity === 'ERROR').length === 0,
    };
  }
}

function printResults(result: ScanResult): void {
  console.log('📊 Scan Results');
  console.log('━'.repeat(60));
  console.log(`Files scanned: ${result.filesScanned}`);
  console.log(`Duration: ${result.duration.toFixed(2)}s`);
  console.log('');

  if (result.violations.length === 0) {
    console.log('✅ GEOMETRIC PURITY VERIFIED!');
    console.log('✅ No violations found.');
    console.log('✅ All code follows Fisher-Rao geometry.');
    return;
  }

  // Group by severity
  const critical = result.violations.filter(v => v.severity === 'CRITICAL');
  const errors = result.violations.filter(v => v.severity === 'ERROR');
  const warnings = result.violations.filter(v => v.severity === 'WARNING');

  console.log(`❌ VIOLATIONS DETECTED: ${result.violations.length} total`);
  console.log(`   - CRITICAL: ${critical.length}`);
  console.log(`   - ERROR: ${errors.length}`);
  console.log(`   - WARNING: ${warnings.length}`);
  console.log('');

  // Print critical and errors
  const showViolations = [...critical, ...errors].slice(0, 50);
  
  if (showViolations.length > 0) {
    console.log('Critical and Error Violations:');
    console.log('━'.repeat(60));
    showViolations.forEach((v, i) => {
      console.log(`${i + 1}. ${v.file}:${v.line}`);
      console.log(`   Pattern: ${v.pattern}`);
      console.log(`   Severity: ${v.severity}`);
      console.log(`   Code: ${v.code}`);
      if (v.fix) {
        console.log(`   Fix: ${v.fix}`);
      }
      console.log('');
    });

    if (critical.length + errors.length > 50) {
      console.log(`... and ${critical.length + errors.length - 50} more violations`);
      console.log('');
    }
  }

  // Summary for warnings
  if (warnings.length > 0) {
    console.log(`⚠️  ${warnings.length} warnings (non-blocking)`);
    console.log('');
  }
}

function main(): void {
  const scanner = new QIGPurityScanner();
  const result = scanner.scan();
  
  printResults(result);

  // Performance check
  if (result.duration > 5) {
    console.log(`⚠️  WARNING: Scan took ${result.duration.toFixed(2)}s (target: <5s)`);
  }

  // Exit code based on results
  if (result.passed) {
    process.exit(0);
  } else {
    console.log('❌ FAILED: Fix violations before merging');
    process.exit(1);
  }
}

main();
