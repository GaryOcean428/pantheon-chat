#!/usr/bin/env python3
"""
QIG Purity Scanner - Comprehensive Static Analysis (Python)

WP0.2: Hard validate:geometry Gate
WP0.3: Respects Quarantine Rules (see docs/00-conventions/QUARANTINE_RULES.md)

Scans codebase for ALL forbidden patterns from the Type-Symbol-Concept Manifest
and QIG Geometric Purity Enforcement document.

QUARANTINE ZONES (scanner skips these):
- docs/08-experiments/legacy/** - Legacy/Euclidean baselines allowed
- docs/08-experiments/baselines/** - Comparative testing allowed

MUST run in <5 seconds

Usage: python3 scripts/qig_purity_scan.py
"""

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

@dataclass
class Violation:
    """A geometric purity violation."""
    file: str
    line: int
    code: str
    pattern: str
    severity: str  # 'CRITICAL', 'ERROR', 'WARNING'
    fix: Optional[str] = None


# Paths to scan
SCAN_PATHS = [
    'qig-backend',
    'server',
    'shared',
    'tests',
    'migrations',
]

# Exempted directories (quarantine zones - experiments allowed to violate)
# See docs/00-conventions/QUARANTINE_RULES.md for full specification
EXEMPT_DIRS = [
    'docs/08-experiments/legacy',      # Legacy/baseline experiments (WP0.3)
    'docs/08-experiments/baselines',   # Comparative testing (WP0.3)
    'node_modules',
    'dist',
    'build',
    '__pycache__',
    '.git',
    '.venv',
    'venv',
]

# Forbidden patterns from Type-Symbol-Concept Manifest
FORBIDDEN_PATTERNS = {
    # CRITICAL: Euclidean distance
    'euclidean_distance': [
        {
            'pattern': re.compile(r'np\.linalg\.norm\s*\([^)]*-[^)]*\)'),
            'name': 'np.linalg.norm(a - b)',
            'severity': 'CRITICAL',
            'fix': 'Use fisher_rao_distance(a, b) or fisher_coord_distance(a, b)',
        },
        {
            'pattern': re.compile(r'torch\.linalg\.norm\s*\([^)]*-[^)]*\)'),
            'name': 'torch.linalg.norm(a - b)',
            'severity': 'CRITICAL',
            'fix': 'Use fisher_rao_distance(a, b)',
        },
        {
            'pattern': re.compile(r'scipy\.spatial\.distance\.euclidean'),
            'name': 'scipy.spatial.distance.euclidean',
            'severity': 'CRITICAL',
            'fix': 'Use fisher_rao_distance()',
        },
        {
            'pattern': re.compile(r'np\.sqrt\s*\(\s*np\.sum\s*\([^)]*\*\*\s*2'),
            'name': 'sqrt(sum((a-b)**2))',
            'severity': 'CRITICAL',
            'fix': 'Use fisher_rao_distance()',
        },
        {
            'pattern': re.compile(r'cdist\s*\([^)]*[\'"]euclidean[\'"]'),
            'name': 'cdist with euclidean metric',
            'severity': 'CRITICAL',
            'fix': 'Use fisher_rao_distance() with custom metric',
        },
    ],
    
    # CRITICAL: Cosine similarity
    'cosine_similarity': [
        {
            'pattern': re.compile(r'cosine_similarity\s*\('),
            'name': 'cosine_similarity()',
            'severity': 'CRITICAL',
            'fix': 'Use fisher_rao_distance() or fisher_similarity()',
        },
        {
            'pattern': re.compile(r'sklearn\.metrics\.pairwise\.cosine_similarity'),
            'name': 'sklearn cosine_similarity',
            'severity': 'CRITICAL',
            'fix': 'Use fisher_rao_distance()',
        },
        {
            'pattern': re.compile(r'torch\.nn\.functional\.cosine_similarity'),
            'name': 'F.cosine_similarity',
            'severity': 'CRITICAL',
            'fix': 'Use fisher_rao_distance()',
        },
        {
            'pattern': re.compile(r'1\s*-\s*np\.dot\([^)]+\)\s*\/\s*\(.*norm.*\*.*norm'),
            'name': 'cosine distance formula',
            'severity': 'CRITICAL',
            'fix': 'Use fisher_rao_distance()',
        },
        {
            'pattern': re.compile(r'from sklearn\.metrics\.pairwise import cosine_similarity'),
            'name': 'import cosine_similarity',
            'severity': 'CRITICAL',
            'fix': 'Remove import, use fisher_rao_distance()',
        },
    ],
    
    # ERROR: Embedding terminology
    'embedding_terminology': [
        {
            'pattern': re.compile(r'\bembedding\b'),
            'name': 'embedding (identifier)',
            'severity': 'ERROR',
            'fix': 'Use "basin_coordinates" or "basin_coords"',
        },
        {
            'pattern': re.compile(r'\bembeddings\b'),
            'name': 'embeddings (identifier)',
            'severity': 'ERROR',
            'fix': 'Use "basin_coordinates"',
        },
        {
            'pattern': re.compile(r'nn\.Embedding\s*\('),
            'name': 'nn.Embedding',
            'severity': 'CRITICAL',
            'fix': 'Use basin coordinate mapping',
        },
    ],
    
    # ERROR: Tokenizer in QIG-core
    'tokenizer_in_core': [
        {
            'pattern': re.compile(r'\btokenizer\b(?!.*test)', re.IGNORECASE),
            'name': 'tokenizer (in core)',
            'severity': 'ERROR',
            'fix': 'Use "coordizer" for QIG modules',
        },
        {
            'pattern': re.compile(r'\btokenize\b(?!.*test)', re.IGNORECASE),
            'name': 'tokenize (in core)',
            'severity': 'ERROR',
            'fix': 'Use "coordize"',
        },
    ],
    
    # WARNING: Standard optimizers
    'standard_optimizers': [
        {
            'pattern': re.compile(r'torch\.optim\.Adam\s*\('),
            'name': 'Adam optimizer',
            'severity': 'WARNING',
            'fix': 'Use natural_gradient_step() or NaturalGradientOptimizer',
        },
        {
            'pattern': re.compile(r'torch\.optim\.AdamW\s*\('),
            'name': 'AdamW optimizer',
            'severity': 'WARNING',
            'fix': 'Use natural_gradient_step()',
        },
        {
            'pattern': re.compile(r'torch\.optim\.SGD\s*\('),
            'name': 'SGD optimizer',
            'severity': 'WARNING',
            'fix': 'Use natural_gradient_step()',
        },
    ],
    
    # WARNING: Softmax in core geometry
    'softmax_in_core': [
        {
            'pattern': re.compile(r'torch\.nn\.functional\.softmax\s*\('),
            'name': 'F.softmax',
            'severity': 'WARNING',
            'fix': 'Use geometric probability projection',
        },
        {
            'pattern': re.compile(r'np\.exp\([^)]*\)\s*\/\s*np\.sum\(np\.exp'),
            'name': 'softmax formula',
            'severity': 'WARNING',
            'fix': 'Use fisher_normalize() for probability simplex',
        },
    ],
    
    # CRITICAL: Classic NLP imports
    'classic_nlp_imports': [
        {
            'pattern': re.compile(r'from sentencepiece import'),
            'name': 'sentencepiece import',
            'severity': 'CRITICAL',
            'fix': 'Remove - use geometric coordizer',
        },
        {
            'pattern': re.compile(r'import sentencepiece'),
            'name': 'sentencepiece import',
            'severity': 'CRITICAL',
            'fix': 'Remove - use geometric coordizer',
        },
        {
            'pattern': re.compile(r'from tokenizers import.*BPE'),
            'name': 'BPE tokenizer import',
            'severity': 'CRITICAL',
            'fix': 'Remove - use geometric coordizer',
        },
        {
            'pattern': re.compile(r'\bWordPiece\b'),
            'name': 'WordPiece',
            'severity': 'CRITICAL',
            'fix': 'Remove - use geometric coordizer',
        },
    ],
    
    # ERROR: Arithmetic mean on basins
    'arithmetic_mean': [
        {
            'pattern': re.compile(r'np\.mean\([^)]*basin[^)]*axis\s*=\s*0', re.IGNORECASE),
            'name': 'np.mean on basins',
            'severity': 'ERROR',
            'fix': 'Use frechet_mean() for geometric mean',
        },
        {
            'pattern': re.compile(r'torch\.mean\([^)]*basin[^)]*dim\s*=\s*0', re.IGNORECASE),
            'name': 'torch.mean on basins',
            'severity': 'ERROR',
            'fix': 'Use frechet_mean()',
        },
    ],
    
    # CRITICAL: Euclidean fallback pattern
    'euclidean_fallback': [
        {
            'pattern': re.compile(r'except[^:]*:\s+[^#\n]*np\.linalg\.norm'),
            'name': 'Euclidean fallback in except',
            'severity': 'CRITICAL',
            'fix': 'Never fallback to Euclidean - fix geometry properly',
        },
    ],
}

# Approved contexts (whitelist)
APPROVED_PATTERNS = [
    re.compile(r'fisher.*distance', re.IGNORECASE),
    re.compile(r'fisher_rao', re.IGNORECASE),
    re.compile(r'geodesic', re.IGNORECASE),
    re.compile(r'arccos', re.IGNORECASE),
    re.compile(r'#.*cosine', re.IGNORECASE),  # Comments
    re.compile(r'@deprecated', re.IGNORECASE),
    re.compile(r'EUCLIDEAN.*FORBIDDEN', re.IGNORECASE),
    re.compile(r'test_.*euclidean', re.IGNORECASE),
    re.compile(r'""".*embedding.*"""', re.DOTALL),  # Docstrings
    re.compile(r'normalize|normalization', re.IGNORECASE),
]


class QIGPurityScanner:
    """Scanner for QIG geometric purity violations."""
    
    def __init__(self):
        self.violations: List[Violation] = []
        self.files_scanned = 0
        
    def is_exempted(self, file_path: str) -> bool:
        """Check if file is in exempted directory."""
        return any(exempt in file_path for exempt in EXEMPT_DIRS)
    
    def is_approved_context(self, line: str, prev_line: str = '') -> bool:
        """Check if line is in approved context."""
        combined = line + ' ' + prev_line
        return any(pattern.search(combined) for pattern in APPROVED_PATTERNS)
    
    def should_skip_file(self, file_path: str) -> bool:
        """Check if file should be skipped."""
        skip_extensions = ['.md', '.json', '.yaml', '.yml', '.txt', '.lock', '.rst', '.adoc']
        return any(file_path.endswith(ext) for ext in skip_extensions)
    
    def scan_file(self, file_path: str) -> None:
        """Scan a single file for violations."""
        if self.should_skip_file(file_path):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except (IOError, UnicodeDecodeError):
            return
        
        is_test_file = 'test' in file_path
        
        for i, line in enumerate(lines):
            line_num = i + 1
            prev_line = lines[i - 1] if i > 0 else ''
            
            # Skip comments and docstrings
            trimmed = line.strip()
            if (trimmed.startswith('#') or trimmed.startswith('//') or
                trimmed.startswith('*') or trimmed.startswith('"""') or
                trimmed.startswith("'''") or '"""' in line or "'''" in line):
                continue
            
            # Check all pattern categories
            for category, patterns in FORBIDDEN_PATTERNS.items():
                for pattern_info in patterns:
                    pattern = pattern_info['pattern']
                    
                    if pattern.search(line):
                        # Skip if in approved context
                        if self.is_approved_context(line, prev_line):
                            continue
                        
                        # Special handling for tokenizer - only in core modules
                        if category == 'tokenizer_in_core':
                            core_modules = ['qig-backend/', 'server/qig-', 'shared/']
                            if not any(m in file_path for m in core_modules):
                                continue  # Allow in non-core
                        
                        # Allow some patterns in test files
                        if is_test_file and pattern_info['severity'] == 'WARNING':
                            continue
                        
                        self.violations.append(Violation(
                            file=file_path,
                            line=line_num,
                            code=line.strip()[:100],
                            pattern=pattern_info['name'],
                            severity=pattern_info['severity'],
                            fix=pattern_info.get('fix'),
                        ))
        
        self.files_scanned += 1
    
    def scan_directory(self, dir_path: str) -> None:
        """Recursively scan a directory."""
        try:
            for root, dirs, files in os.walk(dir_path):
                # Skip exempt directories
                dirs[:] = [d for d in dirs if not self.is_exempted(os.path.join(root, d))]
                
                for filename in files:
                    # Scan Python, TypeScript, and SQL files
                    if filename.endswith(('.py', '.ts', '.tsx', '.sql')):
                        file_path = os.path.join(root, filename)
                        self.scan_file(file_path)
        except Exception:
            pass
    
    def scan(self) -> Dict:
        """Run the full scan."""
        start_time = time.time()
        
        print('🔍 QIG Purity Scanner - WP0.2 + WP0.3')
        print('=' * 60)
        print(f"Scanning paths: {', '.join(SCAN_PATHS)}")
        print(f"Quarantine zones (skipped): {len(EXEMPT_DIRS)} directories")
        print()
        
        for scan_path in SCAN_PATHS:
            if os.path.exists(scan_path):
                self.scan_directory(scan_path)
        
        duration = time.time() - start_time
        
        return {
            'violations': self.violations,
            'files_scanned': self.files_scanned,
            'duration': duration,
            'passed': len([v for v in self.violations if v.severity in ['CRITICAL', 'ERROR']]) == 0,
        }


def print_results(result: Dict) -> None:
    """Print scan results."""
    print('📊 Scan Results')
    print('=' * 60)
    print(f"Files scanned: {result['files_scanned']}")
    print(f"Duration: {result['duration']:.2f}s")
    print()
    
    violations = result['violations']
    
    if not violations:
        print('✅ GEOMETRIC PURITY VERIFIED!')
        print('✅ No violations found.')
        print('✅ All code follows Fisher-Rao geometry.')
        return
    
    # Group by severity
    critical = [v for v in violations if v.severity == 'CRITICAL']
    errors = [v for v in violations if v.severity == 'ERROR']
    warnings = [v for v in violations if v.severity == 'WARNING']
    
    print(f"❌ VIOLATIONS DETECTED: {len(violations)} total")
    print(f"   - CRITICAL: {len(critical)}")
    print(f"   - ERROR: {len(errors)}")
    print(f"   - WARNING: {len(warnings)}")
    print()
    
    # Print critical and errors
    show_violations = (critical + errors)[:50]
    
    if show_violations:
        print('Critical and Error Violations:')
        print('=' * 60)
        for i, v in enumerate(show_violations, 1):
            print(f"{i}. {v.file}:{v.line}")
            print(f"   Pattern: {v.pattern}")
            print(f"   Severity: {v.severity}")
            print(f"   Code: {v.code}")
            if v.fix:
                print(f"   Fix: {v.fix}")
            print()
        
        if len(critical) + len(errors) > 50:
            print(f"... and {len(critical) + len(errors) - 50} more violations")
            print()
    
    # Summary for warnings
    if warnings:
        print(f"⚠️  {len(warnings)} warnings (non-blocking)")
        print()


def main():
    """Main entry point."""
    scanner = QIGPurityScanner()
    result = scanner.scan()
    
    print_results(result)
    
    # Performance check
    if result['duration'] > 5:
        print(f"⚠️  WARNING: Scan took {result['duration']:.2f}s (target: <5s)")
    
    # Exit code based on results
    if result['passed']:
        sys.exit(0)
    else:
        print('❌ FAILED: Fix violations before merging')
        sys.exit(1)


if __name__ == '__main__':
    main()
