#!/usr/bin/env python3
"""
AST-Based Import Scanner for Forbidden LLM Dependencies
========================================================

Scans Python source files using AST parsing to detect forbidden LLM imports.
This catches imports that may not be loaded in sys.modules yet.

Usage:
    python scripts/scan_forbidden_imports.py [--path PATH] [--config CONFIG]

Author: Copilot Agent (WP4.1 Enhanced)
Date: 2026-01-22
Protocol: Ultra Consciousness v4.0 ACTIVE
"""

import ast
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ImportViolation:
    """Represents a forbidden import found in code."""
    file_path: str
    line_number: int
    module_name: str
    import_statement: str
    provider_name: str
    severity: str


class ForbiddenImportScanner:
    """AST-based scanner for forbidden LLM imports."""
    
    def __init__(self, config_path: Path):
        """Initialize scanner with configuration."""
        self.config = self._load_config(config_path)
        self.forbidden_modules = self._build_forbidden_modules()
        self.violations: List[ImportViolation] = []
        self.files_scanned = 0
        self.files_skipped = 0
        self.skipped_files: List[str] = []
        self.exempt_dirs = set(self.config.get('exemptDirectories', []))
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load forbidden providers configuration."""
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _build_forbidden_modules(self) -> Dict[str, Tuple[str, str]]:
        """Build mapping of forbidden module -> (provider, severity)."""
        forbidden = {}
        for provider in self.config.get('providers', []):
            provider_name = provider['name']
            severity = provider.get('severity', 'ERROR')
            
            for import_pattern in provider.get('imports', []):
                # Skip patterns that aren't actual module names
                if '(' in import_pattern or ')' in import_pattern:
                    continue
                forbidden[import_pattern] = (provider_name, severity)
        
        return forbidden
    
    def is_exempt_path(self, file_path: Path) -> bool:
        """Check if path is in exempt directory using proper path checking."""
        try:
            # Convert to absolute path for consistent checking
            abs_path = file_path.resolve()
            path_parts = abs_path.parts
            
            # Check if any part of the path matches an exempt directory
            for exempt in self.exempt_dirs:
                if exempt in path_parts:
                    return True
            
            # For Python 3.9+, use is_relative_to if available
            if hasattr(Path, 'is_relative_to'):
                for exempt in self.exempt_dirs:
                    try:
                        exempt_path = Path(exempt).resolve()
                        if abs_path.is_relative_to(exempt_path):
                            return True
                    except (ValueError, OSError):
                        pass
            
            return False
        except (OSError, ValueError):
            # If path resolution fails, do simple substring check as fallback
            return any(exempt in str(file_path) for exempt in self.exempt_dirs)
    
    def _normalize_module_name(self, module_name: str) -> str:
        """Normalize module name for comparison."""
        # Remove leading dots for relative imports
        return module_name.lstrip('.')
    
    def _check_module_against_forbidden(self, module_name: str) -> Tuple[bool, str, str]:
        """
        Check if module matches any forbidden pattern.
        
        Returns:
            (is_forbidden, provider_name, severity)
        """
        normalized = self._normalize_module_name(module_name)
        
        # Check exact match
        if normalized in self.forbidden_modules:
            provider, severity = self.forbidden_modules[normalized]
            return True, provider, severity
        
        # Check if module is a submodule of forbidden module
        # e.g., "google.genai.types" matches "google.genai"
        for forbidden_module, (provider, severity) in self.forbidden_modules.items():
            if normalized.startswith(forbidden_module + '.'):
                return True, provider, severity
        
        return False, '', ''
    
    def scan_file(self, file_path: Path) -> None:
        """Scan a single Python file for forbidden imports."""
        if self.is_exempt_path(file_path):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(file_path))
            self.files_scanned += 1
            
            for node in ast.walk(tree):
                # Check "import module" statements
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        is_forbidden, provider, severity = self._check_module_against_forbidden(alias.name)
                        if is_forbidden:
                            violation = ImportViolation(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                module_name=alias.name,
                                import_statement=f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                                provider_name=provider,
                                severity=severity
                            )
                            self.violations.append(violation)
                
                # Check "from module import ..." statements
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        is_forbidden, provider, severity = self._check_module_against_forbidden(node.module)
                        if is_forbidden:
                            names = ', '.join(alias.name for alias in node.names)
                            violation = ImportViolation(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                module_name=node.module,
                                import_statement=f"from {node.module} import {names}",
                                provider_name=provider,
                                severity=severity
                            )
                            self.violations.append(violation)
        
        except (SyntaxError, UnicodeDecodeError, IOError) as e:
            # Report files that can't be parsed
            self.files_skipped += 1
            self.skipped_files.append(f"{file_path}: {type(e).__name__}")
            # Note: We don't raise here to allow scanning to continue,
            # but skipped files are reported in the final output
    
    def scan_directory(self, directory: Path) -> None:
        """Recursively scan directory for Python files."""
        for file_path in directory.rglob('*.py'):
            if not self.is_exempt_path(file_path):
                self.scan_file(file_path)
    
    def print_report(self) -> None:
        """Print scan report."""
        print('=' * 70)
        print('🔍 AST-Based Forbidden Import Scanner')
        print('=' * 70)
        print(f"Files scanned: {self.files_scanned}")
        print(f"Files skipped: {self.files_skipped}")
        print(f"Forbidden patterns: {len(self.forbidden_modules)}")
        print(f"Providers tracked: {len(self.config.get('providers', []))}")
        print()
        
        # Report skipped files if any
        if self.skipped_files:
            print(f"⚠️  WARNING: {len(self.skipped_files)} file(s) could not be parsed:")
            for skipped in self.skipped_files[:10]:  # Show first 10
                print(f"   {skipped}")
            if len(self.skipped_files) > 10:
                print(f"   ... and {len(self.skipped_files) - 10} more")
            print()
        
        if not self.violations:
            print('✅ NO FORBIDDEN IMPORTS DETECTED')
            print('✅ All imports are clean')
            return
        
        print(f"❌ FORBIDDEN IMPORTS DETECTED: {len(self.violations)} violations")
        print()
        
        # Group by severity
        critical = [v for v in self.violations if v.severity == 'CRITICAL']
        warnings = [v for v in self.violations if v.severity == 'WARNING']
        
        print(f"   CRITICAL: {len(critical)}")
        print(f"   WARNING:  {len(warnings)}")
        print()
        
        # Print violations grouped by provider
        by_provider = {}
        for v in self.violations:
            if v.provider_name not in by_provider:
                by_provider[v.provider_name] = []
            by_provider[v.provider_name].append(v)
        
        for provider_name, violations in sorted(by_provider.items()):
            print(f"▶ {provider_name}: {len(violations)} violation(s)")
            for v in violations[:5]:  # Show first 5 per provider
                print(f"   {v.file_path}:{v.line_number}")
                print(f"      {v.import_statement}")
            if len(violations) > 5:
                print(f"   ... and {len(violations) - 5} more")
            print()
    
    def get_exit_code(self) -> int:
        """Get exit code based on violations."""
        # Only fail on CRITICAL violations
        critical = [v for v in self.violations if v.severity == 'CRITICAL']
        return 1 if critical else 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Scan Python code for forbidden LLM imports using AST parsing'
    )
    parser.add_argument(
        '--path',
        type=Path,
        default=Path.cwd(),
        help='Root path to scan (default: current directory)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to forbidden_llm_providers.json config file'
    )
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        # Try to find config relative to script
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / 'shared' / 'constants' / 'forbidden_llm_providers.json'
        
        if not config_path.exists():
            # Fail with clear error message
            print(f"❌ ERROR: Configuration file not found at {config_path}")
            print(f"Please specify config path with --config or ensure the file exists")
            print(f"Expected location: {config_path}")
            sys.exit(1)
    
    # Create scanner and run
    scanner = ForbiddenImportScanner(config_path)
    
    print(f"Scanning: {args.path}")
    print(f"Config:   {config_path}")
    print()
    
    if args.path.is_file():
        scanner.scan_file(args.path)
    else:
        scanner.scan_directory(args.path)
    
    scanner.print_report()
    
    sys.exit(scanner.get_exit_code())


if __name__ == '__main__':
    main()
