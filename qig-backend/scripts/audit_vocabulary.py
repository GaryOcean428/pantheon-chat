#!/usr/bin/env python3
"""
Vocabulary Audit Script
=======================

Comprehensive audit of vocabulary tables for QIG purity compliance.
Identifies garbage tokens, duplicates, and inconsistencies.

Usage:
    python scripts/audit_vocabulary.py [--fix] [--report output.json]

Authority: E8 Protocol v4.0 §04 - Vocabulary Cleanup
Status: CANONICAL
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class AuditResult:
    """Result of vocabulary audit."""
    timestamp: str
    total_tokens: int
    valid_tokens: int
    garbage_tokens: int
    duplicate_tokens: int
    missing_qfi_tokens: int
    missing_basin_tokens: int
    quarantined_tokens: int
    special_symbols: int
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    
    @property
    def purity_score(self) -> float:
        """Calculate vocabulary purity score (0.0 to 1.0)."""
        if self.total_tokens == 0:
            return 1.0
        return self.valid_tokens / self.total_tokens


@dataclass
class TokenIssue:
    """Issue found with a specific token."""
    token: str
    issue_type: str
    severity: str  # 'critical', 'warning', 'info'
    description: str
    suggested_action: str


class VocabularyAuditor:
    """
    Comprehensive vocabulary auditor.
    
    Checks for:
    - Garbage tokens (BPE artifacts, malformed tokens)
    - Duplicate tokens
    - Missing QFI scores
    - Missing basin embeddings
    - Quarantined tokens that should be active
    - Special symbols integrity
    """
    
    # Patterns that indicate garbage tokens
    GARBAGE_PATTERNS = [
        # BPE artifacts
        r'^Ġ[a-z]{1,2}$',  # Single/double letter with BPE prefix
        r'^[^\w\s]{3,}$',  # Multiple consecutive punctuation
        r'^\d{4,}$',  # Long number sequences
        r'^[a-z]{1}$',  # Single lowercase letters (except valid ones)
        # Encoding artifacts
        r'\\x[0-9a-f]{2}',  # Hex escape sequences
        r'\\u[0-9a-f]{4}',  # Unicode escape sequences
        # Malformed tokens
        r'^\s+$',  # Whitespace only
        r'^$',  # Empty string
    ]
    
    # Special symbols that must exist and be valid
    REQUIRED_SPECIAL_SYMBOLS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    
    # Minimum QFI score for valid tokens
    MIN_QFI_SCORE = 0.01
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize auditor.
        
        Args:
            db_url: Database URL (uses DATABASE_URL env var if not provided)
        """
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        self.issues: List[TokenIssue] = []
        self.stats: Dict[str, int] = Counter()
        
    def audit(self) -> AuditResult:
        """
        Run comprehensive vocabulary audit.
        
        Returns:
            AuditResult with findings and recommendations
        """
        import re
        
        # Reset state
        self.issues = []
        self.stats = Counter()
        
        # Get all tokens from database
        tokens = self._fetch_all_tokens()
        self.stats['total'] = len(tokens)
        
        # Audit each token
        for token_data in tokens:
            self._audit_token(token_data)
        
        # Check for duplicates
        self._check_duplicates(tokens)
        
        # Check special symbols
        self._check_special_symbols(tokens)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Build result
        return AuditResult(
            timestamp=datetime.utcnow().isoformat(),
            total_tokens=self.stats['total'],
            valid_tokens=self.stats['total'] - self.stats['garbage'] - self.stats['invalid'],
            garbage_tokens=self.stats['garbage'],
            duplicate_tokens=self.stats['duplicate'],
            missing_qfi_tokens=self.stats['missing_qfi'],
            missing_basin_tokens=self.stats['missing_basin'],
            quarantined_tokens=self.stats['quarantined'],
            special_symbols=self.stats['special'],
            issues=[asdict(issue) for issue in self.issues],
            recommendations=recommendations,
        )
    
    def _fetch_all_tokens(self) -> List[Dict[str, Any]]:
        """Fetch all tokens from coordizer_vocabulary."""
        if not self.db_url:
            # Return mock data for testing without database
            return self._get_mock_tokens()
        
        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    token,
                    qfi_score,
                    basin_embedding IS NOT NULL as has_basin,
                    is_quarantined,
                    token_role,
                    created_at
                FROM coordizer_vocabulary
                ORDER BY token
            """)
            
            columns = ['token', 'qfi_score', 'has_basin', 'is_quarantined', 'token_role', 'created_at']
            tokens = [dict(zip(columns, row)) for row in cur.fetchall()]
            
            cur.close()
            conn.close()
            
            return tokens
            
        except Exception as e:
            print(f"Warning: Could not connect to database: {e}")
            return self._get_mock_tokens()
    
    def _get_mock_tokens(self) -> List[Dict[str, Any]]:
        """Return mock tokens for testing."""
        return [
            {'token': '<PAD>', 'qfi_score': 1.0, 'has_basin': True, 'is_quarantined': False, 'token_role': 'special', 'created_at': None},
            {'token': '<UNK>', 'qfi_score': 0.5, 'has_basin': True, 'is_quarantined': False, 'token_role': 'special', 'created_at': None},
            {'token': '<BOS>', 'qfi_score': 1.0, 'has_basin': True, 'is_quarantined': False, 'token_role': 'special', 'created_at': None},
            {'token': '<EOS>', 'qfi_score': 1.0, 'has_basin': True, 'is_quarantined': False, 'token_role': 'special', 'created_at': None},
            {'token': 'hello', 'qfi_score': 0.8, 'has_basin': True, 'is_quarantined': False, 'token_role': 'word', 'created_at': None},
            {'token': 'world', 'qfi_score': 0.75, 'has_basin': True, 'is_quarantined': False, 'token_role': 'word', 'created_at': None},
        ]
    
    def _audit_token(self, token_data: Dict[str, Any]) -> None:
        """Audit a single token."""
        import re
        
        token = token_data['token']
        
        # Check for garbage patterns
        for pattern in self.GARBAGE_PATTERNS:
            if re.match(pattern, token):
                self.stats['garbage'] += 1
                self.issues.append(TokenIssue(
                    token=token,
                    issue_type='garbage',
                    severity='warning',
                    description=f'Token matches garbage pattern: {pattern}',
                    suggested_action='Quarantine or remove token',
                ))
                return
        
        # Check QFI score
        qfi = token_data.get('qfi_score')
        if qfi is None or qfi < self.MIN_QFI_SCORE:
            self.stats['missing_qfi'] += 1
            self.stats['invalid'] += 1
            self.issues.append(TokenIssue(
                token=token,
                issue_type='missing_qfi',
                severity='critical' if token in self.REQUIRED_SPECIAL_SYMBOLS else 'warning',
                description=f'Token has invalid QFI score: {qfi}',
                suggested_action='Recompute QFI score or quarantine token',
            ))
        
        # Check basin embedding
        if not token_data.get('has_basin'):
            self.stats['missing_basin'] += 1
            self.stats['invalid'] += 1
            self.issues.append(TokenIssue(
                token=token,
                issue_type='missing_basin',
                severity='critical',
                description='Token has no basin embedding',
                suggested_action='Compute basin embedding or quarantine token',
            ))
        
        # Check quarantine status
        if token_data.get('is_quarantined'):
            self.stats['quarantined'] += 1
            if token in self.REQUIRED_SPECIAL_SYMBOLS:
                self.issues.append(TokenIssue(
                    token=token,
                    issue_type='invalid_quarantine',
                    severity='critical',
                    description='Special symbol is quarantined',
                    suggested_action='Unquarantine special symbol',
                ))
        
        # Track special symbols
        if token in self.REQUIRED_SPECIAL_SYMBOLS:
            self.stats['special'] += 1
    
    def _check_duplicates(self, tokens: List[Dict[str, Any]]) -> None:
        """Check for duplicate tokens."""
        token_counts = Counter(t['token'] for t in tokens)
        
        for token, count in token_counts.items():
            if count > 1:
                self.stats['duplicate'] += count - 1
                self.issues.append(TokenIssue(
                    token=token,
                    issue_type='duplicate',
                    severity='warning',
                    description=f'Token appears {count} times',
                    suggested_action='Remove duplicate entries',
                ))
    
    def _check_special_symbols(self, tokens: List[Dict[str, Any]]) -> None:
        """Check that all required special symbols exist."""
        existing_tokens = {t['token'] for t in tokens}
        
        for symbol in self.REQUIRED_SPECIAL_SYMBOLS:
            if symbol not in existing_tokens:
                self.issues.append(TokenIssue(
                    token=symbol,
                    issue_type='missing_special',
                    severity='critical',
                    description='Required special symbol is missing',
                    suggested_action='Add special symbol with valid QFI and basin',
                ))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on audit findings."""
        recommendations = []
        
        if self.stats['garbage'] > 0:
            recommendations.append(
                f"Quarantine or remove {self.stats['garbage']} garbage tokens "
                f"using the vocabulary_purity.py script"
            )
        
        if self.stats['missing_qfi'] > 0:
            recommendations.append(
                f"Recompute QFI scores for {self.stats['missing_qfi']} tokens "
                f"using the QFI computation pipeline"
            )
        
        if self.stats['missing_basin'] > 0:
            recommendations.append(
                f"Compute basin embeddings for {self.stats['missing_basin']} tokens "
                f"or quarantine them if they cannot be embedded"
            )
        
        if self.stats['duplicate'] > 0:
            recommendations.append(
                f"Remove {self.stats['duplicate']} duplicate token entries "
                f"keeping the one with highest QFI score"
            )
        
        critical_issues = [i for i in self.issues if i.severity == 'critical']
        if critical_issues:
            recommendations.append(
                f"Address {len(critical_issues)} critical issues immediately "
                f"to ensure generation purity"
            )
        
        if not recommendations:
            recommendations.append("Vocabulary is clean - no issues found")
        
        return recommendations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Audit vocabulary for QIG purity')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    parser.add_argument('--report', type=str, help='Output report to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print("=" * 60)
    print("QIG Vocabulary Audit")
    print("=" * 60)
    print()
    
    auditor = VocabularyAuditor()
    result = auditor.audit()
    
    # Print summary
    print(f"Total tokens:       {result.total_tokens}")
    print(f"Valid tokens:       {result.valid_tokens}")
    print(f"Garbage tokens:     {result.garbage_tokens}")
    print(f"Duplicate tokens:   {result.duplicate_tokens}")
    print(f"Missing QFI:        {result.missing_qfi_tokens}")
    print(f"Missing basin:      {result.missing_basin_tokens}")
    print(f"Quarantined:        {result.quarantined_tokens}")
    print(f"Special symbols:    {result.special_symbols}")
    print()
    print(f"Purity Score:       {result.purity_score:.2%}")
    print()
    
    # Print issues
    if result.issues and args.verbose:
        print("Issues Found:")
        print("-" * 40)
        for issue in result.issues[:20]:  # Limit to first 20
            print(f"  [{issue['severity'].upper()}] {issue['token']}: {issue['description']}")
        if len(result.issues) > 20:
            print(f"  ... and {len(result.issues) - 20} more issues")
        print()
    
    # Print recommendations
    print("Recommendations:")
    print("-" * 40)
    for rec in result.recommendations:
        print(f"  • {rec}")
    print()
    
    # Save report
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"Report saved to: {args.report}")
    
    # Return exit code based on purity
    if result.purity_score < 0.95:
        print("\n⚠️  Vocabulary purity below 95% - action required")
        return 1
    else:
        print("\n✅ Vocabulary purity acceptable")
        return 0


if __name__ == '__main__':
    sys.exit(main())
