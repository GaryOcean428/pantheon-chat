#!/usr/bin/env python3
"""
Quarantine Garbage Tokens Script
=================================

Identifies and quarantines garbage tokens from coordizer_vocabulary:
- BPE tokenizer artifacts (##, @@, ▁, </w>)
- Non-words (random character sequences)
- Invalid entries (empty, too long, special tokens)

This script moves garbage tokens to coordizer_vocabulary_quarantine table
while preserving the original data for forensics.

Usage:
    python quarantine_garbage_tokens.py [--dry-run] [--report output.txt] [--aggressive]

Args:
    --dry-run: Show what would be quarantined without making changes
    --report: Output file for quarantine list
    --aggressive: Use stricter filtering rules

Author: Copilot AI Agent
Date: 2026-01-20
Issue: GaryOcean428/pantheon-chat#97 (E8 Protocol Issue-01)
Reference: docs/10-e8-protocol/issues/20260119-issue-97-qfi-integrity-gate-remediation-1.00W.md
"""

import argparse
import logging
import os
import re
import sys
from typing import List, Dict, Tuple
import psycopg2
from psycopg2.extensions import connection as PgConnection

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Garbage detection patterns
BPE_PATTERNS = [
    r'^##',      # BERT-style subword prefix
    r'@@$',      # SentencePiece suffix
    r'▁',        # SentencePiece space marker
    r'</w>',     # GPT-2 end-of-word
    r'<\|.*?\|>',  # Special tokens
]

# Known valid acronyms without vowels
KNOWN_ACRONYMS = {'phd', 'css', 'html', 'xml', 'sql', 'npm', 'llm', 'qig', 'api', 'url', 'cdn'}


def is_bpe_artifact(word: str) -> bool:
    """Check if word contains BPE tokenizer artifacts."""
    for pattern in BPE_PATTERNS:
        if re.search(pattern, word):
            return True
    return False


def is_garbage_nonword(word: str, aggressive: bool = False) -> bool:
    """
    Check if word is a non-word (random character sequence).
    
    Detection rules:
    - All same character repeated
    - No vowels (except known acronyms)
    - Excessive repetition (same char 4+ times)
    - Random byte sequences
    
    Args:
        word: Word to check
        aggressive: If True, use stricter rules
        
    Returns:
        True if word is garbage
    """
    word_lower = word.lower()
    
    # All same character
    if len(set(word_lower)) == 1 and len(word_lower) > 1:
        return True
    
    # No vowels (except known acronyms)
    if not re.search('[aeiou]', word_lower, re.I):
        if word_lower not in KNOWN_ACRONYMS:
            return True
    
    # Excessive repetition
    if re.search(r'(.)\1{3,}', word_lower):  # Same char 4+ times
        return True
    
    # Aggressive: mostly consonants
    if aggressive:
        vowel_count = len(re.findall('[aeiou]', word_lower))
        consonant_count = len(re.findall('[bcdfghjklmnpqrstvwxyz]', word_lower))
        if consonant_count > 0 and vowel_count / (consonant_count + vowel_count) < 0.2:
            return True
    
    return False


def is_invalid_entry(word: str) -> bool:
    """
    Check if word is an invalid entry.
    
    Detection rules:
    - Empty or whitespace only
    - Too short (< 2 chars) or too long (> 50 chars)
    - Starts/ends with special characters
    - Special token markers
    
    Args:
        word: Word to check
        
    Returns:
        True if word is invalid
    """
    if not word or word.isspace():
        return True
    
    if len(word) < 2 or len(word) > 50:
        return True
    
    # Special token markers
    if word.startswith('<') and word.endswith('>'):
        return True
    
    if word.startswith('[') and word.endswith(']'):
        return True
    
    # Excessive special characters
    special_char_count = len(re.findall(r'[^a-zA-Z0-9\s\-\']', word))
    if special_char_count > len(word) * 0.5:
        return True
    
    return False


def classify_garbage(word: str, aggressive: bool = False) -> Tuple[bool, str]:
    """
    Classify if word is garbage and provide reason.
    
    Args:
        word: Word to classify
        aggressive: If True, use stricter rules
        
    Returns:
        (is_garbage, reason) tuple
    """
    if is_bpe_artifact(word):
        return (True, "BPE artifact")
    
    if is_invalid_entry(word):
        return (True, "Invalid entry")
    
    if is_garbage_nonword(word, aggressive=aggressive):
        return (True, "Non-word")
    
    return (False, "")


def scan_garbage_tokens(
    db_conn: PgConnection,
    aggressive: bool = False
) -> List[Dict[str, any]]:
    """
    Scan coordizer_vocabulary for garbage tokens.
    
    Args:
        db_conn: PostgreSQL connection
        aggressive: If True, use stricter detection rules
        
    Returns:
        List of dicts with keys: token_id, token, reason
    """
    garbage_tokens = []
    
    with db_conn.cursor() as cursor:
        cursor.execute("""
            SELECT token_id, token
            FROM coordizer_vocabulary
            WHERE token_status = 'active'
        """)
        
        for row in cursor.fetchall():
            token_id, token = row
            is_garbage, reason = classify_garbage(token, aggressive=aggressive)
            
            if is_garbage:
                garbage_tokens.append({
                    'token_id': token_id,
                    'token': token,
                    'reason': reason
                })
    
    return garbage_tokens


def quarantine_tokens(
    db_conn: PgConnection,
    garbage_tokens: List[Dict[str, any]],
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Quarantine garbage tokens.
    
    Args:
        db_conn: PostgreSQL connection
        garbage_tokens: List of tokens to quarantine
        dry_run: If True, don't make database changes
        
    Returns:
        Dict with counts: quarantined, failed
    """
    quarantined = 0
    failed = 0
    
    if dry_run:
        logger.info(f"DRY RUN: Would quarantine {len(garbage_tokens)} tokens")
        return {'quarantined': len(garbage_tokens), 'failed': 0}
    
    with db_conn.cursor() as cursor:
        for token_data in garbage_tokens:
            try:
                # Update token_status to quarantined
                cursor.execute("""
                    UPDATE coordizer_vocabulary
                    SET token_status = 'quarantined',
                        updated_at = NOW()
                    WHERE token_id = %s
                    RETURNING token_id
                """, (token_data['token_id'],))
                
                if cursor.fetchone():
                    quarantined += 1
                    logger.debug(f"Quarantined: {token_data['token']} (reason: {token_data['reason']})")
                else:
                    failed += 1
                    logger.warning(f"Failed to quarantine: {token_data['token']}")
                
            except Exception as e:
                failed += 1
                logger.error(f"Error quarantining {token_data['token']}: {e}")
    
    db_conn.commit()
    
    return {'quarantined': quarantined, 'failed': failed}


def generate_report(
    garbage_tokens: List[Dict[str, any]],
    output_file: str = None
):
    """
    Generate human-readable report of garbage tokens.
    
    Args:
        garbage_tokens: List of tokens to report
        output_file: Optional output file path
    """
    # Group by reason
    by_reason = {}
    for token in garbage_tokens:
        reason = token['reason']
        if reason not in by_reason:
            by_reason[reason] = []
        by_reason[reason].append(token['token'])
    
    # Generate report
    lines = []
    lines.append("=" * 80)
    lines.append("GARBAGE TOKEN QUARANTINE REPORT")
    lines.append("=" * 80)
    lines.append(f"Total garbage tokens: {len(garbage_tokens)}")
    lines.append("")
    
    for reason, tokens in sorted(by_reason.items()):
        lines.append(f"\n{reason.upper()} ({len(tokens)} tokens)")
        lines.append("-" * 80)
        for token in sorted(tokens)[:50]:  # Show first 50
            lines.append(f"  - {token}")
        if len(tokens) > 50:
            lines.append(f"  ... and {len(tokens) - 50} more")
    
    lines.append("\n" + "=" * 80)
    
    report = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Report written to {output_file}")
    else:
        print(report)


def main():
    parser = argparse.ArgumentParser(description='Quarantine garbage tokens from coordizer_vocabulary')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--report', type=str, help='Output file for quarantine report')
    parser.add_argument('--aggressive', action='store_true', help='Use stricter detection rules')
    parser.add_argument('--db-url', type=str, help='Database URL (or use DATABASE_URL env var)')
    
    args = parser.parse_args()
    
    # Get database URL
    db_url = args.db_url or os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("ERROR: Database URL not provided. Set DATABASE_URL or use --db-url")
        sys.exit(1)
    
    # Connect to database
    try:
        conn = psycopg2.connect(db_url)
        logger.info("Connected to database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    # Scan for garbage tokens
    logger.info("Scanning for garbage tokens...")
    garbage_tokens = scan_garbage_tokens(conn, aggressive=args.aggressive)
    logger.info(f"Found {len(garbage_tokens)} garbage tokens")
    
    # Generate report
    if args.report or args.dry_run:
        generate_report(garbage_tokens, args.report)
    
    # Quarantine tokens
    if not args.dry_run:
        logger.info("Quarantining tokens...")
        result = quarantine_tokens(conn, garbage_tokens, dry_run=False)
        logger.info(f"Quarantined: {result['quarantined']}, Failed: {result['failed']}")
    else:
        logger.info("DRY RUN: No changes made to database")
    
    conn.close()


if __name__ == '__main__':
    main()
