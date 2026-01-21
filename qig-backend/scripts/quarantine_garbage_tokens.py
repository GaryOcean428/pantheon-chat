#!/usr/bin/env python3
"""
Quarantine Garbage Tokens - Detect and quarantine BPE artifacts and non-words

This script identifies and quarantines tokens that are:
1. BPE artifacts (random character sequences, byte-pair fragments)
2. Truncated words (incomplete words like "cryptogra", "analysi")
3. Non-English character sequences (fgzsnl, jcbhgp, kkjvdc)
4. Overly short or long tokens
5. Tokens with excessive consonants or vowels

Usage:
    python scripts/quarantine_garbage_tokens.py [--dry-run] [--report] [--batch-size 100]

Source: E8 Protocol Issue #97 (Issue-01: QFI Integrity Gate)
"""

import argparse
import logging
import os
import re
import sys
from typing import Dict, List, Tuple, Optional

try:
    import psycopg2
    from psycopg2.extras import execute_values, RealDictCursor
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("ERROR: psycopg2 not installed. Install with: pip install psycopg2-binary", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Special symbols that should never be quarantined
SPECIAL_SYMBOLS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>', '<SEP>', '<CLS>']

# Common prefixes/suffixes that indicate valid words
VALID_PREFIXES = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out', 'anti', 'de', 'non']
VALID_SUFFIXES = ['ing', 'ed', 'er', 'est', 'ly', 'ness', 'ment', 'tion', 'sion', 'ism', 'ist', 'ful', 'less']


def is_garbage_token(token: str) -> Tuple[bool, str]:
    """
    Detect garbage tokens using validation rules.
    
    Returns:
        (is_garbage, reason) tuple
    """
    # Skip special symbols
    if token in SPECIAL_SYMBOLS:
        return False, "special_symbol"
    
    # Rule 1: Empty or whitespace only
    if not token or token.isspace():
        return True, "empty_or_whitespace"
    
    # Rule 2: Too short (< 2 chars, unless common words like 'a', 'I')
    if len(token) < 2 and token.lower() not in ['a', 'i']:
        return True, "too_short"
    
    # Rule 3: Too long (> 45 chars, likely not a real word)
    if len(token) > 45:
        return True, "too_long"
    
    # Rule 4: Contains non-printable or control characters
    if any(ord(c) < 32 or ord(c) == 127 for c in token):
        return True, "control_characters"
    
    # Rule 5: Contains digits and letters mixed oddly (like "abc123def")
    if re.search(r'[a-zA-Z]+\d+[a-zA-Z]+|\d+[a-zA-Z]+\d+', token):
        return True, "mixed_digits_letters"
    
    # Rule 6: All digits or all punctuation
    if token.isdigit() or all(not c.isalnum() for c in token):
        return True, "all_digits_or_punctuation"
    
    # Rule 7: BPE artifacts (starts with ## or @@)
    if token.startswith('##') or token.startswith('@@'):
        return True, "bpe_artifact"
    
    # Rule 8: No vowels (except valid acronyms like HTTP, CSS)
    alpha_only = ''.join(c for c in token if c.isalpha())
    if len(alpha_only) > 3 and not re.search(r'[aeiouAEIOU]', alpha_only):
        # Check if it's a known acronym pattern (all caps, 2-5 chars)
        if not (token.isupper() and 2 <= len(token) <= 5):
            return True, "no_vowels"
    
    # Rule 9: Too many consonants in a row (5+)
    if re.search(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}', token):
        return True, "excessive_consonants"
    
    # Rule 10: Single character repeated excessively (5+ times)
    if re.search(r'(.)\1{4,}', token):
        return True, "excessive_repetition"
    
    # Rule 11: Non-ASCII characters (except common accented characters)
    if not token.isascii():
        # Allow common accented characters in European languages
        allowed_non_ascii = set('àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ')
        if not all(c.isascii() or c in allowed_non_ascii for c in token):
            return True, "non_ascii_characters"
    
    # Rule 12: Truncated words (ends with incomplete suffix)
    # Common truncation patterns: ends with "gra", "si", "nforc", etc.
    truncated_patterns = [
        r'^.{1,6}gra$',      # "cryptogra"
        r'^.{1,6}si$',       # "analysi"
        r'^.{1,6}nforc$',    # "enforc"
        r'^.{1,6}ment$',     # Overly short -ment words
    ]
    for pattern in truncated_patterns:
        if re.match(pattern, token, re.IGNORECASE):
            # Check if it's actually a valid short word
            if len(token) < 4:
                continue
            return True, "truncated_fragment"
    
    # Rule 13: Random character sequences (high entropy, no recognizable patterns)
    # Check for lack of common English digraphs (th, er, on, an, re, etc.)
    if len(token) > 5:
        common_digraphs = ['th', 'er', 'on', 'an', 're', 'he', 'in', 'ed', 'nd', 'ha', 'at', 'en', 'es', 'of', 'or', 'nt', 'ea', 'ti', 'to', 'it', 'st', 'io', 'le', 'is', 'ou', 'ar', 'as', 'de', 'rt', 've']
        token_lower = token.lower()
        has_common_digraph = any(dg in token_lower for dg in common_digraphs)
        
        if not has_common_digraph and len(token) > 7:
            # Very unlikely to be a real English word
            return True, "no_common_digraphs"
    
    return False, "valid"


def get_db_connection(database_url: str):
    """Get PostgreSQL connection."""
    return psycopg2.connect(database_url)


def quarantine_garbage_tokens(
    database_url: str,
    dry_run: bool = False,
    batch_size: int = 100,
    report_path: Optional[str] = None
) -> Dict:
    """
    Identify and quarantine garbage tokens.
    
    Args:
        database_url: PostgreSQL connection string
        dry_run: If True, only report what would be done (no database writes)
        batch_size: Number of tokens to process per batch
        report_path: Optional path to write detailed report
        
    Returns:
        Dict with statistics: total_scanned, quarantined, already_quarantined, reasons
    """
    stats = {
        'total_scanned': 0,
        'quarantined': 0,
        'already_quarantined': 0,
        'valid': 0,
        'reasons': {},
        'errors': []
    }
    
    quarantine_details = []
    
    try:
        conn = get_db_connection(database_url)
        
        # Step 1: Count total tokens to scan
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM coordizer_vocabulary")
            stats['total_scanned'] = cur.fetchone()[0]
        
        logger.info(f"Scanning {stats['total_scanned']} tokens for garbage detection...")
        
        # Step 2: Scan tokens in batches
        offset = 0
        while offset < stats['total_scanned']:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Fetch batch
                cur.execute("""
                    SELECT token, qfi_score, frequency, token_status, is_real_word
                    FROM coordizer_vocabulary
                    ORDER BY token
                    LIMIT %s OFFSET %s
                """, (batch_size, offset))
                
                batch = cur.fetchall()
                
                if not batch:
                    break
                
                # Analyze each token
                to_quarantine = []
                for row in batch:
                    token = row['token']
                    token_status = row['token_status']
                    
                    # Skip already quarantined tokens
                    if token_status == 'quarantined':
                        stats['already_quarantined'] += 1
                        continue
                    
                    # Check if garbage
                    is_garbage, reason = is_garbage_token(token)
                    
                    if is_garbage:
                        stats['quarantined'] += 1
                        stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
                        
                        to_quarantine.append({
                            'token': token,
                            'reason': reason,
                            'qfi_score': row['qfi_score'],
                            'frequency': row['frequency']
                        })
                        
                        quarantine_details.append({
                            'token': token,
                            'reason': reason,
                            'qfi_score': row['qfi_score'],
                            'frequency': row['frequency'],
                            'was_real_word': row['is_real_word']
                        })
                    else:
                        stats['valid'] += 1
                
                # Execute quarantine (if not dry run)
                if not dry_run and to_quarantine:
                    with conn.cursor() as write_cur:
                        # Insert into quarantine table
                        quarantine_inserts = [
                            (
                                item['token'],
                                item['reason'],
                                item['frequency'],
                                item['qfi_score']
                            )
                            for item in to_quarantine
                        ]
                        
                        execute_values(
                            write_cur,
                            """
                            INSERT INTO coordizer_vocabulary_quarantine 
                                (token, reason, frequency, original_qfi_score, quarantined_at)
                            VALUES %s
                            ON CONFLICT (token) DO UPDATE SET
                                reason = EXCLUDED.reason,
                                original_qfi_score = EXCLUDED.original_qfi_score,
                                quarantined_at = NOW()
                            """,
                            quarantine_inserts
                        )
                        
                        # Mark as quarantined in main table
                        tokens_to_mark = [item['token'] for item in to_quarantine]
                        write_cur.execute("""
                            UPDATE coordizer_vocabulary
                            SET token_status = 'quarantined',
                                is_generation_eligible = FALSE,
                                is_real_word = FALSE
                            WHERE token = ANY(%s)
                        """, (tokens_to_mark,))
                        
                        conn.commit()
                
                logger.info(
                    f"Processed batch {offset//batch_size + 1}: "
                    f"Found {len(to_quarantine)} garbage tokens "
                    f"({stats['quarantined']}/{stats['total_scanned']} total)"
                )
            
            offset += batch_size
        
        conn.close()
        
        # Generate report
        logger.info("=" * 70)
        logger.info("GARBAGE TOKEN DETECTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total tokens scanned: {stats['total_scanned']}")
        logger.info(f"Valid tokens: {stats['valid']}")
        logger.info(f"Garbage tokens found: {stats['quarantined']}")
        logger.info(f"Already quarantined: {stats['already_quarantined']}")
        
        if stats['reasons']:
            logger.info("\nBreakdown by reason:")
            for reason, count in sorted(stats['reasons'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {reason}: {count}")
        
        if dry_run:
            logger.info("\nDRY RUN: No changes were made to the database")
        else:
            logger.info(f"\nQuarantined {stats['quarantined']} garbage tokens")
        
        logger.info("=" * 70)
        
        # Write detailed report if requested
        if report_path and quarantine_details:
            with open(report_path, 'w') as f:
                f.write("# Garbage Token Quarantine Report\n\n")
                f.write(f"Total scanned: {stats['total_scanned']}\n")
                f.write(f"Quarantined: {stats['quarantined']}\n\n")
                f.write("## Quarantined Tokens\n\n")
                f.write("| Token | Reason | QFI Score | Frequency | Was Real Word |\n")
                f.write("|-------|--------|-----------|-----------|---------------|\n")
                
                for item in sorted(quarantine_details, key=lambda x: x['frequency'], reverse=True):
                    f.write(
                        f"| `{item['token']}` | {item['reason']} | "
                        f"{item['qfi_score']:.4f if item['qfi_score'] else 'NULL'} | "
                        f"{item['frequency']} | {item['was_real_word']} |\n"
                    )
            
            logger.info(f"\nDetailed report written to: {report_path}")
    
    except Exception as e:
        logger.error(f"Garbage token detection failed: {e}")
        stats['errors'].append(str(e))
        raise
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Detect and quarantine garbage tokens (BPE artifacts, non-words)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help="Number of tokens to process per batch (default: 100)"
    )
    parser.add_argument(
        '--database-url',
        type=str,
        default=None,
        help="PostgreSQL connection URL (default: from DATABASE_URL env var)"
    )
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help="Path to write detailed quarantine report (optional)"
    )
    
    args = parser.parse_args()
    
    database_url = args.database_url or os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("Database URL not provided (use --database-url or set DATABASE_URL)")
        sys.exit(1)
    
    logger.info("Starting garbage token detection...")
    if args.dry_run:
        logger.info("DRY RUN MODE: No changes will be made")
    
    try:
        stats = quarantine_garbage_tokens(
            database_url=database_url,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            report_path=args.report
        )
        
        # Exit with status code based on results
        if stats['errors']:
            sys.exit(1)
        else:
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
