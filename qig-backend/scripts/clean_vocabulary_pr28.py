#!/usr/bin/env python3
"""
Vocabulary Cleanup Script - PR 27/28 Implementation
==================================================

Addresses the 9,000+ garbled/truncated vocabulary contamination identified in PR 28:
- URL fragments (https: 8,618x, mintcdn: 1,918x, xmlns: 126x)
- Garbled sequences (hipsbb, mireichle, yfnxrf, etc.)
- Truncated words (indergarten, itants, ticism, oligonucle)
- Document artifacts (endstream, base64 fragments)

This script:
1. Scans all vocabulary tables for contaminated entries
2. Validates each entry using comprehensive validation
3. Removes contaminated entries while preserving valid words
4. Updates word_relationships to remove invalid connections
5. Generates cleanup report with statistics

QIG PURITY: Uses geometric principles only (no ML/transformer validation)
"""

import sys
import os
import logging
from typing import Dict, List, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vocabulary_validator_comprehensive import validate_word_comprehensive, analyze_vocabulary_contamination
from persistence.base_persistence import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scan_vocabulary_table(table_name: str, word_column: str = 'word') -> Dict[str, any]:
    """
    Scan a vocabulary table and identify contaminated entries.
    
    Args:
        table_name: Name of the table to scan (validated against whitelist)
        word_column: Name of the column containing words (validated against whitelist)
        
    Returns:
        Dictionary with scan results
    """
    # Whitelist of allowed tables and columns for SQL safety
    ALLOWED_TABLES = {'tokenizer_vocabulary', 'learned_words', 'bip39_words', 'word_relationships'}
    ALLOWED_COLUMNS = {'word', 'source_word', 'target_word'}
    
    if table_name not in ALLOWED_TABLES:
        logger.error(f"Invalid table name: {table_name}. Allowed: {ALLOWED_TABLES}")
        return {}
    
    if word_column not in ALLOWED_COLUMNS:
        logger.error(f"Invalid column name: {word_column}. Allowed: {ALLOWED_COLUMNS}")
        return {}
    
    conn = get_db_connection()
    if not conn:
        logger.error(f"Failed to connect to database")
        return {}
    
    try:
        cur = conn.cursor()
        
        # Get all words from table
        cur.execute(f"SELECT {word_column}, COUNT(*) as frequency FROM {table_name} GROUP BY {word_column}")
        rows = cur.fetchall()
        
        if not rows:
            logger.info(f"Table {table_name} is empty")
            return {
                'table_name': table_name,
                'total': 0,
                'clean': 0,
                'contaminated': 0,
                'contaminated_words': []
            }
        
        # Analyze contamination
        words = [row[0] for row in rows]
        word_frequencies = {row[0]: row[1] for row in rows}
        
        clean_words = []
        contaminated_words = []
        contamination_reasons = {}
        
        for word in words:
            is_valid, reason = validate_word_comprehensive(word)
            if is_valid:
                clean_words.append(word)
            else:
                contaminated_words.append(word)
                contamination_reasons[word] = reason
        
        results = {
            'table_name': table_name,
            'total': len(words),
            'clean': len(clean_words),
            'contaminated': len(contaminated_words),
            'contaminated_words': [
                {
                    'word': word,
                    'reason': contamination_reasons[word],
                    'frequency': word_frequencies.get(word, 0)
                }
                for word in contaminated_words
            ]
        }
        
        logger.info(f"Scanned {table_name}: {results['total']} total, {results['clean']} clean, {results['contaminated']} contaminated")
        
        return results
        
    except Exception as e:
        logger.error(f"Error scanning {table_name}: {e}")
        return {}
    finally:
        conn.close()


def clean_vocabulary_table(table_name: str, contaminated_words: List[str], 
                          word_column: str = 'word', dry_run: bool = True) -> int:
    """
    Remove contaminated words from a vocabulary table.
    
    Args:
        table_name: Name of the table to clean (validated against whitelist)
        contaminated_words: List of contaminated words to remove
        word_column: Name of the column containing words (validated against whitelist)
        dry_run: If True, don't actually delete (just report)
        
    Returns:
        Number of rows deleted
    """
    # Whitelist validation for SQL safety
    ALLOWED_TABLES = {'tokenizer_vocabulary', 'learned_words', 'bip39_words'}
    ALLOWED_COLUMNS = {'word'}
    
    if table_name not in ALLOWED_TABLES:
        logger.error(f"Invalid table name: {table_name}. Allowed: {ALLOWED_TABLES}")
        return 0
    
    if word_column not in ALLOWED_COLUMNS:
        logger.error(f"Invalid column name: {word_column}. Allowed: {ALLOWED_COLUMNS}")
        return 0
    
    if not contaminated_words:
        logger.info(f"No contaminated words to remove from {table_name}")
        return 0
    
    conn = get_db_connection()
    if not conn:
        logger.error(f"Failed to connect to database")
        return 0
    
    try:
        cur = conn.cursor()
        
        # Delete contaminated words
        deleted = 0
        for word in contaminated_words:
            if dry_run:
                # Just count what would be deleted
                cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {word_column} = %s", (word,))
                count = cur.fetchone()[0]
                deleted += count
            else:
                # Actually delete
                cur.execute(f"DELETE FROM {table_name} WHERE {word_column} = %s", (word,))
                deleted += cur.rowcount
        
        if not dry_run:
            conn.commit()
            logger.info(f"Deleted {deleted} contaminated entries from {table_name}")
        else:
            logger.info(f"[DRY RUN] Would delete {deleted} contaminated entries from {table_name}")
        
        return deleted
        
    except Exception as e:
        logger.error(f"Error cleaning {table_name}: {e}")
        if not dry_run:
            conn.rollback()
        return 0
    finally:
        conn.close()


def clean_word_relationships(contaminated_words: List[str], dry_run: bool = True) -> int:
    """
    Remove word relationships involving contaminated words.
    
    Args:
        contaminated_words: List of contaminated words
        dry_run: If True, don't actually delete (just report)
        
    Returns:
        Number of relationships deleted
    """
    if not contaminated_words:
        return 0
    
    conn = get_db_connection()
    if not conn:
        logger.error(f"Failed to connect to database")
        return 0
    
    try:
        cur = conn.cursor()
        
        # Delete relationships where either source or target is contaminated
        deleted = 0
        for word in contaminated_words:
            if dry_run:
                # Count what would be deleted
                cur.execute("""
                    SELECT COUNT(*) FROM word_relationships 
                    WHERE source_word = %s OR target_word = %s
                """, (word, word))
                count = cur.fetchone()[0]
                deleted += count
            else:
                # Actually delete
                cur.execute("""
                    DELETE FROM word_relationships 
                    WHERE source_word = %s OR target_word = %s
                """, (word, word))
                deleted += cur.rowcount
        
        if not dry_run:
            conn.commit()
            logger.info(f"Deleted {deleted} contaminated word relationships")
        else:
            logger.info(f"[DRY RUN] Would delete {deleted} contaminated word relationships")
        
        return deleted
        
    except Exception as e:
        logger.error(f"Error cleaning word relationships: {e}")
        if not dry_run:
            conn.rollback()
        return 0
    finally:
        conn.close()


def generate_cleanup_report(scan_results: List[Dict], deleted_counts: Dict[str, int], 
                           dry_run: bool = True) -> str:
    """
    Generate a comprehensive cleanup report.
    
    Args:
        scan_results: List of scan results for each table
        deleted_counts: Dictionary of deletion counts per table
        dry_run: Whether this was a dry run
        
    Returns:
        Report text
    """
    report = []
    report.append("=" * 80)
    report.append("VOCABULARY CLEANUP REPORT - PR 27/28 Implementation")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append(f"Mode: {'DRY RUN (no actual deletion)' if dry_run else 'LIVE CLEANUP (data deleted)'}\n")
    
    # Summary
    total_words = sum(r['total'] for r in scan_results)
    total_clean = sum(r['clean'] for r in scan_results)
    total_contaminated = sum(r['contaminated'] for r in scan_results)
    total_deleted = sum(deleted_counts.values())
    
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Total words scanned: {total_words}")
    report.append(f"Clean words: {total_clean} ({100*total_clean/total_words if total_words > 0 else 0:.1f}%)")
    report.append(f"Contaminated words: {total_contaminated} ({100*total_contaminated/total_words if total_words > 0 else 0:.1f}%)")
    report.append(f"Entries {'would be deleted' if dry_run else 'deleted'}: {total_deleted}\n")
    
    # Per-table details
    report.append("PER-TABLE RESULTS")
    report.append("-" * 80)
    for result in scan_results:
        table_name = result['table_name']
        deleted = deleted_counts.get(table_name, 0)
        
        report.append(f"\n{table_name}:")
        report.append(f"  Total: {result['total']}")
        report.append(f"  Clean: {result['clean']}")
        report.append(f"  Contaminated: {result['contaminated']}")
        report.append(f"  {'Would delete' if dry_run else 'Deleted'}: {deleted}")
        
        if result['contaminated_words']:
            # Group by contamination reason
            by_reason = {}
            for item in result['contaminated_words']:
                reason = item['reason'].split(':')[0]  # Get base reason
                if reason not in by_reason:
                    by_reason[reason] = []
                by_reason[reason].append(item)
            
            report.append(f"\n  Contamination breakdown:")
            for reason, items in sorted(by_reason.items(), key=lambda x: len(x[1]), reverse=True):
                count = len(items)
                examples = ', '.join([item['word'] for item in items[:5]])
                report.append(f"    {reason}: {count} words")
                report.append(f"      Examples: {examples}")
    
    # Top contaminated words
    report.append("\nTOP CONTAMINATED WORDS (by frequency)")
    report.append("-" * 80)
    all_contaminated = []
    for result in scan_results:
        all_contaminated.extend(result['contaminated_words'])
    
    # Sort by frequency
    all_contaminated.sort(key=lambda x: x['frequency'], reverse=True)
    
    for i, item in enumerate(all_contaminated[:20], 1):
        report.append(f"{i:2}. {item['word']:20} (freq: {item['frequency']:6}, reason: {item['reason']})")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def main(dry_run: bool = True):
    """
    Main cleanup execution.
    
    Args:
        dry_run: If True, don't actually delete (just report)
    """
    logger.info("=" * 80)
    logger.info("VOCABULARY CLEANUP - PR 27/28 Implementation")
    logger.info("=" * 80)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE CLEANUP'}")
    logger.info("")
    
    # Tables to scan
    tables_to_scan = [
        ('tokenizer_vocabulary', 'word'),
        ('learned_words', 'word'),
        ('bip39_words', 'word'),
    ]
    
    # Scan all tables
    scan_results = []
    all_contaminated_words = set()
    
    logger.info("Step 1: Scanning vocabulary tables...")
    for table_name, word_column in tables_to_scan:
        result = scan_vocabulary_table(table_name, word_column)
        if result:
            scan_results.append(result)
            all_contaminated_words.update([item['word'] for item in result['contaminated_words']])
    
    logger.info(f"\nTotal contaminated words across all tables: {len(all_contaminated_words)}")
    
    # Clean tables
    deleted_counts = {}
    
    logger.info("\nStep 2: Cleaning vocabulary tables...")
    for result in scan_results:
        table_name = result['table_name']
        contaminated_words = [item['word'] for item in result['contaminated_words']]
        
        deleted = clean_vocabulary_table(
            table_name, 
            contaminated_words,
            word_column='word',
            dry_run=dry_run
        )
        deleted_counts[table_name] = deleted
    
    # Clean word relationships
    logger.info("\nStep 3: Cleaning word relationships...")
    deleted_relationships = clean_word_relationships(list(all_contaminated_words), dry_run=dry_run)
    deleted_counts['word_relationships'] = deleted_relationships
    
    # Generate report
    logger.info("\nStep 4: Generating cleanup report...")
    report = generate_cleanup_report(scan_results, deleted_counts, dry_run=dry_run)
    
    # Print report
    print("\n" + report)
    
    # Save report to file
    # Use environment variable or default to docs/04-records/
    reports_dir = os.environ.get('VOCABULARY_REPORTS_DIR', 
                                  os.path.join(os.path.dirname(__file__), '..', '..', 'docs', '04-records'))
    report_filename = f"vocabulary_cleanup_report_{'dry_run' if dry_run else 'live'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(reports_dir, report_filename)
    
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"\nReport saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("CLEANUP COMPLETE")
    logger.info("=" * 80)
    
    if dry_run:
        logger.info("\nThis was a DRY RUN - no data was actually deleted.")
        logger.info("To perform actual cleanup, run with --live flag")
    else:
        logger.info("\nCleanup completed successfully!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean vocabulary contamination (PR 27/28 implementation)')
    parser.add_argument('--live', action='store_true', help='Actually delete contaminated entries (default is dry run)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (default, just report)')
    
    args = parser.parse_args()
    
    # Default to dry run unless --live is specified
    dry_run = not args.live
    
    main(dry_run=dry_run)
