#!/usr/bin/env python3
"""
Validate Special Symbol Basins in Database

Script to validate that special symbols (UNK, PAD, BOS, EOS) in the database
have geometrically valid basin coordinates according to WP2.3 requirements.

Usage:
    python3 validate_special_symbols.py
    python3 validate_special_symbols.py --database-url postgresql://...
    python3 validate_special_symbols.py --fix  # Apply fixes if validation fails

Requirements:
- Special symbols must exist in coordizer_vocabulary table
- Basin embeddings must be 64D
- Must satisfy simplex constraints (non-negative, sum=1)
- Must match geometric definitions from FisherCoordizer
"""

import sys
import os
import argparse
import logging
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from qig_geometry.representation import validate_basin
from qig_geometry.canonical import fisher_rao_distance

# Database connection
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Warning: psycopg2 not available - database validation disabled")

logger = logging.getLogger(__name__)


class SpecialSymbolValidator:
    """Validates special symbols in coordizer_vocabulary table."""
    
    REQUIRED_SYMBOLS = ['UNK', 'PAD', 'BOS', 'EOS']
    BASIN_DIM = 64
    
    def __init__(self, database_url: str):
        """Initialize validator with database connection."""
        self.database_url = database_url
        self.conn = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def connect(self):
        """Establish database connection."""
        if not DB_AVAILABLE:
            raise RuntimeError("psycopg2 not available - cannot connect to database")
        
        try:
            self.conn = psycopg2.connect(self.database_url)
            logger.info("Connected to database")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to database: {e}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def fetch_special_symbols(self) -> Dict[str, Dict]:
        """
        Fetch special symbols from database.
        
        Returns:
            Dict mapping symbol name to row data (token, basin_embedding, etc.)
        """
        if not self.conn:
            raise RuntimeError("Not connected to database")
        
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # Query for special symbols
        query = """
            SELECT token, basin_embedding, qfi_score, token_status
            FROM coordizer_vocabulary
            WHERE token IN ('<UNK>', '<PAD>', '<BOS>', '<EOS>')
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        
        # Map to symbol names
        result = {}
        for row in rows:
            token = row['token']
            symbol = token.strip('<>')
            result[symbol] = dict(row)
        
        return result
    
    def validate_symbol_exists(self, symbols: Dict[str, Dict]) -> bool:
        """Verify all required special symbols exist."""
        missing = []
        for symbol in self.REQUIRED_SYMBOLS:
            if symbol not in symbols:
                missing.append(f"<{symbol}>")
        
        if missing:
            self.errors.append(f"Missing required special symbols: {', '.join(missing)}")
            return False
        
        logger.info(f"✓ All {len(self.REQUIRED_SYMBOLS)} required special symbols exist")
        return True
    
    def validate_basin_dimension(self, symbols: Dict[str, Dict]) -> bool:
        """Verify basin embeddings are 64D."""
        valid = True
        
        for symbol, data in symbols.items():
            basin = data.get('basin_embedding')
            
            if basin is None:
                self.errors.append(f"{symbol}: basin_embedding is NULL")
                valid = False
                continue
            
            if len(basin) != self.BASIN_DIM:
                self.errors.append(
                    f"{symbol}: basin dimension is {len(basin)}, expected {self.BASIN_DIM}"
                )
                valid = False
        
        if valid:
            logger.info(f"✓ All special symbols have {self.BASIN_DIM}D basin embeddings")
        
        return valid
    
    def validate_simplex_constraints(self, symbols: Dict[str, Dict]) -> bool:
        """Verify basin embeddings satisfy simplex constraints."""
        valid = True
        
        for symbol, data in symbols.items():
            basin = np.array(data['basin_embedding'], dtype=np.float64)
            
            # Use canonical validation
            is_valid, reason = validate_basin(basin)
            
            if not is_valid:
                self.errors.append(f"{symbol}: {reason}")
                valid = False
            else:
                logger.info(f"✓ {symbol}: valid simplex point")
        
        return valid
    
    def validate_geometric_meaning(self, symbols: Dict[str, Dict]) -> bool:
        """Verify special symbols match their geometric definitions."""
        valid = True
        
        # UNK should be uniform (maximum entropy)
        if 'UNK' in symbols:
            unk = np.array(symbols['UNK']['basin_embedding'], dtype=np.float64)
            unk_std = np.std(unk)
            
            if unk_std > 0.001:
                self.warnings.append(
                    f"UNK: std={unk_std:.6f}, expected uniform (std≈0)"
                )
                valid = False
            else:
                logger.info(f"✓ UNK is uniform distribution (std={unk_std:.6f})")
        
        # PAD should be sparse (minimal entropy)
        if 'PAD' in symbols:
            pad = np.array(symbols['PAD']['basin_embedding'], dtype=np.float64)
            pad_max = np.max(pad)
            
            if pad_max < 0.9:
                self.warnings.append(
                    f"PAD: max={pad_max:.6f}, expected sparse (max≈1.0)"
                )
                valid = False
            else:
                logger.info(f"✓ PAD is sparse distribution (max={pad_max:.6f})")
        
        # BOS should be at a vertex
        if 'BOS' in symbols:
            bos = np.array(symbols['BOS']['basin_embedding'], dtype=np.float64)
            bos_max = np.max(bos)
            
            if bos_max < 0.9:
                self.warnings.append(
                    f"BOS: max={bos_max:.6f}, expected vertex (max≈1.0)"
                )
                valid = False
            else:
                bos_argmax = np.argmax(bos)
                logger.info(f"✓ BOS is vertex at dimension {bos_argmax} (max={bos_max:.6f})")
        
        # EOS should be at opposite vertex
        if 'EOS' in symbols:
            eos = np.array(symbols['EOS']['basin_embedding'], dtype=np.float64)
            eos_max = np.max(eos)
            
            if eos_max < 0.9:
                self.warnings.append(
                    f"EOS: max={eos_max:.6f}, expected vertex (max≈1.0)"
                )
                valid = False
            else:
                eos_argmax = np.argmax(eos)
                logger.info(f"✓ EOS is vertex at dimension {eos_argmax} (max={eos_max:.6f})")
        
        # Verify BOS and EOS are at different vertices
        if 'BOS' in symbols and 'EOS' in symbols:
            bos = np.array(symbols['BOS']['basin_embedding'], dtype=np.float64)
            eos = np.array(symbols['EOS']['basin_embedding'], dtype=np.float64)
            
            bos_argmax = np.argmax(bos)
            eos_argmax = np.argmax(eos)
            
            if bos_argmax == eos_argmax:
                self.errors.append(
                    f"BOS and EOS are at same vertex (dimension {bos_argmax})"
                )
                valid = False
            else:
                logger.info(f"✓ BOS (dim {bos_argmax}) and EOS (dim {eos_argmax}) at different vertices")
        
        return valid
    
    def validate_distances(self, symbols: Dict[str, Dict]) -> bool:
        """Verify Fisher-Rao distances are in valid range."""
        valid = True
        max_distance = np.pi / 2
        
        # Check a few key distances
        pairs = [
            ('UNK', 'PAD'),
            ('UNK', 'BOS'),
            ('BOS', 'EOS'),
        ]
        
        for sym1, sym2 in pairs:
            if sym1 not in symbols or sym2 not in symbols:
                continue
            
            basin1 = np.array(symbols[sym1]['basin_embedding'], dtype=np.float64)
            basin2 = np.array(symbols[sym2]['basin_embedding'], dtype=np.float64)
            
            try:
                dist = fisher_rao_distance(basin1, basin2)
                
                if not (0 <= dist <= max_distance):
                    self.errors.append(
                        f"Distance {sym1}-{sym2} out of range: {dist:.4f}"
                    )
                    valid = False
                else:
                    logger.info(f"✓ Distance {sym1}-{sym2}: {dist:.4f}")
            except Exception as e:
                self.errors.append(f"Failed to compute distance {sym1}-{sym2}: {e}")
                valid = False
        
        return valid
    
    def validate_token_status(self, symbols: Dict[str, Dict]) -> bool:
        """Verify special symbols have valid token_status."""
        valid = True
        
        for symbol, data in symbols.items():
            status = data.get('token_status')
            
            if status not in ['active', 'quarantined', 'deprecated']:
                self.errors.append(
                    f"{symbol}: invalid token_status '{status}'"
                )
                valid = False
            elif status != 'active':
                self.warnings.append(
                    f"{symbol}: token_status is '{status}' (expected 'active')"
                )
        
        if valid:
            logger.info("✓ All special symbols have valid token_status")
        
        return valid
    
    def run_validation(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run complete validation suite.
        
        Returns:
            Tuple of (success, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        try:
            self.connect()
            
            # Fetch special symbols
            logger.info("Fetching special symbols from database...")
            symbols = self.fetch_special_symbols()
            
            # Run validation checks
            checks = [
                ("Symbol existence", self.validate_symbol_exists(symbols)),
                ("Basin dimension", self.validate_basin_dimension(symbols)),
                ("Simplex constraints", self.validate_simplex_constraints(symbols)),
                ("Geometric meaning", self.validate_geometric_meaning(symbols)),
                ("Fisher-Rao distances", self.validate_distances(symbols)),
                ("Token status", self.validate_token_status(symbols)),
            ]
            
            # Print summary
            print("\n" + "=" * 60)
            print("VALIDATION SUMMARY")
            print("=" * 60)
            
            for check_name, passed in checks:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"{check_name:.<40} {status}")
            
            success = all(passed for _, passed in checks)
            
            if self.errors:
                print(f"\n❌ {len(self.errors)} ERROR(S):")
                for error in self.errors:
                    print(f"  - {error}")
            
            if self.warnings:
                print(f"\n⚠️  {len(self.warnings)} WARNING(S):")
                for warning in self.warnings:
                    print(f"  - {warning}")
            
            if success and not self.warnings:
                print("\n✅ ALL VALIDATION CHECKS PASSED")
            elif success:
                print("\n✅ VALIDATION PASSED (with warnings)")
            else:
                print("\n❌ VALIDATION FAILED")
            
            return success, self.errors, self.warnings
            
        finally:
            self.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate special symbol basins in coordizer_vocabulary table"
    )
    parser.add_argument(
        '--database-url',
        default=os.environ.get('DATABASE_URL'),
        help='PostgreSQL database URL (default: from DATABASE_URL env var)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Apply fixes if validation fails (not yet implemented)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check database URL
    if not args.database_url:
        print("Error: No database URL provided")
        print("Set DATABASE_URL environment variable or use --database-url")
        return 1
    
    # Run validation
    validator = SpecialSymbolValidator(args.database_url)
    
    try:
        success, errors, warnings = validator.run_validation()
        
        if args.fix and not success:
            print("\n⚠️  --fix flag not yet implemented")
            print("Please manually fix the errors listed above")
            return 1
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        print(f"\n❌ VALIDATION ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
