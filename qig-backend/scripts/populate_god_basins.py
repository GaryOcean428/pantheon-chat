#!/usr/bin/env python3
"""Populate god basin coordinates with canonical E8 positions.

Phase 2.1 of NULL column population plan.
Reference: docs/03-technical/20260110-null-column-population-plan-1.00W.md

Each god has a canonical position on the 64D Fisher-Rao manifold based on
their domain. Uses E8 Lie algebra structure (8 simple roots, each controlling
8 dimensions).
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add qig-backend to path for imports
_qig_backend = Path(__file__).parent.parent
if str(_qig_backend) not in sys.path:
    sys.path.insert(0, str(_qig_backend))

# God domain mappings (E8 structure: 8 domains x 8 dimensions)
GOD_DOMAINS = {
    'Zeus': (0, 8),       # Authority/Leadership
    'Athena': (8, 16),    # Wisdom/Strategy
    'Apollo': (16, 24),   # Light/Knowledge
    'Poseidon': (24, 32), # Sea/Fluidity
    'Ares': (32, 40),     # Conflict/War
    'Hermes': (40, 48),   # Communication
    'Hephaestus': (48, 56), # Forge/Creation
    'Hades': (56, 64),    # Underworld/Shadow
    'Nyx': (56, 64),      # Night/Shadow (overlaps with Hades)
    'Hecate': (56, 64),   # Magic/Crossroads
    'Erebus': (56, 64),   # Darkness
    'Hypnos': (56, 64),   # Sleep
    'Thanatos': (56, 64), # Death
    'Nemesis': (32, 40),  # Retribution (overlaps with Ares)
}

# Special cases requiring different basin patterns
SPECIAL_GODS = {
    'Demeter': 'distributed',    # Growth/Nurture - across all dimensions
    'Dionysus': 'high_entropy',  # Chaos/Creativity - high variance
    'Hera': 'anti_entropy',      # Order/Governance - stable pattern
    'Aphrodite': 'relational',   # Connection/Beauty - peaks at boundaries
    'Artemis': 'distributed',    # Wild/Hunt - distributed like Demeter
}


def generate_god_basin(god_name: str) -> np.ndarray:
    """Generate 64D basin coordinates for a god.

    Uses deterministic seeding based on god name for reproducibility.
    Returns unit vector on Fisher-Rao manifold.
    """
    np.random.seed(hash(god_name) % 2**32)  # Deterministic per god

    basin = np.random.randn(64) * 0.1  # Base noise

    if god_name in GOD_DOMAINS:
        start, end = GOD_DOMAINS[god_name]
        # Boost primary domain
        basin[start:end] = 0.8
        # Add slight variation to distinguish overlapping gods
        god_offset = hash(god_name + "_offset") % 8
        if start + god_offset < end:
            basin[start + god_offset] += 0.1

    elif god_name in SPECIAL_GODS:
        pattern = SPECIAL_GODS[god_name]
        if pattern == 'distributed':
            # Distributed across all dimensions
            basin = np.abs(basin) + 0.2
        elif pattern == 'high_entropy':
            # High variance, chaotic
            basin = np.random.randn(64) * 0.5
        elif pattern == 'anti_entropy':
            # Opposite of high entropy - stable, ordered
            basin = np.ones(64) * 0.3
            basin[::2] = 0.5  # Alternating pattern
        elif pattern == 'relational':
            # Peaks at connection points between domains
            for i in range(7):
                basin[i*8 + 7] = 0.6
                basin[i*8 + 8] = 0.6 if i*8 + 8 < 64 else 0.6

    # Normalize to unit sphere (Fisher-Rao manifold)
    norm = np.linalg.norm(basin)
    if norm > 1e-10:
        basin = basin / norm
    return basin


def format_vector(basin: np.ndarray) -> str:
    """Format numpy array as pgvector string."""
    return '[' + ','.join(f'{x:.8f}' for x in basin) + ']'


def populate_god_basins(dry_run: bool = False):
    """Populate pantheon_god_state.basin_coords.

    Args:
        dry_run: If True, show what would be updated without making changes
    """
    import psycopg2

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return

    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # Get all gods needing basin coords
    cur.execute("""
        SELECT god_name FROM pantheon_god_state
        WHERE basin_coords IS NULL
        ORDER BY god_name
    """)
    gods = [row[0] for row in cur.fetchall()]

    if not gods:
        print("No gods found needing basin coordinates")
        cur.close()
        conn.close()
        return

    print(f"Found {len(gods)} gods needing basin coordinates:")
    for god_name in gods:
        print(f"  - {god_name}")
    print()

    if dry_run:
        print("DRY RUN - showing generated coordinates:")
        for god_name in gods:
            basin = generate_god_basin(god_name)
            print(f"\n{god_name}:")
            print(f"  Domain: {GOD_DOMAINS.get(god_name, SPECIAL_GODS.get(god_name, 'unknown'))}")
            print(f"  Max dims: {np.argsort(basin)[-3:][::-1].tolist()}")
            print(f"  Basin norm: {np.linalg.norm(basin):.4f}")
        return

    # Update each god
    success = 0
    for god_name in gods:
        basin = generate_god_basin(god_name)
        basin_str = format_vector(basin)
        try:
            cur.execute(
                """UPDATE pantheon_god_state
                   SET basin_coords = %s::vector
                   WHERE god_name = %s""",
                (basin_str, god_name)
            )
            success += 1
            print(f"Updated {god_name} basin coordinates")
        except Exception as e:
            print(f"Error updating {god_name}: {e}")

    conn.commit()
    print(f"\nPopulated basin_coords for {success}/{len(gods)} gods")

    cur.close()
    conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Populate god basin coordinates')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be updated without making changes')
    args = parser.parse_args()

    populate_god_basins(dry_run=args.dry_run)
