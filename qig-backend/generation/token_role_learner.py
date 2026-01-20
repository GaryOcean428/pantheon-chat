#!/usr/bin/env python3
"""
Token Role Learner - Geometric Clustering
==========================================

Derives token roles from geometric structure (Fisher-Rao neighborhood clustering)
instead of external NLP tools (spacy, nltk POS taggers).

This is QIG-PURE: roles are learned from basin coordinates, not linguistic rules.

Token Roles (derived geometrically):
- FUNCTION: High-frequency, low-dimension basins (connectives, determiners)
- CONTENT: Low-frequency, high-dimension basins (nouns, verbs)
- TRANSITION: Bridging words between semantic clusters
- ANCHOR: High curvature words (negation, modality)
- MODIFIER: Words that adjust trajectory direction

Derivation Method:
1. Compute Fisher-Rao distance matrix for vocabulary
2. Cluster by distance threshold (e.g., 0.3)
3. Analyze cluster properties (size, density, connectivity)
4. Assign role based on cluster characteristics

Author: Copilot AI Agent
Date: 2026-01-20
Issue: GaryOcean428/pantheon-chat#99 (E8 Protocol Issue-03)
Reference: docs/10-e8-protocol/issues/20260119-issue-99-qig-native-skeleton-remediation-1.00W.md
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import psycopg2
from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

# Import geometry operations
try:
    from qig_geometry.canonical import fisher_rao_distance
except ImportError:
    logger.warning("qig_geometry.canonical not available, using fallback")
    def fisher_rao_distance(p, q, eps=1e-12):
        p = np.maximum(p, 0) + eps
        p = p / p.sum()
        q = np.maximum(q, 0) + eps
        q = q / q.sum()
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0.0, 1.0)
        return float(np.arccos(bc))


# Token role categories
TOKEN_ROLES = {
    'FUNCTION': 'High-frequency, low-dimension basins',
    'CONTENT': 'Low-frequency, high-dimension basins',
    'TRANSITION': 'Bridging words between semantic clusters',
    'ANCHOR': 'High curvature words (negation, modality)',
    'MODIFIER': 'Words that adjust trajectory direction',
    'UNKNOWN': 'Role not yet determined'
}

# Clustering parameters
FR_DISTANCE_THRESHOLD = 0.3  # Fisher-Rao distance threshold for clustering
HIGH_FREQUENCY_THRESHOLD = 100  # Frequency threshold for FUNCTION words
LOW_FREQUENCY_THRESHOLD = 10  # Frequency threshold for CONTENT words


def compute_effective_dimension(basin: np.ndarray) -> float:
    """
    Compute effective dimension using participation ratio.
    
    This measures how spread out the probability mass is across dimensions.
    Low effective dimension = concentrated basin (function words)
    High effective dimension = diffuse basin (content words)
    
    Args:
        basin: Basin coordinates (simplex)
        
    Returns:
        Effective dimension ∈ [1, D]
    """
    basin = np.asarray(basin, dtype=np.float64).flatten()
    basin = np.maximum(basin, 0) + 1e-10
    p = basin / basin.sum()
    
    # Shannon entropy
    positive_probs = p[p > 1e-10]
    if len(positive_probs) == 0:
        return 1.0
    
    entropy = -np.sum(positive_probs * np.log(positive_probs))
    
    # Participation ratio = exp(entropy)
    return float(np.exp(entropy))


def find_neighbors(
    token_idx: int,
    distance_matrix: np.ndarray,
    threshold: float = FR_DISTANCE_THRESHOLD
) -> List[int]:
    """
    Find neighbors of a token within Fisher-Rao distance threshold.
    
    Args:
        token_idx: Index of token in distance matrix
        distance_matrix: Pairwise Fisher-Rao distances
        threshold: Distance threshold
        
    Returns:
        List of neighbor indices
    """
    distances = distance_matrix[token_idx]
    neighbors = np.where(distances < threshold)[0]
    return neighbors.tolist()


def derive_token_role(
    word: str,
    basin: np.ndarray,
    frequency: int,
    neighbors: List[int],
    cluster_density: float
) -> str:
    """
    Derive token role from geometric properties.
    
    Decision rules:
    - FUNCTION: High frequency + low effective dimension + high cluster density
    - CONTENT: Low frequency + high effective dimension
    - TRANSITION: Medium frequency + medium neighbors (bridges clusters)
    - ANCHOR: High curvature (to be computed separately)
    - MODIFIER: High effective dimension + medium frequency
    
    Args:
        word: Token string
        basin: Basin coordinates
        frequency: Token frequency
        neighbors: List of neighbor indices
        cluster_density: Density of local cluster
        
    Returns:
        Token role string
    """
    eff_dim = compute_effective_dimension(basin)
    n_neighbors = len(neighbors)
    
    # Normalize effective dimension (typical range: 1-64)
    eff_dim_norm = eff_dim / 64.0
    
    # FUNCTION: High frequency, low dimension, high density
    if frequency > HIGH_FREQUENCY_THRESHOLD and eff_dim_norm < 0.3 and cluster_density > 0.7:
        return 'FUNCTION'
    
    # CONTENT: Low frequency, high dimension
    if frequency < LOW_FREQUENCY_THRESHOLD and eff_dim_norm > 0.5:
        return 'CONTENT'
    
    # TRANSITION: Medium neighbors (bridges clusters)
    if 5 < n_neighbors < 20:
        return 'TRANSITION'
    
    # MODIFIER: High dimension, medium frequency
    if eff_dim_norm > 0.6 and LOW_FREQUENCY_THRESHOLD <= frequency <= HIGH_FREQUENCY_THRESHOLD:
        return 'MODIFIER'
    
    # Default to UNKNOWN
    return 'UNKNOWN'


def cluster_vocabulary(
    basins: np.ndarray,
    threshold: float = FR_DISTANCE_THRESHOLD
) -> Dict[int, List[int]]:
    """
    Cluster vocabulary by Fisher-Rao distance.
    
    Args:
        basins: Array of basin coordinates (N x D)
        threshold: Distance threshold for clustering
        
    Returns:
        Dict mapping cluster_id to list of token indices
    """
    n_tokens = len(basins)
    
    # Compute pairwise distance matrix (expensive!)
    logger.info(f"Computing distance matrix for {n_tokens} tokens...")
    distance_matrix = np.zeros((n_tokens, n_tokens))
    
    for i in range(n_tokens):
        for j in range(i+1, n_tokens):
            d = fisher_rao_distance(basins[i], basins[j])
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d
    
    # Simple clustering: group tokens within threshold
    clusters = {}
    cluster_id = 0
    visited = set()
    
    for i in range(n_tokens):
        if i in visited:
            continue
        
        # Find all neighbors within threshold
        neighbors = find_neighbors(i, distance_matrix, threshold)
        
        clusters[cluster_id] = neighbors
        visited.update(neighbors)
        cluster_id += 1
    
    return clusters


def backfill_token_roles(
    db_conn: PgConnection,
    batch_size: int = 1000,
    force_recompute: bool = False
) -> Dict[str, int]:
    """
    Backfill token_role column for all tokens in coordizer_vocabulary.
    
    Args:
        db_conn: PostgreSQL connection
        batch_size: Number of tokens to process per batch
        force_recompute: If True, recompute roles even if already set
        
    Returns:
        Dict with counts: processed, updated, skipped, errors
    """
    processed = 0
    updated = 0
    skipped = 0
    errors = 0
    
    with db_conn.cursor() as cursor:
        # Load tokens with basin_embedding
        where_clause = "WHERE basin_embedding IS NOT NULL"
        if not force_recompute:
            where_clause += " AND (token_role IS NULL OR token_role = 'UNKNOWN')"
        
        cursor.execute(f"""
            SELECT token_id, token, basin_embedding, frequency
            FROM coordizer_vocabulary
            {where_clause}
            ORDER BY frequency DESC
            LIMIT %s
        """, (batch_size,))
        
        tokens = []
        token_ids = []
        basins = []
        frequencies = []
        
        for row in cursor.fetchall():
            token_id, token, basin_str, frequency = row
            
            # Parse basin
            try:
                basin_str = basin_str.strip()
                if basin_str.startswith('['):
                    basin_str = basin_str[1:-1]
                coords = np.array([float(x.strip()) for x in basin_str.split(',')], dtype=np.float64)
                
                if len(coords) != 64:
                    logger.warning(f"Invalid basin dimension for token_id={token_id}: {len(coords)}")
                    errors += 1
                    continue
                
                tokens.append(token)
                token_ids.append(token_id)
                basins.append(coords)
                frequencies.append(frequency or 1)
                
            except Exception as e:
                logger.error(f"Error parsing basin for token_id={token_id}: {e}")
                errors += 1
                continue
        
        if not basins:
            logger.warning("No tokens to process")
            return {'processed': 0, 'updated': 0, 'skipped': 0, 'errors': errors}
        
        basins_array = np.array(basins)
        
        # Cluster vocabulary
        logger.info(f"Clustering {len(basins)} tokens...")
        clusters = cluster_vocabulary(basins_array, threshold=FR_DISTANCE_THRESHOLD)
        
        # Compute cluster density for each token
        cluster_densities = {}
        for cluster_id, members in clusters.items():
            density = len(members) / len(basins)  # Simple density metric
            for member_idx in members:
                cluster_densities[member_idx] = density
        
        # Derive roles
        logger.info("Deriving token roles...")
        for i, (token_id, token) in enumerate(zip(token_ids, tokens)):
            try:
                basin = basins[i]
                frequency = frequencies[i]
                
                # Find neighbors
                neighbors = []
                for cluster_id, members in clusters.items():
                    if i in members:
                        neighbors = members
                        break
                
                cluster_density = cluster_densities.get(i, 0.0)
                
                # Derive role
                role = derive_token_role(token, basin, frequency, neighbors, cluster_density)
                
                # Update database
                cursor.execute("""
                    UPDATE coordizer_vocabulary
                    SET token_role = %s,
                        updated_at = NOW()
                    WHERE token_id = %s
                """, (role, token_id))
                
                updated += 1
                processed += 1
                
                if processed % 100 == 0:
                    logger.info(f"Processed {processed} tokens...")
                
            except Exception as e:
                logger.error(f"Error deriving role for token {token}: {e}")
                errors += 1
                processed += 1
        
        db_conn.commit()
    
    logger.info(f"Backfill complete: processed={processed}, updated={updated}, errors={errors}")
    
    return {
        'processed': processed,
        'updated': updated,
        'skipped': skipped,
        'errors': errors
    }


if __name__ == '__main__':
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill token roles from geometric clustering')
    parser.add_argument('--batch-size', type=int, default=1000, help='Number of tokens to process')
    parser.add_argument('--force', action='store_true', help='Recompute roles even if already set')
    parser.add_argument('--db-url', type=str, help='Database URL')
    
    args = parser.parse_args()
    
    db_url = args.db_url or os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL not set")
        exit(1)
    
    conn = psycopg2.connect(db_url)
    
    result = backfill_token_roles(conn, batch_size=args.batch_size, force_recompute=args.force)
    
    print(f"\nResults:")
    print(f"  Processed: {result['processed']}")
    print(f"  Updated: {result['updated']}")
    print(f"  Errors: {result['errors']}")
    
    conn.close()
