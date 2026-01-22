"""
Genetic Lineage Persistence - E8 Protocol v4.0 Phase 4E
========================================================

Database persistence layer for kernel genetic lineage system.
Provides functions to store and retrieve genomes, lineage records,
merge events, cannibalism events, and genome archives.

Authority: E8 Protocol v4.0 WP5.2 Phase 4E
Status: ACTIVE
Created: 2026-01-22
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import json

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from kernels.genome import (
    KernelGenome,
    E8Faculty,
    serialize_genome,
    deserialize_genome,
)
from kernels.kernel_lineage import (
    LineageRecord,
    MergeRecord,
)
from kernels.cannibalism import (
    CannibalismRecord,
    GenomeArchive,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_db_connection(database_url: Optional[str] = None) -> psycopg2.extensions.connection:
    """
    Get database connection for genetic lineage operations.
    
    Args:
        database_url: Optional PostgreSQL connection string
    
    Returns:
        Database connection
    """
    import os
    
    if database_url is None:
        database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        raise ValueError("DATABASE_URL not set")
    
    return psycopg2.connect(database_url)


# =============================================================================
# GENOME PERSISTENCE
# =============================================================================

def save_genome(genome: KernelGenome, conn: Optional[psycopg2.extensions.connection] = None) -> None:
    """
    Save kernel genome to database.
    
    Args:
        genome: Kernel genome to save
        conn: Optional database connection (creates new if None)
    """
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # Convert basin seed to PostgreSQL vector format
            basin_seed_vec = f"[{','.join(str(x) for x in genome.basin_seed)}]"
            
            # Convert faculties to arrays
            active_faculties = [f.value for f in genome.faculties.active_faculties]
            activation_strengths = {
                f.value: s for f, s in genome.faculties.activation_strengths.items()
            }
            faculty_coupling = {
                f"{f1.value}-{f2.value}": s 
                for (f1, f2), s in genome.faculties.faculty_coupling.items()
            }
            
            # Convert forbidden regions to JSONB
            forbidden_regions = [
                {'center': center.tolist(), 'radius': float(radius)}
                for center, radius in genome.constraints.forbidden_regions
            ]
            
            # Coupling preferences
            coupling_strengths = genome.coupling_prefs.coupling_strengths
            
            cur.execute("""
                INSERT INTO kernel_genomes (
                    genome_id, kernel_id, basin_seed,
                    active_faculties, activation_strengths, primary_faculty, faculty_coupling,
                    phi_threshold, kappa_range_min, kappa_range_max,
                    forbidden_regions, field_penalties, max_fisher_distance,
                    hemisphere_affinity, preferred_couplings, coupling_strengths, anti_couplings,
                    parent_genomes, generation,
                    created_at, fitness_score, mutation_count, metadata
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (genome_id) DO UPDATE SET
                    kernel_id = EXCLUDED.kernel_id,
                    basin_seed = EXCLUDED.basin_seed,
                    active_faculties = EXCLUDED.active_faculties,
                    activation_strengths = EXCLUDED.activation_strengths,
                    primary_faculty = EXCLUDED.primary_faculty,
                    faculty_coupling = EXCLUDED.faculty_coupling,
                    fitness_score = EXCLUDED.fitness_score,
                    mutation_count = EXCLUDED.mutation_count
            """, (
                genome.genome_id, genome.kernel_id, basin_seed_vec,
                active_faculties, json.dumps(activation_strengths),
                genome.faculties.primary_faculty.value if genome.faculties.primary_faculty else None,
                json.dumps(faculty_coupling),
                genome.constraints.phi_threshold,
                genome.constraints.kappa_range[0],
                genome.constraints.kappa_range[1],
                json.dumps(forbidden_regions),
                json.dumps(genome.constraints.field_penalties),
                genome.constraints.max_fisher_distance,
                genome.coupling_prefs.hemisphere_affinity,
                genome.coupling_prefs.preferred_couplings,
                json.dumps(coupling_strengths),
                genome.coupling_prefs.anti_couplings,
                genome.parent_genomes,
                genome.generation,
                genome.created_at,
                genome.fitness_score,
                genome.mutation_count,
                json.dumps({})  # metadata
            ))
            
            conn.commit()
            logger.info(f"Saved genome {genome.genome_id} to database")
            
    finally:
        if should_close:
            conn.close()


def load_genome(genome_id: str, conn: Optional[psycopg2.extensions.connection] = None) -> Optional[KernelGenome]:
    """
    Load kernel genome from database.
    
    Args:
        genome_id: Genome ID to load
        conn: Optional database connection
    
    Returns:
        KernelGenome or None if not found
    """
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM kernel_genomes WHERE genome_id = %s
            """, (genome_id,))
            
            row = cur.fetchone()
            if not row:
                return None
            
            # Convert from database format to genome dict
            genome_dict = {
                'genome_id': row['genome_id'],
                'kernel_id': row['kernel_id'],
                'basin_seed': row['basin_seed'],
                'faculties': {
                    'active_faculties': row['active_faculties'],
                    'activation_strengths': json.loads(row['activation_strengths']) if isinstance(row['activation_strengths'], str) else row['activation_strengths'],
                    'primary_faculty': row['primary_faculty'],
                    'faculty_coupling': json.loads(row['faculty_coupling']) if isinstance(row['faculty_coupling'], str) else row['faculty_coupling'],
                },
                'constraints': {
                    'phi_threshold': row['phi_threshold'],
                    'kappa_range': (row['kappa_range_min'], row['kappa_range_max']),
                    'forbidden_regions': json.loads(row['forbidden_regions']) if isinstance(row['forbidden_regions'], str) else row['forbidden_regions'],
                    'field_penalties': json.loads(row['field_penalties']) if isinstance(row['field_penalties'], str) else row['field_penalties'],
                    'max_fisher_distance': row['max_fisher_distance'],
                },
                'coupling_prefs': {
                    'hemisphere_affinity': row['hemisphere_affinity'],
                    'preferred_couplings': row['preferred_couplings'],
                    'coupling_strengths': json.loads(row['coupling_strengths']) if isinstance(row['coupling_strengths'], str) else row['coupling_strengths'],
                    'anti_couplings': row['anti_couplings'],
                },
                'parent_genomes': row['parent_genomes'],
                'generation': row['generation'],
                'created_at': row['created_at'].isoformat(),
                'fitness_score': row['fitness_score'],
                'mutation_count': row['mutation_count'],
            }
            
            # Convert genome dict to JSON then deserialize
            genome_json = json.dumps(genome_dict)
            return deserialize_genome(genome_json)
            
    finally:
        if should_close:
            conn.close()


# =============================================================================
# LINEAGE PERSISTENCE
# =============================================================================

def save_lineage_record(
    lineage: LineageRecord,
    conn: Optional[psycopg2.extensions.connection] = None
) -> None:
    """Save lineage record to database."""
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            inherited_faculties = {
                f.value: parent_id 
                for f, parent_id in lineage.inherited_faculties.items()
            }
            
            cur.execute("""
                INSERT INTO kernel_lineage (
                    lineage_id, child_genome_id, parent_genome_ids,
                    merge_type, fisher_distance, inherited_faculties,
                    created_at, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (lineage_id) DO NOTHING
            """, (
                lineage.lineage_id,
                lineage.child_genome_id,
                lineage.parent_genome_ids,
                lineage.merge_type,
                lineage.fisher_distance,
                json.dumps(inherited_faculties),
                lineage.created_at,
                json.dumps(lineage.metadata)
            ))
            
            conn.commit()
            logger.info(f"Saved lineage record {lineage.lineage_id}")
    finally:
        if should_close:
            conn.close()


def save_merge_record(
    merge: MergeRecord,
    conn: Optional[psycopg2.extensions.connection] = None
) -> None:
    """Save merge record to database."""
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            faculty_contract = {
                f.value: parent_id 
                for f, parent_id in merge.faculty_contract.items()
            }
            
            basin_distances = {
                f"{k[0]}-{k[1]}": v 
                for k, v in merge.basin_distances.items()
            }
            
            cur.execute("""
                INSERT INTO merge_events (
                    merge_id, parent_genome_ids, child_genome_id,
                    merge_weights, interpolation_t,
                    faculty_contract, basin_distances,
                    created_at, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (merge_id) DO NOTHING
            """, (
                merge.merge_id,
                merge.parent_genome_ids,
                merge.child_genome_id,
                merge.merge_weights,
                merge.interpolation_t,
                json.dumps(faculty_contract),
                json.dumps(basin_distances),
                merge.created_at,
                json.dumps(merge.metadata)
            ))
            
            conn.commit()
            logger.info(f"Saved merge record {merge.merge_id}")
    finally:
        if should_close:
            conn.close()


# =============================================================================
# CANNIBALISM PERSISTENCE
# =============================================================================

def save_cannibalism_record(
    record: CannibalismRecord,
    conn: Optional[psycopg2.extensions.connection] = None
) -> None:
    """Save cannibalism record to database."""
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # Convert basin vectors
            winner_before_vec = f"[{','.join(str(x) for x in record.winner_before.basin_seed)}]"
            winner_after_vec = f"[{','.join(str(x) for x in record.winner_after.basin_seed)}]"
            
            absorbed_faculties = [f.value for f in record.absorbed_faculties]
            
            cur.execute("""
                INSERT INTO cannibalism_events (
                    event_id, winner_genome_id, loser_genome_id,
                    winner_basin_before, winner_basin_after,
                    absorbed_faculties, absorption_rate, fisher_distance,
                    resurrection_eligible,
                    created_at, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
            """, (
                record.event_id,
                record.winner_genome_id,
                record.loser_genome_id,
                winner_before_vec,
                winner_after_vec,
                absorbed_faculties,
                record.absorption_rate,
                record.fisher_distance,
                record.resurrection_eligible,
                record.created_at,
                json.dumps(record.metadata)
            ))
            
            conn.commit()
            logger.info(f"Saved cannibalism record {record.event_id}")
    finally:
        if should_close:
            conn.close()


def save_genome_archive(
    archive: GenomeArchive,
    conn: Optional[psycopg2.extensions.connection] = None
) -> None:
    """Save genome archive to database."""
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO genome_archives (
                    archive_id, genome_id,
                    archival_reason, final_fitness,
                    resurrection_conditions, resurrection_eligible, resurrection_count,
                    archived_at, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (archive_id) DO UPDATE SET
                    resurrection_count = EXCLUDED.resurrection_count,
                    resurrection_eligible = EXCLUDED.resurrection_eligible
            """, (
                archive.archive_id,
                archive.genome.genome_id,
                archive.archival_reason,
                archive.final_fitness,
                json.dumps(archive.resurrection_conditions),
                archive.resurrection_eligible,
                archive.resurrection_count,
                archive.archived_at,
                json.dumps(archive.metadata)
            ))
            
            conn.commit()
            logger.info(f"Saved genome archive {archive.archive_id}")
    finally:
        if should_close:
            conn.close()


# =============================================================================
# QUERY FUNCTIONS
# =============================================================================

def get_genome_lineage(
    genome_id: str,
    max_depth: int = 10,
    conn: Optional[psycopg2.extensions.connection] = None
) -> List[Dict[str, Any]]:
    """
    Get lineage tree for a genome.
    
    Args:
        genome_id: Root genome ID
        max_depth: Maximum ancestor depth
        conn: Optional database connection
    
    Returns:
        List of lineage records
    """
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM kernel_genealogy
                WHERE genome_id = %s
                LIMIT 1
            """, (genome_id,))
            
            result = cur.fetchone()
            if result:
                return [dict(result)]
            return []
    finally:
        if should_close:
            conn.close()


def get_descendants(
    genome_id: str,
    max_depth: int = 10,
    conn: Optional[psycopg2.extensions.connection] = None
) -> List[Dict[str, Any]]:
    """
    Get all descendants of a genome.
    
    Args:
        genome_id: Genome ID to find descendants for
        max_depth: Maximum descendant depth
        conn: Optional database connection
    
    Returns:
        List of descendant records
    """
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM get_descendants(%s, %s)
            """, (genome_id, max_depth))
            
            return [dict(row) for row in cur.fetchall()]
    finally:
        if should_close:
            conn.close()


def get_evolution_summary(
    genome_id: str,
    conn: Optional[psycopg2.extensions.connection] = None
) -> Optional[Dict[str, Any]]:
    """
    Get evolution summary for a genome.
    
    Args:
        genome_id: Genome ID
        conn: Optional database connection
    
    Returns:
        Evolution summary dict or None
    """
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM kernel_evolution_summary
                WHERE genome_id = %s
            """, (genome_id,))
            
            result = cur.fetchone()
            if result:
                return dict(result)
            return None
    finally:
        if should_close:
            conn.close()
