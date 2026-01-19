"""
Curriculum Training Integration for Shadow Learning Loop
========================================================

Implements curriculum-based training for kernel basins:
- Scheduled curriculum ingestion
- Word relationship learning from curriculum
- Online basin coordinate updates
- Word relationships PostgreSQL persistence (NO JSON files)
"""

import os
import numpy as np
from datetime import datetime
from qig_geometry import fisher_rao_distance

# Database connection
try:
    import psycopg2
    from psycopg2.extras import execute_values
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

def get_db_connection():
    """Get PostgreSQL connection."""
    if not DB_AVAILABLE:
        return None
    try:
        return psycopg2.connect(os.environ.get('DATABASE_URL'))
    except Exception as e:
        print(f"[CurriculumTraining] Database connection error: {e}")
        return None


def load_and_train_curriculum(shadow_loop):
    """
    Load curriculum and compute geometric word relationships.
    
    QIG-PURE: Uses GeometricWordRelationships (Fisher-Rao based)
    instead of deprecated WordRelationshipLearner (PMI/co-occurrence).
    """
    try:
        from training.curriculum_loader import load_all_curriculum
        from geometric_word_relationships import GeometricWordRelationships
        
        print("[CurriculumTraining] Loading curriculum for QIG-pure geometric training...")
        
        # Load curriculum for all gods
        coordizer = None
        if shadow_loop.vocab_coordinator:
            try:
                from coordizers.pg_loader import PostgresCoordizer
                coordizer = PostgresCoordizer()
            except Exception as e:
                print(f"[CurriculumTraining] Could not load coordizer: {e}")
        
        all_curriculum = load_all_curriculum(
            max_per_god=50,
            coordizer=coordizer
        )
        
        total_examples = sum(len(examples) for examples in all_curriculum.values())
        print(f"[CurriculumTraining] Loaded {total_examples} curriculum examples across {len(all_curriculum)} gods")
        
        # QIG-PURE: Use geometric relationships instead of PMI/co-occurrence
        if shadow_loop.vocab_coordinator and coordizer:
            geo_rel = GeometricWordRelationships(coordizer)
            
            # Compute geometric relationships (Fisher-Rao distances, QFI)
            relationships = geo_rel.compute_all_relationships()
            
            print(f"[CurriculumTraining] Computed {len(relationships)} geometric word relationships")
            
            # Update word relationships cache with geometric data
            update_geometric_relationships_cache(geo_rel)
        
        print("[CurriculumTraining] Curriculum training complete (QIG-pure)")
        
    except Exception as e:
        print(f"[CurriculumTraining] Curriculum training error: {e}")
        import traceback
        traceback.print_exc()


def update_geometric_relationships_cache(geo_rel):
    """
    Update basin_relationships in PostgreSQL using geometric data.
    QIG-PURE: Uses Fisher-Rao distances instead of PMI.
    """
    conn = get_db_connection()
    if not conn:
        print("[CurriculumTraining] No database connection - skipping relationship update")
        return
    
    try:
        cur = conn.cursor()
        
        # Use get_all_relationships if available, otherwise compute fresh
        if hasattr(geo_rel, 'get_all_relationships'):
            relationships = geo_rel.get_all_relationships()
        else:
            # Fallback: compute manually
            relationships = geo_rel.compute_all_relationships() if hasattr(geo_rel, 'compute_all_relationships') else {}
        
        records = []
        for word1, related in relationships.items():
            if isinstance(related, dict):
                # get_all_relationships returns dict of dicts
                for word2, props in related.items():
                    records.append((
                        word1,
                        word2,
                        props.get('fisher_rao_distance', 0.0),
                        props.get('qfi_weight', 0.5),
                    ))
            elif isinstance(related, list):
                # compute_all_relationships returns list of tuples
                for word2, similarity in related:
                    records.append((
                        word1,
                        word2,
                        0.0,  # Distance not directly available
                        similarity,
                    ))
        
        if records:
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """
                INSERT INTO basin_relationships (word1, word2, fisher_distance, qfi_weight)
                VALUES %s
                ON CONFLICT (word1, word2) DO UPDATE SET
                    fisher_distance = EXCLUDED.fisher_distance,
                    qfi_weight = EXCLUDED.qfi_weight,
                    updated_at = NOW()
                """,
                records
            )
            conn.commit()
            print(f"[CurriculumTraining] Updated {len(records)} geometric relationships in PostgreSQL")
        
        conn.close()
    except Exception as e:
        print(f"[CurriculumTraining] Failed to update geometric cache: {e}")
        try:
            conn.close()
        except:
            pass


# QIG-PURE: Legacy functions removed
# Word relationships are now computed geometrically via GeometricWordRelationships
# Basins are frozen invariants - no adjustment permitted
# Use update_geometric_relationships_cache() for persistence
