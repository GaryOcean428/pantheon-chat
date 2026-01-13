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
    Update word_relationships in PostgreSQL using geometric data.
    QIG-PURE: Uses Fisher-Rao distances instead of PMI.
    """
    conn = get_db_connection()
    if not conn:
        print("[CurriculumTraining] No database connection - skipping relationship update")
        return
    
    try:
        cur = conn.cursor()
        relationships = geo_rel.get_all_relationships()
        
        records = []
        for word1, related in relationships.items():
            for word2, props in related.items():
                records.append((
                    word1,
                    word2,
                    props.get('fisher_rao_distance', 0.0),
                    props.get('qfi_weight', 0.5),
                ))
        
        if records:
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """
                INSERT INTO word_relationships (word1, word2, fisher_distance, qfi_weight)
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
        conn.close()


def update_word_relationships_cache(learner):
    """
    Update word_relationships in PostgreSQL.
    
    This enables online relationship updates so generation improves over time.
    PERSISTENCE: Uses PostgreSQL word_relationships table (NO JSON files).
    """
    conn = get_db_connection()
    if not conn:
        print("[CurriculumTraining] No database connection - skipping relationship update")
        return
    
    try:
        # Prepare batch data for upsert
        records = []
        for word, neighbors in learner.cooccurrence.items():
            for neighbor, count in neighbors.items():
                if word != neighbor:  # Prevent self-referential entries (DB constraint)
                    records.append((word, neighbor, float(count)))
        
        if not records:
            print("[CurriculumTraining] No relationships to update")
            conn.close()
            return
        
        with conn.cursor() as cur:
            # Upsert using ON CONFLICT - update count if higher
            execute_values(
                cur,
                """
                INSERT INTO word_relationships (word, neighbor, cooccurrence_count, updated_at)
                VALUES %s
                ON CONFLICT (word, neighbor) 
                DO UPDATE SET 
                    cooccurrence_count = GREATEST(word_relationships.cooccurrence_count, EXCLUDED.cooccurrence_count),
                    updated_at = NOW()
                """,
                records,
                template="(%s, %s, %s, NOW())"
            )
        
        conn.commit()
        conn.close()
        
        print(f"[CurriculumTraining] Saved {len(records)} word relationships to PostgreSQL")
        
    except Exception as e:
        print(f"[CurriculumTraining] PostgreSQL update error: {e}")
        try:
            conn.close()
        except:
            pass


def adjust_kernel_basins_from_relationships(learner, coordizer):
    """
    Adjust kernel basin coordinates based on learned word relationships.
    
    This implements the training loop: learned relationships update basins,
    which affects future generation.
    """
    try:
        print("[CurriculumTraining] Adjusting kernel basins from learned relationships...")
        
        # Get current basins from coordizer
        basins = dict(coordizer.basin_coords)
        
        # Adjust basins using word relationship learner
        adjusted_basins = learner.adjust_basin_coordinates(
            basins=basins,
            learning_rate=0.05,  # Small learning rate for stability
            iterations=5  # Few iterations to avoid overfitting
        )
        
        # Update coordizer with adjusted basins (if it supports updates)
        updated_count = 0
        for word, new_basin in adjusted_basins.items():
            if word in coordizer.basin_coords:
                # Check if adjustment is within reasonable bounds using Fisher-Rao distance
                old_basin = coordizer.basin_coords[word]
                distance = fisher_rao_distance(new_basin, old_basin)

                # Only update if change is moderate (not too radical)
                # Fisher-Rao max is π, so 0.5 corresponds to ~16% of max distance
                if distance < 0.5:  # Threshold for safety
                    coordizer.basin_coords[word] = new_basin
                    updated_count += 1
        
        print(f"[CurriculumTraining] Updated {updated_count} kernel basin coordinates")
        
    except Exception as e:
        print(f"[CurriculumTraining] Basin adjustment error: {e}")
