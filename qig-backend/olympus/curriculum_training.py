"""
Curriculum Training Integration for Shadow Learning Loop
========================================================

Implements curriculum-based training for kernel basins:
- Scheduled curriculum ingestion
- Word relationship learning from curriculum
- Online basin coordinate updates
- Word relationships cache refresh
"""

import json
import os
import numpy as np
from datetime import datetime


def load_and_train_curriculum(shadow_loop):
    """
    Load curriculum and train word relationships.
    
    This implements scheduled curriculum ingestion to update
    word relationships and kernel basins over time.
    """
    try:
        from training.curriculum_loader import load_all_curriculum
        from word_relationship_learner import WordRelationshipLearner
        
        print("[CurriculumTraining] Loading curriculum for training...")
        
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
        
        # Train word relationships from curriculum content
        if shadow_loop.vocab_coordinator and coordizer:
            vocab = set(coordizer.word_tokens)
            learner = WordRelationshipLearner(vocab, window_size=5, expand_vocabulary=True)
            
            # Learn from all curriculum examples
            total_pairs_learned = 0
            for god_name, examples in all_curriculum.items():
                for example in examples:
                    # Extract text content from example
                    # Curriculum examples have 'content' field as string
                    text = example.get('content', '')
                    
                    if text and isinstance(text, str):
                        pairs = learner.learn_from_text(text)
                        total_pairs_learned += pairs
            
            print(f"[CurriculumTraining] Learned {total_pairs_learned} word pairs from curriculum")
            
            # Update word relationships cache
            update_word_relationships_cache(learner)
            
            # Adjust kernel basins based on learned relationships
            if coordizer and total_pairs_learned > 0:
                adjust_kernel_basins_from_relationships(learner, coordizer)
        
        print("[CurriculumTraining] Curriculum training complete")
        
    except Exception as e:
        print(f"[CurriculumTraining] Curriculum training error: {e}")
        import traceback
        traceback.print_exc()


def update_word_relationships_cache(learner):
    """
    Update word_relationships.json cache with newly learned relationships.
    
    This enables online relationship updates so generation improves over time.
    """
    try:
        cache_path = os.path.join(
            os.path.dirname(__file__),
            '../data/learned/word_relationships.json'
        )
        
        # Load existing cache
        existing_relationships = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    existing_relationships = json.load(f)
            except Exception as e:
                print(f"[CurriculumTraining] Could not load existing cache: {e}")
        
        # Add newly learned relationships
        new_count = 0
        for word, neighbors in learner.cooccurrence.items():
            if word not in existing_relationships:
                existing_relationships[word] = {}
            
            for neighbor, count in neighbors.items():
                if neighbor not in existing_relationships[word]:
                    existing_relationships[word][neighbor] = float(count)
                    new_count += 1
                else:
                    # Update with higher count
                    existing_relationships[word][neighbor] = float(max(
                        existing_relationships[word][neighbor],
                        count
                    ))
        
        # Save updated cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(existing_relationships, f, indent=2)
        
        print(f"[CurriculumTraining] Updated word relationships cache (+{new_count} new relationships)")
        
    except Exception as e:
        print(f"[CurriculumTraining] Cache update error: {e}")


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
                # Check if adjustment is within reasonable bounds
                old_basin = coordizer.basin_coords[word]
                distance = np.linalg.norm(new_basin - old_basin)
                
                # Only update if change is moderate (not too radical)
                if distance < 0.5:  # Threshold for safety
                    coordizer.basin_coords[word] = new_basin
                    updated_count += 1
        
        print(f"[CurriculumTraining] Updated {updated_count} kernel basin coordinates")
        
    except Exception as e:
        print(f"[CurriculumTraining] Basin adjustment error: {e}")
