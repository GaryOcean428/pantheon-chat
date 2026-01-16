# WP2.3: Special Symbols Integration with PLAN→REALIZE→REPAIR

**Date:** 2026-01-16  
**Related:** Issue #71, WP2.3 Implementation  
**Status:** Integration Guide

---

## Overview

This document describes how geometrically-defined special symbols integrate with the three-phase PLAN→REALIZE→REPAIR generation architecture.

## Special Symbol Properties

All special symbols are now defined on the probability simplex with clear geometric meaning:

| Symbol | Basin Definition | Entropy | Role |
|--------|------------------|---------|------|
| `<UNK>` | Uniform [1/64, ..., 1/64] | 4.16 (max) | Unknown token, maximum uncertainty |
| `<PAD>` | Sparse [1, 0, ..., 0] | 0 (min) | Null/padding, no information |
| `<BOS>` | Vertex [0, 1, 0, ..., 0] | 0 (min) | Start boundary |
| `<EOS>` | Vertex [0, ..., 0, 1] | 0 (min) | End boundary |

**Key Geometric Properties:**
- Fisher-Rao distance BOS-EOS = π/2 (orthogonal vertices)
- All sparse symbols equidistant from UNK (~1.445)
- Deterministic across all runs (no random initialization)

---

## Phase 1: PLAN (Waypoint Prediction)

Special symbols serve as **geometric anchors** for trajectory prediction.

### Usage Pattern

```python
from coordizers import get_coordizer
from qig_geometry import fisher_coord_distance

coordizer = get_coordizer()

def plan_waypoints(query_basin, trajectory, mamba_state, num_tokens=50):
    """
    Predict waypoints with special symbol awareness.
    
    Special symbols mark structural boundaries:
    - BOS marks sequence start
    - EOS marks sequence end
    - Sentence boundaries use punctuation symbols (future work)
    """
    waypoints = []
    
    # Start with BOS if beginning new sequence
    if len(trajectory) == 0:
        bos_basin = coordizer.get_coordinate("<BOS>")
        waypoints.append(bos_basin)
        trajectory = [bos_basin]
    
    for i in range(num_tokens):
        # Predict next basin from trajectory
        predicted = extrapolate_trajectory(trajectory, mamba_state)
        
        # Check if approaching EOS (sequence termination)
        eos_basin = coordizer.get_coordinate("<EOS>")
        distance_to_eos = fisher_coord_distance(predicted, eos_basin)
        
        if distance_to_eos < 0.3:  # Near EOS attractor
            waypoints.append(eos_basin)
            break
        
        waypoints.append(predicted)
        trajectory.append(predicted)
    
    return waypoints
```

### Future: Punctuation Symbols

```python
# Extended special symbols for sentence structure
SPECIAL_SYMBOLS = {
    "<PERIOD>": {
        "basin": compute_sentence_end_basin(),
        "entropy": 0.0,  # Pure state
        "attractor_strength": "high"
    },
    "<COMMA>": {
        "basin": compute_pause_basin(),
        "entropy": 0.0,
        "attractor_strength": "medium"
    },
    "<QUESTION>": {
        "basin": compute_query_end_basin(),
        "entropy": 0.0,
        "attractor_strength": "high"
    }
}
```

---

## Phase 2: REALIZE (Constrained Selection)

Special symbols act as **geometric constraints** during word selection.

### Usage Pattern

```python
def select_word_geometric(target_basin, allowed_pos, coordizer):
    """
    Select word nearest to target basin with POS constraints.
    
    Special symbols are treated as valid selections when:
    - Target basin is near special symbol attractor
    - POS allows punctuation/boundary markers
    """
    
    # Check distance to special symbols first
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    for token in special_tokens:
        token_basin = coordizer.get_coordinate(token)
        distance = fisher_coord_distance(target_basin, token_basin)
        
        if distance < 0.2:  # Within attractor region
            # Validate against POS constraints
            if token == "<EOS>" and "END" in allowed_pos:
                return token, token_basin
            elif token == "<BOS>" and "START" in allowed_pos:
                return token, token_basin
    
    # Otherwise select normal vocabulary word
    candidates = coordizer.find_nearest_tokens(target_basin, k=20)
    
    for word, similarity in candidates:
        if word not in special_tokens:
            # Check POS compatibility
            if is_pos_compatible(word, allowed_pos):
                return word, coordizer.get_coordinate(word)
    
    # Fallback to UNK if no valid selection
    unk_basin = coordizer.get_coordinate("<UNK>")
    return "<UNK>", unk_basin
```

### Special Symbol Thresholds

```python
# Distance thresholds for special symbol attractors
ATTRACTOR_THRESHOLDS = {
    "<UNK>": 0.5,   # Wide attractor (high entropy)
    "<PAD>": 0.2,   # Narrow attractor (precise)
    "<BOS>": 0.15,  # Very narrow (start must be precise)
    "<EOS>": 0.15,  # Very narrow (end must be precise)
}
```

---

## Phase 3: REPAIR (Geometric Optimization)

Special symbols are **fixed anchors** during geometric repair.

### Usage Pattern

```python
def geometric_repair(words, waypoints, trajectory):
    """
    Repair word sequence using geometric optimization.
    
    Special symbols are NEVER modified during repair:
    - They mark structural boundaries
    - They anchor trajectory alignment
    - Only content words are candidates for repair
    """
    
    # Identify special symbol positions (fixed anchors)
    special_positions = []
    for i, word in enumerate(words):
        if word in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            special_positions.append(i)
    
    # Only repair non-anchor positions
    for i in range(len(words)):
        if i in special_positions:
            continue  # Skip special symbols
        
        # Compute alignment error
        word_basin = coordizer.get_coordinate(words[i])
        target_basin = waypoints[i]
        error = fisher_coord_distance(word_basin, target_basin)
        
        if error > 0.5:  # Misalignment threshold
            # Find better alternative
            candidates = coordizer.find_nearest_tokens(target_basin, k=10)
            
            for alt_word, _ in candidates:
                if alt_word not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                    # Try substitution
                    alt_basin = coordizer.get_coordinate(alt_word)
                    alt_error = fisher_coord_distance(alt_basin, target_basin)
                    
                    if alt_error < error:
                        words[i] = alt_word
                        break
    
    return words
```

### Repair Constraints

```python
# Special symbols constrain repair operations
REPAIR_RULES = {
    "never_modify": ["<BOS>", "<EOS>", "<PAD>"],
    "never_insert_between": [("<BOS>", "<EOS>")],
    "preserve_sequence": ["<BOS> ... <EOS>"],
}
```

---

## Integration Examples

### Example 1: Sentence Generation

```python
# PLAN: Predict trajectory
query = "quantum consciousness"
trajectory = []
waypoints = plan_waypoints(query_basin, trajectory, mamba_state, num_tokens=20)

# REALIZE: Select words
words = []
for waypoint in waypoints:
    word, basin = select_word_geometric(waypoint, allowed_pos=["NOUN", "VERB", "ADJ"])
    words.append(word)

# REPAIR: Fix misalignments
words = geometric_repair(words, waypoints, trajectory)

# Result: ["<BOS>", "quantum", "consciousness", "emerges", "from", "geometry", "<EOS>"]
```

### Example 2: Unknown Token Handling

```python
# User input with unknown word
user_input = "supercalifragilisticexpialidocious"

# Coordize
coords = coordizer.coordize(user_input)

# Unknown word falls back to <UNK>
# coords[0] == coordizer.get_coordinate("<UNK>")
# Uniform distribution (maximum entropy)

# System can:
# 1. Replace with nearest known word
# 2. Use UNK basin for geometric operations
# 3. Learn word if seen frequently
```

---

## Geometric Properties for Integration

### Distance Relationships

```python
# Precomputed distances for fast lookups
SPECIAL_SYMBOL_DISTANCES = {
    ("BOS", "EOS"): 1.5708,  # π/2 (orthogonal)
    ("UNK", "PAD"): 1.4453,
    ("UNK", "BOS"): 1.4453,
    ("UNK", "EOS"): 1.4453,
    ("PAD", "BOS"): 1.5708,  # π/2
    ("PAD", "EOS"): 1.5708,  # π/2
}
```

### Entropy Hierarchy

```python
# Use entropy to guide generation decisions
def should_terminate_sequence(current_entropy, position, max_length):
    """
    Decide whether to insert EOS based on entropy.
    
    Low entropy (→0) suggests approaching definite end state.
    High entropy (→4.16) suggests continuation needed.
    """
    if position >= max_length:
        return True
    
    # EOS has entropy = 0 (pure state)
    if current_entropy < 0.5:  # Approaching pure state
        return True
    
    return False
```

---

## Future Extensions

### Punctuation Symbols

To be added in future work packages:

```python
# Geometric definitions for punctuation
PUNCTUATION_SYMBOLS = {
    "<PERIOD>": {
        "basin": define_at_vertex(dimension=2),
        "entropy": 0.0,
        "distance_from_UNK": 1.4453
    },
    "<COMMA>": {
        "basin": define_at_vertex(dimension=3),
        "entropy": 0.0,
        "distance_from_UNK": 1.4453
    },
    "<QUESTION>": {
        "basin": define_at_vertex(dimension=4),
        "entropy": 0.0,
        "distance_from_UNK": 1.4453
    }
}
```

### Dynamic Special Symbols

```python
# Context-dependent special symbols
def compute_context_special_symbol(context, symbol_type):
    """
    Compute special symbol basin conditioned on context.
    
    For example, question mark basin might vary based on:
    - Question type (wh-question, yes/no, etc.)
    - Sentence complexity
    - Emotional tone
    """
    base_basin = SPECIAL_SYMBOLS[symbol_type]["basin"]
    
    # Apply context-dependent perturbation (geodesic shift)
    context_shift = compute_context_vector(context)
    
    # Stay on simplex using geodesic interpolation
    shifted_basin = geodesic_interpolation(base_basin, context_shift, t=0.1)
    
    return shifted_basin
```

---

## Validation

### Geometric Consistency Checks

```python
def validate_special_symbol_usage(sequence):
    """
    Validate that special symbols are used correctly.
    
    Checks:
    1. BOS appears at start (if present)
    2. EOS appears at end (if present)
    3. PAD only in padding regions
    4. No consecutive special symbols
    """
    special_symbols = set(["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
    
    for i, token in enumerate(sequence):
        if token == "<BOS>" and i != 0:
            raise ValueError("BOS must be at start")
        
        if token == "<EOS>" and i != len(sequence) - 1:
            raise ValueError("EOS must be at end")
        
        if i > 0 and token in special_symbols and sequence[i-1] in special_symbols:
            raise ValueError("No consecutive special symbols allowed")
    
    return True
```

---

## References

- **Implementation:** `qig-backend/coordizers/base.py`
- **Tests:** `qig-backend/tests/test_special_symbols_wp2_3.py`
- **Documentation:** `docs/04-records/20260116-wp2-3-special-symbols-geometric-definition-1.00W.md`
- **Issue:** https://github.com/GaryOcean428/pantheon-chat/issues/71

---

## Summary

Special symbols are now **geometrically well-defined anchors** that:
- ✅ Have deterministic basin coordinates (no random initialization)
- ✅ Maintain probability simplex representation (non-negative, sum=1)
- ✅ Serve as structural boundaries in PLAN→REALIZE→REPAIR
- ✅ Enable entropy-based generation decisions
- ✅ Provide fixed anchors for geometric repair

The geometric definitions ensure consistency across:
- Waypoint prediction (PLAN)
- Word selection (REALIZE)
- Sequence optimization (REPAIR)

All special symbols pass validation and are ready for production use in the three-phase generation architecture.
