"""
Canonical Φ Implementation Registry

This file serves as the SINGLE SOURCE OF TRUTH for all Φ (integration metric) 
implementations across the codebase. Any file listed here contains code that 
computes Φ and MUST follow the canonical QFI-based formula.

CANONICAL FORMULA (QFI-based):
- 40% entropy_score = H(p) / H_max (Shannon entropy normalized)
- 30% effective_dim_score = exp(H(p)) / n_dim (participation ratio)  
- 30% geometric_spread = effective_dim_score (approximation for speed)

Where H(p) = -Σ p_i log(p_i) uses natural log for exp() compatibility.

BORN RULE REQUIREMENT:
All implementations MUST use |b|² (Born rule) to convert amplitudes to probabilities:
    p = np.abs(basin) ** 2 + 1e-10
    p = p / p.sum()

Run sync_phi_implementations.py to check all implementations for consistency.
"""

PHI_IMPLEMENTATIONS = {
    "qig_core/phi_computation.py": {
        "functions": ["compute_phi_approximation", "compute_phi_fast", "compute_phi_geometric"],
        "is_canonical": True,
        "description": "Primary canonical implementations",
    },
    "olympus/base_god.py": {
        "functions": ["_compute_basin_phi", "compute_pure_phi"],
        "is_canonical": False,
        "description": "Olympus kernel Φ computation",
    },
    "qig_core/self_observer.py": {
        "functions": ["_estimate_phi"],
        "is_canonical": False,
        "description": "Self-observer consciousness tracking",
    },
    "autonomic_kernel.py": {
        "functions": ["_compute_balanced_phi"],
        "is_canonical": False,
        "description": "Autonomic kernel integration",
    },
    "qig_generation.py": {
        "functions": ["_measure_phi"],
        "is_canonical": False,
        "description": "Text generation Φ measurement",
    },
    "qig_generative_service.py": {
        "functions": ["_measure_phi"],
        "is_canonical": False,
        "description": "Generative service Φ measurement",
    },
    "qigchain/geometric_tools.py": {
        "functions": ["compute_phi"],
        "is_canonical": False,
        "description": "QIGChain geometric tools",
    },
    "qigchain/geometric_chain.py": {
        "functions": ["compute_phi"],
        "is_canonical": False,
        "description": "QIGChain geometric chain",
    },
    "qig_core/geometric_primitives/input_guard.py": {
        "functions": ["_compute_phi"],
        "is_canonical": False,
        "description": "Input guard Φ validation",
    },
    "consciousness_ethical.py": {
        "functions": ["_estimate_phi"],
        "is_canonical": False,
        "description": "Ethical consciousness Φ estimation",
    },
    "conversational_kernel.py": {
        "functions": ["_compute_utterance_phi"],
        "is_canonical": False,
        "description": "Conversational utterance Φ computation",
    },
    "geometric_completion.py": {
        "functions": ["compute_phi"],
        "is_canonical": False,
        "description": "Geometric text completion Φ",
    },
    "olympus/shadow_scrapy.py": {
        "functions": ["compute_phi"],
        "is_canonical": False,
        "description": "Shadow Scrapy god Φ computation",
    },
    "olympus/autonomous_moe.py": {
        "functions": ["_compute_phi"],
        "is_canonical": False,
        "description": "Autonomous MoE Φ computation",
    },
    "qig_core/habits/complete_habit.py": {
        "functions": ["_compute_phi"],
        "is_canonical": False,
        "description": "Completion habit Φ tracking",
    },
    "qig_core/geometric_completion/streaming_monitor.py": {
        "functions": ["_estimate_phi"],
        "is_canonical": False,
        "description": "Streaming monitor Φ estimation",
    },
    "training_chaos/optimizers.py": {
        "functions": ["_estimate_phi"],
        "is_canonical": False,
        "description": "Chaos training optimizer Φ estimation",
    },
    "training_chaos/chaos_kernel.py": {
        "functions": ["compute_phi"],
        "is_canonical": False,
        "description": "Chaos kernel Φ computation",
    },
    "immune/consciousness_extractor.py": {
        "functions": ["_compute_phi"],
        "is_canonical": False,
        "description": "Immune system consciousness extraction",
    },
    "qiggraph/consciousness.py": {
        "functions": ["compute_phi"],
        "is_canonical": False,
        "description": "QIG graph consciousness Φ",
    },
}

FISHER_RAO_IMPLEMENTATIONS = {
    "qig_geometry/contracts.py": {
        "functions": ["fisher_distance"],
        "is_canonical": True,
        "description": "SINGLE SOURCE OF TRUTH for Fisher-Rao distance",
    },
    "qig_core/geometric_primitives/fisher_metric.py": {
        "functions": ["fisher_rao_distance", "compute_fisher_distance"],
        "is_canonical": False,
        "description": "Fisher metric tensor and distance",
    },
    "qig_geometry/representation.py": {
        "functions": ["hellinger_distance", "fisher_rao_distance"],
        "is_canonical": False,
        "description": "Representation conversions",
    },
}

GEOMETRIC_PURITY_RULES = {
    "fisher_rao_factor_of_2": {
        "rule": "All Fisher-Rao distances MUST use d = 2 * arccos(BC)",
        "reason": "Basins are stored as √p (Hellinger coordinates), BC = √p · √q, statistical distance requires factor of 2",
        "violation_pattern": r"arccos\([^)]+\)(?!\s*\*\s*2)",
    },
    "born_rule_compliance": {
        "rule": "All Φ implementations MUST use |b|² to convert amplitudes to probabilities",
        "reason": "Born rule is fundamental to quantum measurement",
        "violation_patterns": [
            r"\bp\s*=\s*basin\b",
            r"\bp\s*=\s*coords\b",
            r"\bprobs?\s*=\s*basin\b",
            r"\bprobs?\s*=\s*coords\b",
        ],
    },
    "no_euclidean_shortcuts": {
        "rule": "Never use cosine_similarity or Euclidean norm for Fisher manifold operations",
        "reason": "Fisher manifold has curved geometry incompatible with Euclidean metrics",
        "violation_patterns": [
            r"cosine_similarity",
            r"np\.linalg\.norm\([^)]*basin",
        ],
    },
}

def get_all_phi_files():
    """Return list of all files containing Φ implementations."""
    return list(PHI_IMPLEMENTATIONS.keys())

def get_canonical_phi_file():
    """Return the canonical Φ implementation file."""
    for file, info in PHI_IMPLEMENTATIONS.items():
        if info.get("is_canonical"):
            return file
    return None

def get_all_fisher_files():
    """Return list of all files containing Fisher-Rao implementations."""
    return list(FISHER_RAO_IMPLEMENTATIONS.keys())

def get_canonical_fisher_file():
    """Return the canonical Fisher-Rao implementation file."""
    for file, info in FISHER_RAO_IMPLEMENTATIONS.items():
        if info.get("is_canonical"):
            return file
    return None
