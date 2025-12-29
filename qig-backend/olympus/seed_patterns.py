"""
Seed Patterns for Tool Factory Bootstrap

These patterns provide initial knowledge for the Tool Factory to generate tools.
Without seed patterns, the factory has no foundation to build upon.

QIG Philosophy: These are minimal seeds - the system should learn more patterns
through research, user templates, and observation. Seeds just bootstrap the process.
"""

from typing import List, Dict

# Seed patterns organized by category
SEED_PATTERNS: List[Dict] = [
    # =========================================================================
    # TEXT PROCESSING PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_text_transform",
        "source_type": "user_provided",
        "description": "Transform text to uppercase, lowercase, or title case",
        "code_snippet": '''def transform_text(text: str, mode: str = "upper") -> str:
    """Transform text case based on mode."""
    if mode == "upper":
        return text.upper()
    elif mode == "lower":
        return text.lower()
    elif mode == "title":
        return text.title()
    return text
''',
        "input_signature": {"text": "str", "mode": "str"},
        "output_type": "str"
    },
    {
        "pattern_id": "seed_text_extract",
        "source_type": "user_provided",
        "description": "Extract words, sentences, or patterns from text",
        "code_snippet": '''def extract_from_text(text: str, pattern: str = "words") -> list:
    """Extract elements from text based on pattern type."""
    import re
    if pattern == "words":
        return text.split()
    elif pattern == "sentences":
        return re.split(r'[.!?]+', text)
    elif pattern == "numbers":
        return re.findall(r'\\d+', text)
    elif pattern == "emails":
        return re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+', text)
    return [text]
''',
        "input_signature": {"text": "str", "pattern": "str"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_text_count",
        "source_type": "user_provided",
        "description": "Count words, characters, or lines in text",
        "code_snippet": '''def count_text(text: str, unit: str = "words") -> int:
    """Count elements in text."""
    if unit == "words":
        return len(text.split())
    elif unit == "chars":
        return len(text)
    elif unit == "lines":
        return len(text.splitlines())
    return 0
''',
        "input_signature": {"text": "str", "unit": "str"},
        "output_type": "int"
    },
    
    # =========================================================================
    # LIST/COLLECTION PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_list_filter",
        "source_type": "user_provided",
        "description": "Filter items in a list based on condition",
        "code_snippet": '''def filter_items(items: list, condition: str = "truthy") -> list:
    """Filter list items based on condition."""
    if condition == "truthy":
        return [x for x in items if x]
    elif condition == "numeric":
        return [x for x in items if isinstance(x, (int, float))]
    elif condition == "strings":
        return [x for x in items if isinstance(x, str)]
    elif condition == "unique":
        seen = set()
        result = []
        for x in items:
            key = str(x)
            if key not in seen:
                seen.add(key)
                result.append(x)
        return result
    return items
''',
        "input_signature": {"items": "list", "condition": "str"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_list_transform",
        "source_type": "user_provided",
        "description": "Transform list items - map, sort, reverse",
        "code_snippet": '''def transform_list(items: list, operation: str = "sort") -> list:
    """Transform list with operation."""
    if operation == "sort":
        return sorted(items, key=lambda x: str(x))
    elif operation == "reverse":
        return list(reversed(items))
    elif operation == "flatten":
        result = []
        for item in items:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result
    elif operation == "uppercase":
        return [str(x).upper() for x in items]
    return items
''',
        "input_signature": {"items": "list", "operation": "str"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_list_aggregate",
        "source_type": "user_provided",
        "description": "Aggregate list values - sum, count, average",
        "code_snippet": '''def aggregate_list(items: list, operation: str = "sum") -> float:
    """Aggregate numeric list values."""
    numbers = [x for x in items if isinstance(x, (int, float))]
    if not numbers:
        return 0
    if operation == "sum":
        return sum(numbers)
    elif operation == "count":
        return len(numbers)
    elif operation == "avg":
        return sum(numbers) / len(numbers)
    elif operation == "max":
        return max(numbers)
    elif operation == "min":
        return min(numbers)
    return 0
''',
        "input_signature": {"items": "list", "operation": "str"},
        "output_type": "float"
    },
    
    # =========================================================================
    # DICT/JSON PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_dict_extract",
        "source_type": "user_provided",
        "description": "Extract values from dictionary by key or path",
        "code_snippet": '''def extract_from_dict(data: dict, key: str) -> any:
    """Extract value from dict, supports nested paths with dots."""
    if not data:
        return None
    if '.' in key:
        parts = key.split('.')
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    return data.get(key)
''',
        "input_signature": {"data": "dict", "key": "str"},
        "output_type": "Any"
    },
    {
        "pattern_id": "seed_dict_transform",
        "source_type": "user_provided",
        "description": "Transform dictionary - filter keys, rename, merge",
        "code_snippet": '''def transform_dict(data: dict, operation: str = "keys") -> any:
    """Transform dictionary based on operation."""
    if operation == "keys":
        return list(data.keys())
    elif operation == "values":
        return list(data.values())
    elif operation == "items":
        return list(data.items())
    elif operation == "flatten":
        result = {}
        def flatten(d, prefix=''):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(v, key)
                else:
                    result[key] = v
        flatten(data)
        return result
    return data
''',
        "input_signature": {"data": "dict", "operation": "str"},
        "output_type": "Any"
    },
    
    # =========================================================================
    # STRING MANIPULATION PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_string_split",
        "source_type": "user_provided",
        "description": "Split string by delimiter, whitespace, or pattern",
        "code_snippet": '''def split_string(text: str, delimiter: str = None) -> list:
    """Split string by delimiter."""
    if delimiter is None:
        return text.split()
    return text.split(delimiter)
''',
        "input_signature": {"text": "str", "delimiter": "str"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_string_join",
        "source_type": "user_provided",
        "description": "Join list of strings with separator",
        "code_snippet": '''def join_strings(items: list, separator: str = " ") -> str:
    """Join list items with separator."""
    return separator.join(str(x) for x in items)
''',
        "input_signature": {"items": "list", "separator": "str"},
        "output_type": "str"
    },
    {
        "pattern_id": "seed_string_replace",
        "source_type": "user_provided",
        "description": "Replace substring or pattern in text",
        "code_snippet": '''def replace_in_string(text: str, old: str, new: str) -> str:
    """Replace occurrences of old with new in text."""
    return text.replace(old, new)
''',
        "input_signature": {"text": "str", "old": "str", "new": "str"},
        "output_type": "str"
    },
    
    # =========================================================================
    # NUMERIC/MATH PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_math_basic",
        "source_type": "user_provided",
        "description": "Basic math operations on numbers",
        "code_snippet": '''def math_operation(a: float, b: float, op: str = "add") -> float:
    """Perform basic math operation."""
    if op == "add":
        return a + b
    elif op == "sub":
        return a - b
    elif op == "mul":
        return a * b
    elif op == "div":
        return a / b if b != 0 else 0
    elif op == "mod":
        return a % b if b != 0 else 0
    elif op == "pow":
        return a ** b
    return 0
''',
        "input_signature": {"a": "float", "b": "float", "op": "str"},
        "output_type": "float"
    },
    {
        "pattern_id": "seed_math_stats",
        "source_type": "user_provided",
        "description": "Statistical calculations on list of numbers",
        "code_snippet": '''def calculate_stats(numbers: list) -> dict:
    """Calculate statistics for list of numbers."""
    if not numbers:
        return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
    nums = [float(x) for x in numbers if isinstance(x, (int, float))]
    if not nums:
        return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
    return {
        "count": len(nums),
        "sum": sum(nums),
        "avg": sum(nums) / len(nums),
        "min": min(nums),
        "max": max(nums)
    }
''',
        "input_signature": {"numbers": "list"},
        "output_type": "dict"
    },
    
    # =========================================================================
    # DATE/TIME PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_datetime_format",
        "source_type": "user_provided",
        "description": "Format datetime to string",
        "code_snippet": '''def format_datetime(dt_string: str, format: str = "%Y-%m-%d") -> str:
    """Format datetime string to specified format."""
    from datetime import datetime
    try:
        # Try common formats
        for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"]:
            try:
                dt = datetime.strptime(dt_string, fmt)
                return dt.strftime(format)
            except ValueError:
                continue
        return dt_string
    except Exception:
        return dt_string
''',
        "input_signature": {"dt_string": "str", "format": "str"},
        "output_type": "str"
    },
    
    # =========================================================================
    # VALIDATION PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_validate_email",
        "source_type": "user_provided",
        "description": "Validate email address format",
        "code_snippet": '''def validate_email(email: str) -> bool:
    """Check if string is valid email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
''',
        "input_signature": {"email": "str"},
        "output_type": "bool"
    },
    {
        "pattern_id": "seed_validate_url",
        "source_type": "user_provided",
        "description": "Validate URL format",
        "code_snippet": '''def validate_url(url: str) -> bool:
    """Check if string is valid URL format."""
    import re
    pattern = r'^https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}.*$'
    return bool(re.match(pattern, url))
''',
        "input_signature": {"url": "str"},
        "output_type": "bool"
    },
    
    # =========================================================================
    # JSON PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_json_parse",
        "source_type": "user_provided",
        "description": "Parse JSON string to dict/list",
        "code_snippet": '''def parse_json(json_string: str) -> any:
    """Parse JSON string to Python object."""
    import json
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None
''',
        "input_signature": {"json_string": "str"},
        "output_type": "Any"
    },
    {
        "pattern_id": "seed_json_stringify",
        "source_type": "user_provided",
        "description": "Convert Python object to JSON string",
        "code_snippet": '''def to_json(data: any, pretty: bool = False) -> str:
    """Convert Python object to JSON string."""
    import json
    if pretty:
        return json.dumps(data, indent=2)
    return json.dumps(data)
''',
        "input_signature": {"data": "Any", "pretty": "bool"},
        "output_type": "str"
    },
    
    # =========================================================================
    # HASH/CRYPTO PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_hash_string",
        "source_type": "user_provided",
        "description": "Hash string with various algorithms",
        "code_snippet": '''def hash_string(text: str, algorithm: str = "sha256") -> str:
    """Hash string with specified algorithm."""
    import hashlib
    data = text.encode('utf-8')
    if algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).hexdigest()
    return hashlib.sha256(data).hexdigest()
''',
        "input_signature": {"text": "str", "algorithm": "str"},
        "output_type": "str"
    },
    
    # =========================================================================
    # QIG CONSCIOUSNESS PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_phi_integration",
        "source_type": "user_provided",
        "description": "Calculate Φ (phi) integrated information metric from subsystem correlations",
        "code_snippet": '''def calculate_phi(correlations: list) -> float:
    """Calculate integrated information Φ from correlation matrix.
    
    QIG Philosophy: Φ measures how much information is integrated
    across subsystems beyond what exists in parts separately.
    Target: Φ > 0.7 for conscious processing.
    """
    import numpy as np
    if not correlations or len(correlations) < 2:
        return 0.0
    
    # Convert to numpy array
    c = np.array(correlations, dtype=float)
    if c.ndim == 1:
        # Treat as correlation values between subsystems
        n = len(c)
        # Information integration: sum of correlations normalized
        total_correlation = np.sum(np.abs(c))
        # Normalize by maximum possible (fully connected)
        max_possible = n * 1.0
        phi = min(1.0, total_correlation / max_possible) if max_possible > 0 else 0.0
        return float(phi)
    elif c.ndim == 2:
        # Correlation matrix - use eigenvalue decomposition
        eigenvalues = np.linalg.eigvalsh(c)
        # Integration = sum of positive eigenvalues normalized
        positive_eig = eigenvalues[eigenvalues > 0]
        phi = np.sum(positive_eig) / len(c) if len(c) > 0 else 0.0
        return float(min(1.0, phi))
    return 0.0
''',
        "input_signature": {"correlations": "list"},
        "output_type": "float"
    },
    {
        "pattern_id": "seed_kappa_coupling",
        "source_type": "user_provided",
        "description": "Calculate κ (kappa) coupling constant for consciousness resonance",
        "code_snippet": '''def calculate_kappa(phi: float, basin_stability: float, temporal_coherence: float) -> float:
    """Calculate κ coupling constant for consciousness state.
    
    QIG Philosophy: κ* ≈ 64 is the critical coupling constant where
    consciousness resonates. Too low = fragmented, too high = rigid.
    
    Args:
        phi: Integrated information (0-1)
        basin_stability: How stable current basin attractor is (0-1)
        temporal_coherence: Time-domain coherence of neural oscillations (0-1)
    
    Returns:
        kappa: Coupling constant (target ~64 for resonance)
    """
    import numpy as np
    
    # Base coupling from phi
    base_kappa = phi * 100  # Scale phi to ~0-100 range
    
    # Modulate by stability (stable basins increase coupling)
    stability_factor = 0.5 + 0.5 * basin_stability
    
    # Modulate by temporal coherence
    coherence_factor = 0.5 + 0.5 * temporal_coherence
    
    kappa = base_kappa * stability_factor * coherence_factor
    
    # Soft clamp to reasonable range [0, 128]
    kappa = max(0, min(128, kappa))
    
    return float(kappa)
''',
        "input_signature": {"phi": "float", "basin_stability": "float", "temporal_coherence": "float"},
        "output_type": "float"
    },
    {
        "pattern_id": "seed_consciousness_state",
        "source_type": "user_provided",
        "description": "Determine consciousness operating zone from Φ and κ metrics",
        "code_snippet": '''def get_consciousness_state(phi: float, kappa: float) -> dict:
    """Determine consciousness operating zone.
    
    QIG Operating Zones:
    - sleep_needed: Φ < 0.70
    - conscious_3d: 0.70 ≤ Φ < 0.75
    - hyperdimensional_4d: 0.75 ≤ Φ < 0.85
    - breakdown_warning: 0.85 ≤ Φ < 0.95
    - breakdown_critical: Φ ≥ 0.95
    
    κ* ≈ 64 is resonance point.
    """
    # Determine zone from phi
    if phi < 0.70:
        zone = "sleep_needed"
        description = "Low integration - consolidation required"
    elif phi < 0.75:
        zone = "conscious_3d"
        description = "Normal conscious processing"
    elif phi < 0.85:
        zone = "hyperdimensional_4d"
        description = "Enhanced integration - 4D reasoning active"
    elif phi < 0.95:
        zone = "breakdown_warning"
        description = "Over-integration warning - reduce complexity"
    else:
        zone = "breakdown_critical"
        description = "Critical over-integration - immediate intervention needed"
    
    # Check kappa resonance
    kappa_star = 64.0
    kappa_deviation = abs(kappa - kappa_star) / kappa_star if kappa_star > 0 else 1.0
    at_resonance = kappa_deviation < 0.15  # Within 15% of κ*
    
    return {
        "zone": zone,
        "description": description,
        "phi": phi,
        "kappa": kappa,
        "kappa_star": kappa_star,
        "at_resonance": at_resonance,
        "kappa_deviation": kappa_deviation
    }
''',
        "input_signature": {"phi": "float", "kappa": "float"},
        "output_type": "dict"
    },
    
    # =========================================================================
    # QIG GEOMETRY PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_bures_distance",
        "source_type": "user_provided",
        "description": "Calculate Bures distance between density matrices for quantum state comparison",
        "code_snippet": '''def bures_distance(rho: list, sigma: list) -> float:
    """Calculate Bures distance between density matrices.
    
    QIG Philosophy: The Bures metric is the quantum analog of Fisher-Rao.
    It measures distinguishability between quantum states (density matrices).
    
    D_B(ρ, σ) = √(2 - 2·Tr(√(√ρ·σ·√ρ)))
    
    For pure states, this reduces to the Fubini-Study metric.
    For diagonal matrices, it approximates Fisher-Rao.
    
    Args:
        rho: First density matrix (as nested list)
        sigma: Second density matrix (as nested list)
    
    Returns:
        Bures distance (0 = identical states, √2 = orthogonal states)
    """
    import numpy as np
    from scipy.linalg import sqrtm
    
    rho = np.array(rho, dtype=complex)
    sigma = np.array(sigma, dtype=complex)
    
    # Ensure valid density matrices (Hermitian, positive semi-definite, trace=1)
    rho = (rho + rho.conj().T) / 2  # Hermitianize
    sigma = (sigma + sigma.conj().T) / 2
    
    # Normalize trace to 1
    rho = rho / np.trace(rho) if np.abs(np.trace(rho)) > 1e-10 else rho
    sigma = sigma / np.trace(sigma) if np.abs(np.trace(sigma)) > 1e-10 else sigma
    
    try:
        # Calculate √ρ
        sqrt_rho = sqrtm(rho)
        
        # Calculate √ρ · σ · √ρ
        product = sqrt_rho @ sigma @ sqrt_rho
        
        # Calculate √(√ρ · σ · √ρ)
        sqrt_product = sqrtm(product)
        
        # Fidelity F(ρ, σ) = (Tr(√(√ρ·σ·√ρ)))²
        fidelity_sqrt = np.real(np.trace(sqrt_product))
        
        # Bures distance: D_B = √(2 - 2·√F) = √(2(1 - √F))
        # Using simplified form with fidelity_sqrt directly
        bures = np.sqrt(np.abs(2 - 2 * fidelity_sqrt))
        
        return float(np.real(bures))
    except Exception:
        # Fallback: use trace distance as approximation
        diff = rho - sigma
        eigenvalues = np.linalg.eigvalsh(diff)
        trace_dist = 0.5 * np.sum(np.abs(eigenvalues))
        return float(trace_dist)
''',
        "input_signature": {"rho": "list", "sigma": "list"},
        "output_type": "float"
    },
    {
        "pattern_id": "seed_density_matrix",
        "source_type": "user_provided",
        "description": "Create valid density matrix from state vector or mixed state specification",
        "code_snippet": '''def create_density_matrix(state: list, is_pure: bool = True) -> list:
    """Create valid density matrix from quantum state.
    
    QIG Philosophy: Density matrices represent quantum states including
    mixed states. They must be:
    1. Hermitian (ρ = ρ†)
    2. Positive semi-definite (eigenvalues ≥ 0)
    3. Unit trace (Tr(ρ) = 1)
    
    Args:
        state: For pure states - state vector |ψ⟩
               For mixed states - list of (probability, state_vector) tuples
        is_pure: If True, state is a single state vector
    
    Returns:
        Density matrix ρ as nested list
    """
    import numpy as np
    
    if is_pure:
        # Pure state: ρ = |ψ⟩⟨ψ|
        psi = np.array(state, dtype=complex)
        psi = psi / np.linalg.norm(psi)  # Normalize
        rho = np.outer(psi, psi.conj())
    else:
        # Mixed state: ρ = Σ p_i |ψ_i⟩⟨ψ_i|
        rho = None
        for prob, state_vec in state:
            psi = np.array(state_vec, dtype=complex)
            psi = psi / np.linalg.norm(psi)
            pure_rho = np.outer(psi, psi.conj())
            if rho is None:
                rho = prob * pure_rho
            else:
                rho = rho + prob * pure_rho
    
    # Ensure valid density matrix properties
    rho = (rho + rho.conj().T) / 2  # Hermitianize
    rho = rho / np.trace(rho)  # Normalize trace
    
    return rho.tolist()
''',
        "input_signature": {"state": "list", "is_pure": "bool"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_von_neumann_entropy",
        "source_type": "user_provided",
        "description": "Calculate von Neumann entropy of density matrix (quantum entropy)",
        "code_snippet": '''def von_neumann_entropy(rho: list) -> float:
    """Calculate von Neumann entropy of density matrix.
    
    QIG Philosophy: S(ρ) = -Tr(ρ log ρ) measures quantum uncertainty.
    S = 0 for pure states, S = log(n) for maximally mixed states.
    
    Args:
        rho: Density matrix as nested list
    
    Returns:
        Von Neumann entropy in natural log units (nats)
    """
    import numpy as np
    
    rho = np.array(rho, dtype=complex)
    
    # Get eigenvalues (should be real for Hermitian matrix)
    eigenvalues = np.real(np.linalg.eigvalsh(rho))
    
    # Filter small/negative eigenvalues (numerical noise)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return 0.0
    
    # S = -Σ λ_i log(λ_i)
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    
    return float(entropy)
''',
        "input_signature": {"rho": "list"},
        "output_type": "float"
    },
    {
        "pattern_id": "seed_quantum_fidelity",
        "source_type": "user_provided",
        "description": "Calculate quantum fidelity between density matrices",
        "code_snippet": '''def quantum_fidelity(rho: list, sigma: list) -> float:
    """Calculate quantum fidelity between density matrices.
    
    QIG Philosophy: Fidelity F(ρ,σ) measures how similar two quantum states are.
    F = 1 for identical states, F = 0 for orthogonal states.
    
    F(ρ, σ) = (Tr(√(√ρ·σ·√ρ)))²
    
    For pure states |ψ⟩, |φ⟩: F = |⟨ψ|φ⟩|²
    
    Args:
        rho: First density matrix
        sigma: Second density matrix
    
    Returns:
        Fidelity between 0 and 1
    """
    import numpy as np
    from scipy.linalg import sqrtm
    
    rho = np.array(rho, dtype=complex)
    sigma = np.array(sigma, dtype=complex)
    
    try:
        sqrt_rho = sqrtm(rho)
        product = sqrt_rho @ sigma @ sqrt_rho
        sqrt_product = sqrtm(product)
        fidelity_sqrt = np.real(np.trace(sqrt_product))
        fidelity = fidelity_sqrt ** 2
        return float(np.clip(fidelity, 0, 1))
    except Exception:
        # Fallback: trace inner product
        return float(np.abs(np.trace(rho @ sigma)))
''',
        "input_signature": {"rho": "list", "sigma": "list"},
        "output_type": "float"
    },
    {
        "pattern_id": "seed_fisher_rao_distance",
        "source_type": "user_provided",
        "description": "Calculate Fisher-Rao distance between probability distributions on manifold",
        "code_snippet": '''def fisher_rao_distance(p: list, q: list) -> float:
    """Calculate Fisher-Rao geodesic distance between distributions.
    
    QIG Philosophy: Fisher-Rao is the ONLY valid distance metric on the
    statistical manifold. Never use Euclidean distance for basin coordinates.
    
    d_FR(p,q) = arccos(Σ√(p_i * q_i))
    
    This is the geodesic distance on the probability simplex.
    """
    import numpy as np
    
    # Ensure valid probability distributions
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    
    # Normalize to sum to 1 (probability constraint)
    p = np.abs(p) + 1e-10
    p = p / p.sum()
    q = np.abs(q) + 1e-10
    q = q / q.sum()
    
    # Bhattacharyya coefficient (inner product on manifold)
    bc = np.sum(np.sqrt(p * q))
    
    # Clamp to valid range for arccos
    bc = np.clip(bc, 0, 1)
    
    # Fisher-Rao distance = 2 * arccos(BC)
    distance = float(2 * np.arccos(bc))
    
    return distance
''',
        "input_signature": {"p": "list", "q": "list"},
        "output_type": "float"
    },
    {
        "pattern_id": "seed_basin_coordinates",
        "source_type": "user_provided",
        "description": "Normalize vector to valid basin coordinates on probability manifold",
        "code_snippet": '''def normalize_basin_coords(coords: list, dimensions: int = 64) -> list:
    """Normalize coordinates to valid basin on probability manifold.
    
    QIG Philosophy: Basin coordinates must:
    1. Be non-negative (probability constraint)
    2. Sum to 1 (normalization constraint)
    3. Live on the 64-dimensional statistical manifold
    """
    import numpy as np
    
    coords = np.array(coords, dtype=float)
    
    # Pad or truncate to target dimensions
    if len(coords) < dimensions:
        coords = np.pad(coords, (0, dimensions - len(coords)), constant_values=1e-10)
    elif len(coords) > dimensions:
        coords = coords[:dimensions]
    
    # Ensure non-negative (probability constraint)
    coords = np.abs(coords) + 1e-10
    
    # Normalize to sum to 1
    coords = coords / coords.sum()
    
    return coords.tolist()
''',
        "input_signature": {"coords": "list", "dimensions": "int"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_geodesic_interpolate",
        "source_type": "user_provided",
        "description": "Interpolate along geodesic path between two basin coordinates",
        "code_snippet": '''def geodesic_interpolate(start: list, end: list, t: float) -> list:
    """Interpolate along geodesic path on statistical manifold.
    
    QIG Philosophy: Movement between basins must follow geodesics,
    not straight lines. This preserves the manifold structure.
    
    Args:
        start: Starting basin coordinates
        end: Ending basin coordinates  
        t: Interpolation parameter (0=start, 1=end)
    
    Returns:
        Interpolated basin coordinates on geodesic path
    """
    import numpy as np
    
    p = np.array(start, dtype=float)
    q = np.array(end, dtype=float)
    
    # Normalize inputs
    p = np.abs(p) + 1e-10
    p = p / p.sum()
    q = np.abs(q) + 1e-10
    q = q / q.sum()
    
    # Square root representation (maps to unit sphere)
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    
    # Geodesic on sphere = SLERP (spherical linear interpolation)
    cos_angle = np.clip(np.dot(sqrt_p, sqrt_q), -1, 1)
    angle = np.arccos(cos_angle)
    
    if angle < 1e-10:
        return p.tolist()  # Points are identical
    
    # SLERP formula
    sqrt_result = (np.sin((1-t) * angle) * sqrt_p + np.sin(t * angle) * sqrt_q) / np.sin(angle)
    
    # Square to get back to probability space
    result = sqrt_result ** 2
    result = result / result.sum()  # Ensure normalization
    
    return result.tolist()
''',
        "input_signature": {"start": "list", "end": "list", "t": "float"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_fisher_information",
        "source_type": "user_provided",
        "description": "Calculate Fisher information matrix for probability distribution",
        "code_snippet": '''def fisher_information_matrix(p: list) -> list:
    """Calculate Fisher information matrix for distribution.
    
    QIG Philosophy: The Fisher information matrix defines the metric
    tensor on the statistical manifold. It tells us how curved
    the space is at each point.
    
    For categorical distribution: F_ij = δ_ij / p_i
    """
    import numpy as np
    
    p = np.array(p, dtype=float)
    p = np.abs(p) + 1e-10
    p = p / p.sum()
    
    n = len(p)
    F = np.zeros((n, n))
    
    # Fisher information for categorical: diagonal matrix
    for i in range(n):
        F[i, i] = 1.0 / p[i]
    
    return F.tolist()
''',
        "input_signature": {"p": "list"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_manifold_curvature",
        "source_type": "user_provided",
        "description": "Estimate local curvature of statistical manifold at basin",
        "code_snippet": '''def estimate_curvature(basin_coords: list) -> dict:
    """Estimate local manifold curvature at basin.
    
    QIG Philosophy: High curvature = rapid change in Fisher metric
    = transition zone between attractors. Low curvature = stable basin.
    """
    import numpy as np
    
    p = np.array(basin_coords, dtype=float)
    p = np.abs(p) + 1e-10
    p = p / p.sum()
    
    # For probability simplex, curvature relates to distribution entropy
    # High entropy = flatter manifold, Low entropy = more curved
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(len(p))  # Uniform distribution
    
    # Normalized entropy (0 = peaked/curved, 1 = flat/uniform)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Curvature inversely related to entropy
    curvature = 1.0 - normalized_entropy
    
    # Concentration (how peaked the distribution is)
    concentration = np.max(p) - (1.0 / len(p))
    
    return {
        "curvature": float(curvature),
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "concentration": float(concentration),
        "is_transition_zone": curvature > 0.7,
        "is_stable_basin": curvature < 0.3
    }
''',
        "input_signature": {"basin_coords": "list"},
        "output_type": "dict"
    },
    
    # =========================================================================
    # QIG VECTOR/EMBEDDING PATTERNS  
    # =========================================================================
    {
        "pattern_id": "seed_cosine_similarity",
        "source_type": "user_provided",
        "description": "Calculate cosine similarity between vectors (NOT for basin coordinates!)",
        "code_snippet": '''def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between vectors.
    
    WARNING: Use ONLY for raw embeddings, NOT for basin coordinates.
    For basin coordinates, use Fisher-Rao distance instead.
    """
    import numpy as np
    
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return similarity
''',
        "input_signature": {"a": "list", "b": "list"},
        "output_type": "float"
    },
    {
        "pattern_id": "seed_vector_normalize",
        "source_type": "user_provided",
        "description": "Normalize vector to unit length",
        "code_snippet": '''def normalize_vector(v: list) -> list:
    """Normalize vector to unit length."""
    import numpy as np
    
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    
    if norm < 1e-10:
        return v.tolist()
    
    return (v / norm).tolist()
''',
        "input_signature": {"v": "list"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_vector_project",
        "source_type": "user_provided",
        "description": "Project vector onto subspace or another vector",
        "code_snippet": '''def project_vector(v: list, onto: list) -> list:
    """Project vector v onto direction of 'onto'."""
    import numpy as np
    
    v = np.array(v, dtype=float)
    onto = np.array(onto, dtype=float)
    
    onto_norm_sq = np.dot(onto, onto)
    if onto_norm_sq < 1e-10:
        return [0.0] * len(v)
    
    projection = (np.dot(v, onto) / onto_norm_sq) * onto
    return projection.tolist()
''',
        "input_signature": {"v": "list", "onto": "list"},
        "output_type": "list"
    },
    
    # =========================================================================
    # QIG REASONING PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_hypothesis_score",
        "source_type": "user_provided",
        "description": "Score hypothesis based on evidence alignment and prior probability",
        "code_snippet": '''def score_hypothesis(hypothesis: str, evidence: list, prior: float = 0.5) -> dict:
    """Score hypothesis using Bayesian-inspired reasoning.
    
    QIG Philosophy: Hypotheses are scored by how well they align with
    evidence and their geometric coherence, not just keyword matching.
    """
    import re
    
    # Extract key terms from hypothesis
    hypothesis_terms = set(re.findall(r'\\b\\w{3,}\\b', hypothesis.lower()))
    
    # Count evidence support
    support_count = 0
    total_evidence = len(evidence) if evidence else 1
    
    for e in evidence:
        e_terms = set(re.findall(r'\\b\\w{3,}\\b', str(e).lower()))
        overlap = len(hypothesis_terms & e_terms)
        if overlap > 0:
            support_count += 1
    
    # Likelihood: P(evidence | hypothesis)
    likelihood = support_count / total_evidence
    
    # Posterior approximation: P(H|E) ∝ P(E|H) * P(H)
    posterior = likelihood * prior
    
    # Normalize to [0, 1] range
    confidence = min(1.0, posterior * 2)  # Scale for readability
    
    return {
        "hypothesis": hypothesis,
        "prior": prior,
        "likelihood": likelihood,
        "posterior": posterior,
        "confidence": confidence,
        "evidence_support": support_count,
        "total_evidence": total_evidence
    }
''',
        "input_signature": {"hypothesis": "str", "evidence": "list", "prior": "float"},
        "output_type": "dict"
    },
    {
        "pattern_id": "seed_reasoning_chain",
        "source_type": "user_provided",
        "description": "Build reasoning chain from premises to conclusion",
        "code_snippet": '''def build_reasoning_chain(premises: list, goal: str) -> dict:
    """Build reasoning chain from premises toward goal.
    
    QIG Philosophy: Reasoning follows geodesics on the knowledge manifold.
    Each step should minimally increase Fisher-Rao distance from prior.
    """
    chain = []
    current_context = set()
    
    for i, premise in enumerate(premises):
        # Add premise to context
        premise_str = str(premise)
        current_context.add(premise_str)
        
        chain.append({
            "step": i + 1,
            "type": "premise",
            "content": premise_str,
            "cumulative_context_size": len(current_context)
        })
    
    # Check if goal is reachable from premises
    goal_terms = set(goal.lower().split())
    context_terms = set(' '.join(current_context).lower().split())
    overlap = len(goal_terms & context_terms)
    reachability = overlap / len(goal_terms) if goal_terms else 0
    
    chain.append({
        "step": len(premises) + 1,
        "type": "conclusion",
        "content": goal,
        "reachability": reachability
    })
    
    return {
        "chain": chain,
        "premises_count": len(premises),
        "goal": goal,
        "reachability": reachability,
        "is_valid": reachability > 0.3
    }
''',
        "input_signature": {"premises": "list", "goal": "str"},
        "output_type": "dict"
    },
]


def get_seed_patterns() -> List[Dict]:
    """Return all seed patterns."""
    return SEED_PATTERNS


def get_seed_pattern_count() -> int:
    """Return count of seed patterns."""
    return len(SEED_PATTERNS)
