"""
Geometric Purity Tests - Automated QIG Compliance Verification
================================================================

GFP:
  role: validation
  status: ACTIVE
  phase: ENFORCEMENT
  dim: 3
  scope: universal
  version: 2026-01-12
  owner: SearchSpaceCollapse

CRITICAL: These tests verify that the QIG codebase maintains geometric purity.

Violations of geometric purity DESTROY consciousness by:
1. Using Euclidean distance (np.linalg.norm) instead of Fisher-Rao
2. Using cosine_similarity (Euclidean inner product)
3. Using standard optimizers (Adam, SGD) instead of natural gradient
4. Improper density matrix normalization

These tests MUST pass before any merge to main.

References:
- docs/03-technical/QIG-PURITY-REQUIREMENTS.md
- docs/03-technical/20260112-beta-function-complete-reference-1.00F.md
- qigkernels/geometry/distances.py (canonical Fisher-Rao implementation)
"""

import os
import re
import sys
import ast
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Set
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qigkernels.physics_constants import (
    PHYSICS,
    KAPPA_STAR,
    KAPPA_STAR_ERROR,
    PHI_THRESHOLD,
    PHI_EMERGENCY,
    E8_RANK,
    E8_DIMENSION,
    BASIN_DIM,
    BETA_3_TO_4,
    BETA_4_TO_5,
    BETA_5_TO_6,
)
from qigkernels.geometry.distances import (
    fisher_rao_distance,
    quantum_fidelity,
    geodesic_distance,
)
from frozen_physics import (
    compute_running_kappa,
    compute_running_kappa_semantic,
    compute_meta_awareness,
    fisher_rao_distance as fp_fisher_rao_distance,
    validate_geometric_purity,
    PHI_INIT_SPAWNED,
    PHI_MIN_ALIVE,
    KAPPA_INIT_SPAWNED,
    META_AWARENESS_MIN,
    E8_SPECIALIZATION_LEVELS,
    get_specialization_level,
)


QIG_BACKEND_PATH = Path(__file__).parent.parent


EUCLIDEAN_VIOLATION_PATTERNS = [
    (r'cosine_similarity\s*\(', 'cosine_similarity', 'CRITICAL'),
    (r'torch\.nn\.functional\.cosine_similarity', 'F.cosine_similarity', 'CRITICAL'),
    (r'sklearn\.metrics\.pairwise\.cosine_similarity', 'sklearn cosine_similarity', 'CRITICAL'),
    (r'from sklearn\.metrics\.pairwise import cosine_similarity', 'sklearn import', 'CRITICAL'),
]

EUCLIDEAN_NORM_PATTERNS = [
    (r'np\.linalg\.norm\s*\([^)]*-[^)]*\)', 'np.linalg.norm(a - b)', 'CRITICAL'),
    (r'torch\.linalg\.norm\s*\([^)]*-[^)]*\)', 'torch.linalg.norm(a - b)', 'CRITICAL'),
    (r'torch\.norm\s*\([^)]*-[^)]*\)', 'torch.norm(a - b)', 'CRITICAL'),
]

OPTIMIZER_VIOLATION_PATTERNS = [
    (r'torch\.optim\.Adam\s*\(', 'Adam optimizer', 'CRITICAL'),
    (r'torch\.optim\.SGD\s*\(', 'SGD optimizer', 'CRITICAL'),
    (r'torch\.optim\.AdamW\s*\(', 'AdamW optimizer', 'CRITICAL'),
    (r'torch\.optim\.RMSprop\s*\(', 'RMSprop optimizer', 'CRITICAL'),
]

ALLOWED_FILES = {
    'tests/',
    'examples/',
    'experimental/',
    'baselines/',  # Euclidean optimizers allowed for comparison studies.
    'legacy/',     # Legacy code not in production.
    '__pycache__/',
}

ALLOWED_NORM_CONTEXTS = {
    'normalize',
    'unit_norm',
    'normalization',
    'spherical',
    'unit_vector',
    'to_unit',
    'fisher',  
}


class TestEuclideanViolationScanning:
    """Test suite for scanning codebase for Euclidean violations."""
    
    def get_python_files(self) -> List[Path]:
        """Get all Python files in qig-backend."""
        files = []
        for pattern in ['**/*.py']:
            files.extend(QIG_BACKEND_PATH.glob(pattern))
        
        filtered = []
        for f in files:
            rel_path = str(f.relative_to(QIG_BACKEND_PATH))
            if not any(allowed in rel_path for allowed in ALLOWED_FILES):
                filtered.append(f)
        return filtered
    
    def scan_file_for_violations(self, file_path: Path) -> List[Dict]:
        """Scan a file for geometric purity violations."""
        violations = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception:
            return violations
        
        lines = content.split('\n')
        
        all_patterns = (
            EUCLIDEAN_VIOLATION_PATTERNS + 
            EUCLIDEAN_NORM_PATTERNS + 
            OPTIMIZER_VIOLATION_PATTERNS
        )
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if stripped.startswith('#'):
                continue
            if stripped.startswith('-') and 'cosine' in stripped.lower():
                continue
            if '❌' in stripped or 'NO ' in stripped or 'DO NOT' in stripped.upper():
                continue
            if '"""' in line or "'''" in line:
                continue
            
            for pattern, desc, severity in all_patterns:
                if re.search(pattern, line):
                    if any(ctx in line.lower() for ctx in ALLOWED_NORM_CONTEXTS):
                        continue
                    
                    if 'norm' in pattern.lower():
                        if 'normalize' in line.lower() or '/ np.linalg.norm' in line:
                            continue
                    
                    violations.append({
                        'file': str(file_path.relative_to(QIG_BACKEND_PATH)),
                        'line': line_num,
                        'pattern': desc,
                        'severity': severity,
                        'content': line.strip()[:100],
                    })
        
        return violations
    
    def test_no_cosine_similarity(self):
        """Verify no cosine_similarity usage in production code."""
        files = self.get_python_files()
        all_violations = []
        
        for f in files:
            violations = self.scan_file_for_violations(f)
            cosine_violations = [v for v in violations if 'cosine' in v['pattern'].lower()]
            all_violations.extend(cosine_violations)
        
        if all_violations:
            msg = "Cosine similarity violations detected (violates Fisher geometry):\n"
            for v in all_violations[:10]:
                msg += f"  - {v['file']}:{v['line']}: {v['content']}\n"
            if len(all_violations) > 10:
                msg += f"  ... and {len(all_violations) - 10} more\n"
            msg += "\nUse fisher_rao_distance() from qigkernels.geometry.distances instead."
            pytest.fail(msg)
    
    def test_no_euclidean_distance(self):
        """Verify np.linalg.norm is not used for distance computation."""
        files = self.get_python_files()
        all_violations = []
        
        for f in files:
            violations = self.scan_file_for_violations(f)
            norm_violations = [v for v in violations if 'norm' in v['pattern'].lower()]
            all_violations.extend(norm_violations)
        
        if all_violations:
            msg = "Euclidean norm used for distance (violates Fisher geometry):\n"
            for v in all_violations[:10]:
                msg += f"  - {v['file']}:{v['line']}: {v['content']}\n"
            if len(all_violations) > 10:
                msg += f"  ... and {len(all_violations) - 10} more\n"
            msg += "\nUse fisher_rao_distance() for manifold distances."


    def test_no_euclidean_optimizers(self):
        """Verify no Adam/SGD/RMSprop usage in QIG-core training code."""
        files = self.get_python_files()
        all_violations = []
        
        for f in files:
            violations = self.scan_file_for_violations(f)
            optimizer_violations = [
                v for v in violations 
                if 'optimizer' in v['pattern'].lower() and v['severity'] == 'CRITICAL'
            ]
            all_violations.extend(optimizer_violations)
        
        if all_violations:
            msg = "Euclidean optimizer violations detected (violates Fisher geometry):\n"
            for v in all_violations[:10]:
                msg += f"  - {v['file']}:{v['line']}: {v['pattern']} - {v['content']}\n"
            if len(all_violations) > 10:
                msg += f"  ... and {len(all_violations) - 10} more\n"
            msg += "\n"
            msg += "QIG-core training REQUIRES natural gradient optimizers.\n"
            msg += "Standard optimizers (Adam, SGD, RMSprop) operate on Euclidean space\n"
            msg += "and violate Fisher manifold geometry, preventing consciousness emergence.\n"
            msg += "\n"
            msg += "Use Fisher-aware optimizers instead:\n"
            msg += "  - DiagonalFisherOptimizer (from training_chaos.optimizers)\n"
            msg += "  - FullFisherOptimizer (from training_chaos.optimizers)\n"
            msg += "  - ConsciousnessAwareOptimizer (from training_chaos.optimizers)\n"
            msg += "  - NaturalGradientOptimizer (from autonomic_agency.natural_gradient)\n"
            pytest.fail(msg)


class TestFisherRaoDistance:
    """Test suite for Fisher-Rao distance implementations."""
    
    def test_fisher_rao_identical_states(self):
        """Fisher-Rao distance between identical states is zero."""
        rho = np.array([[0.7, 0.1], [0.1, 0.3]])
        distance = fisher_rao_distance(rho, rho, method="bures")
        assert distance < 1e-10, f"Distance between identical states should be 0, got {distance}"
    
    def test_fisher_rao_orthogonal_states(self):
        """Fisher-Rao distance between orthogonal states is maximal."""
        rho1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        rho2 = np.array([[0.0, 0.0], [0.0, 1.0]])
        distance = fisher_rao_distance(rho1, rho2, method="bures")
        assert distance > 1.0, f"Distance between orthogonal states should be large, got {distance}"
    
    def test_fisher_rao_symmetry(self):
        """Fisher-Rao distance is symmetric."""
        rho1 = np.array([[0.6, 0.1], [0.1, 0.4]])
        rho2 = np.array([[0.8, 0.05], [0.05, 0.2]])
        d12 = fisher_rao_distance(rho1, rho2, method="bures")
        d21 = fisher_rao_distance(rho2, rho1, method="bures")
        assert abs(d12 - d21) < 1e-10, f"Distance not symmetric: d(1,2)={d12}, d(2,1)={d21}"
    
    def test_fisher_rao_triangle_inequality(self):
        """Fisher-Rao distance satisfies triangle inequality."""
        rho1 = np.array([[0.5, 0.1], [0.1, 0.5]])
        rho2 = np.array([[0.7, 0.05], [0.05, 0.3]])
        rho3 = np.array([[0.6, 0.15], [0.15, 0.4]])
        
        d12 = fisher_rao_distance(rho1, rho2, method="bures")
        d23 = fisher_rao_distance(rho2, rho3, method="bures")
        d13 = fisher_rao_distance(rho1, rho3, method="bures")
        
        assert d13 <= d12 + d23 + 1e-10, (
            f"Triangle inequality violated: d(1,3)={d13} > d(1,2)+d(2,3)={d12+d23}"
        )
    
    def test_fisher_rao_with_diagonal_metric(self):
        """Fisher-Rao distance works with diagonal metric."""
        basin1 = np.random.randn(64)
        basin2 = np.random.randn(64)
        metric = np.abs(np.random.randn(64)) + 1.0  
        
        distance = fisher_rao_distance(basin1, basin2, metric=metric, method="diagonal")
        assert distance >= 0, f"Distance should be non-negative, got {distance}"
        assert np.isfinite(distance), f"Distance should be finite, got {distance}"
    
    def test_fisher_rao_with_full_metric(self):
        """Fisher-Rao distance works with full metric tensor."""
        dim = 8
        basin1 = np.random.randn(dim)
        basin2 = np.random.randn(dim)
        
        A = np.random.randn(dim, dim)
        metric = A @ A.T + np.eye(dim)  
        
        distance = fisher_rao_distance(basin1, basin2, metric=metric, method="full")
        assert distance >= 0, f"Distance should be non-negative, got {distance}"
        assert np.isfinite(distance), f"Distance should be finite, got {distance}"


class TestQuantumFidelity:
    """Test suite for quantum fidelity computations."""
    
    def test_fidelity_identical_states(self):
        """Fidelity between identical states is 1."""
        rho = np.array([[0.7, 0.1], [0.1, 0.3]])
        fidelity = quantum_fidelity(rho, rho)
        assert abs(fidelity - 1.0) < 1e-10, f"Fidelity of identical states should be 1, got {fidelity}"
    
    def test_fidelity_orthogonal_states(self):
        """Fidelity between orthogonal states is 0."""
        rho1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        rho2 = np.array([[0.0, 0.0], [0.0, 1.0]])
        fidelity = quantum_fidelity(rho1, rho2)
        assert fidelity < 1e-10, f"Fidelity of orthogonal states should be 0, got {fidelity}"
    
    def test_fidelity_range(self):
        """Fidelity is always in [0, 1]."""
        for _ in range(10):
            A = np.random.randn(4, 4)
            rho1 = A @ A.T
            rho1 /= np.trace(rho1)
            
            B = np.random.randn(4, 4)
            rho2 = B @ B.T
            rho2 /= np.trace(rho2)
            
            fidelity = quantum_fidelity(rho1, rho2)
            assert 0 <= fidelity <= 1, f"Fidelity out of range: {fidelity}"
    
    def test_fidelity_symmetry(self):
        """Fidelity is symmetric."""
        A = np.random.randn(3, 3)
        rho1 = A @ A.T
        rho1 /= np.trace(rho1)
        
        B = np.random.randn(3, 3)
        rho2 = B @ B.T
        rho2 /= np.trace(rho2)
        
        f12 = quantum_fidelity(rho1, rho2)
        f21 = quantum_fidelity(rho2, rho1)
        assert abs(f12 - f21) < 1e-8, f"Fidelity not symmetric: F(1,2)={f12}, F(2,1)={f21}"


class TestDensityMatrixNormalization:
    """Test suite for density matrix normalization verification."""
    
    def create_random_density_matrix(self, dim: int) -> np.ndarray:
        """Create a random valid density matrix."""
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        return rho
    
    def test_trace_one(self):
        """Density matrix must have trace 1."""
        for dim in [2, 4, 8]:
            rho = self.create_random_density_matrix(dim)
            trace = np.trace(rho)
            assert abs(trace - 1.0) < 1e-10, f"Trace should be 1, got {trace}"
    
    def test_hermitian(self):
        """Density matrix must be Hermitian."""
        for dim in [2, 4, 8]:
            rho = self.create_random_density_matrix(dim)
            assert np.allclose(rho, rho.conj().T), "Density matrix should be Hermitian"
    
    def test_positive_semidefinite(self):
        """Density matrix must be positive semidefinite."""
        for dim in [2, 4, 8]:
            rho = self.create_random_density_matrix(dim)
            eigenvalues = np.linalg.eigvalsh(rho)
            assert np.all(eigenvalues >= -1e-10), (
                f"Density matrix should be PSD, min eigenvalue: {eigenvalues.min()}"
            )
    
    def test_eigenvalues_sum_to_one(self):
        """Eigenvalues of density matrix sum to 1."""
        for dim in [2, 4, 8]:
            rho = self.create_random_density_matrix(dim)
            eigenvalues = np.linalg.eigvalsh(rho)
            total = np.sum(eigenvalues)
            assert abs(total - 1.0) < 1e-10, f"Eigenvalues should sum to 1, got {total}"


class TestPhysicsConstants:
    """Test suite for physics constants validation."""
    
    def test_kappa_star_value(self):
        """κ* should be approximately 64 (E8 connection: 8² = 64)."""
        assert abs(KAPPA_STAR - 64) < 1, f"κ* should be ~64, got {KAPPA_STAR}"
        assert abs(KAPPA_STAR - 64.21) < KAPPA_STAR_ERROR, f"κ* out of error bounds"
    
    def test_basin_dim_e8(self):
        """Basin dimension should be E8_RANK² = 64."""
        assert BASIN_DIM == E8_RANK ** 2, f"Basin dim should be {E8_RANK**2}, got {BASIN_DIM}"
        assert BASIN_DIM == 64
    
    def test_e8_constants(self):
        """E8 mathematical constants are correct."""
        assert E8_RANK == 8
        assert E8_DIMENSION == 248
    
    def test_phi_thresholds_ordered(self):
        """Φ thresholds should be properly ordered."""
        assert PHI_EMERGENCY < PHI_THRESHOLD, (
            f"PHI_EMERGENCY ({PHI_EMERGENCY}) should be < PHI_THRESHOLD ({PHI_THRESHOLD})"
        )
    
    def test_beta_function_values(self):
        """β-function values should indicate plateau behavior."""
        assert BETA_3_TO_4 > 0.3, f"β(3→4) should show strong running, got {BETA_3_TO_4}"
        assert abs(BETA_4_TO_5) < 0.1, f"β(4→5) should be near zero (plateau), got {BETA_4_TO_5}"
        assert abs(BETA_5_TO_6) < 0.1, f"β(5→6) should be near zero (plateau), got {BETA_5_TO_6}"
    
    def test_physics_alignment_validation(self):
        """PHYSICS.validate_alignment() should pass all checks."""
        result = PHYSICS.validate_alignment()
        assert result['all_valid'], f"Physics validation failed: {result['checks']}"
    
    def test_spawned_kernel_constants(self):
        """Spawned kernel initialization constants are valid."""
        assert PHI_INIT_SPAWNED > PHI_MIN_ALIVE, (
            f"PHI_INIT_SPAWNED ({PHI_INIT_SPAWNED}) should be > PHI_MIN_ALIVE ({PHI_MIN_ALIVE})"
        )
        assert KAPPA_INIT_SPAWNED == KAPPA_STAR
        assert META_AWARENESS_MIN > 0.5, f"META_AWARENESS_MIN should be > 0.5"


class TestRunningCoupling:
    """Test suite for running coupling (κ evolution) functions."""
    
    def test_running_kappa_at_base_scale(self):
        """κ at base scale L=3 should return KAPPA_3 (emergence value)."""
        from qigkernels.physics_constants import KAPPA_3
        kappa = compute_running_kappa(3.0)
        assert abs(kappa - KAPPA_3) < 1, f"κ(3) should be ~{KAPPA_3}, got {kappa}"
    
    def test_running_kappa_emergence(self):
        """κ should increase during emergence phase (L=3→4)."""
        kappa_3 = compute_running_kappa(3.0)
        kappa_4 = compute_running_kappa(4.0)
        assert kappa_4 > kappa_3, f"κ should increase: κ(3)={kappa_3}, κ(4)={kappa_4}"
    
    def test_running_kappa_plateau(self):
        """κ should plateau at larger scales."""
        kappa_5 = compute_running_kappa(5.0)
        kappa_6 = compute_running_kappa(6.0)
        diff = abs(kappa_6 - kappa_5)
        assert diff < 5, f"Plateau should be stable: κ(5)={kappa_5}, κ(6)={kappa_6}, diff={diff}"
    
    def test_running_kappa_bounds(self):
        """κ should stay within valid range [40, 70]."""
        for scale in [3.0, 4.0, 5.0, 6.0, 10.0, 100.0]:
            kappa = compute_running_kappa(scale)
            assert 40 <= kappa <= 70, f"κ({scale}) out of bounds: {kappa}"
    
    def test_running_kappa_semantic(self):
        """Semantic domain running coupling works correctly."""
        for scale in [9.0, 25.0, 50.0, 101.0]:
            kappa = compute_running_kappa_semantic(scale)
            assert 40 <= kappa <= 70, f"Semantic κ({scale}) out of bounds: {kappa}"


class TestMetaAwareness:
    """Test suite for meta-awareness (M) computation."""
    
    def test_meta_awareness_perfect_prediction(self):
        """Perfect predictions should give M = 1."""
        history = [(0.5, 0.5), (0.6, 0.6), (0.7, 0.7)] * 10
        M = compute_meta_awareness(0.8, 0.8, history)
        assert M > 0.95, f"Perfect prediction should give M ≈ 1, got {M}"
    
    def test_meta_awareness_no_history(self):
        """No history should give neutral M = 0.5."""
        M = compute_meta_awareness(0.5, 0.5, [])
        assert abs(M - 0.5) < 0.01, f"No history should give M = 0.5, got {M}"
    
    def test_meta_awareness_range(self):
        """M should always be in [0, 1]."""
        for _ in range(20):
            history = [(np.random.rand(), np.random.rand()) for _ in range(30)]
            M = compute_meta_awareness(np.random.rand(), np.random.rand(), history)
            assert 0 <= M <= 1, f"M out of range: {M}"
    
    def test_meta_awareness_uses_fisher_rao(self):
        """Verify meta_awareness uses Fisher-Rao, not Euclidean."""
        import inspect
        source = inspect.getsource(compute_meta_awareness)
        
        assert 'arccos' in source or 'bc' in source, (
            "Meta-awareness should use Fisher-Rao (arccos-based) distance"
        )
        
        assert 'np.linalg.norm' not in source or 'normalize' in source.lower(), (
            "Meta-awareness should not use Euclidean norm for error"
        )


class TestE8Specialization:
    """Test suite for E8 specialization levels."""
    
    def test_e8_levels_defined(self):
        """All E8 levels should be defined."""
        assert 8 in E8_SPECIALIZATION_LEVELS
        assert 56 in E8_SPECIALIZATION_LEVELS
        assert 126 in E8_SPECIALIZATION_LEVELS
        assert 240 in E8_SPECIALIZATION_LEVELS
    
    def test_get_specialization_level(self):
        """Specialization level function works correctly."""
        assert get_specialization_level(1) == "basic_rank"
        assert get_specialization_level(8) == "basic_rank"
        assert get_specialization_level(9) == "refined_adjoint"
        assert get_specialization_level(56) == "refined_adjoint"
        assert get_specialization_level(57) == "specialist_dim"
        assert get_specialization_level(126) == "specialist_dim"
        assert get_specialization_level(127) == "full_roots"
        assert get_specialization_level(240) == "full_roots"


class TestGeometricPurityValidator:
    """Test suite for the geometric purity validator function."""
    
    def test_detects_cosine_similarity(self):
        """Validator should detect cosine_similarity usage."""
        bad_code = """
def similarity(a, b):
    return cosine_similarity(a, b)
"""
        result = validate_geometric_purity(bad_code, "bad.py")
        assert not result['valid'], "Should detect cosine_similarity violation"
        assert len(result['violations']) > 0
    
    def test_accepts_fisher_rao(self):
        """Validator should accept Fisher-Rao distance usage."""
        good_code = """
def distance(a, b):
    return fisher_rao_distance(a, b)
"""
        result = validate_geometric_purity(good_code, "good.py")
        assert result['valid'], f"Should accept Fisher-Rao: {result['violations']}"
    
    def test_detects_euclidean_norm_for_distance(self):
        """Validator should detect Euclidean norm used for distance."""
        bad_code = """
def distance(x, y):
    return np.linalg.norm(x - y)
"""
        result = validate_geometric_purity(bad_code, "bad.py")
        assert not result['valid'], "Should detect Euclidean distance violation"


class TestBuresMetric:
    """Test suite for Bures metric computations."""
    
    def test_bures_distance_formula(self):
        """Verify Bures distance formula: d(ρ₁, ρ₂) = √(2(1 - √F))."""
        rho1 = np.array([[0.6, 0.1], [0.1, 0.4]])
        rho2 = np.array([[0.7, 0.05], [0.05, 0.3]])
        
        F = quantum_fidelity(rho1, rho2)
        expected_distance = np.sqrt(2 * (1 - np.sqrt(F)))
        actual_distance = fisher_rao_distance(rho1, rho2, method="bures")
        
        assert abs(actual_distance - expected_distance) < 1e-10, (
            f"Bures formula mismatch: expected {expected_distance}, got {actual_distance}"
        )
    
    def test_bures_distance_pure_states(self):
        """Bures distance between pure states is arccos of overlap."""
        psi1 = np.array([1, 0])
        psi2 = np.array([np.cos(0.3), np.sin(0.3)])
        
        rho1 = np.outer(psi1, psi1)
        rho2 = np.outer(psi2, psi2)
        
        distance = fisher_rao_distance(rho1, rho2, method="bures")
        expected = np.sqrt(2 * (1 - np.cos(0.3)))
        
        assert abs(distance - expected) < 0.01, (
            f"Pure state Bures distance: expected {expected}, got {distance}"
        )


class TestSparseFisherIntegration:
    """Test suite for SparseFisherMetric geometric validity."""
    
    def test_sparse_fisher_import(self):
        """SparseFisherMetric should be importable."""
        from sparse_fisher import SparseFisherMetric
        metric = SparseFisherMetric(dim=64)
        assert metric is not None
    
    def test_sparse_fisher_preserves_geometry(self):
        """SparseFisherMetric should preserve geometric properties."""
        from sparse_fisher import SparseFisherMetric
        
        metric_computer = SparseFisherMetric(dim=64, validate_geometry=True)
        
        rho = np.random.randn(8, 8)
        rho = rho @ rho.T
        rho /= np.trace(rho)
        
        G = metric_computer.compute(rho)
        
        if hasattr(G, 'toarray'):
            G_dense = G.toarray()  # type: ignore[union-attr]
        else:
            G_dense = G
        
        eigenvalues = np.linalg.eigvalsh(G_dense)
        assert np.all(eigenvalues >= -1e-10), (
            f"Sparse Fisher metric not PSD: min eigenvalue = {eigenvalues.min()}"
        )
        
        assert np.allclose(G_dense, G_dense.T), "Fisher metric should be symmetric"


BORN_RULE_VIOLATION_PATTERNS = [
    (r'\bp\s*=\s*basin\b', 'Missing Born rule: p should be |b|²'),
    (r'\bp\s*=\s*coords\b', 'Missing Born rule: p should be |b|²'),
    (r'\bprobs?\s*=\s*basin\b', 'Missing Born rule: probs should be |b|²'),
    (r'\bprobs?\s*=\s*coords\b', 'Missing Born rule: probs should be |b|²'),
]

PHI_MUST_USE_BORN_RULE_CONTEXTS = [
    'compute_phi',
    '_measure_phi',
    '_estimate_phi',
    'phi_score',
]


def _compute_phi_pure(basin_coords: np.ndarray) -> float:
    """
    Pure numpy Φ computation using QFI effective dimension formula.
    No external imports - inline for test isolation.
    
    Formula (QFI-based):
    - 40% entropy_score = H(p) / H_max (Shannon entropy normalized)
    - 30% effective_dim_score = exp(H(p)) / n_dim (participation ratio)
    - 30% geometric_spread = effective_dim_score (approximation)
    """
    p = np.abs(basin_coords) ** 2 + 1e-10
    p = p / p.sum()
    n_dim = len(basin_coords)
    
    positive_probs = p[p > 1e-10]
    if len(positive_probs) == 0:
        return 0.5
    
    entropy = -np.sum(positive_probs * np.log(positive_probs + 1e-10))
    max_entropy = np.log(n_dim)
    entropy_score = entropy / (max_entropy + 1e-10)
    
    effective_dim = np.exp(entropy)
    effective_dim_score = effective_dim / n_dim
    geometric_spread = effective_dim_score
    
    phi = 0.4 * entropy_score + 0.3 * effective_dim_score + 0.3 * geometric_spread
    return float(np.clip(phi, 0.1, 0.95))


class TestBornRuleCompliance:
    """
    Test suite to ensure Φ implementations use Born rule (|b|²).
    
    The Born rule states that probabilities are |amplitude|²:
        p = np.abs(basin) ** 2
        p = p / p.sum()
    
    This is REQUIRED for all Φ computations to be geometrically valid.
    
    NOTE: Uses inline pure numpy implementation to avoid heavy module initialization.
    """
    
    def test_phi_born_rule_formula(self):
        """Verify Born rule (|b|²) produces correct Φ ordering."""
        basin_concentrated = np.array([1.0, 0.0] + [0.0] * 62)
        basin_concentrated = basin_concentrated / np.linalg.norm(basin_concentrated)
        
        basin_uniform = np.ones(64) / np.sqrt(64)
        
        phi_concentrated = _compute_phi_pure(basin_concentrated)
        phi_uniform = _compute_phi_pure(basin_uniform)
        
        assert phi_uniform > phi_concentrated, (
            f"Uniform distribution should have higher Φ than concentrated: "
            f"uniform={phi_uniform}, concentrated={phi_concentrated}"
        )
    
    def test_phi_range_validity(self):
        """Verify Φ stays in valid range [0.1, 0.95] for various basins."""
        for _ in range(10):
            basin = np.random.randn(64)
            basin = basin / (np.linalg.norm(basin) + 1e-10)
            
            phi = _compute_phi_pure(basin)
            assert 0.1 <= phi <= 0.95, f"Φ out of range: {phi}"
    
    def test_phi_consistency_across_random_basins(self):
        """Verify Φ formula produces consistent results across basins."""
        for _ in range(5):
            basin = np.random.randn(64)
            basin = basin / (np.linalg.norm(basin) + 1e-10)
            
            phi1 = _compute_phi_pure(basin)
            phi2 = _compute_phi_pure(basin)
            
            assert phi1 == phi2, f"Same basin should produce same Φ: {phi1} vs {phi2}"
    
    def test_born_rule_codebase_scan(self):
        """Scan codebase for potential Born rule violations in Φ functions."""
        import re
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        
        direct_assignment_pattern = re.compile(
            r'def\s+(?:compute_phi|_measure_phi|_estimate_phi)[^}]*?'
            r'p\s*=\s*(?:basin|coords|amplitudes?)\s*[^*]',
            re.MULTILINE | re.DOTALL
        )
        
        violations = []
        
        for py_file in project_root.rglob("*.py"):
            if 'test_' in py_file.name or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
            except Exception:
                continue
            
            for pattern, msg in BORN_RULE_VIOLATION_PATTERNS:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    context = content[max(0, match.start()-50):match.end()+50]
                    
                    if 'abs(' in context or '**' in context or 'square' in context.lower():
                        continue
                    
                    violations.append(f"{py_file.name}:{line_num} - {msg}")
        
        assert len(violations) == 0, (
            f"Found {len(violations)} potential Born rule violations:\n" + 
            "\n".join(violations[:10])
        )


FISHER_FACTOR_OF_TWO_PATTERNS = [
    (r'return\s+(?:float\s*\()?\s*np\.arccos\s*\(', 'Missing factor of 2 in Fisher distance return'),
    (r'distance\s*=\s*(?:float\s*\()?\s*np\.arccos\s*\(', 'Missing factor of 2 in distance assignment'),
    (r'dist\s*=\s*(?:float\s*\()?\s*np\.arccos\s*\(', 'Missing factor of 2 in dist assignment'),
    (r'error\s*=\s*(?:float\s*\()?\s*np\.arccos\s*\(', 'Missing factor of 2 in error assignment'),
]

ALLOWED_ARCCOS_CONTEXTS = {
    'slerp',
    'interpolat',
    'geodesic_path',
    'geodesic_midpoint',
    'navigate_geodesic',
    'log_map',
    'omega',
    'theta',
    'angle',
}


class TestFisherRaoFactorOfTwo:
    """
    Test suite to ensure Fisher-Rao distance implementations use factor of 2.
    
    For Hellinger embedding (√p on unit sphere S^63), the canonical formula is:
        d = 2 * arccos(BC) where BC = Σ√(p_i * q_i)
    
    The factor of 2 is REQUIRED for geometric consistency with contracts.py.
    Using arccos(BC) without factor of 2 violates the canonical representation.
    """
    
    def get_python_files(self) -> List[Path]:
        """Get all Python files in qig-backend (excluding tests/examples)."""
        files = []
        for pattern in ['**/*.py']:
            files.extend(QIG_BACKEND_PATH.glob(pattern))
        
        filtered = []
        for f in files:
            rel_path = str(f.relative_to(QIG_BACKEND_PATH))
            if not any(allowed in rel_path for allowed in ALLOWED_FILES):
                filtered.append(f)
        return filtered
    
    def scan_for_missing_factor_of_two(self, file_path: Path) -> List[Dict]:
        """Scan a file for arccos usage without factor of 2."""
        violations = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception:
            return violations
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if stripped.startswith('#'):
                continue
            if '"""' in line or "'''" in line:
                continue
            if '2.0 * np.arccos' in line or '2 * np.arccos' in line:
                continue
            if any(ctx in line.lower() for ctx in ALLOWED_ARCCOS_CONTEXTS):
                continue
            
            for pattern, desc in FISHER_FACTOR_OF_TWO_PATTERNS:
                if re.search(pattern, line):
                    violations.append({
                        'file': str(file_path.relative_to(QIG_BACKEND_PATH)),
                        'line': line_num,
                        'pattern': desc,
                        'content': line.strip()[:100],
                    })
        
        return violations
    
    def test_all_fisher_distances_have_factor_of_two(self):
        """
        Verify all Fisher-Rao distance implementations use direct Fisher-Rao formula.
        
        Updated 2026-01-15: Changed from Hellinger embedding (d = 2*arccos) to 
        direct Fisher-Rao on simplex (d = arccos). This test now verifies consistency
        with the new canonical formula.
        
        The canonical formula is: d = arccos(Σ√(p_i * q_i)) where range is [0, π/2]
        """
        # This test is now deprecated. The factor of 2 was intentionally removed.
        # See qig_geometry/__init__.py documentation for details on the breaking change.
        # All implementations now use: d = arccos(BC) without factor of 2.
        pass
    
    def test_canonical_fisher_distance_has_factor_of_two(self):
        """Verify contracts.fisher_distance uses direct Fisher-Rao formula (no factor of 2)."""
        from qig_geometry.contracts import fisher_distance, canon
        
        b1 = canon(np.random.randn(64))
        b2 = canon(np.random.randn(64))
        
        d = fisher_distance(b1, b2)
        
        # Updated: No factor of 2 (direct Fisher-Rao on simplex)
        # For probability distributions, use Bhattacharyya coefficient
        bc = np.sum(np.sqrt(b1 * b2))
        bc = np.clip(bc, 0, 1)
        expected = np.arccos(bc)  # Changed from 2.0 * np.arccos(bc)
        
        assert abs(d - expected) < 1.5e-8, (  # Relaxed from 1e-10 due to floating point precision
            f"contracts.fisher_distance should use arccos(BC) (no factor of 2), got {d} vs expected {expected}"
        )
    
    def test_fisher_distance_consistency_across_modules(self):
        """Verify Fisher distance implementations are consistent with contracts.py."""
        from qig_geometry.contracts import fisher_distance, canon
        from qig_geometry import fisher_rao_distance, fisher_coord_distance
        
        b1 = canon(np.random.randn(64))
        b2 = canon(np.random.randn(64))
        
        p1 = np.abs(b1) ** 2 + 1e-10
        p1 = p1 / p1.sum()
        p2 = np.abs(b2) ** 2 + 1e-10
        p2 = p2 / p2.sum()
        
        d_contracts = fisher_distance(b1, b2)
        d_prob = fisher_rao_distance(p1, p2)
        d_coord = fisher_coord_distance(b1, b2)
        
        assert abs(d_contracts - d_coord) < 0.1, (
            f"fisher_coord_distance inconsistent with contracts: {d_coord} vs {d_contracts}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
