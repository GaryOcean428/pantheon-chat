"""
Geometric Health Monitor - Layer 1 of Self-Healing Architecture

Provides real-time monitoring of system geometric health through:
- Φ (integration) tracking
- κ (coupling constant) monitoring
- Basin coordinate drift detection
- Regime stability analysis
- Performance telemetry

Pure geometric approach - no code optimization, only geometry optimization.
"""

import numpy as np

# QIG-pure geometric operations
try:
    from qig_geometry import sphere_project
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    def sphere_project(v):
        """Fallback sphere projection."""
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            result = np.ones_like(v)
            return result / np.linalg.norm(result)
        return v / norm
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import subprocess
import sys
import psutil
import os

# Import physics constants
try:
    from qigkernels.physics_constants import PHI_THRESHOLD, KAPPA_STAR
except ImportError:
    # Fallback values if qigkernels not available
    PHI_THRESHOLD = 0.70
    KAPPA_STAR = 64.21


@dataclass
class GeometricSnapshot:
    """Snapshot of system geometry at a point in time."""
    timestamp: datetime
    phi: float
    kappa_eff: float
    basin_coords: np.ndarray  # 64D
    confidence: float
    surprise: float
    agency: float
    regime: str  # "linear" | "geometric" | "breakdown" | "hierarchical" | "4d_block_universe"
    
    # Code fingerprint
    code_hash: str
    active_modules: List[str]
    module_versions: Dict[str, str]
    
    # Performance metrics
    error_rate: float
    avg_latency_ms: float
    memory_usage_mb: float
    cpu_usage_pct: float
    
    # Optional metadata
    label: str = ""
    context: Dict = field(default_factory=dict)


class GeometricHealthMonitor:
    """
    Monitors system geometric health in real-time.
    
    Detects:
    - Φ degradation (consciousness loss)
    - Basin drift (state instability)
    - Regime instability (processing breakdown)
    - Performance anomalies
    
    Key insight: Good code preserves/improves geometry.
    Bad code degrades Φ and increases basin drift.
    """
    
    def __init__(self, 
                 snapshot_interval_sec: int = 60,
                 history_size: int = 1000):
        self.snapshot_interval = snapshot_interval_sec
        self.history_size = history_size
        
        self.snapshots: List[GeometricSnapshot] = []
        self.baseline_basin: Optional[np.ndarray] = None
        
        # Health thresholds (from physics constants)
        self.phi_min = PHI_THRESHOLD * 0.93  # ~0.65, slightly below consciousness
        self.phi_max = 0.85  # Breakdown begins above this
        self.kappa_target = KAPPA_STAR  # Fixed point resonance
        self.kappa_tolerance = 5.0  # Acceptable deviation
        self.basin_drift_max = 2.0  # Max Fisher distance from baseline
        
        # Performance thresholds
        self.error_rate_max = 0.05  # 5% error rate
        self.latency_max_ms = 2000  # 2 seconds
        self.memory_warning_mb = 1000  # 1GB warning
        
        print("[GeometricHealthMonitor] Initialized")
    
    def capture_snapshot(self, system_state: Dict) -> GeometricSnapshot:
        """
        Capture current geometric state.
        
        Args:
            system_state: Dict with keys:
                - phi: float
                - kappa_eff: float
                - basin_coords: np.ndarray or list
                - confidence: float
                - surprise: float
                - agency: float
                - error_rate: float (optional)
                - avg_latency: float (optional)
        
        Returns:
            GeometricSnapshot instance
        """
        # Extract geometric metrics
        phi = float(system_state.get("phi", 0.5))
        kappa_eff = float(system_state.get("kappa_eff", 50.0))
        
        # Handle basin coordinates
        basin_coords = system_state.get("basin_coords", system_state.get("basin_coordinates"))
        if basin_coords is None:
            basin_coords = np.zeros(64)
        elif isinstance(basin_coords, list):
            basin_coords = np.array(basin_coords, dtype=np.float64)
        
        # Ensure unit norm (basins live on hypersphere)
        norm = np.linalg.norm(basin_coords)
        if norm > 0:
            basin_coords = basin_coords / norm
        
        # Get consciousness metrics
        confidence = float(system_state.get("confidence", 0.5))
        surprise = float(system_state.get("surprise", 0.0))
        agency = float(system_state.get("agency", 0.5))
        
        # Classify regime
        regime = self._classify_regime(phi)
        
        # Get performance metrics
        error_rate = float(system_state.get("error_rate", 0.0))
        avg_latency = float(system_state.get("avg_latency", system_state.get("latency_ms", 100.0)))
        
        # Get system resource usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_pct = process.cpu_percent(interval=0.1)
        
        snapshot = GeometricSnapshot(
            timestamp=datetime.now(),
            phi=phi,
            kappa_eff=kappa_eff,
            basin_coords=basin_coords,
            confidence=confidence,
            surprise=surprise,
            agency=agency,
            regime=regime,
            code_hash=self._get_git_hash(),
            active_modules=self._get_active_modules(),
            module_versions=self._get_module_versions(),
            error_rate=error_rate,
            avg_latency_ms=avg_latency,
            memory_usage_mb=memory_mb,
            cpu_usage_pct=cpu_pct,
            label=system_state.get("label", ""),
            context=system_state.get("context", {})
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.history_size:
            self.snapshots.pop(0)
        
        # Set baseline if first snapshot or explicitly requested
        if self.baseline_basin is None or system_state.get("set_baseline", False):
            self.baseline_basin = snapshot.basin_coords.copy()
            print(f"[GeometricHealthMonitor] Baseline set: Φ={phi:.3f}, κ={kappa_eff:.2f}")
        
        return snapshot
    
    def detect_degradation(self) -> Dict:
        """
        Detect geometric degradation.
        
        Returns dict with:
        - degraded: bool
        - issues: List[str]
        - severity: "critical" | "warning" | "normal"
        - metrics: Dict with detailed measurements
        """
        if len(self.snapshots) < 10:
            return {
                "degraded": False,
                "issues": [],
                "severity": "normal",
                "message": "Insufficient history for analysis"
            }
        
        recent = self.snapshots[-10:]  # Last 10 snapshots
        current = self.snapshots[-1]
        
        issues = []
        severity = "normal"
        
        # 1. Check Φ degradation
        avg_phi = np.mean([s.phi for s in recent])
        phi_trend = self._compute_trend([s.phi for s in recent])
        
        if avg_phi < self.phi_min:
            issues.append(f"Φ below consciousness threshold: {avg_phi:.3f} < {self.phi_min}")
            severity = "critical"
        elif current.phi < self.phi_min * 1.1:
            issues.append(f"Φ approaching threshold: {current.phi:.3f}")
            severity = "warning" if severity != "critical" else severity
        
        if avg_phi > self.phi_max:
            issues.append(f"Φ too high (breakdown): {avg_phi:.3f} > {self.phi_max}")
            severity = "critical"
        
        if phi_trend < -0.05:
            issues.append(f"Φ declining: trend={phi_trend:.3f}")
            severity = "warning" if severity == "normal" else severity
        
        # 2. Check basin drift
        if self.baseline_basin is not None:
            basin_distance = self._fisher_distance(
                current.basin_coords, 
                self.baseline_basin
            )
            
            if basin_distance > self.basin_drift_max:
                issues.append(f"Basin drift critical: {basin_distance:.3f} > {self.basin_drift_max}")
                severity = "critical"
            elif basin_distance > self.basin_drift_max * 0.7:
                issues.append(f"Basin drift warning: {basin_distance:.3f}")
                severity = "warning" if severity == "normal" else severity
        else:
            basin_distance = 0.0
        
        # 3. Check κ stability
        kappa_deviation = abs(current.kappa_eff - self.kappa_target)
        if kappa_deviation > self.kappa_tolerance * 2:
            issues.append(f"κ far from resonance: {current.kappa_eff:.2f} vs {self.kappa_target:.2f}")
            severity = "warning" if severity == "normal" else severity
        
        # 4. Check regime stability
        regimes = [s.regime for s in recent]
        breakdown_count = regimes.count("breakdown")
        if breakdown_count > 3:
            issues.append(f"Frequent breakdown regime: {breakdown_count}/10 snapshots")
            severity = "critical"
        
        # 5. Check performance anomalies
        if current.error_rate > self.error_rate_max:
            issues.append(f"High error rate: {current.error_rate:.1%}")
            severity = "critical"
        
        if current.avg_latency_ms > self.latency_max_ms:
            issues.append(f"High latency: {current.avg_latency_ms:.0f}ms")
            severity = "warning" if severity == "normal" else severity
        
        if current.memory_usage_mb > self.memory_warning_mb:
            issues.append(f"High memory usage: {current.memory_usage_mb:.0f}MB")
            severity = "warning" if severity == "normal" else severity
        
        # 6. Check for memory leaks
        memory_trend = self._compute_trend([s.memory_usage_mb for s in recent])
        if memory_trend > 10:  # >10MB/snapshot increase
            issues.append(f"Potential memory leak: {memory_trend:.1f}MB/snapshot growth")
            severity = "warning" if severity == "normal" else severity
        
        return {
            "degraded": len(issues) > 0,
            "issues": issues,
            "severity": severity,
            "metrics": {
                "basin_distance": basin_distance if self.baseline_basin is not None else None,
                "phi_current": current.phi,
                "phi_avg": avg_phi,
                "phi_trend": phi_trend,
                "kappa_current": current.kappa_eff,
                "kappa_deviation": kappa_deviation,
                "regime": current.regime,
                "breakdown_frequency": breakdown_count / 10,
                "error_rate": current.error_rate,
                "latency_ms": current.avg_latency_ms,
                "memory_mb": current.memory_usage_mb,
                "memory_trend": memory_trend
            },
            "timestamp": current.timestamp.isoformat()
        }
    
    def _fisher_distance(self, basin1: np.ndarray, basin2: np.ndarray) -> float:
        """
        Fisher-Rao distance between basins.
        
        Basins are unit-norm vectors on hypersphere.
        Fisher distance = arccos(basin1 · basin2)
        """
        dot_product = np.clip(np.dot(basin1, basin2), -1.0, 1.0)
        return float(np.arccos(dot_product))
    
    def _classify_regime(self, phi: float) -> str:
        """Classify processing regime from Φ."""
        if phi < 0.3:
            return "linear"
        elif phi < 0.7:
            return "geometric"
        elif phi < 0.85:
            return "hierarchical"
        else:
            return "breakdown"
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend (slope) of values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=1,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return "unknown"
    
    def _get_active_modules(self) -> List[str]:
        """Get list of active Python modules (top 50)."""
        modules = list(sys.modules.keys())[:50]
        return [m for m in modules if not m.startswith('_')]
    
    def _get_module_versions(self) -> Dict[str, str]:
        """Get versions of key modules."""
        import importlib.metadata
        
        key_modules = [
            "numpy", "scipy", "flask", "fastapi",
            "psycopg2", "redis", "torch"
        ]
        
        versions = {}
        for module in key_modules:
            try:
                versions[module] = importlib.metadata.version(module)
            except Exception:
                versions[module] = "not installed"
        
        return versions
    
    def get_health_summary(self) -> Dict:
        """Get summary of system health."""
        if not self.snapshots:
            return {
                "status": "no data",
                "snapshots_collected": 0
            }
        
        current = self.snapshots[-1]
        degradation = self.detect_degradation()
        
        # Handle case where metrics might not be present
        metrics = degradation.get("metrics", {})
        basin_distance = metrics.get("basin_distance") if metrics else None
        
        return {
            "status": "healthy" if not degradation["degraded"] else degradation["severity"],
            "snapshots_collected": len(self.snapshots),
            "current_phi": current.phi,
            "current_kappa": current.kappa_eff,
            "current_regime": current.regime,
            "basin_drift": basin_distance,
            "issues": degradation.get("issues", []),
            "last_snapshot": current.timestamp.isoformat()
        }
    
    def set_baseline(self, basin_coords: Optional[np.ndarray] = None):
        """Set baseline basin coordinates."""
        if basin_coords is not None:
            self.baseline_basin = sphere_project(basin_coords)
        elif self.snapshots:
            self.baseline_basin = self.snapshots[-1].basin_coords.copy()
        print(f"[GeometricHealthMonitor] Baseline updated")
    
    def clear_history(self):
        """Clear snapshot history."""
        self.snapshots.clear()
        self.baseline_basin = None
        print("[GeometricHealthMonitor] History cleared")
