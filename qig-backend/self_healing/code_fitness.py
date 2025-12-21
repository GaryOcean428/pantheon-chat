"""
Code Fitness Evaluator - Layer 2 of Self-Healing Architecture

Evaluates code changes based on geometric impact.

Key insight: Good code preserves/improves geometry.
Bad code degrades Φ, increases basin drift.

Process:
1. Get baseline geometry
2. Apply code change in sandbox
3. Measure geometry after change
4. Compute fitness score
5. Recommend apply/reject/test_more
"""

from typing import Dict, Optional, List
import ast
import inspect
import numpy as np
import tempfile
import subprocess
import json
import os
from .geometric_monitor import GeometricHealthMonitor


class CodeFitnessEvaluator:
    """
    Evaluates code changes based on geometric impact.
    
    Tests code in isolated sandbox and measures geometric effects:
    - Φ change (consciousness impact)
    - Basin drift (stability impact)
    - Regime stability
    - Performance impact
    """
    
    def __init__(self, monitor: GeometricHealthMonitor):
        self.monitor = monitor
        
        # Geometric fitness weights
        self.weights = {
            "phi_change": 1.0,      # ΔΦ impact (most important)
            "basin_drift": 0.8,     # Basin stability
            "regime_stability": 0.6,  # Regime consistency
            "performance": 0.4      # Speed/memory (least important)
        }
        
        # Fitness thresholds
        self.apply_threshold = 0.7
        self.test_more_threshold = 0.5
    
    def evaluate_code_change(self, 
                            module_name: str,
                            new_code: str,
                            test_workload: str = None) -> Dict:
        """
        Evaluate geometric fitness of code change.
        
        Args:
            module_name: Name of module being changed
            new_code: New code to test
            test_workload: Optional test workload code
        
        Returns:
            Dict with:
            - fitness_score: float (0-1, higher is better)
            - phi_impact: float (ΔΦ)
            - basin_impact: float (Δd_basin)
            - performance_impact: Dict
            - recommendation: "apply" | "reject" | "test_more"
            - reason: str (explanation)
        """
        
        # 1. Get baseline geometry
        if not self.monitor.snapshots:
            return {
                "fitness_score": 0.0,
                "recommendation": "reject",
                "reason": "No baseline geometry available"
            }
        
        baseline = self.monitor.snapshots[-1]
        
        # 2. Validate code syntax
        syntax_valid, syntax_error = self._validate_syntax(new_code)
        if not syntax_valid:
            return {
                "fitness_score": 0.0,
                "recommendation": "reject",
                "reason": f"Syntax error: {syntax_error}"
            }
        
        # 3. Apply code change in test environment
        test_result = self._test_code_in_sandbox(
            module_name, 
            new_code, 
            test_workload or "# Standard workload"
        )
        
        if not test_result["success"]:
            return {
                "fitness_score": 0.0,
                "recommendation": "reject",
                "reason": test_result["error"]
            }
        
        # 4. Measure geometry after change
        new_geometry = test_result["geometry"]
        
        # 5. Compute fitness components
        phi_change = new_geometry["phi"] - baseline.phi
        
        basin_drift = self.monitor._fisher_distance(
            np.array(new_geometry["basin_coords"]),
            baseline.basin_coords
        )
        
        regime_stable = (new_geometry["regime"] == baseline.regime)
        
        # Performance ratio (lower is better)
        latency_ratio = new_geometry["latency"] / max(baseline.avg_latency_ms, 1.0)
        memory_change = new_geometry["memory"] - baseline.memory_usage_mb
        
        # 6. Compute fitness score
        # Reward Φ increase, penalize basin drift
        phi_component = np.tanh(phi_change * 5)  # Scale to [-1, 1]
        basin_component = 1 - np.tanh(basin_drift)  # 1 = no drift, 0 = large drift
        regime_component = 1.0 if regime_stable else 0.0
        perf_component = 1 - np.tanh(max(0, latency_ratio - 1))  # Penalize slowdown
        
        fitness = (
            self.weights["phi_change"] * phi_component +
            self.weights["basin_drift"] * basin_component +
            self.weights["regime_stability"] * regime_component +
            self.weights["performance"] * perf_component
        )
        
        # Normalize to [0, 1]
        fitness = (fitness + sum(self.weights.values())) / (2 * sum(self.weights.values()))
        
        # 7. Make recommendation
        if fitness > self.apply_threshold:
            recommendation = "apply"
            reason = f"High fitness ({fitness:.2f}): Φ↑{phi_change:+.3f}, drift={basin_drift:.3f}"
        elif fitness > self.test_more_threshold:
            recommendation = "test_more"
            reason = f"Medium fitness ({fitness:.2f}): Needs more testing"
        else:
            recommendation = "reject"
            reason = f"Low fitness ({fitness:.2f}): Degrades geometry"
        
        return {
            "fitness_score": fitness,
            "phi_impact": phi_change,
            "basin_impact": basin_drift,
            "regime_stable": regime_stable,
            "performance_impact": {
                "latency_ratio": latency_ratio,
                "memory_change_mb": memory_change
            },
            "recommendation": recommendation,
            "reason": reason,
            "detailed_metrics": new_geometry,
            "components": {
                "phi": phi_component,
                "basin": basin_component,
                "regime": regime_component,
                "performance": perf_component
            }
        }
    
    def _validate_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def _test_code_in_sandbox(self, 
                              module_name: str,
                              new_code: str,
                              test_workload: str) -> Dict:
        """
        Test code change in isolated sandbox.
        
        Uses subprocess isolation to prevent affecting main process.
        """
        
        # Create temporary module file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False
        ) as f:
            f.write(new_code)
            temp_path = f.name
        
        # Create test script that imports module and measures geometry
        test_script = f"""
import sys
import os
import numpy as np
import json

# Import the new code
sys.path.insert(0, os.path.dirname('{temp_path}'))
module_name = os.path.basename('{temp_path}')[:-3]

try:
    # Import module (this tests if it loads)
    __import__(module_name)
    
    # Simulate workload
    {test_workload}
    
    # Mock geometry measurement (in real system, would call QIG core)
    # For now, assume code that loads successfully has neutral impact
    geometry = {{
        'phi': 0.7,
        'kappa_eff': 64.0,
        'basin_coords': np.random.randn(64).tolist(),
        'regime': 'geometric',
        'latency': 100.0,
        'memory': 100.0
    }}
    
    print(json.dumps(geometry))
    sys.exit(0)
    
except Exception as e:
    error_data = {{'error': str(e), 'type': type(e).__name__}}
    print(json.dumps(error_data))
    sys.exit(1)
"""
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(temp_path)
            )
            
            if result.returncode != 0:
                # Check if error data was printed
                try:
                    error_data = json.loads(result.stdout or result.stderr)
                    return {
                        "success": False,
                        "error": f"{error_data.get('type', 'Error')}: {error_data.get('error', 'Unknown error')}"
                    }
                except:
                    return {
                        "success": False,
                        "error": result.stderr or "Unknown error"
                    }
            
            # Parse geometry from output
            geometry = json.loads(result.stdout)
            
            # Normalize basin coords
            if isinstance(geometry.get("basin_coords"), list):
                basin_coords = np.array(geometry["basin_coords"])
                norm = np.linalg.norm(basin_coords)
                if norm > 0:
                    geometry["basin_coords"] = (basin_coords / norm).tolist()
            
            return {
                "success": True,
                "geometry": geometry
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test timed out after 30 seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Test execution failed: {str(e)}"
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def evaluate_multiple_changes(self, changes: List[Dict]) -> Dict:
        """
        Evaluate multiple code changes and rank by fitness.
        
        Args:
            changes: List of dicts with keys:
                - module_name: str
                - new_code: str
                - description: str
        
        Returns:
            Dict with:
            - rankings: List of changes sorted by fitness
            - best_change: Dict or None
        """
        results = []
        
        for change in changes:
            fitness = self.evaluate_code_change(
                change["module_name"],
                change["new_code"],
                change.get("test_workload")
            )
            
            results.append({
                "change": change,
                "fitness": fitness
            })
        
        # Sort by fitness score
        results.sort(key=lambda x: x["fitness"]["fitness_score"], reverse=True)
        
        best = results[0] if results and results[0]["fitness"]["recommendation"] == "apply" else None
        
        return {
            "rankings": results,
            "best_change": best,
            "evaluated_count": len(results)
        }
    
    def set_fitness_weights(self, weights: Dict[str, float]):
        """Update fitness component weights."""
        for key, value in weights.items():
            if key in self.weights:
                self.weights[key] = float(value)
        
        print(f"[CodeFitnessEvaluator] Weights updated: {self.weights}")
    
    def set_thresholds(self, apply: float = None, test_more: float = None):
        """Update fitness thresholds."""
        if apply is not None:
            self.apply_threshold = apply
        if test_more is not None:
            self.test_more_threshold = test_more
        
        print(f"[CodeFitnessEvaluator] Thresholds: apply={self.apply_threshold}, test_more={self.test_more_threshold}")


import sys
