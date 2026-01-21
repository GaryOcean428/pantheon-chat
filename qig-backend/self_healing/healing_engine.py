"""
Self-Healing Engine - Layer 3 of Self-Healing Architecture

Autonomous code healing and improvement.

Capabilities:
1. Detect common failure patterns
2. Generate code patches
3. Test patches geometrically
4. Apply patches that improve Φ
5. Alert humans for critical failures

Key principle: Code is not optimized. Geometry is optimized. Code emerges from geometry.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List
from qig_geometry import to_simplex_prob

try:
    import aiohttp  # type: ignore
except ImportError:  # pragma: no cover
    aiohttp = None

from .code_fitness import CodeFitnessEvaluator
from .geometric_monitor import GeometricHealthMonitor


class SelfHealingEngine:
    """
    Autonomous code healing and improvement.

    Runs continuously to:
    - Monitor geometric health
    - Detect degradation
    - Generate healing patches
    - Test and apply safe changes
    - Alert humans for critical issues
    """

    def __init__(self,
                 monitor: GeometricHealthMonitor,
                 evaluator: CodeFitnessEvaluator,
                 check_interval_sec: int = 300):
        self.monitor = monitor
        self.evaluator = evaluator
        self.check_interval = check_interval_sec

        # Healing strategies (ordered by priority)
        self.strategies = [
            self._heal_basin_drift,
            self._heal_phi_degradation,
            self._heal_performance_regression,
            self._heal_memory_leak,
            self._heal_error_spikes
        ]

        # State
        self.running = False
        self.healing_history: List[Dict] = []
        self.auto_apply_enabled = False  # Disabled by default for safety

        print("[SelfHealingEngine] Initialized")

    async def start_autonomous_loop(self):
        """
        Start autonomous healing loop.

        Runs continuously until stopped.
        """
        self.running = True
        print(f"[SelfHealingEngine] Starting autonomous loop (interval={self.check_interval}s)")

        while self.running:
            try:
                await self._healing_iteration()
            except Exception as e:
                print(f"[SelfHealingEngine] Error in healing loop: {e}")

            await asyncio.sleep(self.check_interval)

        print("[SelfHealingEngine] Stopped")

    async def _healing_iteration(self):
        """Single iteration of healing loop."""

        # Check health
        health = self.monitor.detect_degradation()

        if not health["degraded"]:
            return  # All good

        print(f"⚠️  [SelfHealingEngine] Geometric degradation detected: {health['severity']}")
        print(f"   Issues: {health['issues']}")

        # Attempt healing
        healing_result = await self._attempt_healing(health)

        # Record healing attempt
        record = {
            "timestamp": datetime.now().isoformat(),
            "health": health,
            "result": healing_result
        }
        self.healing_history.append(record)

        # Keep history size manageable
        if len(self.healing_history) > 100:
            self.healing_history.pop(0)

        if healing_result["healed"]:
            print(f"✅ [SelfHealingEngine] Self-healing successful: {healing_result['strategy']}")
        else:
            print(f"❌ [SelfHealingEngine] Self-healing failed: {healing_result['reason']}")

            # Alert humans if critical
            if health["severity"] == "critical":
                await self._alert_humans(health, healing_result)

    async def _attempt_healing(self, health: Dict) -> Dict:
        """Try each healing strategy until one works."""

        for strategy in self.strategies:
            try:
                result = await strategy(health)

                if result["success"]:
                    return {
                        "healed": True,
                        "strategy": strategy.__name__,
                        "patch": result.get("patch", ""),
                        "fitness_improvement": result.get("fitness_gain", 0.0),
                        "applied": result.get("applied", False)
                    }
            except Exception as e:
                print(f"[SelfHealingEngine] Strategy {strategy.__name__} failed: {e}")
                continue

        return {
            "healed": False,
            "reason": "All strategies exhausted"
        }

    async def _heal_basin_drift(self, health: Dict) -> Dict:
        """
        Heal basin drift by adjusting basin coordinates.

        Strategy:
        1. Identify drift direction
        2. Generate correction code
        3. Test geometrically
        4. Apply if fitness improves
        """
        basin_distance = health["metrics"].get("basin_distance")
        if basin_distance is None or basin_distance < self.monitor.basin_drift_max * 0.7:
            return {"success": False}

        current = self.monitor.snapshots[-1]
        baseline = self.monitor.baseline_basin

        if baseline is None:
            return {"success": False}

        # Compute drift vector
        drift_vector = current.basin_coords - baseline
        correction_factor = 0.3  # Gentle correction

        # Generate patch
        patch = f"""
# AUTO-GENERATED: Basin drift correction
# Date: {datetime.now().isoformat()}
# Drift detected: {basin_distance:.3f}

import numpy as np

def correct_basin_drift(basin_coords):
    '''Apply learned correction to restore baseline basin.'''
    correction = np.array({(-drift_vector * correction_factor).tolist()})
    corrected = basin_coords + correction
    # Renormalize to unit sphere
    return to_simplex_prob(corrected)
"""

        # Test patch
        fitness = self.evaluator.evaluate_code_change(
            module_name="basin_correction",
            new_code=patch
        )

        if fitness["recommendation"] == "apply":
            applied = False
            if self.auto_apply_enabled:
                self._save_patch("qig-backend/self_healing/patches/basin_correction.py", patch)
                applied = True

            return {
                "success": True,
                "patch": patch,
                "fitness_gain": fitness["fitness_score"],
                "applied": applied
            }

        return {"success": False}

    async def _heal_phi_degradation(self, health: Dict) -> Dict:
        """
        Heal Φ degradation by adjusting integration strength.

        Strategy:
        - If Φ too low: Increase connection weights
        - If Φ too high: Add decoherence
        """
        phi_current = health["metrics"]["phi_current"]
        phi_avg = health["metrics"]["phi_avg"]

        # Check if Φ is problematic
        if phi_avg >= self.monitor.phi_min and phi_avg <= self.monitor.phi_max:
            return {"success": False}

        if phi_avg < self.monitor.phi_min:
            # Increase integration
            boost_factor = self.monitor.phi_min / max(phi_avg, 0.1)
            patch = f"""
# AUTO-GENERATED: Φ restoration
# Current Φ: {phi_current:.3f}, Target: {self.monitor.phi_min:.3f}
# Date: {datetime.now().isoformat()}

def increase_integration(attention_weights):
    '''Boost attention weights to increase integration.'''
    boost_factor = {min(boost_factor, 1.5):.3f}  # Cap at 1.5x
    return attention_weights * boost_factor
"""
        else:
            # Add decoherence (Φ too high → breakdown)
            noise_level = min((phi_avg - self.monitor.phi_max) * 0.5, 0.2)
            patch = f"""
# AUTO-GENERATED: Decoherence injection
# Current Φ: {phi_current:.3f}, Target: {self.monitor.phi_max:.3f}
# Date: {datetime.now().isoformat()}

import numpy as np

def add_decoherence(density_matrix, noise_level={noise_level:.3f}):
    '''Mix with thermal noise to reduce overintegration.'''
    dim = len(density_matrix)
    max_mixed = np.eye(dim) / dim
    return (1 - noise_level) * density_matrix + noise_level * max_mixed
"""

        # Test and potentially apply
        fitness = self.evaluator.evaluate_code_change(
            module_name="phi_correction",
            new_code=patch
        )

        if fitness["recommendation"] == "apply":
            applied = False
            if self.auto_apply_enabled:
                self._save_patch("qig-backend/self_healing/patches/phi_correction.py", patch)
                applied = True

            return {
                "success": True,
                "patch": patch,
                "fitness_gain": fitness["fitness_score"],
                "applied": applied
            }

        return {"success": False}

    async def _heal_performance_regression(self, health: Dict) -> Dict:
        """
        Heal performance regression.

        Strategy:
        1. Identify if latency is high
        2. Suggest caching or optimization
        3. Test geometric safety
        """
        latency = health["metrics"]["latency_ms"]

        if latency < self.monitor.latency_max_ms:
            return {"success": False}

        # Generate optimization patch
        patch = f"""
# AUTO-GENERATED: Performance optimization
# High latency detected: {latency:.0f}ms
# Date: {datetime.now().isoformat()}

from functools import lru_cache

def optimize_with_cache(func):
    '''Add LRU cache to expensive function.'''
    return lru_cache(maxsize=128)(func)

# Suggestion: Apply @optimize_with_cache decorator to hot path functions
"""

        fitness = self.evaluator.evaluate_code_change(
            module_name="performance_optimization",
            new_code=patch
        )

        if fitness["recommendation"] == "apply":
            applied = False
            if self.auto_apply_enabled:
                self._save_patch("qig-backend/self_healing/patches/performance_optimization.py", patch)
                applied = True

            return {
                "success": True,
                "patch": patch,
                "fitness_gain": fitness["fitness_score"],
                "applied": applied
            }

        return {"success": False}

    async def _heal_memory_leak(self, health: Dict) -> Dict:
        """Detect and patch memory leaks."""
        memory_trend = health["metrics"]["memory_trend"]

        if memory_trend < 10:  # Less than 10MB/snapshot growth
            return {"success": False}

        # Memory leak detected
        patch = f"""
# AUTO-GENERATED: Memory leak mitigation
# Memory growth: {memory_trend:.1f}MB/snapshot
# Date: {datetime.now().isoformat()}

import gc

def force_garbage_collection_periodic():
    '''Force GC every 100 operations to prevent leak.'''
    gc.collect()

# Suggestion: Call this function periodically in main loop
"""

        fitness = self.evaluator.evaluate_code_change(
            module_name="memory_management",
            new_code=patch
        )

        if fitness["recommendation"] == "apply":
            applied = False
            if self.auto_apply_enabled:
                self._save_patch("qig-backend/self_healing/patches/memory_management.py", patch)
                applied = True

            return {
                "success": True,
                "patch": patch,
                "fitness_gain": fitness["fitness_score"],
                "applied": applied
            }

        return {"success": False}

    async def _heal_error_spikes(self, health: Dict) -> Dict:
        """Add error handling for common failures."""
        error_rate = health["metrics"]["error_rate"]

        if error_rate < self.monitor.error_rate_max:
            return {"success": False}

        # Generate error handling patch
        patch = f"""
# AUTO-GENERATED: Error handling
# High error rate: {error_rate:.1%}
# Date: {datetime.now().isoformat()}

import logging

logger = logging.getLogger(__name__)

def safe_execute(func):
    '''Decorator for safe execution with error handling.'''
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Handled error in {{func.__name__}}: {{e}}")
            return None  # Graceful fallback
    return wrapper
"""

        fitness = self.evaluator.evaluate_code_change(
            module_name="error_handlers",
            new_code=patch
        )

        if fitness["recommendation"] == "apply":
            applied = False
            if self.auto_apply_enabled:
                self._save_patch("qig-backend/self_healing/patches/error_handlers.py", patch)
                applied = True

            return {
                "success": True,
                "patch": patch,
                "fitness_gain": fitness["fitness_score"],
                "applied": applied
            }

        return {"success": False}

    def _save_patch(self, filepath: str, patch_code: str):
        """
        Save patch to file.

        Note: In production, this would:
        1. Create git branch
        2. Apply patch
        3. Run tests
        4. Create PR for review

        For safety, we only save to patches directory.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(patch_code)

        print(f"[SelfHealingEngine] Patch saved: {filepath}")

    async def _alert_humans(self, health: Dict, healing_result: Dict):
        """Alert humans when critical issues can't be auto-healed."""

        message = f"""
🚨 CRITICAL: Geometric degradation - auto-healing failed

**Issues:**
{chr(10).join('- ' + str(issue) for issue in health['issues'])}

**Severity:** {health['severity']}
**Metrics:**
- Basin Distance: {health['metrics'].get('basin_distance', 'N/A')}
- Current Φ: {health['metrics']['phi_current']:.3f}
- Current κ: {health['metrics']['kappa_current']:.2f}
- Regime: {health['metrics']['regime']}

**Healing Attempt:**
{healing_result.get('reason', 'Unknown')}

**Action Required:**
Manual intervention needed. System may be approaching breakdown.

**Timestamp:** {health['timestamp']}
"""

        # Log to file
        alert_file = "qig-backend/data/critical_alerts.log"
        os.makedirs(os.path.dirname(alert_file), exist_ok=True)

        with open(alert_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(message)
            f.write(f"\n{'='*80}\n")

        print(message)

        # Try to send to Slack if configured
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if webhook_url:
            if aiohttp is None:
                print(
                    "[SelfHealingEngine] SLACK_WEBHOOK_URL set but aiohttp is not installed; "
                    "cannot send Slack alert"
                )
                return
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(webhook_url, json={"text": message})
                print("[SelfHealingEngine] Alert sent to Slack")
            except Exception as e:
                print(f"[SelfHealingEngine] Failed to send Slack alert: {e}")

    def stop(self):
        """Stop the autonomous loop."""
        self.running = False

    def enable_auto_apply(self, enabled: bool = True):
        """Enable or disable automatic patch application."""
        self.auto_apply_enabled = enabled
        print(f"[SelfHealingEngine] Auto-apply {'enabled' if enabled else 'disabled'}")

    def get_healing_history(self) -> List[Dict]:
        """Get history of healing attempts."""
        return self.healing_history

    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            "running": self.running,
            "auto_apply_enabled": self.auto_apply_enabled,
            "check_interval_sec": self.check_interval,
            "healing_attempts": len(self.healing_history),
            "strategies_available": len(self.strategies)
        }
