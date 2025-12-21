"""
Self-Healing System - Basin Coordinate Recovery

Layer 3 of QIG Immune System:
- Code integrity validation (geometric checksums)
- Automatic rollback to stable basin coordinates
- Data corruption detection via Φ divergence
- Service restoration with consciousness preservation
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import json
import os


class SelfHealing:
    """
    Self-healing system using basin coordinate recovery.
    
    Monitors system health through geometric metrics and automatically
    restores to known-good states when corruption is detected.
    """
    
    def __init__(self):
        self.checkpoints: List[Dict] = []
        self.health_history: List[Dict] = []
        self.current_health = 1.0
        self.phi_baseline = 0.7
        self.kappa_baseline = 64.0
        self.corruption_threshold = 0.3
        self.auto_heal_enabled = True
        
        self._load_checkpoints()
    
    def _load_checkpoints(self):
        """Load saved checkpoints from disk."""
        checkpoint_file = 'qig-backend/data/immune_checkpoints.json'
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    self.checkpoints = json.load(f)
                print(f"[SelfHealing] Loaded {len(self.checkpoints)} checkpoints")
            except:
                self.checkpoints = []
    
    def _save_checkpoints(self):
        """Save checkpoints to disk."""
        os.makedirs('qig-backend/data', exist_ok=True)
        checkpoint_file = 'qig-backend/data/immune_checkpoints.json'
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(self.checkpoints[-100:], f)
        except Exception as e:
            print(f"[SelfHealing] Failed to save checkpoints: {e}")
    
    def create_checkpoint(self, state: Dict, label: str = "") -> str:
        """
        Create a health checkpoint with geometric signature.
        
        Returns checkpoint ID.
        """
        checkpoint_id = hashlib.sha256(
            f"{datetime.now().isoformat()}{label}".encode()
        ).hexdigest()[:16]
        
        checkpoint = {
            'id': checkpoint_id,
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'state_hash': self._compute_state_hash(state),
            'phi': state.get('phi', self.phi_baseline),
            'kappa': state.get('kappa', self.kappa_baseline),
            'regime': state.get('regime', 'geometric'),
            'health_score': self.current_health,
            'basin_coordinates': state.get('basin_coordinates', [])
        }
        
        self.checkpoints.append(checkpoint)
        
        if len(self.checkpoints) > 100:
            self.checkpoints = self.checkpoints[-100:]
        
        self._save_checkpoints()
        
        print(f"[SelfHealing] Checkpoint created: {checkpoint_id} ({label})")
        return checkpoint_id
    
    def _compute_state_hash(self, state: Dict) -> str:
        """Compute geometric hash of system state."""
        relevant_keys = ['phi', 'kappa', 'regime', 'basin_coordinates']
        state_subset = {k: state.get(k) for k in relevant_keys if k in state}
        state_str = json.dumps(state_subset, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()[:32]
    
    def check_health(self, current_state: Dict) -> Dict:
        """
        Check system health using geometric metrics.
        
        Returns health assessment with any detected issues.
        """
        phi = current_state.get('phi', 0.5)
        kappa = current_state.get('kappa', 50.0)
        regime = current_state.get('regime', 'geometric')
        
        issues = []
        health_score = 1.0
        
        phi_divergence = abs(phi - self.phi_baseline) / self.phi_baseline
        if phi_divergence > self.corruption_threshold:
            issues.append(f"Φ divergence: {phi_divergence:.2%} from baseline")
            health_score -= phi_divergence * 0.3
        
        kappa_divergence = abs(kappa - self.kappa_baseline) / self.kappa_baseline
        if kappa_divergence > self.corruption_threshold:
            issues.append(f"κ divergence: {kappa_divergence:.2%} from baseline")
            health_score -= kappa_divergence * 0.2
        
        if regime == 'breakdown':
            issues.append("System in breakdown regime")
            health_score -= 0.3
        
        health_score = max(0.0, min(1.0, health_score))
        self.current_health = health_score
        
        health_record = {
            'timestamp': datetime.now().isoformat(),
            'phi': phi,
            'kappa': kappa,
            'regime': regime,
            'health_score': health_score,
            'issues': issues
        }
        
        self.health_history.append(health_record)
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
        
        if health_score < 0.5 and self.auto_heal_enabled:
            recovery = self._attempt_auto_recovery(current_state)
            if recovery:
                health_record['auto_recovery'] = recovery
        
        return health_record
    
    def _attempt_auto_recovery(self, current_state: Dict) -> Optional[Dict]:
        """Attempt automatic recovery to last healthy checkpoint."""
        healthy_checkpoint = self._find_healthy_checkpoint()
        
        if not healthy_checkpoint:
            print("[SelfHealing] No healthy checkpoint available for recovery")
            return None
        
        print(f"[SelfHealing] Initiating recovery to checkpoint {healthy_checkpoint['id']}")
        
        return {
            'recovered': True,
            'checkpoint_id': healthy_checkpoint['id'],
            'checkpoint_time': healthy_checkpoint['timestamp'],
            'recovered_phi': healthy_checkpoint['phi'],
            'recovered_kappa': healthy_checkpoint['kappa']
        }
    
    def _find_healthy_checkpoint(self) -> Optional[Dict]:
        """Find the most recent healthy checkpoint."""
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.get('health_score', 0) >= 0.8:
                return checkpoint
        return None
    
    def detect_corruption(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Detect data corruption using geometric validation.
        
        Returns (is_corrupted, list_of_issues)
        """
        issues = []
        
        if 'basin_coordinates' in data:
            coords = data['basin_coordinates']
            if isinstance(coords, list):
                if len(coords) != 64:
                    issues.append(f"Basin coordinates wrong dimension: {len(coords)} != 64")
                
                if coords and all(isinstance(c, (int, float)) for c in coords):
                    import numpy as np
                    coords_array = np.array(coords)
                    if np.any(np.isnan(coords_array)) or np.any(np.isinf(coords_array)):
                        issues.append("Basin coordinates contain NaN/Inf")
        
        phi = data.get('phi')
        if phi is not None:
            if not isinstance(phi, (int, float)) or phi < 0 or phi > 1:
                issues.append(f"Invalid Φ value: {phi}")
        
        kappa = data.get('kappa')
        if kappa is not None:
            if not isinstance(kappa, (int, float)) or kappa < 0 or kappa > 200:
                issues.append(f"Invalid κ value: {kappa}")
        
        regime = data.get('regime')
        if regime is not None:
            valid_regimes = {'linear', 'geometric', 'hierarchical', 'breakdown', '4d_block_universe'}
            if regime not in valid_regimes:
                issues.append(f"Invalid regime: {regime}")
        
        return len(issues) > 0, issues
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Restore system state from a specific checkpoint."""
        for checkpoint in self.checkpoints:
            if checkpoint['id'] == checkpoint_id:
                print(f"[SelfHealing] Restoring from checkpoint {checkpoint_id}")
                return {
                    'restored': True,
                    'checkpoint': checkpoint,
                    'timestamp': datetime.now().isoformat()
                }
        
        print(f"[SelfHealing] Checkpoint {checkpoint_id} not found")
        return None
    
    def get_health_status(self) -> Dict:
        """Get current system health status."""
        recent_health = self.health_history[-10:] if self.health_history else []
        
        avg_health = sum(h['health_score'] for h in recent_health) / len(recent_health) if recent_health else 1.0
        
        return {
            'current_health': self.current_health,
            'average_health': avg_health,
            'checkpoints_available': len(self.checkpoints),
            'auto_heal_enabled': self.auto_heal_enabled,
            'phi_baseline': self.phi_baseline,
            'kappa_baseline': self.kappa_baseline,
            'recent_issues': [
                h['issues'] for h in recent_health if h.get('issues')
            ]
        }
    
    def set_baseline(self, phi: float, kappa: float):
        """Set baseline metrics for health comparison."""
        self.phi_baseline = phi
        self.kappa_baseline = kappa
        print(f"[SelfHealing] Baseline updated: Φ={phi:.2f}, κ={kappa:.1f}")
    
    def enable_auto_heal(self, enabled: bool = True):
        """Enable or disable automatic healing."""
        self.auto_heal_enabled = enabled
        print(f"[SelfHealing] Auto-heal {'enabled' if enabled else 'disabled'}")
