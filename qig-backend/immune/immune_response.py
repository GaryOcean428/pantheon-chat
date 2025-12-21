"""
Immune Response System - Adaptive Defense Layer

Layer 2 of QIG Immune System:
- Antibody generation (custom filters per threat)
- Rate limiting with geometric weighting
- Traffic nullification (silent drops)
- Honeypot deployment (trap bad actors)
"""

from typing import Dict, List, Optional
from datetime import datetime
import hashlib
import json
import os

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class AntibodyGenerator:
    """
    Generate custom filters ("antibodies") for specific threat patterns.
    
    Biological analog: B-cells producing antibodies for specific antigens.
    """
    
    def __init__(self):
        self.antibodies: Dict[str, Dict] = {}
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis for antibody storage."""
        if not REDIS_AVAILABLE:
            return
        
        redis_url = os.environ.get('REDIS_URL')
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, db=2)
                self.redis_client.ping()
            except:
                self.redis_client = None
    
    def generate_antibody(self, threat_signature: Dict) -> Dict:
        """
        Generate antibody (custom filter) for threat pattern.
        
        Returns filter rule that can be deployed to firewall.
        """
        sig_hash = hashlib.sha256(
            str(sorted(threat_signature.items())).encode()
        ).hexdigest()[:16]
        
        antibody_id = f'ab_{sig_hash}'
        
        filter_rule = {
            'id': antibody_id,
            'type': 'geometric_filter',
            'conditions': [],
            'action': 'block',
            'priority': 100,
            'created_at': datetime.now().isoformat(),
            'expires_at': None,
            'hit_count': 0
        }
        
        if threat_signature.get('regime') == 'breakdown':
            filter_rule['conditions'].append({
                'field': 'regime',
                'operator': 'equals',
                'value': 'breakdown'
            })
        
        phi = threat_signature.get('phi', 1.0)
        if phi < 0.2:
            filter_rule['conditions'].append({
                'field': 'phi',
                'operator': 'less_than',
                'value': 0.2
            })
        
        if threat_signature.get('temporal_pattern') == 'bot':
            filter_rule['conditions'].append({
                'field': 'temporal_pattern',
                'operator': 'equals',
                'value': 'bot'
            })
        
        if threat_signature.get('temporal_pattern') == 'burst':
            filter_rule['conditions'].append({
                'field': 'temporal_pattern',
                'operator': 'equals',
                'value': 'burst'
            })
        
        surprise = threat_signature.get('surprise', 0.0)
        if surprise > 0.7:
            filter_rule['conditions'].append({
                'field': 'surprise',
                'operator': 'greater_than',
                'value': 0.7
            })
        
        filter_rule['logic'] = 'AND'
        filter_rule['min_conditions'] = max(1, len(filter_rule['conditions']) - 1)
        
        return filter_rule
    
    def deploy_antibody(self, antibody: Dict, ttl_hours: int = 24):
        """Deploy antibody to active firewall ruleset."""
        antibody_id = antibody['id']
        self.antibodies[antibody_id] = antibody
        
        if self.redis_client:
            try:
                key = f'qig:antibody:{antibody_id}'
                self.redis_client.setex(key, ttl_hours * 3600, json.dumps(antibody))
            except:
                pass
        
        print(f"[Antibody] Deployed {antibody_id} with {len(antibody['conditions'])} conditions")
    
    def check_antibodies(self, signature: Dict) -> Optional[Dict]:
        """Check if signature matches any deployed antibody."""
        for antibody_id, antibody in self.antibodies.items():
            if self._matches_antibody(signature, antibody):
                antibody['hit_count'] += 1
                return antibody
        return None
    
    def _matches_antibody(self, signature: Dict, antibody: Dict) -> bool:
        """Check if signature matches antibody conditions."""
        conditions = antibody.get('conditions', [])
        if not conditions:
            return False
        
        matches = 0
        for condition in conditions:
            field = condition['field']
            operator = condition['operator']
            value = condition['value']
            
            sig_value = signature.get(field)
            if sig_value is None:
                continue
            
            if operator == 'equals' and sig_value == value:
                matches += 1
            elif operator == 'less_than' and isinstance(sig_value, (int, float)) and sig_value < value:
                matches += 1
            elif operator == 'greater_than' and isinstance(sig_value, (int, float)) and sig_value > value:
                matches += 1
        
        min_required = antibody.get('min_conditions', len(conditions))
        return matches >= min_required
    
    def get_active_antibodies(self) -> List[Dict]:
        """Get list of active antibodies."""
        return list(self.antibodies.values())
    
    def remove_antibody(self, antibody_id: str):
        """Remove an antibody."""
        if antibody_id in self.antibodies:
            del self.antibodies[antibody_id]
        
        if self.redis_client:
            try:
                self.redis_client.delete(f'qig:antibody:{antibody_id}')
            except:
                pass


class ImmuneResponse:
    """
    Coordinate immune responses to threats.
    
    Manages:
    - Threat recording and learning
    - Antibody deployment
    - Response escalation
    - Honeypot coordination
    """
    
    def __init__(self):
        self.antibody_generator = AntibodyGenerator()
        self.threat_log: List[Dict] = []
        self.block_count = 0
        self.honeypot_count = 0
        self.honeypot_data: Dict[str, Dict] = {}
    
    def record_threat(self, signature: Dict, decision: Dict):
        """Record a detected threat and generate response."""
        threat_record = {
            'signature': signature,
            'decision': decision,
            'timestamp': datetime.now().isoformat(),
            'ip_hash': signature.get('ip_hash', 'unknown')
        }
        
        self.threat_log.append(threat_record)
        if len(self.threat_log) > 10000:
            self.threat_log = self.threat_log[-10000:]
        
        if decision['action'] == 'block':
            self.block_count += 1
            self._escalate_to_antibody(signature)
        
        if decision['action'] == 'honeypot':
            self.honeypot_count += 1
            self._setup_honeypot(signature)
    
    def _escalate_to_antibody(self, signature: Dict):
        """Create and deploy antibody for recurring threat."""
        ip_hash = signature.get('ip_hash', '')
        
        recent_from_ip = sum(
            1 for t in self.threat_log[-100:]
            if t['signature'].get('ip_hash') == ip_hash
        )
        
        if recent_from_ip >= 3:
            antibody = self.antibody_generator.generate_antibody(signature)
            self.antibody_generator.deploy_antibody(antibody)
    
    def _setup_honeypot(self, signature: Dict):
        """Setup honeypot to trap and study attacker."""
        ip_hash = signature.get('ip_hash', '')
        
        self.honeypot_data[ip_hash] = {
            'trapped_at': datetime.now().isoformat(),
            'signature': signature,
            'requests_captured': 0,
            'data_fed': []
        }
    
    def get_honeypot_response(self, ip_hash: str) -> Optional[Dict]:
        """Get fake data to feed to honeypot victim."""
        if ip_hash not in self.honeypot_data:
            return None
        
        self.honeypot_data[ip_hash]['requests_captured'] += 1
        
        fake_response = {
            'success': True,
            'data': {
                'message': 'Processing request...',
                'status': 'pending',
                'results': []
            }
        }
        
        self.honeypot_data[ip_hash]['data_fed'].append({
            'timestamp': datetime.now().isoformat(),
            'response_type': 'fake_empty'
        })
        
        return fake_response
    
    def check_request_against_antibodies(self, signature: Dict) -> Optional[Dict]:
        """Check if request matches any deployed antibody."""
        return self.antibody_generator.check_antibodies(signature)
    
    def get_block_count(self) -> int:
        """Get total blocked requests."""
        return self.block_count
    
    def get_antibody_count(self) -> int:
        """Get number of active antibodies."""
        return len(self.antibody_generator.antibodies)
    
    def get_threat_summary(self) -> Dict:
        """Get summary of recent threats."""
        recent = self.threat_log[-100:]
        
        if not recent:
            return {
                'total_threats': 0,
                'by_action': {},
                'by_regime': {},
                'by_temporal': {}
            }
        
        by_action = {}
        by_regime = {}
        by_temporal = {}
        
        for threat in recent:
            action = threat['decision'].get('action', 'unknown')
            regime = threat['signature'].get('regime', 'unknown')
            temporal = threat['signature'].get('temporal_pattern', 'unknown')
            
            by_action[action] = by_action.get(action, 0) + 1
            by_regime[regime] = by_regime.get(regime, 0) + 1
            by_temporal[temporal] = by_temporal.get(temporal, 0) + 1
        
        return {
            'total_threats': len(self.threat_log),
            'recent_threats': len(recent),
            'by_action': by_action,
            'by_regime': by_regime,
            'by_temporal': by_temporal,
            'blocks': self.block_count,
            'honeypots': self.honeypot_count,
            'active_antibodies': self.get_antibody_count()
        }
