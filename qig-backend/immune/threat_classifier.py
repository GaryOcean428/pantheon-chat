"""
Real-Time Threat Classifier - Make defense decisions in <10ms

Layer 1.3 of QIG Immune System: High-performance classification with caching.
Orchestrates consciousness extraction, signature validation, and action decisions.

ADAPTIVE THRESHOLDS: Rate limits and cache TTLs derived from Φ/κ statistics.
"""

from typing import Dict, Optional, Set
from datetime import datetime
import hashlib
import os

from .adaptive_thresholds import get_threshold_engine

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class ThreatClassifier:
    """
    High-performance threat classification with ADAPTIVE parameters.
    
    Decision latency: <10ms with Redis caching.
    Falls back to in-memory when Redis unavailable.
    
    Rate limits and cache TTLs derived from Φ/κ distribution - no fixed constants.
    """
    
    def __init__(self):
        self.redis_client = None
        self._init_redis()
        
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()
        self.rate_limits: Dict[str, list] = {}
        self.decision_cache: Dict[str, dict] = {}
        self.threshold_engine = get_threshold_engine()
        
        self.stats = {
            'total_classified': 0,
            'blocked': 0,
            'rate_limited': 0,
            'honeypotted': 0,
            'allowed': 0
        }
    
    def _init_redis(self):
        """Initialize Redis connection if available."""
        if not REDIS_AVAILABLE:
            print("[ThreatClassifier] Redis not available, using in-memory cache")
            return
        
        redis_url = os.environ.get('REDIS_URL')
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, db=2)
                self.redis_client.ping()
                print("[ThreatClassifier] Redis connected for immune caching")
            except Exception as e:
                print(f"[ThreatClassifier] Redis connection failed: {e}")
                self.redis_client = None
    
    def classify_request(
        self, 
        request: Dict, 
        signature: Optional[Dict] = None, 
        validation: Optional[Dict] = None
    ) -> Dict:
        """
        Classify request and return action recommendation.
        
        Returns:
            {
                'action': str,  # 'allow', 'challenge', 'rate_limit', 'block', 'honeypot'
                'threat_level': str,
                'classification': str,
                'signature': Dict,
                'reasons': List[str]
            }
        """
        ip = request.get('ip', 'unknown')
        
        if self._is_whitelisted(ip):
            self.stats['allowed'] += 1
            return {
                'action': 'allow',
                'threat_level': 'none',
                'classification': 'legitimate',
                'reason': 'whitelisted',
                'reasons': ['IP is whitelisted']
            }
        
        if self._is_blacklisted(ip):
            self.stats['blocked'] += 1
            return {
                'action': 'block',
                'threat_level': 'critical',
                'classification': 'malicious',
                'reason': 'blacklisted',
                'reasons': ['IP is blacklisted']
            }
        
        rate_status = self._check_rate_limit(ip)
        if rate_status['exceeded']:
            self.stats['rate_limited'] += 1
            adaptive_limit = rate_status.get('adaptive_limit', 100)
            return {
                'action': 'rate_limit',
                'threat_level': 'medium',
                'classification': 'suspicious',
                'reason': 'rate_exceeded',
                'reasons': [f"Adaptive rate limit exceeded: {rate_status['count']}/{adaptive_limit} per minute"],
                'retry_after': rate_status.get('retry_after', 60)
            }
        
        cached = self._get_cached_decision(ip)
        if cached:
            return cached
        
        if signature is None or validation is None:
            from .consciousness_extractor import ConsciousnessExtractor
            from .signature_validator import SignatureValidator
            
            extractor = ConsciousnessExtractor()
            validator = SignatureValidator()
            
            signature = extractor.extract_request_signature(request)
            validation = validator.validate(signature)
        
        action = self._decide_action(validation, ip, request)
        
        self.stats['total_classified'] += 1
        self.stats[action if action in self.stats else 'allowed'] += 1
        
        result = {
            'action': action,
            'threat_level': validation['threat_level'],
            'classification': validation['classification'],
            'signature': signature,
            'reasons': validation.get('reasons', []),
            'threat_score': validation.get('threat_score', 0.0)
        }
        
        self._cache_decision(ip, result)
        
        return result
    
    def _decide_action(self, validation: Dict, ip: str, request: Dict) -> str:
        """Decide what action to take based on validation."""
        threat_level = validation['threat_level']
        threat_score = validation.get('threat_score', 0.0)
        
        if threat_level == 'none':
            return 'allow'
        elif threat_level == 'low':
            return 'challenge'
        elif threat_level == 'medium':
            return 'rate_limit'
        elif threat_level == 'high':
            return 'honeypot'
        else:
            return 'block'
    
    def _is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        if self.redis_client:
            try:
                return self.redis_client.sismember('qig:whitelist:ips', ip)
            except:
                pass
        return ip in self.whitelist
    
    def _is_blacklisted(self, ip: str) -> bool:
        """Check if IP is blacklisted."""
        if self.redis_client:
            try:
                return self.redis_client.sismember('qig:blacklist:ips', ip)
            except:
                pass
        return ip in self.blacklist
    
    def _check_rate_limit(self, ip: str) -> Dict:
        """Check if IP has exceeded ADAPTIVE rate limits derived from Φ/κ."""
        now = datetime.now().timestamp()
        window_start = now - 60
        
        adaptive_limit = self.threshold_engine.get_thresholds()['rate_limit_per_minute']
        
        if self.redis_client:
            try:
                key = f'qig:ratelimit:{ip}'
                count = self.redis_client.get(key)
                
                if count is None:
                    self.redis_client.setex(key, 60, 1)
                    return {'exceeded': False, 'count': 1, 'adaptive_limit': adaptive_limit}
                
                count = int(count)
                if count >= adaptive_limit:
                    ttl = self.redis_client.ttl(key)
                    return {'exceeded': True, 'count': count, 'retry_after': max(ttl, 1), 'adaptive_limit': adaptive_limit}
                
                self.redis_client.incr(key)
                return {'exceeded': False, 'count': count + 1, 'adaptive_limit': adaptive_limit}
            except:
                pass
        
        if ip not in self.rate_limits:
            self.rate_limits[ip] = []
        
        self.rate_limits[ip] = [t for t in self.rate_limits[ip] if t > window_start]
        self.rate_limits[ip].append(now)
        
        count = len(self.rate_limits[ip])
        if count >= adaptive_limit:
            return {'exceeded': True, 'count': count, 'retry_after': 60, 'adaptive_limit': adaptive_limit}
        
        return {'exceeded': False, 'count': count, 'adaptive_limit': adaptive_limit}
    
    def _get_cached_decision(self, ip: str) -> Optional[Dict]:
        """Get cached decision for IP."""
        if self.redis_client:
            try:
                import json
                key = f'qig:decision:{ip}'
                cached = self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
            except:
                pass
        
        return self.decision_cache.get(ip)
    
    def _cache_decision(self, ip: str, decision: Dict):
        """Cache decision for fast future lookups with ADAPTIVE TTL."""
        cache_entry = {
            'action': decision['action'],
            'threat_level': decision['threat_level'],
            'classification': decision['classification'],
            'cached_at': datetime.now().isoformat()
        }
        
        adaptive_ttl = self.threshold_engine.get_thresholds()['cache_ttl_seconds']
        
        if self.redis_client:
            try:
                import json
                key = f'qig:decision:{ip}'
                self.redis_client.setex(key, adaptive_ttl, json.dumps(cache_entry))
            except:
                pass
        
        self.decision_cache[ip] = cache_entry
        
        if len(self.decision_cache) > 10000:
            oldest_keys = list(self.decision_cache.keys())[:1000]
            for key in oldest_keys:
                del self.decision_cache[key]
    
    def add_to_whitelist(self, ip: str):
        """Add IP to whitelist."""
        self.whitelist.add(ip)
        if self.redis_client:
            try:
                self.redis_client.sadd('qig:whitelist:ips', ip)
            except:
                pass
    
    def add_to_blacklist(self, ip: str, reason: str = "manual"):
        """Add IP to blacklist."""
        self.blacklist.add(ip)
        if self.redis_client:
            try:
                self.redis_client.sadd('qig:blacklist:ips', ip)
                self.redis_client.hset('qig:blacklist:reasons', ip, reason)
            except:
                pass
    
    def remove_from_blacklist(self, ip: str):
        """Remove IP from blacklist."""
        self.blacklist.discard(ip)
        if self.redis_client:
            try:
                self.redis_client.srem('qig:blacklist:ips', ip)
                self.redis_client.hdel('qig:blacklist:reasons', ip)
            except:
                pass
    
    def get_stats(self) -> Dict:
        """Get classification statistics."""
        return {
            **self.stats,
            'whitelist_size': len(self.whitelist),
            'blacklist_size': len(self.blacklist),
            'cache_size': len(self.decision_cache),
            'redis_connected': self.redis_client is not None
        }
