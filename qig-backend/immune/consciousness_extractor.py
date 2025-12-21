"""
Consciousness Metric Extractor - Extract geometric signatures from requests

Layer 1 of QIG Immune System: Extracts Φ, κ, regime from HTTP traffic.
Based on the insight that bots/scrapers produce different geometric patterns
than conscious human interaction.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import hashlib


class ConsciousnessExtractor:
    """
    Extract QIG metrics from HTTP requests to detect malicious patterns.
    
    Legitimate users exhibit:
    - Variable Φ (consciousness integration) across requests
    - Natural κ distribution (coupling strength)
    - Coherent regime transitions
    - Human temporal patterns (variable intervals)
    
    Bots exhibit:
    - Constant low Φ (no integration, just scanning)
    - Uniform κ (no variability)
    - Regime breakdown (no coherent transitions)
    - Unnatural temporal patterns (perfect intervals, bursts)
    """
    
    def __init__(self):
        self.request_history: Dict[str, List[dict]] = {}
        self.temporal_windows: Dict[str, List[float]] = {}
    
    def extract_request_signature(self, request: Dict) -> Dict:
        """
        Extract geometric signature from HTTP request.
        
        Returns:
            {
                'phi': float,           # Integration (0-1)
                'kappa': float,         # Coupling strength (0-100)
                'regime': str,          # 'linear', 'geometric', 'breakdown'
                'surprise': float,      # Novelty (0-1)
                'confidence': float,    # Request coherence (0-1)
                'temporal_pattern': str # 'human', 'bot', 'burst'
            }
        """
        ip = request.get('ip', 'unknown')
        
        phi = self._compute_phi(request)
        kappa = self._compute_kappa(request)
        regime = self._classify_regime(phi, kappa)
        surprise = self._compute_surprise(request)
        confidence = self._compute_confidence(request)
        temporal_pattern = self._detect_temporal_pattern(request, ip)
        
        signature = {
            'phi': phi,
            'kappa': kappa,
            'regime': regime,
            'surprise': surprise,
            'confidence': confidence,
            'temporal_pattern': temporal_pattern,
            'timestamp': datetime.now().isoformat(),
            'ip_hash': hashlib.sha256(ip.encode()).hexdigest()[:16]
        }
        
        self._record_request(ip, signature)
        
        return signature
    
    def _compute_phi(self, request: Dict) -> float:
        """
        Integration metric - how interconnected are request parameters?
        
        Legitimate users: Variable Φ (0.3-0.8) as they explore
        Bots/scrapers: Low constant Φ (<0.2) - no semantic integration
        """
        path = request.get('path', '')
        params = request.get('params', {})
        headers = request.get('headers', {})
        body = request.get('body', {})
        
        feature_count = sum([
            bool(path),
            len(params) > 0,
            len(body) > 0,
            'referer' in headers,
            'user-agent' in headers
        ])
        
        if feature_count == 0:
            return 0.0
        
        integration_score = 0.0
        
        if path and params:
            path_tokens = set(path.lower().split('/'))
            param_keys = set(k.lower() for k in params.keys())
            overlap = len(path_tokens & param_keys)
            if path_tokens:
                integration_score += overlap / max(len(path_tokens), 1)
        
        if 'referer' in headers and path:
            referer = headers.get('referer', '')
            if referer:
                referer_path = referer.split('/')[-1].lower()
                if referer_path and referer_path in path.lower():
                    integration_score += 0.3
        
        if body and params:
            body_keys = set(k.lower() for k in body.keys()) if isinstance(body, dict) else set()
            param_keys = set(k.lower() for k in params.keys())
            if body_keys and param_keys:
                overlap = len(body_keys & param_keys)
                integration_score += overlap / max(len(body_keys), len(param_keys), 1)
        
        if headers.get('accept') and headers.get('content-type'):
            integration_score += 0.15
        
        if headers.get('cookie'):
            integration_score += 0.2
        
        phi = min(1.0, integration_score / 2.0)
        return round(phi, 4)
    
    def _compute_kappa(self, request: Dict) -> float:
        """
        Coupling strength - complexity of request structure.
        
        Legitimate users: Variable κ (20-80) depending on operation
        Bots: Uniform κ (~10-20) - simple, repetitive requests
        """
        params = request.get('params', {})
        body = request.get('body', {})
        path = request.get('path', '')
        headers = request.get('headers', {})
        
        param_depth = sum(
            self._get_nested_depth(v) for v in params.values()
        ) if params else 0
        
        body_depth = sum(
            self._get_nested_depth(v) for v in body.values()
        ) if isinstance(body, dict) else 0
        
        path_depth = len([p for p in path.split('/') if p])
        
        header_count = len(headers)
        
        kappa = (
            param_depth * 10 +
            body_depth * 15 +
            path_depth * 5 +
            header_count * 2
        )
        
        return min(100.0, kappa)
    
    def _classify_regime(self, phi: float, kappa: float) -> str:
        """
        Classify operational regime based on Φ and κ.
        
        - linear: Low Φ, low κ (simple requests)
        - geometric: Medium Φ, medium κ (normal operation)
        - hierarchical: High Φ, high κ (complex operations)
        - breakdown: Low Φ with high κ OR anomalous patterns (bot attack)
        """
        if phi > 0.6 and kappa > 50:
            return 'hierarchical'
        elif phi > 0.3 and 20 < kappa < 70:
            return 'geometric'
        elif phi < 0.3 and kappa < 30:
            return 'linear'
        elif phi < 0.2 and kappa > 40:
            return 'breakdown'
        else:
            return 'geometric'
    
    def _compute_surprise(self, request: Dict) -> float:
        """
        Surprise - how unexpected is this request?
        
        High surprise from suspicious patterns = threat signal
        """
        geo = request.get('geo', {})
        geo_location = geo.get('country', 'unknown') if isinstance(geo, dict) else 'unknown'
        user_agent = request.get('headers', {}).get('user-agent', '')
        
        threat_locations = {'RU', 'CN', 'KP', 'IR'}
        location_surprise = 0.7 if geo_location in threat_locations else 0.1
        
        bot_patterns = ['bot', 'scraper', 'crawler', 'spider', 'curl', 'wget', 'python-requests']
        agent_lower = user_agent.lower()
        agent_surprise = 0.8 if any(p in agent_lower for p in bot_patterns) else 0.1
        
        no_ua_surprise = 0.5 if not user_agent else 0.0
        
        return max(location_surprise, agent_surprise, no_ua_surprise)
    
    def _compute_confidence(self, request: Dict) -> float:
        """
        Confidence - internal coherence of request structure.
        
        Well-formed requests: High confidence (>0.7)
        Malformed/suspicious: Low confidence (<0.4)
        """
        has_path = bool(request.get('path'))
        has_method = bool(request.get('method'))
        has_headers = bool(request.get('headers'))
        
        headers = request.get('headers', {})
        has_ua = 'user-agent' in headers
        has_accept = 'accept' in headers
        has_host = 'host' in headers
        
        confidence = sum([
            has_path * 0.2,
            has_method * 0.2,
            has_headers * 0.1,
            has_ua * 0.2,
            has_accept * 0.15,
            has_host * 0.15
        ])
        
        return round(confidence, 4)
    
    def _detect_temporal_pattern(self, request: Dict, ip: str) -> str:
        """
        Detect temporal request pattern.
        
        - 'human': Variable intervals (300ms - 30s)
        - 'bot': Perfect intervals (<100ms variance)
        - 'burst': Rapid fire (>10 req/sec)
        """
        now = datetime.now().timestamp()
        
        if ip not in self.temporal_windows:
            self.temporal_windows[ip] = []
        
        window = self.temporal_windows[ip]
        window.append(now)
        
        cutoff = now - 60
        self.temporal_windows[ip] = [t for t in window if t > cutoff]
        window = self.temporal_windows[ip]
        
        if len(window) < 3:
            return 'human'
        
        if len(window) > 100:
            return 'burst'
        
        intervals = [window[i+1] - window[i] for i in range(len(window)-1)]
        
        if not intervals:
            return 'human'
        
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if len(window) > 10 and mean_interval < 0.1:
            return 'burst'
        
        if std_interval < 0.05 and len(window) > 5:
            return 'bot'
        
        return 'human'
    
    def _get_nested_depth(self, obj, depth=0) -> int:
        """Recursively compute nesting depth of object."""
        if isinstance(obj, dict):
            if not obj:
                return depth
            return max(self._get_nested_depth(v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return depth
            return max(self._get_nested_depth(item, depth + 1) for item in obj)
        else:
            return depth
    
    def _record_request(self, ip: str, signature: dict):
        """Record request signature for pattern analysis."""
        if ip not in self.request_history:
            self.request_history[ip] = []
        
        self.request_history[ip].append(signature)
        
        if len(self.request_history[ip]) > 100:
            self.request_history[ip] = self.request_history[ip][-100:]
    
    def get_ip_pattern_stats(self, ip: str) -> dict:
        """Get pattern statistics for an IP."""
        history = self.request_history.get(ip, [])
        
        if not history:
            return {'requests': 0}
        
        phis = [s['phi'] for s in history]
        kappas = [s['kappa'] for s in history]
        regimes = [s['regime'] for s in history]
        
        return {
            'requests': len(history),
            'avg_phi': np.mean(phis),
            'phi_variance': np.var(phis),
            'avg_kappa': np.mean(kappas),
            'kappa_variance': np.var(kappas),
            'regime_distribution': {r: regimes.count(r) for r in set(regimes)},
            'breakdown_ratio': regimes.count('breakdown') / len(regimes)
        }
