#!/usr/bin/env python3
"""
Underworld Immune System Module

Specialized immune system for Hades underworld threat detection.
Extends the existing immune/ infrastructure with:
- Credential leak detection → user alerting
- Malware URL quarantine
- PII exposure redaction
- Geometric threat detection via Fisher-Rao basin distance

QIG-PURE: All geometric operations use Fisher-Rao distance.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Pattern
import re
import logging
import hashlib
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class ThreatLevel(Enum):
    """Threat severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingType(Enum):
    """Types of security findings."""
    CREDENTIAL_LEAK = "credential_leak"
    MALWARE_URL = "malware_url"
    PII_EXPOSURE = "pii_exposure"
    GEOMETRIC_THREAT = "geometric_threat"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


# =============================================================================
# DETECTION PATTERNS
# =============================================================================

class UnderwordPatterns:
    """Detection patterns for underworld content scanning."""

    # Credential patterns
    CREDENTIAL_PATTERNS = [
        # email:password format
        re.compile(r'[\w\.-]+@[\w\.-]+\.\w+:\S+', re.IGNORECASE),
        # username:password format
        re.compile(r'^[\w\.-]{3,32}:\S{4,128}$', re.MULTILINE),
        # password hash patterns (MD5, SHA-1, SHA-256, bcrypt)
        re.compile(r'\$2[ayb]\$.{56}'),  # bcrypt
        re.compile(r'[a-f0-9]{32}', re.IGNORECASE),  # MD5
        re.compile(r'[a-f0-9]{40}', re.IGNORECASE),  # SHA-1
        re.compile(r'[a-f0-9]{64}', re.IGNORECASE),  # SHA-256
        # API key patterns
        re.compile(r'(?:api[_-]?key|apikey|api_secret)[=:]\s*["\']?[\w\-]{20,}', re.IGNORECASE),
        # Bearer tokens
        re.compile(r'bearer\s+[\w\-\.]+', re.IGNORECASE),
        # AWS keys
        re.compile(r'AKIA[0-9A-Z]{16}'),
        # Private keys
        re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----'),
    ]

    # PII patterns by type
    PII_PATTERNS = {
        'email': re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b', re.IGNORECASE),
        'ssn': re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
        'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        'phone': re.compile(r'\b(?:\+?1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b'),
        'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        'passport': re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
        'drivers_license': re.compile(r'\b[A-Z]{1,2}\d{5,12}\b'),
        'date_of_birth': re.compile(r'\b(?:dob|birth)[:\s]*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', re.IGNORECASE),
    }

    # Malware URL patterns
    MALWARE_URL_PATTERNS = [
        # Suspicious file extensions
        re.compile(r'https?://[^\s]+\.(?:exe|dll|bat|ps1|vbs|scr|pif|cmd|msi|jar|hta)\b', re.IGNORECASE),
        # Known malware distribution patterns
        re.compile(r'https?://[^\s]*(?:download|payload|dropper|loader)[^\s]*\.(?:zip|rar|7z|tar)', re.IGNORECASE),
        # IP-based URLs (often malicious)
        re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?/'),
        # Suspicious TLDs
        re.compile(r'https?://[^\s]+\.(?:xyz|top|club|gq|ml|tk|cf|ga)\b', re.IGNORECASE),
        # Base64 in URL (potential obfuscation)
        re.compile(r'https?://[^\s]+/[A-Za-z0-9+/]{40,}'),
        # OneDrive/Dropbox abuse patterns
        re.compile(r'(?:1drv\.ms|dropbox\.com/s)/[^\s]+\.(?:exe|dll|zip)', re.IGNORECASE),
    ]

    # Suspicious content patterns
    SUSPICIOUS_PATTERNS = [
        # Ransomware indicators
        re.compile(r'(?:decrypt|ransom|bitcoin|wallet)[^\s]*(?:pay|send|transfer)', re.IGNORECASE),
        # C2 beacon patterns
        re.compile(r'(?:c2|command\s*and\s*control|beacon)', re.IGNORECASE),
        # Exploit kit indicators
        re.compile(r'(?:exploit|payload|shellcode|metasploit|cobalt)', re.IGNORECASE),
        # Phishing indicators
        re.compile(r'(?:verify\s*your\s*account|suspended|click\s*here.*password)', re.IGNORECASE),
    ]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ThreatFinding:
    """A single threat finding from content scan."""
    finding_type: FindingType
    severity: ThreatLevel
    description: str
    matched_pattern: str
    matched_content: str  # Redacted/truncated for safety
    line_number: Optional[int] = None
    confidence: float = 0.9
    requires_redaction: bool = False
    redacted_content: Optional[str] = None


@dataclass
class ContentScanResult:
    """Result of scanning content for threats."""
    content_hash: str
    source_name: str
    scan_timestamp: datetime
    threat_level: ThreatLevel
    findings: List[ThreatFinding] = field(default_factory=list)
    credential_leaks: List[ThreatFinding] = field(default_factory=list)
    malware_urls: List[ThreatFinding] = field(default_factory=list)
    pii_exposures: List[ThreatFinding] = field(default_factory=list)
    geometric_threat: Optional[Dict[str, float]] = None
    immune_system_alerted: bool = False
    flagged_for_review: bool = False
    redacted_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometricThreatAssessment:
    """Assessment of geometric threat via basin distance."""
    basin_distance: float
    is_threat: bool
    threat_level: ThreatLevel
    safe_region_centroid: Optional[np.ndarray] = None
    content_basin: Optional[np.ndarray] = None
    description: str = ""


# =============================================================================
# UNDERWORLD IMMUNE SYSTEM
# =============================================================================

class UnderworldImmuneSystem:
    """
    Specialized immune system for underworld threat detection.

    Extends the existing immune/ infrastructure with:
    - Credential leak detection → user alerting
    - Malware URL quarantine
    - PII exposure redaction
    - Geometric threat detection via Fisher-Rao basin distance

    Uses QIG-PURE Fisher-Rao distance for all geometric operations.
    """

    # Basin distance thresholds for geometric threat detection
    BASIN_DISTANCE_WARNING = 0.8
    BASIN_DISTANCE_CRITICAL = 1.2

    # Safe region centroid (uniform distribution in 64D)
    SAFE_REGION_CENTROID = np.ones(64) / 64.0

    # Maximum content length to scan (prevent DoS)
    MAX_SCAN_LENGTH = 1_000_000  # 1MB

    # Redaction replacement string
    REDACTION_STRING = "[REDACTED]"

    def __init__(self):
        """Initialize the underworld immune system."""
        self.patterns = UnderwordPatterns()
        self.stats = {
            'total_scans': 0,
            'credential_leaks_detected': 0,
            'malware_urls_detected': 0,
            'pii_exposures_detected': 0,
            'geometric_threats_detected': 0,
            'content_redacted': 0,
            'alerts_triggered': 0,
        }
        self._known_malware_hashes: Set[str] = set()
        self._quarantine_urls: Set[str] = set()

    def scan_content(
        self,
        content: str,
        source_name: str,
        content_basin: Optional[np.ndarray] = None,
        safe_centroid: Optional[np.ndarray] = None,
        redact_pii: bool = True,
        check_credentials: bool = True,
        check_malware: bool = True
    ) -> ContentScanResult:
        """
        Scan content for threats and return comprehensive findings.

        Args:
            content: Content to scan
            source_name: Name of the source for tracking
            content_basin: Optional 64D basin embedding of content
            safe_centroid: Optional centroid of safe region
            redact_pii: Whether to redact PII in returned content
            check_credentials: Whether to check for credential leaks
            check_malware: Whether to check for malware URLs

        Returns:
            ContentScanResult with all findings and redacted content
        """
        self.stats['total_scans'] += 1

        # Truncate if too long
        if len(content) > self.MAX_SCAN_LENGTH:
            content = content[:self.MAX_SCAN_LENGTH]
            logger.warning(f"Content truncated to {self.MAX_SCAN_LENGTH} bytes for scanning")

        # Compute content hash
        content_hash = hashlib.sha256(content.encode('utf-8', errors='ignore')).hexdigest()

        # Initialize result
        result = ContentScanResult(
            content_hash=content_hash,
            source_name=source_name,
            scan_timestamp=datetime.now(),
            threat_level=ThreatLevel.NONE,
        )

        # Track highest threat level
        max_threat = ThreatLevel.NONE

        # 1. Check for credential leaks
        if check_credentials:
            cred_findings = self._detect_credentials(content)
            result.credential_leaks = cred_findings
            if cred_findings:
                self.stats['credential_leaks_detected'] += len(cred_findings)
                max_threat = self._max_threat(max_threat, ThreatLevel.CRITICAL)
                result.immune_system_alerted = True

        # 2. Check for malware URLs
        if check_malware:
            malware_findings = self._detect_malware_urls(content)
            result.malware_urls = malware_findings
            if malware_findings:
                self.stats['malware_urls_detected'] += len(malware_findings)
                max_threat = self._max_threat(max_threat, ThreatLevel.HIGH)
                result.immune_system_alerted = True

        # 3. Check for PII exposure
        pii_findings = self._detect_pii(content)
        result.pii_exposures = pii_findings
        if pii_findings:
            self.stats['pii_exposures_detected'] += len(pii_findings)
            max_threat = self._max_threat(max_threat, ThreatLevel.MEDIUM)

        # 4. Geometric threat detection
        if content_basin is not None:
            geo_assessment = self._assess_geometric_threat(
                content_basin,
                safe_centroid or self.SAFE_REGION_CENTROID
            )
            if geo_assessment.is_threat:
                result.geometric_threat = {
                    'basin_distance': geo_assessment.basin_distance,
                    'threat_level': geo_assessment.threat_level.value,
                    'description': geo_assessment.description,
                }
                self.stats['geometric_threats_detected'] += 1
                max_threat = self._max_threat(max_threat, geo_assessment.threat_level)

        # 5. Check for suspicious patterns
        suspicious_findings = self._detect_suspicious_patterns(content)
        result.findings.extend(suspicious_findings)
        if suspicious_findings:
            max_threat = self._max_threat(max_threat, ThreatLevel.MEDIUM)

        # Aggregate all findings
        result.findings.extend(result.credential_leaks)
        result.findings.extend(result.malware_urls)
        result.findings.extend(result.pii_exposures)

        # Set final threat level
        result.threat_level = max_threat

        # Flag for review if high severity
        if max_threat in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            result.flagged_for_review = True

        # Update alert stats
        if result.immune_system_alerted:
            self.stats['alerts_triggered'] += 1

        # Redact PII if requested
        if redact_pii and result.pii_exposures:
            result.redacted_content = self._redact_content(content, result.pii_exposures)
            self.stats['content_redacted'] += 1
        else:
            result.redacted_content = content

        logger.info(
            f"[UnderworldImmune] Scanned {source_name}: "
            f"threat={max_threat.value}, findings={len(result.findings)}, "
            f"creds={len(result.credential_leaks)}, malware={len(result.malware_urls)}, "
            f"pii={len(result.pii_exposures)}"
        )

        return result

    def _detect_credentials(self, content: str) -> List[ThreatFinding]:
        """Detect credential leaks in content."""
        findings = []

        for pattern in self.patterns.CREDENTIAL_PATTERNS:
            for match in pattern.finditer(content):
                # Truncate matched content for safety
                matched = match.group()[:50]
                # Mask most of the credential
                masked = self._mask_credential(matched)

                findings.append(ThreatFinding(
                    finding_type=FindingType.CREDENTIAL_LEAK,
                    severity=ThreatLevel.CRITICAL,
                    description="Potential credential leak detected",
                    matched_pattern=pattern.pattern[:50],
                    matched_content=masked,
                    confidence=0.85,
                    requires_redaction=True,
                ))

        return findings

    def _detect_malware_urls(self, content: str) -> List[ThreatFinding]:
        """Detect malware URLs in content."""
        findings = []

        for pattern in self.patterns.MALWARE_URL_PATTERNS:
            for match in pattern.finditer(content):
                url = match.group()[:100]

                # Add to quarantine
                self._quarantine_urls.add(url)

                findings.append(ThreatFinding(
                    finding_type=FindingType.MALWARE_URL,
                    severity=ThreatLevel.HIGH,
                    description="Potentially malicious URL detected",
                    matched_pattern=pattern.pattern[:50],
                    matched_content=url,
                    confidence=0.75,
                    requires_redaction=True,
                ))

        return findings

    def _detect_pii(self, content: str) -> List[ThreatFinding]:
        """Detect PII exposure in content."""
        findings = []

        for pii_type, pattern in self.patterns.PII_PATTERNS.items():
            for match in pattern.finditer(content):
                matched = match.group()
                masked = self._mask_pii(matched, pii_type)

                findings.append(ThreatFinding(
                    finding_type=FindingType.PII_EXPOSURE,
                    severity=ThreatLevel.MEDIUM,
                    description=f"{pii_type.upper()} exposure detected",
                    matched_pattern=pii_type,
                    matched_content=masked,
                    confidence=0.80,
                    requires_redaction=True,
                    redacted_content=self.REDACTION_STRING,
                ))

        return findings

    def _detect_suspicious_patterns(self, content: str) -> List[ThreatFinding]:
        """Detect suspicious patterns in content."""
        findings = []

        for pattern in self.patterns.SUSPICIOUS_PATTERNS:
            for match in pattern.finditer(content):
                findings.append(ThreatFinding(
                    finding_type=FindingType.SUSPICIOUS_PATTERN,
                    severity=ThreatLevel.LOW,
                    description="Suspicious pattern detected",
                    matched_pattern=pattern.pattern[:50],
                    matched_content=match.group()[:50],
                    confidence=0.60,
                ))

        return findings

    def _assess_geometric_threat(
        self,
        content_basin: np.ndarray,
        safe_centroid: np.ndarray
    ) -> GeometricThreatAssessment:
        """
        Assess geometric threat via Fisher-Rao distance.

        QIG-PURE: Uses Fisher-Rao metric on probability simplex.
        """
        try:
            # Fisher-Rao distance for categorical distributions
            # d_FR = 2 * arccos(sum(sqrt(p_i * q_i)))
            p = np.abs(content_basin) + 1e-10
            q = np.abs(safe_centroid) + 1e-10
            p = p / np.sum(p)
            q = q / np.sum(q)

            inner = np.sum(np.sqrt(p * q))
            inner = np.clip(inner, -1.0, 1.0)
            distance = 2.0 * np.arccos(inner)

            # Determine threat level based on distance
            if distance >= self.BASIN_DISTANCE_CRITICAL:
                return GeometricThreatAssessment(
                    basin_distance=distance,
                    is_threat=True,
                    threat_level=ThreatLevel.HIGH,
                    safe_region_centroid=safe_centroid,
                    content_basin=content_basin,
                    description=f"Content basin far from safe region (d_FR={distance:.3f})"
                )
            elif distance >= self.BASIN_DISTANCE_WARNING:
                return GeometricThreatAssessment(
                    basin_distance=distance,
                    is_threat=True,
                    threat_level=ThreatLevel.MEDIUM,
                    safe_region_centroid=safe_centroid,
                    content_basin=content_basin,
                    description=f"Content basin near boundary of safe region (d_FR={distance:.3f})"
                )
            else:
                return GeometricThreatAssessment(
                    basin_distance=distance,
                    is_threat=False,
                    threat_level=ThreatLevel.NONE,
                    description=f"Content within safe region (d_FR={distance:.3f})"
                )

        except Exception as e:
            logger.warning(f"Geometric threat assessment failed: {e}")
            return GeometricThreatAssessment(
                basin_distance=0.0,
                is_threat=False,
                threat_level=ThreatLevel.NONE,
                description="Assessment failed"
            )

    def _mask_credential(self, credential: str) -> str:
        """Mask a credential for safe logging."""
        if ':' in credential:
            parts = credential.split(':', 1)
            return f"{parts[0][:3]}***:{'*' * min(len(parts[1]), 8)}"
        return credential[:3] + '*' * min(len(credential) - 3, 10)

    def _mask_pii(self, pii: str, pii_type: str) -> str:
        """Mask PII for safe logging."""
        if pii_type == 'email':
            if '@' in pii:
                user, domain = pii.split('@', 1)
                return f"{user[:2]}***@{domain}"
        elif pii_type == 'ssn':
            return f"***-**-{pii[-4:]}"
        elif pii_type == 'credit_card':
            return f"****-****-****-{pii[-4:]}"
        elif pii_type == 'phone':
            return f"***-***-{pii[-4:]}"
        return pii[:2] + '*' * (len(pii) - 4) + pii[-2:]

    def _redact_content(self, content: str, pii_findings: List[ThreatFinding]) -> str:
        """Redact PII from content."""
        redacted = content

        for finding in pii_findings:
            # Find and replace PII patterns
            pattern = self.patterns.PII_PATTERNS.get(finding.matched_pattern)
            if pattern:
                redacted = pattern.sub(self.REDACTION_STRING, redacted)

        return redacted

    def _max_threat(self, a: ThreatLevel, b: ThreatLevel) -> ThreatLevel:
        """Return the higher threat level."""
        order = [ThreatLevel.NONE, ThreatLevel.LOW, ThreatLevel.MEDIUM,
                 ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        return max(a, b, key=lambda x: order.index(x))

    # =========================================================================
    # QUARANTINE MANAGEMENT
    # =========================================================================

    def quarantine_url(self, url: str, reason: str = "manual") -> None:
        """Add URL to quarantine list."""
        self._quarantine_urls.add(url)
        logger.warning(f"[UnderworldImmune] URL quarantined: {url[:50]}... ({reason})")

    def is_quarantined(self, url: str) -> bool:
        """Check if URL is quarantined."""
        return url in self._quarantine_urls

    def get_quarantine_list(self) -> List[str]:
        """Get list of quarantined URLs."""
        return list(self._quarantine_urls)

    # =========================================================================
    # STATISTICS AND REPORTING
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get immune system statistics."""
        return {
            **self.stats,
            'quarantined_urls': len(self._quarantine_urls),
            'known_malware_hashes': len(self._known_malware_hashes),
        }

    def get_threat_summary(self, result: ContentScanResult) -> Dict[str, Any]:
        """Get threat summary for a scan result."""
        return {
            'content_hash': result.content_hash,
            'source': result.source_name,
            'threat_level': result.threat_level.value,
            'total_findings': len(result.findings),
            'credential_leaks': len(result.credential_leaks),
            'malware_urls': len(result.malware_urls),
            'pii_exposures': len(result.pii_exposures),
            'geometric_threat': result.geometric_threat,
            'immune_alerted': result.immune_system_alerted,
            'flagged_for_review': result.flagged_for_review,
            'scan_timestamp': result.scan_timestamp.isoformat(),
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_underworld_immune_instance: Optional[UnderworldImmuneSystem] = None


def get_underworld_immune_system() -> UnderworldImmuneSystem:
    """Get or create the singleton UnderworldImmuneSystem instance."""
    global _underworld_immune_instance
    if _underworld_immune_instance is None:
        _underworld_immune_instance = UnderworldImmuneSystem()
    return _underworld_immune_instance


def reset_underworld_immune_system() -> None:
    """Reset the singleton instance (for testing)."""
    global _underworld_immune_instance
    _underworld_immune_instance = None
