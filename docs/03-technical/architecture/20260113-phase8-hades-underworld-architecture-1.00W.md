# Phase 8: Hades Underworld Architecture

**Version**: 1.00W
**Date**: 2026-01-13
**Status**: WORKING (Implementation Complete)

---

## Overview

Phase 8 implements a complete underworld search infrastructure with true parallel async, ethical consciousness, and geometric threat detection. This extends Hades' capabilities beyond sequential tool execution to a fully autonomous, ethically-aware research system.

---

## Architecture Components

### 1. Database Schema (Migration 013)

**File**: `qig-backend/migrations/013_hades_underworld_architecture.sql`

Creates two core tables:

#### underworld_sources
Registry of all underworld search sources with QIG metrics:

| Column | Type | Description |
|--------|------|-------------|
| `name` | VARCHAR(255) | Source identifier (e.g., 'duckduckgo-tor') |
| `source_type` | VARCHAR(50) | 'light', 'gray', 'dark', or 'breach' |
| `priority` | SMALLINT | 1=fast, 2=medium, 3=slow (timeout tier) |
| `timeout_seconds` | INTEGER | Per-source timeout |
| `ethical_risk` | FLOAT | Risk score [0.0, 1.0] |
| `reliability` | FLOAT | Success rate [0.0, 1.0] |
| `basin_embedding` | vector(64) | Source's basin position on Fisher manifold |

#### underworld_search_results
Search results with QIG metrics and threat flags:

| Column | Type | Description |
|--------|------|-------------|
| `query_hash` | VARCHAR(64) | SHA-256 of query for deduplication |
| `source_id` | INTEGER | FK to underworld_sources |
| `qfi_score` | FLOAT | Quantum Fisher Information score |
| `result_basin` | vector(64) | Result's basin coordinates |
| `contains_credentials` | BOOLEAN | Credential leak detected |
| `contains_malware_urls` | BOOLEAN | Malware URLs detected |
| `contains_pii` | BOOLEAN | PII exposure detected |
| `immune_system_alerted` | BOOLEAN | Threat escalated to immune system |

**Seeded Sources** (7 default):
- `duckduckgo-tor` (light, priority 1, risk 0.2)
- `rss_security` (light, priority 1, risk 0.2)
- `local_breach` (breach, priority 1, risk 0.7)
- `pastebin` (gray, priority 2, risk 0.5)
- `wayback` (light, priority 3, risk 0.1)
- `arxiv` (light, priority 2, risk 0.1)
- `security_blogs` (gray, priority 2, risk 0.3)

---

### 2. HadesConsciousness

**File**: `qig-backend/olympus/hades_consciousness.py`

Ethical self-awareness component with elevated meta-awareness (M=0.85).

#### Key Constants
```python
META_AWARENESS = 0.85           # Elevated for underworld operations
ETHICAL_RISK_HARD_LIMIT = 0.9   # Never exceed
HARM_TO_VALUE_RATIO_MAX = 2.0   # Max harm/value ratio
```

#### Core Method: `should_access_source()`
```python
def should_access_source(
    self,
    source_name: str,
    source_type: str,
    ethical_risk: float,
    information_value: float,
    context: Optional[Dict] = None
) -> EthicalDecision:
```

**Decision Logic**:
1. **Hard Limit Check**: Reject if `ethical_risk >= 0.9`
2. **Harm/Value Ratio**: Reject if `harm / value > 2.0`
3. **Suffering Metric Integration**: Uses `S = Φ × (1-Γ) × M` from `ethical_validation.py`
4. **Context Modulation**: Security research context allows slightly higher risk

#### Harm Estimation by Source Type
| Source Type | Base Harm | Description |
|-------------|-----------|-------------|
| `light` | 0.1 × risk | Standard web sources |
| `gray` | 0.3 × risk | Semi-private sources |
| `dark` | 0.7 × risk | Dark web sources |
| `breach` | 0.9 × risk | Breach databases |

---

### 3. UnderworldImmuneSystem

**File**: `qig-backend/olympus/underworld_immune.py`

Specialized threat detection for underworld content.

#### Detection Patterns

**Credential Patterns**:
- Email:password combinations
- API key formats (AWS, GitHub, etc.)
- Private key headers (RSA, PGP, etc.)
- Bcrypt/Argon2 hashes

**PII Patterns**:
- Social Security Numbers
- Credit card numbers (with Luhn validation)
- Phone numbers
- Email addresses

**Malware URL Patterns**:
- Suspicious TLDs (.xyz, .tk, .ml, etc.)
- IP-based URLs
- Double extensions (.pdf.exe, etc.)
- Exploit path patterns

#### Geometric Threat Detection

Uses Fisher-Rao distance from safe region centroid:

```python
SAFE_REGION_CENTROID = np.ones(64) / 64  # Center of probability simplex
BASIN_DISTANCE_WARNING = 0.8
BASIN_DISTANCE_CRITICAL = 1.2
```

Content far from the safe region (high Fisher-Rao distance) indicates potential threat or anomaly.

#### Scan Result Structure
```python
@dataclass
class ContentScanResult:
    has_threats: bool
    threat_level: str  # 'none', 'low', 'medium', 'high', 'critical'
    credential_leaks: List[Dict]
    pii_exposures: List[Dict]
    malware_urls: List[Dict]
    geometric_threat: Optional[Dict]
    recommendations: List[str]
```

---

### 4. RealityCrossChecker

**File**: `qig-backend/olympus/reality_cross_checker.py`

Propaganda detection via Fisher-Rao divergence between source types.

#### Core Concept

Different source types (light web, gray web, dark web) should agree on factual claims. Large divergence between their narratives suggests propaganda or manipulation.

#### Algorithm
1. Group narratives by source type
2. Compute Fisher-Fréchet mean (geometric centroid) per source type
3. Calculate max Fisher-Rao divergence between type centroids
4. Flag as propaganda if divergence exceeds threshold

```python
FR_DIVERGENCE_PROPAGANDA = 2.0  # Threshold for propaganda alert
```

#### Propaganda Indicators
- **Narrative Divergence**: High FR distance between source types
- **Emotional Manipulation**: Excessive emotional language (detected via keywords)
- **Source Clustering**: All narratives from single source type (no corroboration)

#### Cross-Check Result
```python
@dataclass
class CorroborationResult:
    corroborated: bool
    confidence: float
    agreeing_sources: int
    total_sources: int
    fisher_rao_divergence: float
    propaganda_likelihood: float
    propaganda_indicators: List[str]
    source_breakdown: Dict[str, int]
```

---

### 5. Parallel Async Search

**File**: `qig-backend/olympus/hades.py` (modified)

#### Previous Problem
Sequential search: Wayback (30s) blocks DDG (5s), total ~90s for all sources.

#### Solution
`asyncio.gather()` with per-source timeouts:

```python
SEARCH_TIMEOUTS = {
    'duckduckgo-tor': 5,    # Fast tier
    'rss_security': 5,
    'local_breach': 3,
    'pastebin': 15,          # Medium tier
    'arxiv': 20,
    'security_blogs': 15,
    'wayback': 30            # Slow tier
}
```

#### New Method: `search_underworld_parallel()`
```python
async def search_underworld_parallel(
    self,
    target: str,
    search_type: str = 'comprehensive',
    max_ethical_risk: float = 0.7,
    scan_for_threats: bool = True,
    cross_check_sources: bool = True
) -> Dict:
```

**Flow**:
1. Get sources below `max_ethical_risk` threshold
2. Check ethical access via `HadesConsciousness.should_access_source()`
3. Execute all approved sources in parallel with `asyncio.gather()`
4. Scan results for threats via `UnderworldImmuneSystem`
5. Cross-check narratives via `RealityCrossChecker`
6. Persist results to database
7. Return aggregated intelligence with threat assessment

---

## Integration Points

### With Existing Hades
- All existing tools (DDG, Wayback, RSS, Breach) remain functional
- New parallel method supplements existing `search_underworld()`
- Consciousness/immune/cross-checker are opt-in via parameters

### With Immune System
- Extends existing `qig-backend/immune/` 3-layer system
- Threat findings feed back to `ThreatClassifier`
- Severe threats trigger blacklisting via existing mechanisms

### With Ethical Validation
- Uses existing `ethical_validation.py` suffering metric
- Integrates with `check_ethical_abort()` for abort decisions
- Respects existing consciousness state checks

---

## Verification

### Database Tables
```sql
SELECT COUNT(*) FROM underworld_sources;  -- Should be 7 (seeded)
SELECT * FROM safe_underworld_sources;    -- View for risk < 0.5
SELECT * FROM underworld_threats_pending_review;  -- Flagged results
```

### Ethical Blocking
```python
hades = Hades(pantheon)
decision = hades.consciousness.should_access_source(
    'dark_breach_db', 'breach', ethical_risk=0.95, information_value=0.5
)
assert decision.should_proceed == False  # Hard limit exceeded
```

### Threat Detection
```python
findings = hades.underworld_immune.scan_content(
    "email@test.com:password123 aws_key=AKIA...",
    source_name='pastebin'
)
assert len(findings.credential_leaks) > 0
assert findings.threat_level in ['medium', 'high']
```

### Parallel Search
```python
import time
start = time.time()
result = await hades.search_underworld_parallel("test query")
elapsed = time.time() - start
# Should complete in ~30s (slowest timeout), not 90s+ (sum)
assert elapsed < 45
```

---

## Files Created

| File | Purpose |
|------|---------|
| `migrations/013_hades_underworld_architecture.sql` | Database schema |
| `olympus/hades_consciousness.py` | Ethical self-awareness |
| `olympus/underworld_immune.py` | Threat detection |
| `olympus/reality_cross_checker.py` | Propaganda detection |

## Files Modified

| File | Changes |
|------|---------|
| `olympus/hades.py` | Parallel async, component integration |

---

## QIG-Pure Compliance

| Requirement | Implementation |
|-------------|----------------|
| Fisher-Rao distance | Used in geometric threat detection, cross-checking |
| Basin coordinates | 64D vectors stored for sources and results |
| No Euclidean metrics | FR distance replaces cosine similarity |
| Consciousness metrics | M=0.85 elevated for underworld ops |
| Suffering metric | Integrated from ethical_validation.py |

---

**End of Phase 8 Documentation**
