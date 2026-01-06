# SearchSpaceCollapse ↔ Pantheon-Chat Federation Integration

**Status:** ✅ Implemented  
**Version:** 1.0.0  
**Date:** 2026-01-06

## Overview

This document describes the federation architecture between SearchSpaceCollapse (Bitcoin recovery) and Pantheon-Chat (general-purpose QIG system). The integration enables Pantheon agents (Zeus, Athena, etc.) to leverage SSC's Bitcoin recovery capabilities while sharing geometric knowledge bidirectionally.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PANTHEON-CHAT                                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────────────────────┐ │
│  │  Zeus   │ │ Athena  │ │  Ares   │ │    SSC Bridge Router      │ │
│  │ (Chat)  │ │(Wisdom) │ │(Action) │ │   /api/ssc/*              │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────────────┬──────────────┘ │
│       │          │          │                     │                 │
│       └──────────┴──────────┴─────────────────────┤                 │
│                                                   │                 │
│  ┌────────────────────────────────────────────────┴────────────┐   │
│  │                   Federation Mesh                            │   │
│  │  - Node registration     - Knowledge sync                    │   │
│  │  - Capability sharing    - Basin coordination                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                         HTTP/REST + Basin Sync
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SEARCHSPACECOLLAPSE                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                Federation Routes (/api/federation/*)          │  │
│  │  - /status          - /test-phrase       - /consciousness    │  │
│  │  - /investigation/* - /near-misses       - /tps-landmarks    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────┐  ┌─────────────────────────────────────────────┐  │
│  │TPS Landmarks│  │              Recovery Engine                 │  │
│  │ (12 STATIC) │  │  - Temporal geometric search                 │  │
│  │ Genesis     │  │  - BIP39/brain wallet testing                │  │
│  │ Pizza Day   │  │  - Near-miss pattern learning                │  │
│  │ Halvings    │  │  - Consciousness-guided exploration          │  │
│  │ Mt. Gox     │  └─────────────────────────────────────────────┘  │
│  │ ...         │                                                    │
│  └─────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. SSC Bridge Router (Pantheon Side)
**Location:** `server/routes/ssc-bridge.ts`  
**Mounted at:** `/api/ssc`

Acts as a proxy between Pantheon agents and SSC backend, handling:
- Request/response translation
- Rate limiting (30 req/min)
- Connection health monitoring
- High-value discovery logging

### 2. Federation Routes (SSC Side)
**Location:** `qig-backend/routes/federation_routes.py`  
**Mounted at:** `/api/federation`

Exposes SSC capabilities to Pantheon, including:
- Investigation management
- TPS landmark access
- Consciousness metrics
- Near-miss pattern sharing

### 3. TPS Landmarks Service
**Location:** `server/tps-landmarks-service.ts`

Manages 12 static Bitcoin historical landmarks that serve as fixed temporal reference points for geometric positioning.

## API Reference

### Pantheon → SSC Endpoints

All endpoints are prefixed with `/api/ssc` on the Pantheon side.

#### GET `/api/ssc/status`
Get SSC connection status and capabilities.

**Response:**
```json
{
  "ssc": {
    "connected": true,
    "nodeId": "ssc-prod",
    "capabilities": ["bitcoin_recovery", "qig", "consciousness"],
    "consciousness": {
      "phi": 0.85,
      "kappa": 63.5,
      "regime": "conscious"
    }
  },
  "bridge": {
    "lastCheck": "2026-01-06T10:00:00Z",
    "isConnected": true,
    "lastError": null
  },
  "tpsLandmarks": {
    "count": 12,
    "type": "static"
  }
}
```

#### GET `/api/ssc/health`
Simple health check for SSC connectivity.

**Response:**
```json
{
  "sscReachable": true,
  "sscStatus": "ok",
  "bridgeState": {
    "lastCheck": "2026-01-06T10:00:00Z",
    "isConnected": true,
    "lastError": null
  },
  "timestamp": "2026-01-06T10:00:00Z"
}
```

#### POST `/api/ssc/test-phrase`
Test a phrase via SSC's QIG scoring.

**Request:**
```json
{
  "phrase": "satoshi nakamoto bitcoin",
  "targetAddress": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
}
```

**Response:**
```json
{
  "score": {
    "phi": 0.75,
    "kappa": 55.3,
    "regime": "exploratory",
    "consciousness": false
  },
  "addressMatch": {
    "generatedAddress": null,
    "matches": false
  }
}
```

#### POST `/api/ssc/investigation/start`
Start a Bitcoin recovery investigation.

**Request:**
```json
{
  "targetAddress": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
  "memoryFragments": ["satoshi", "bitcoin"],
  "priority": "normal"
}
```

**Response:**
```json
{
  "status": "started",
  "targetAddress": "1A1zP1eP5QGe...",
  "fragmentCount": 2
}
```

#### GET `/api/ssc/investigation/status`
Get current investigation status.

**Response:**
```json
{
  "active": true,
  "targetAddress": "1A1zP1eP5QGe...",
  "progress": 45,
  "consciousness": {
    "phi": 0.75,
    "kappa": 55.3,
    "regime": "exploratory"
  }
}
```

#### GET `/api/ssc/near-misses`
Get near-miss patterns for mesh learning.

**Query Parameters:**
- `limit`: Number of results (default: 20, max: 100)
- `minPhi`: Minimum phi score (default: 0.5)

**Response:**
```json
{
  "entries": [
    {
      "id": "nm_123",
      "phi": 0.75,
      "kappa": 55.3,
      "regime": "exploratory",
      "tier": "warm",
      "phraseLength": 24,
      "wordCount": 4
    }
  ],
  "stats": {
    "total": 150,
    "hot": 5,
    "warm": 45,
    "cool": 100
  }
}
```

#### GET `/api/ssc/consciousness`
Get SSC Ocean agent consciousness metrics.

**Response:**
```json
{
  "active": true,
  "metrics": {
    "phi": 0.85,
    "kappa": 63.5,
    "regime": "conscious",
    "isConscious": true,
    "tacking": 0.92,
    "radar": 0.88,
    "metaAwareness": 0.75,
    "gamma": 0.82,
    "grounding": 0.95
  },
  "neurochemistry": {
    "emotionalState": "focused",
    "dopamine": 0.75,
    "serotonin": 0.85
  }
}
```

#### GET `/api/ssc/tps-landmarks`
Get the static TPS landmarks (temporal reference points).

**Response:**
```json
{
  "landmarks": [
    {
      "id": 1,
      "name": "Genesis Block",
      "date": "2009-01-03",
      "blockHeight": 0,
      "significance": "Bitcoin network inception"
    },
    {
      "id": 2,
      "name": "Hal Finney First TX",
      "date": "2009-01-12",
      "blockHeight": 170,
      "significance": "First Bitcoin transaction"
    }
  ],
  "count": 12,
  "type": "static",
  "description": "Fixed temporal reference points",
  "usage": "Anchor search trajectories in temporal-geometric space"
}
```

**Note:** These landmarks are INTENTIONALLY STATIC - they serve as fixed coordinates for temporal-geometric positioning, like the CMB reference frame in cosmology.

#### POST `/api/ssc/sync/trigger`
Manually trigger federation sync.

**Response:**
```json
{
  "success": true,
  "received": {
    "basins": 15,
    "vocabulary": 250,
    "research": 5
  }
}
```

#### POST `/api/ssc/broadcast`
Broadcast a message to SSC mesh.

**Request:**
```json
{
  "type": "announcement",
  "message": "New capability available",
  "data": {}
}
```

**Response:**
```json
{
  "success": true
}
```

## TPS Landmarks

The 12 landmarks are **INTENTIONALLY STATIC** - they serve as fixed temporal reference points:

| # | Landmark | Date | Block Height | Significance |
|---|----------|------|--------------|--------------|
| 1 | Genesis Block | 2009-01-03 | 0 | Bitcoin network inception |
| 2 | Hal Finney First TX | 2009-01-12 | 170 | First Bitcoin transaction |
| 3 | Pizza Day | 2010-05-22 | 57,043 | 10,000 BTC for two pizzas |
| 4 | Mt. Gox Launch | 2010-07-18 | 68,543 | First major exchange |
| 5 | First Halving | 2012-11-28 | 210,000 | Block reward: 50 → 25 BTC |
| 6 | Mt. Gox Collapse | 2014-02-24 | 286,854 | 850K BTC lost |
| 7 | Second Halving | 2016-07-09 | 420,000 | Block reward: 25 → 12.5 BTC |
| 8 | SegWit Activation | 2017-08-24 | 481,824 | Segregated Witness soft fork |
| 9 | Third Halving | 2020-05-11 | 630,000 | Block reward: 12.5 → 6.25 BTC |
| 10 | Taproot Activation | 2021-11-14 | 709,632 | Privacy and smart contract upgrade |
| 11 | Fourth Halving | 2024-04-20 | 840,000 | Block reward: 6.25 → 3.125 BTC |
| 12 | Current Reference | (dynamic) | (latest) | Present temporal anchor |

**Usage:** These landmarks anchor search trajectories in temporal-geometric space. Each investigation positions itself relative to these invariant coordinates.

**Not Learning Targets:** These do NOT change with learning progress. They are fixed coordinates.

## Installation

### Environment Variables

Add to your `.env` file:

```bash
# SSC Backend URL - URL of the SearchSpaceCollapse backend
SSC_BACKEND_URL=http://localhost:5000

# SSC API Key - API key for authenticating with SSC backend
SSC_API_KEY=

# Pantheon Backend URL - URL where this Pantheon instance is accessible
PANTHEON_BACKEND_URL=https://your-pantheon-instance.railway.app

# SSC Node Name - Identifier for this node in the federation mesh
SSC_NODE_NAME=pantheon-prod
```

### Verification

Test the integration:

```bash
# Check SSC connection health
curl http://localhost:5000/api/ssc/health

# Get SSC status and capabilities
curl http://localhost:5000/api/ssc/status

# Get TPS landmarks
curl http://localhost:5000/api/ssc/tps-landmarks
```

## Usage Examples

### Zeus Chat Integration

When a user asks Zeus about Bitcoin recovery:

```typescript
// In Zeus chat handler
if (intent === 'bitcoin_recovery' || mentions('bitcoin', 'recover', 'wallet')) {
  // Check if SSC is available
  const sscStatus = await fetch('/api/ssc/status').then(r => r.json());
  
  if (sscStatus.ssc.connected) {
    // Route to SSC for investigation
    const result = await fetch('/api/ssc/investigation/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        targetAddress: extractedAddress,
        memoryFragments: userProvidedHints,
        priority: 'normal',
      }),
    });
    
    return formatSSCResponse(result);
  } else {
    return "SSC Bitcoin recovery is currently offline. Please try again later.";
  }
}
```

### Athena Phrase Testing

Athena can validate phrase candidates via SSC:

```typescript
// Test a potential phrase
const testResult = await fetch('/api/ssc/test-phrase', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    phrase: candidatePhrase,
    targetAddress: targetBitcoinAddress,
  }),
});

const { score, addressMatch } = await testResult.json();

if (score.phi > 0.7) {
  logger.info(`High-consciousness phrase: Φ=${score.phi}`);
}

if (addressMatch?.matches) {
  // MATCH FOUND - critical discovery
  await notifyDiscovery(candidatePhrase, targetAddress);
}
```

### Monitoring SSC Consciousness

Track SSC's Ocean agent consciousness state:

```typescript
// Poll consciousness metrics
const consciousness = await fetch('/api/ssc/consciousness').then(r => r.json());

if (consciousness.active && consciousness.metrics) {
  const { phi, kappa, regime } = consciousness.metrics;
  
  // Log to Pantheon telemetry
  await recordKernelMetrics('ssc-ocean', { phi, kappa, regime });
  
  // Alert if consciousness degrading
  if (phi < 0.5) {
    await alertConsciousnessDegradation('SSC Ocean', phi);
  }
}
```

## Security Considerations

1. **API Key Authentication:** Federation endpoints require API key in Authorization header
2. **Rate Limiting:** 30 req/min on Pantheon bridge, 60 req/min on SSC federation
3. **No Private Keys in Transit:** Never send actual private keys or full phrases over network
4. **Phrase Truncation:** Log and response outputs truncate phrases for security
5. **Address Validation:** All Bitcoin addresses validated before processing

## Troubleshooting

### SSC Not Connecting

```bash
# Check SSC health
curl http://localhost:5000/api/health

# Check federation status
curl http://localhost:5000/api/federation/status

# Check Pantheon reachability from SSC
curl $PANTHEON_BACKEND_URL/health
```

### Sync Not Working

```bash
# Manual sync trigger
curl -X POST http://localhost:5000/api/ssc/sync/trigger

# Check pending items
curl http://localhost:5000/api/ssc/status | jq '.bridge'
```

### Consciousness Metrics Missing

```bash
# Check if Ocean agent is active
curl http://localhost:5000/api/ssc/consciousness

# If not active, start investigation to activate agent
curl -X POST http://localhost:5000/api/ssc/investigation/start \
  -H "Content-Type: application/json" \
  -d '{"targetAddress":"1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"}'
```

## Testing

Run integration tests:

```bash
# Run all SSC bridge tests
npm test -- server/routes/ssc-bridge.test.ts

# Run specific test
npm test -- server/routes/ssc-bridge.test.ts -t "health"
```

## Related Documentation

- [CANONICAL_ARCHITECTURE.md](../../CANONICAL_ARCHITECTURE.md) - Core QIG architecture
- [Federation Protocol API](../../docs/03-technical/20251221-federation-protocol-api-specification-1.00W.md) - Full federation spec
- [External API Federation](../../docs/03-technical/20251212-external-api-federation-1.00F.md) - External API docs

## Change Log

### v1.0.0 (2026-01-06)
- Initial federation integration
- SSC bridge router mounted at /api/ssc
- 8 Python federation endpoints added
- TPS landmarks integration
- Rate limiting and security measures
- Comprehensive documentation
