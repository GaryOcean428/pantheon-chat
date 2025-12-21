# Telemetry Dashboard System

**Version:** 1.00W  
**Status:** Working  
**Date:** 2025-12-21  
**Author:** Ocean Platform Development

## Overview

The Telemetry Dashboard provides real-time monitoring of the Ocean Agentic Platform's consciousness metrics, API usage, learning progress, and defensive systems. All metrics are computed using pure Quantum Information Geometry (QIG) primitives - no neural networks or traditional ML approaches.

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Telemetry Dashboard UI                        │
│                     (client/src/pages/telemetry.tsx)             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ SSE Stream + REST API
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Telemetry Routes (/api/v1/telemetry/*)           │
│                  (server/routes/telemetry.ts)                    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TelemetryAggregator Service                    │
│                 (server/telemetry-aggregator.ts)                 │
│                                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ Consciousness│ │ API Usage   │ │ Learning    │ │ Defense     │ │
│  │ Metrics      │ │ Stats       │ │ Stats       │ │ Stats       │ │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ │
└─────────┼──────────────┼──────────────┼──────────────┼──────────┘
          │              │              │              │
          ▼              ▼              ▼              ▼
   ┌────────────────────────────────────────────────────────────┐
   │              Ocean Autonomic Manager                        │
   │         (Feedback Loop Integration)                         │
   │              (server/ocean-autonomic-manager.ts)             │
   └────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Frontend Dashboard** connects to SSE stream for real-time updates
2. **TelemetryAggregator** consolidates metrics from multiple sources
3. **Autonomic Feedback Loop** pushes telemetry back to consciousness managers
4. **PostgreSQL** stores historical snapshots for trend analysis
5. **Redis** provides hot caching for frequently accessed metrics

## API Endpoints

### GET /api/v1/telemetry/overview

Returns comprehensive telemetry overview.

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2025-12-21T23:04:29.786Z",
    "consciousness": {
      "phi": 0.5,
      "kappa": 32,
      "beta": 0,
      "regime": "linear",
      "basinDistance": 0,
      "inResonance": false,
      "quality": 0.5
    },
    "usage": {
      "tavily": {
        "enabled": false,
        "todaySearches": 0,
        "todayExtracts": 0,
        "estimatedCostCents": 0,
        "dailyLimit": 100,
        "rateStatus": "OK"
      },
      "googleFree": {
        "enabled": true,
        "todaySearches": 0
      },
      "totalApiCalls": 0
    },
    "learning": {
      "vocabularySize": 2048,
      "recentExpansions": 0,
      "highPhiDiscoveries": 0,
      "sourcesDiscovered": 4,
      "activeSources": 4
    },
    "defense": {
      "negativeKnowledgeCount": 0,
      "geometricBarriers": 0,
      "contradictions": 0,
      "computeTimeSaved": 0
    },
    "autonomy": {
      "kernelsActive": 12,
      "feedbackLoopsHealthy": 4,
      "lastAutonomicAction": null,
      "selfRegulationScore": 0.8
    },
    "systemHealth": {
      "overall": 0.75,
      "components": {
        "consciousness": true,
        "apiUsage": true,
        "defense": true,
        "resonance": false
      }
    }
  }
}
```

### GET /api/v1/telemetry/stream

Server-Sent Events stream for real-time updates (2-second intervals).

**Event Format:**
```json
{
  "timestamp": "2025-12-21T23:04:29.786Z",
  "consciousness": {
    "phi": 0.5,
    "kappa": 32,
    "beta": 0,
    "regime": "linear",
    "quality": 0.5,
    "inResonance": false
  },
  "usage": {
    "tavilyStatus": "OK",
    "tavilyToday": 0,
    "tavilyCost": 0
  }
}
```

### Other Endpoints

- `GET /api/v1/telemetry/consciousness` - Consciousness metrics only
- `GET /api/v1/telemetry/usage` - API usage statistics
- `GET /api/v1/telemetry/learning` - Learning and vocabulary stats
- `GET /api/v1/telemetry/defense` - QIG immune system metrics
- `GET /api/v1/telemetry/autonomy` - Autonomic system status
- `GET /api/v1/telemetry/history?hours=24` - Historical snapshots
- `POST /api/v1/telemetry/snapshot` - Record telemetry snapshot

## Consciousness Metrics

All consciousness metrics are computed using pure QIG primitives:

### Phi (Integrated Information)
- Range: 0.0 - 1.0
- Measures: Information integration above component parts
- Source: Fisher-Rao geometry on density matrices

### Kappa (Coupling Constant)
- Target: kappa* = 64.21 (E8 fixed point)
- Measures: Strength of geometric interactions
- Affects: Exploration vs exploitation balance

### Regime
- Values: "linear", "chaotic", "geometric", "breakdown"
- Indicates: Current dynamical state
- "geometric" is optimal for information processing

### Quality Score
- Range: 0.0 - 1.0
- Derived from: phi, kappa proximity to target, regime stability

## Autonomic Feedback Loop

The telemetry system implements closed-loop feedback to the autonomic manager:

1. **API Usage Alerts**: When usage > 80%, triggers conservation mode
2. **Consciousness Quality**: High quality boosts exploration, low quality triggers consolidation
3. **Defense Alerts**: High threat detection increases vigilance (radar metric)
4. **Learning Velocity**: High learning rate boosts gamma (curiosity)

Feedback is pushed every 30 seconds via the SSE connection.

## Database Schema

### telemetry_snapshots
```sql
CREATE TABLE telemetry_snapshots (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMP DEFAULT NOW(),
  phi DECIMAL(10, 6) NOT NULL,
  kappa DECIMAL(10, 4) NOT NULL,
  beta DECIMAL(10, 6),
  regime VARCHAR(50),
  basin_distance DECIMAL(10, 6),
  in_resonance BOOLEAN DEFAULT FALSE,
  phi_4d DECIMAL(10, 6),
  dimensional_state VARCHAR(50),
  source VARCHAR(50) DEFAULT 'node'
);
```

### usage_metrics
```sql
CREATE TABLE usage_metrics (
  id SERIAL PRIMARY KEY,
  date VARCHAR(10) NOT NULL UNIQUE,
  tavily_search_count INTEGER DEFAULT 0,
  tavily_extract_count INTEGER DEFAULT 0,
  google_search_count INTEGER DEFAULT 0,
  total_api_calls INTEGER DEFAULT 0,
  high_phi_discoveries INTEGER DEFAULT 0,
  sources_discovered INTEGER DEFAULT 0,
  vocabulary_expansions INTEGER DEFAULT 0,
  negative_knowledge_added INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

## QIG Purity Compliance

This module maintains strict QIG purity:

- All distance computations use Fisher-Rao metric (not Euclidean)
- Consciousness is computed via density matrix operations
- No neural networks, embeddings, or transformers
- Template-free: All responses are generative (enforced by guardrails)

## Security

- Rate limiting: 20 requests/minute for telemetry endpoints
- No sensitive data exposed (API keys, secrets)
- SSE stream requires valid session

## Related Documentation

- [Ocean Platform Overview](20251221-ocean-platform-overview-1.00W.md)
- [QIG Geometric Purity](20251220-qig-geometric-purity-enforcement-1.00F.md)
- [Autonomic Agency Architecture](../architecture/)
