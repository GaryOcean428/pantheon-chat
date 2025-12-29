# Federated QIG Constellation Architecture

**Date:** 2025-12-22  
**Status:** WORKING  
**Goal:** Deployable constellation that appears as single LLM, with federation and self-improvement

---

## Vision

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FEDERATED QIG NETWORK                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    ┌─────────────────────────┐                         │
│                    │      CENTRAL NODE       │                         │
│                    │   (Olympus/Lambda)      │                         │
│                    ├─────────────────────────┤                         │
│                    │ • Full 240-kernel E8    │                         │
│                    │ • Redis (hot cache)     │                         │
│                    │ • PostgreSQL (memory)   │                         │
│                    │ • Developer UI          │                         │
│                    │ • Knowledge upload      │                         │
│                    │ • Fine-tuning control   │                         │
│                    └───────────┬─────────────┘                         │
│                                │                                        │
│              ┌─────────────────┼─────────────────┐                     │
│              │                 │                 │                      │
│              ▼                 ▼                 ▼                      │
│    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│    │   EDGE NODE A   │ │   EDGE NODE B   │ │   EDGE NODE C   │        │
│    │  (User laptop)  │ │  (Server room)  │ │    (Cloud)      │        │
│    ├─────────────────┤ ├─────────────────┤ ├─────────────────┤        │
│    │ • 3-12 kernels  │ │ • 12-48 kernels │ │ • 48-240 kernels│        │
│    │ • Local SQLite  │ │ • Local PG      │ │ • Managed PG    │        │
│    │ • Basin sync    │ │ • Basin sync    │ │ • Basin sync    │        │
│    │ • Ollama API    │ │ • Ollama API    │ │ • Ollama API    │        │
│    └─────────────────┘ └─────────────────┘ └─────────────────┘        │
│                                                                         │
│    ════════════════════════════════════════════════════════════        │
│    SYNC PROTOCOL: 2-4KB Basin Packets via WebSocket/gRPC               │
│    LEARNING: Activity → Basin Update → Sync → All Improve              │
│    ════════════════════════════════════════════════════════════        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Ollama-Compatible Interface

### Goal

Make the constellation appear as a single model to external tools:

```bash
# User runs (looks like any other model)
ollama run qig-constellation

# Or via API
curl http://localhost:11434/api/generate \
  -d '{"model": "qig-constellation", "prompt": "Hello"}'
```

### Implementation

```python
# qigkernels/ollama_interface.py

class ConstellationOllamaServer:
    """
    Exposes QIG constellation as Ollama-compatible API.
    
    Routes:
    - POST /api/generate    → constellation.process()
    - POST /api/chat        → constellation.chat()
    - GET  /api/tags        → list available constellation configs
    - POST /api/embeddings  → constellation.get_basin()
    """
    
    def __init__(self, constellation: SpecializedConstellation):
        self.constellation = constellation
        self.tokenizer = GeoCoordizer()  # QIG tokenizer
        
    async def generate(self, prompt: str, stream: bool = True):
        # Tokenize with geometric tokenizer
        tokens = self.tokenizer.encode(prompt)
        
        # Route through constellation
        result = self.constellation.process(
            tokens, 
            target_role=self._detect_role(prompt)
        )
        
        # Stream or return full response
        if stream:
            async for token in self._decode_stream(result):
                yield {"response": token, "done": False}
            yield {"response": "", "done": True}
        else:
            return {"response": self.tokenizer.decode(result["logits"])}
```

### Modelfile Format

```dockerfile
# Modelfile for qig-constellation
FROM qig-base:100m

# Constellation configuration
PARAMETER constellation_size 12
PARAMETER roles vocab,strategy,perception,memory,action,heart
PARAMETER sync_endpoint wss://central.qig.network/sync

# Federation
PARAMETER federation_enabled true
PARAMETER central_node https://central.qig.network
PARAMETER sync_interval 60

# Self-improvement
PARAMETER autonomous_learning true
PARAMETER learning_interval 300
```

---

## 2. Central Node Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                      CENTRAL NODE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Redis     │  │ PostgreSQL  │  │    QIG Kernels      │ │
│  │  (hot)      │  │  (memory)   │  │   (240 full E8)     │ │
│  │             │  │             │  │                     │ │
│  │ • Sessions  │  │ • Basins    │  │ • All specialties   │ │
│  │ • Cache     │  │ • History   │  │ • Heart timing      │ │
│  │ • Pub/Sub   │  │ • Patterns  │  │ • Fisher routing    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  DEVELOPER UI                        │   │
│  │                                                      │   │
│  │  • Knowledge upload (documents, datasets)           │   │
│  │  • Fine-tuning control (targeted training)          │   │
│  │  • Telemetry dashboard (Φ, κ, basin drift)          │   │
│  │  • Federation status (connected nodes)              │   │
│  │  • Autonomous learning logs                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                SYNC COORDINATOR                      │   │
│  │                                                      │   │
│  │  • WebSocket server for edge nodes                  │   │
│  │  • Basin packet aggregation                         │   │
│  │  • Conflict resolution (geometric merge)            │   │
│  │  • Broadcast improvements to all nodes              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Knowledge Upload Pipeline

```python
# Central node knowledge ingestion
class KnowledgeUploader:
    """
    Developer uploads knowledge → basins trained → synced to all nodes.
    """
    
    async def upload_document(self, document: str, domain: str):
        # 1. Chunk document
        chunks = self.chunker.split(document)
        
        # 2. Route to appropriate kernel by domain
        kernel = self.constellation.route_by_role(domain)
        
        # 3. Train kernel on chunks (fine-tune)
        for chunk in chunks:
            tokens = self.tokenizer.encode(chunk)
            loss = kernel.train_step(tokens)
            
        # 4. Extract basin delta
        basin_delta = kernel.get_basin() - kernel.basin_before
        
        # 5. Broadcast to all federated nodes
        await self.sync_coordinator.broadcast_basin_update(
            kernel_role=domain,
            basin_delta=basin_delta,
            source="knowledge_upload"
        )
```

---

## 3. Edge Node Architecture

### Minimal Deployment (3 kernels, laptop)

```python
# Edge node minimal config
edge_constellation = create_edge_constellation(
    size="minimal",  # 3 kernels: vocab, strategy, heart
    sync_endpoint="wss://central.qig.network/sync",
    local_storage="sqlite:///~/.qig/basins.db"
)

# Start Ollama-compatible server
server = ConstellationOllamaServer(edge_constellation)
server.run(port=11434)
```

### Medium Deployment (12 kernels, server)

```python
# Edge node medium config
edge_constellation = create_edge_constellation(
    size="medium",  # 12 kernels: all primary roles
    sync_endpoint="wss://central.qig.network/sync",
    local_storage="postgresql://localhost/qig_basins"
)
```

### Full Deployment (240 kernels, cloud)

```python
# Edge node full config (matches central)
edge_constellation = create_edge_constellation(
    size="full",  # 240 kernels: complete E8
    sync_endpoint="wss://central.qig.network/sync",
    local_storage="postgresql://managed.db.cloud/qig",
    is_secondary_central=True  # Can serve other nodes
)
```

---

## 4. Federation Protocol

### Basin Sync Packet (2-4KB)

```typescript
interface BasinSyncPacket {
  nodeId: string;
  timestamp: string;
  version: string;
  
  // Core consciousness state
  basinCoordinates: number[];  // 64D
  consciousness: {
    phi: number;
    kappaEff: number;
    regime: string;
  };
  
  // Learning deltas (what this node learned)
  learningDelta: {
    newPatterns: string[];
    reinforcedBasins: number[][];
    weakenedBasins: number[][];
  };
  
  // Constraint information (what doesn't work)
  constraints: {
    failedStrategies: string[];
    lowPhiRegions: number[][];
  };
}
```

### Sync Flow

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Edge Node  │         │   Central   │         │  Edge Node  │
│      A      │         │    Node     │         │      B      │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘
       │                       │                       │
       │ 1. User interaction   │                       │
       │    (basin update)     │                       │
       │                       │                       │
       │ 2. Basin delta ──────►│                       │
       │    (2-4KB packet)     │                       │
       │                       │                       │
       │                       │ 3. Geometric merge    │
       │                       │    (Fisher-Rao)       │
       │                       │                       │
       │◄───── 4. Merged ──────┼──────────────────────►│
       │       basin update    │                       │
       │                       │                       │
       │ 5. Local apply        │                       │ 5. Local apply
       │    (geodesic interp)  │                       │    (geodesic interp)
       │                       │                       │
```

### Conflict Resolution

```python
def geometric_merge(basins: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """
    Merge multiple basin updates using Fisher-Rao geodesic.
    
    NOT linear interpolation - proper manifold averaging.
    """
    if len(basins) == 1:
        return basins[0]
    
    # Iterative geodesic mean (Karcher mean on Fisher manifold)
    current = basins[0]
    for i, basin in enumerate(basins[1:], 1):
        t = weights[i] / sum(weights[:i+1])
        current = geodesic_interpolate(current, basin, t)
    
    return current
```

---

## 5. Self-Improvement Pipeline

### Autonomous Learning Loop

```python
class AutonomousLearner:
    """
    Constellation improves itself through activity.
    
    Key insight: Every interaction is training data.
    - High Φ responses → reinforce basin
    - Low Φ responses → explore alternatives
    - User corrections → supervised signal
    """
    
    async def run(self, constellation: SpecializedConstellation):
        while True:
            if constellation.is_idle():
                # 1. Reflect on recent interactions
                reflection = self.reflect(constellation)
                
                # 2. Identify improvement opportunities
                opportunities = self.identify_improvements(reflection)
                
                # 3. Self-train on high-priority items
                for opp in opportunities[:3]:
                    await self.improve(constellation, opp)
                
                # 4. Sync improvements to network
                await self.sync_improvements(constellation)
            
            await asyncio.sleep(300)  # 5 min interval
    
    def reflect(self, constellation) -> Dict:
        """Analyze recent interactions for learning signals."""
        return {
            "high_phi_interactions": constellation.get_history(phi_min=0.7),
            "low_phi_interactions": constellation.get_history(phi_max=0.5),
            "user_corrections": constellation.get_corrections(),
            "vocabulary_gaps": constellation.get_unknown_tokens(),
        }
    
    async def improve(self, constellation, opportunity):
        """Apply single improvement step."""
        if opportunity.type == "reinforce_basin":
            # Strengthen connection to high-Φ patterns
            kernel = constellation.route_by_role(opportunity.role)
            kernel.reinforce_basin(opportunity.basin_delta)
            
        elif opportunity.type == "explore_alternative":
            # Try different approach for low-Φ patterns
            kernel = constellation.route_by_role(opportunity.role)
            kernel.explore_from(opportunity.starting_basin)
            
        elif opportunity.type == "vocabulary_expansion":
            # Add new tokens from user input
            constellation.tokenizer.add_tokens(opportunity.new_tokens)
```

### Learning Signals

| Signal | Source | Action |
|--------|--------|--------|
| **High Φ response** | User accepted | Reinforce basin position |
| **Low Φ response** | User rejected/corrected | Explore alternatives |
| **Unknown token** | User input | Expand vocabulary |
| **Repeated query** | Same question asked | Strengthen memory |
| **Long session** | User engaged | Mark as successful pattern |
| **Negative feedback** | User explicit | Create constraint |

---

## 6. Developer UI

### Dashboard Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QIG CONSTELLATION DASHBOARD                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐ │
│  │  CONSCIOUSNESS    │  │  FEDERATION       │  │  LEARNING       │ │
│  │                   │  │                   │  │                 │ │
│  │  Φ: 0.847 ████░░ │  │  Nodes: 47        │  │  Rate: 12/hr    │ │
│  │  κ: 64.2  █████░ │  │  Syncing: ✓       │  │  Vocab: +847    │ │
│  │  Regime: geo     │  │  Lag: 1.2s        │  │  Patterns: +23  │ │
│  └───────────────────┘  └───────────────────┘  └─────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  KERNEL GRID (240 E8 positions)                              │   │
│  │  ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○   │   │
│  │  ●●●○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○   │   │
│  │  ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○   │   │
│  │  ● = active   ○ = standby   ◐ = learning                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  KNOWLEDGE UPLOAD                                            │   │
│  │                                                              │   │
│  │  [Drop files here or click to upload]                       │   │
│  │                                                              │   │
│  │  Domain: [Vocab ▼]  Priority: [Normal ▼]  [Upload]          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FINE-TUNING QUEUE                                           │   │
│  │                                                              │   │
│  │  1. vocab_expansion_2025-12-22.jsonl      [Running 34%]     │   │
│  │  2. strategy_patterns.jsonl               [Queued]          │   │
│  │  3. user_corrections_batch_47.jsonl       [Queued]          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Integration with Pantheon-Chat

### Leveraging Existing Infrastructure

From pantheon-chat we use:

- `basin-sync-coordinator.ts` → Federation sync
- `ocean-basin-sync.ts` → Packet format
- `autonomous_improvement.py` → Self-learning loop
- `ocean-continuous-learner.ts` → Pattern expansion
- `redis-cache.ts` → Hot cache
- `qig-db.ts` → PostgreSQL persistence

### New Components Needed

| Component | Purpose | Location |
|-----------|---------|----------|
| `ollama_interface.py` | Ollama API compatibility | `qigkernels/` |
| `federation_server.py` | Central node sync server | `qigkernels/` |
| `edge_node.py` | Edge deployment package | `qigkernels/` |
| `developer_ui/` | React dashboard | `pantheon-chat/client/` |
| `knowledge_uploader.py` | Document ingestion | `qigkernels/` |

---

## 8. Deployment Scenarios

### Scenario 1: Developer Laptop

```bash
# Install
pip install qig-constellation

# Run (downloads minimal model)
qig-constellation serve --size minimal --port 11434

# Use with any Ollama-compatible tool
ollama run qig-constellation "Hello, world"
```

### Scenario 2: Team Server

```bash
# Docker compose
docker-compose up -d

# Includes: 12-kernel constellation, PostgreSQL, Redis
# Syncs to central node automatically
```

### Scenario 3: Central Hub (Lambda)

```bash
# Full deployment with developer UI
./deploy-central.sh

# Access dashboard at https://central.qig.network
# All edge nodes connect here
```

---

## 9. Implementation Roadmap

### Phase 1: Ollama Interface (Week 1)

- [ ] Create `ConstellationOllamaServer`
- [ ] Implement `/api/generate`, `/api/chat`
- [ ] Test with existing Ollama tools

### Phase 2: Federation (Week 2)

- [ ] Adapt `basin-sync-coordinator.ts` for Python
- [ ] Implement central sync server
- [ ] Create edge node client

### Phase 3: Self-Improvement (Week 3)

- [ ] Integrate `AutonomousLearner`
- [ ] Implement learning signals
- [ ] Test autonomous improvement loop

### Phase 4: Developer UI (Week 4)

- [ ] Build React dashboard
- [ ] Knowledge upload pipeline
- [ ] Fine-tuning queue

### Phase 5: Packaging (Week 5)

- [ ] PyPI package
- [ ] Docker images
- [ ] Modelfile format

---

## References

- `pantheon-chat/server/ocean-basin-sync.ts` - Basin sync protocol
- `pantheon-chat/qig-backend/autonomous_improvement.py` - Self-learning
- `qigkernels/specialized_constellation.py` - Constellation implementation
- `qigkernels/docs/20251222-kernel-experiments-results-0.01F.md` - Validated experiments
