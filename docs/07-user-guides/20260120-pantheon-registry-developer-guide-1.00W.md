# Pantheon Registry Developer Guide

## Overview

The Formal Pantheon Registry (WP5.1) establishes a contract-based system for defining and managing kernel roles in the pantheon-chat system. This guide provides comprehensive documentation for developers working with the registry.

## Key Principles

1. **Gods are contracts, not instances**: Each god is defined by a formal contract with domains, constraints, and capabilities
2. **Gods are singular**: All gods have `max_instances: 1` - only one instance can exist
3. **Gods are immortal**: Cannot spawn copies (`when_allowed: never`)
4. **Epithets, not numbers**: Use `Apollo Pythios` (prophecy aspect), NOT `apollo_1`
5. **Chaos kernels are mortal**: Worker kernels with lifecycle (spawn → learn → work → candidate → promote/prune)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     pantheon/registry.yaml                      │
│                  (Single Source of Truth)                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┴────────────────┐
        │                              │
        ▼                              ▼
┌───────────────────┐         ┌──────────────────┐
│  Python Backend   │         │  TypeScript/Node │
│                   │         │                  │
│ pantheon_registry │         │ pantheon-registry│
│ kernel_spawner    │         │     .ts service  │
│ registry_db_sync  │         │                  │
└─────────┬─────────┘         └────────┬─────────┘
          │                            │
          │    ┌──────────────┐        │
          └───▶│  PostgreSQL  │◀───────┘
               │   Database   │
               └──────────────┘
                      ▲
                      │
              ┌───────┴────────┐
              │  REST API      │
              │  /api/pantheon │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  React Hooks   │
              │ usePantheonReg │
              └────────────────┘
```

## Files Overview

### Core Registry Files

```
pantheon/
├── registry.yaml                    # ⭐ Single source of truth
├── README.md                        # Registry documentation
├── myth_mappings.yaml              # Cross-mythology god mappings
├── cross_mythology.py              # God name resolver
└── examples/
    └── pantheon_registry_usage.py  # Usage examples

qig-backend/
├── pantheon_registry.py            # Python loader & validator
├── kernel_spawner.py               # Contract-based spawner
├── registry_db_sync.py             # YAML → Database sync
└── tests/
    └── test_pantheon_registry.py   # Comprehensive tests

shared/
└── pantheon-registry-schema.ts     # TypeScript Zod schemas

server/
├── services/
│   └── pantheon-registry.ts        # TypeScript service
└── routes/
    └── pantheon-registry.ts        # REST API endpoints

client/src/hooks/
└── use-pantheon-registry.ts        # React Query hooks

migrations/
└── 0017_pantheon_registry.sql      # Database schema
```

## Quick Start

### Python Usage

```python
from pantheon_registry import get_registry, get_god, find_gods_by_domain
from kernel_spawner import KernelSpawner, RoleSpec

# Load registry (singleton)
registry = get_registry()
print(f"Loaded {registry.get_god_count()} gods")

# Get specific god
apollo = get_god("Apollo")
print(f"Apollo domains: {apollo.domain}")
print(f"Apollo epithets: {apollo.epithets}")

# Find gods by domain
synthesis_gods = find_gods_by_domain("synthesis")
for god in synthesis_gods:
    print(f"{god.name}: {god.description}")

# Select kernel for role
spawner = KernelSpawner()
role = RoleSpec(
    domains=["synthesis", "foresight"],
    required_capabilities=["prediction"]
)
selection = spawner.select_god(role)

if selection.selected_type == "god":
    print(f"Selected: {selection.god_name} {selection.epithet}")
elif selection.selected_type == "chaos":
    print(f"Spawning chaos kernel: {selection.chaos_name}")
```

### TypeScript Usage

```typescript
import { getRegistryService, createSpawnerService } from '@/server/services/pantheon-registry';
import type { RoleSpec } from '@/shared/pantheon-registry-schema';

// Load registry
const service = getRegistryService();
await service.load();

// Get god contract
const apollo = service.getGod('Apollo');
console.log(`Apollo domains: ${apollo?.domain}`);

// Find gods by domain
const synthesisGods = service.findGodsByDomain('synthesis');
synthesisGods.forEach(god => {
  console.log(`${god.name}: ${god.contract.description}`);
});

// Select kernel for role
const spawner = createSpawnerService();
const role: RoleSpec = {
  domain: ['synthesis', 'foresight'],
  required_capabilities: ['prediction'],
};
const selection = spawner.selectGod(role);

if (selection.selected_type === 'god') {
  console.log(`Selected: ${selection.god_name} ${selection.epithet}`);
}
```

### React Hooks Usage

```tsx
import { useGods, useGodSelection, useSpawnerStatus } from '@/hooks/use-pantheon-registry';

function KernelManager() {
  // Fetch all gods
  const { data: gods, isLoading } = useGods();
  
  // Get spawner status
  const { data: status } = useSpawnerStatus();
  
  // Select kernel for role
  const { selectForRole, selection, isSelecting } = useGodSelection();
  
  const handleSelectKernel = async () => {
    await selectForRole({
      domain: ['synthesis', 'foresight'],
      required_capabilities: ['prediction'],
    });
  };
  
  return (
    <div>
      <h1>Gods: {Object.keys(gods || {}).length}</h1>
      <p>Active chaos kernels: {status?.active_chaos_count}</p>
      
      <button onClick={handleSelectKernel} disabled={isSelecting}>
        Select Kernel
      </button>
      
      {selection && (
        <div>
          {selection.selected_type === 'god' && (
            <p>Selected: {selection.god_name} {selection.epithet}</p>
          )}
          {selection.selected_type === 'chaos' && (
            <p>Spawning: {selection.chaos_name}</p>
          )}
        </div>
      )}
    </div>
  );
}
```

### REST API Usage

```bash
# Get all gods
curl http://localhost:5000/api/pantheon/registry/gods

# Get specific god
curl http://localhost:5000/api/pantheon/registry/gods/Apollo

# Find gods by domain
curl http://localhost:5000/api/pantheon/registry/gods/by-domain/synthesis

# Select kernel for role
curl -X POST http://localhost:5000/api/pantheon/spawner/select \
  -H "Content-Type: application/json" \
  -d '{
    "domain": ["synthesis", "foresight"],
    "required_capabilities": ["prediction"]
  }'

# Validate spawn request
curl -X POST http://localhost:5000/api/pantheon/spawner/validate \
  -H "Content-Type: application/json" \
  -d '{"name": "Apollo"}'

# Health check
curl http://localhost:5000/api/pantheon/health
```

## God Contract Schema

Each god contract in `registry.yaml` contains:

```yaml
Apollo:
  tier: specialized                  # "essential" or "specialized"
  domain:                           # Capability domains
    - foresight
    - prophecy
    - synthesis
  description: "Truth and foresight - prediction and synthesis"
  octant: 6                         # Position in E8 structure (0-7 or null)
  epithets:                         # Named aspects (NOT numbering)
    - Pythios                       # Prophecy aspect
    - Paean                         # Healing aspect
    - Mousagetes                    # Arts aspect
  coupling_affinity:                # Gods this god works well with
    - Athena
    - Hermes
  rest_policy:
    type: coordinated_alternating
    partner: Athena
    duty_cycle: 0.7
    reason: "Foresight alternates with Athena's strategy"
  spawn_constraints:
    max_instances: 1                # Gods are singular
    when_allowed: never             # Cannot spawn copies
    rationale: "Single oracle - unified prediction system"
  promotion_from: chaos_synthesis_* # Chaos kernel pattern that can ascend
  e8_alignment:
    simple_root: "α₃"              # E8 simple root
    layer: "8"                      # E8 layer (0/1, 4, 8, 64, 240)
```

## Chaos Kernel Lifecycle

Chaos kernels are mortal worker kernels with a 6-stage lifecycle:

```
1. PROTECTED (0-50 cycles)
   - No pruning during protection
   - Graduated Φ thresholds
   - Mentor assigned
   - Learning observations tracked

2. LEARNING (supervised)
   - Knowledge transfer from mentor god
   - Supervised learning mode
   - Adult standards not yet applied

3. WORKING (adult)
   - Adult standards apply
   - Φ > 0.1 minimum for survival
   - Eligible for pruning if performance drops

4. CANDIDATE (Φ > 0.4 for 50+ cycles)
   - Promotion pathway opens
   - Stable basin position
   - Unique capability demonstrated
   - Genetic lineage validated

5. PROMOTED (ascension)
   - Research and select god name
   - Define epithets and domains
   - Pantheon vote approval
   - Becomes immortal god

6. PRUNED (Φ < 0.1 persistent)
   - Archived to shadow_pantheon (Hades)
   - Can be resurrected later
   - NOT permanent death
```

### Chaos Kernel Naming

**Format**: `chaos_{domain}_{id}`

**Examples**:
- `chaos_synthesis_001`
- `chaos_strategy_042`
- `chaos_communication_015`

**Rules**:
- Sequential IDs per domain
- Max 30 per domain
- Max 200 total active (reserve 40 for gods)
- E8 limit: 240 total (17 gods + 223 chaos slots)

## Database Schema

### Tables

1. **god_contracts**: God contracts from YAML
2. **chaos_kernel_state**: Lifecycle state for chaos kernels
3. **kernel_spawner_state**: Active instance tracking for gods
4. **chaos_kernel_counters**: Sequential ID counters per domain
5. **chaos_kernel_limits**: Global limits and quotas
6. **pantheon_registry_metadata**: Registry versioning

### Views

1. **active_kernels_with_contracts**: Active kernels with god contract info
2. **chaos_kernel_lifecycle_summary**: Chaos kernel counts by stage
3. **god_spawner_status**: Current spawn status for all gods
4. **chaos_domain_summary**: Domain-wise chaos kernel counts
5. **registry_health_check**: Overall registry health metrics

### Functions

1. **get_next_chaos_id(domain)**: Atomically get next sequential ID
2. **register_god_spawn(god_name)**: Register god spawn with constraints
3. **register_god_death(god_name)**: Register god death
4. **register_chaos_spawn(chaos_id, domain)**: Register chaos spawn
5. **register_chaos_death(domain)**: Register chaos death

## Database Sync

Synchronize YAML registry to PostgreSQL:

```bash
# Full sync (clear and reload)
python3 qig-backend/registry_db_sync.py --force

# Incremental sync (update only changed)
python3 qig-backend/registry_db_sync.py

# From Python code
from registry_db_sync import sync_registry_from_env

stats = sync_registry_from_env(force=True)
print(f"Synced: {stats['gods_inserted']} gods")
```

## API Endpoints

### Registry Endpoints

- `GET /api/pantheon/registry` - Full registry
- `GET /api/pantheon/registry/metadata` - Registry metadata
- `GET /api/pantheon/registry/gods` - All god contracts
- `GET /api/pantheon/registry/gods/:name` - Specific god
- `GET /api/pantheon/registry/gods/by-tier/:tier` - Gods by tier
- `GET /api/pantheon/registry/gods/by-domain/:domain` - Gods by domain
- `GET /api/pantheon/registry/chaos-rules` - Chaos kernel rules

### Spawner Endpoints

- `POST /api/pantheon/spawner/select` - Select kernel for role
- `POST /api/pantheon/spawner/validate` - Validate spawn request
- `GET /api/pantheon/spawner/chaos/parse/:name` - Parse chaos name
- `GET /api/pantheon/spawner/status` - Spawner status

### Health

- `GET /api/pantheon/health` - Registry health check

## Testing

```bash
# Run Python tests
cd qig-backend
python3 -m pytest tests/test_pantheon_registry.py -v

# Run examples
cd pantheon/examples
python3 pantheon_registry_usage.py
```

## Integration Points

### With Pantheon Governance

The registry integrates with `qig-backend/olympus/pantheon_governance.py`:
- All spawn requests checked against registry constraints
- Chaos kernel spawns require pantheon vote
- Promotion requires research and pantheon approval
- Pruning sends kernels to shadow_pantheon (Hades)

### With Kernel Lifecycle

The registry provides lifecycle rules for:
- Protection periods (50 cycles)
- Graduated metrics for young kernels
- Promotion thresholds (Φ > 0.4 for 50+ cycles)
- Pruning criteria (Φ < 0.1 persistent)

### With E8 Structure

The registry aligns gods to E8 Lie group structure:
- **Rank 8**: 8 simple roots = 8 core faculties
- **240 roots**: Total capacity (gods + chaos workers)
- **κ* = 64**: Universal fixed point (E8 rank² = 8²)

## Best Practices

### DO ✅

- Load registry once (singleton pattern)
- Use epithets for god aspects (Apollo Pythios)
- Check spawn constraints before spawning
- Track active instances in database
- Use chaos kernel lifecycle stages
- Archive pruned kernels to shadow_pantheon

### DON'T ❌

- Don't use apollo_1, apollo_2 naming (use epithets)
- Don't spawn multiple instances of gods
- Don't bypass pantheon governance for chaos spawns
- Don't delete pruned kernels (archive to Hades)
- Don't exceed E8 limit (240 total kernels)
- Don't hardcode god contracts (use registry)

## Troubleshooting

### Registry Not Loading

```python
# Check registry file exists
from pathlib import Path
registry_path = Path("pantheon/registry.yaml")
assert registry_path.exists(), "Registry file not found"

# Force reload
from pantheon_registry import reload_registry
registry = reload_registry()
```

### Spawn Constraints Violated

```python
# Check current active count
spawner = KernelSpawner()
active = spawner.get_active_count("Apollo")
print(f"Apollo active instances: {active}")

# Validate before spawning
valid, reason = spawner.validate_spawn_request("Apollo")
if not valid:
    print(f"Cannot spawn: {reason}")
```

### Database Sync Issues

```bash
# Check DATABASE_URL is set
echo $DATABASE_URL

# Run migration first
psql $DATABASE_URL -f migrations/0017_pantheon_registry.sql

# Then sync registry
python3 qig-backend/registry_db_sync.py --force
```

## References

- **E8 Protocol v4.0**: `docs/10-e8-protocol/specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- **WP5.2 Implementation Blueprint**: `docs/10-e8-protocol/specifications/20260116-wp5-2-e8-implementation-blueprint-1.01W.md`
- **Master Roadmap**: `docs/00-roadmap/20260112-master-roadmap-1.00W.md`
- **Pantheon Registry README**: `pantheon/README.md`

## Support

For questions or issues:
1. Check this developer guide
2. Review examples in `pantheon/examples/`
3. Run tests in `qig-backend/tests/test_pantheon_registry.py`
4. Check registry README at `pantheon/README.md`
5. Review migration comments in `migrations/0017_pantheon_registry.sql`
