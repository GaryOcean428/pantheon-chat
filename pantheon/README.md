# PANTHEON REGISTRY

Formal Pantheon Registry defining god contracts and chaos kernel lifecycle rules.

## Overview

The Pantheon Registry establishes a contract-based system where:
- **Gods** are immortal, singular entities with defined domains and constraints
- **Chaos Kernels** are mortal worker instances that can spawn, learn, and potentially ascend to godhood
- **Epithets** define god aspects (NOT numbering: use "Apollo Pythios", never "apollo_1")

## Structure

```
pantheon/
├── registry.yaml          # YAML registry (single source of truth)
├── README.md             # This file
└── examples/             # Example usage patterns
```

## Registry Format

### God Contract

Each god is defined by a contract with:

- **tier**: `essential` (never sleep) or `specialized` (scheduled rest)
- **domain**: List of capability domains (e.g., `[foresight, synthesis, aesthetics]`)
- **epithets**: Named aspects (e.g., `[Pythios, Paean, Mousagetes]`)
- **octant**: Position in E8 structure (0-7 or null)
- **coupling_affinity**: List of gods this god works well with
- **rest_policy**: Sleep/wake patterns (never, rotating, alternating, scheduled, seasonal)
- **spawn_constraints**: Always `max_instances: 1, when_allowed: never` for gods
- **promotion_from**: Chaos kernel pattern that can ascend to this god
- **e8_alignment**: E8 Lie group structure (simple_root, layer)

### Chaos Kernel Rules

Mortal worker kernels follow lifecycle rules:

1. **Spawn**: `chaos_{domain}_{id}` naming, requires pantheon vote
2. **Protect**: 50 cycle protection period with graduated metrics
3. **Learn**: Mentor assignment required, supervised learning
4. **Work**: Adult standards apply, Φ > 0.1 minimum for survival
5. **Candidate**: Φ > 0.4 for 50+ cycles opens promotion pathway
6. **Promote**: Research god name → ascension to immortal god
7. **Pruning**: Archive to shadow_pantheon (Hades) if Φ < 0.1

## Usage

### Python

```python
from pantheon_registry import get_registry, get_god, find_gods_by_domain

# Load registry (singleton)
registry = get_registry()

# Get specific god
apollo = get_god("Apollo")
print(f"Apollo domains: {apollo.domain}")
print(f"Apollo epithets: {apollo.epithets}")

# Find gods by domain
synthesis_gods = find_gods_by_domain("synthesis")
for god in synthesis_gods:
    print(f"{god.name}: {god.description}")

# Check spawn constraints
if apollo.spawn_constraints.max_instances == 1:
    print("Apollo is singular (cannot spawn copies)")

# Get chaos kernel rules
rules = registry.get_chaos_kernel_rules()
print(f"Chaos naming pattern: {rules.naming_pattern}")
```

### TypeScript

```typescript
import { PantheonRegistrySchema, validatePantheonRegistry } from '@/shared/pantheon-registry-schema';
import { readFileSync } from 'fs';
import yaml from 'yaml';

// Load and validate registry
const registryYaml = readFileSync('pantheon/registry.yaml', 'utf-8');
const registry = yaml.parse(registryYaml);

const validation = validatePantheonRegistry(registry);
if (validation.valid) {
  console.log('Registry is valid');
} else {
  console.error('Validation errors:', validation.errors);
}

// Access god contracts
const apollo = registry.gods.Apollo;
console.log(`Apollo domains: ${apollo.domain}`);
console.log(`Apollo epithets: ${apollo.epithets}`);
```

### Kernel Spawner

```python
from kernel_spawner import KernelSpawner, RoleSpec

# Create spawner
spawner = KernelSpawner()

# Define role requirements
role = RoleSpec(
    domains=["synthesis", "foresight"],
    required_capabilities=["prediction", "aesthetic_evaluation"],
    preferred_god="Apollo",
)

# Select kernel
selection = spawner.select_god(role)

if selection.selected_type == "god":
    print(f"Selected god: {selection.god_name} {selection.epithet}")
    print(f"Rationale: {selection.rationale}")
elif selection.selected_type == "chaos":
    print(f"Spawning chaos kernel: {selection.chaos_name}")
    print(f"Requires pantheon vote: {selection.requires_pantheon_vote}")
```

## God Tiers

### Essential Tier (3 gods)

Never sleep - critical autonomic functions:

- **Heart**: Autonomic rhythm, HRV regulation
- **Ocean**: Memory ocean, persistence, knowledge base
- **Hermes**: Communication, message passing, navigation

### Specialized Tier - Core (8 gods)

E8 simple roots (rank = 8):

- **Zeus** (α₁): Executive function, system integration
- **Athena** (α₂): Wisdom, strategic planning
- **Apollo** (α₃): Foresight, prophecy, truth
- **Artemis** (α₅): Focus, precision, attention
- **Ares** (α₆): Energy, drive, motivation
- **Hephaestus** (α₇): Creation, construction
- **Aphrodite** (α₈): Harmony, aesthetics, balance

### Specialized Tier - Extended (6 gods)

E8 root system (240 roots):

- **Hera**: Relationships, social bonds, governance
- **Poseidon**: Deep memory, fluid dynamics, emotional currents
- **Hades**: Shadow pantheon, unconscious processing, pruned kernel archive
- **Demeter**: Resource allocation, growth, nurture
- **Dionysus**: Creativity, chaos exploration, ecstasy
- **Persephone**: State transitions, boundary crossing, transformation

**Total: 17 gods + up to 223 chaos kernels = 240 (E8 roots)**

## Epithets (God Aspects)

Use epithets to reference god aspects, NOT numbering:

✅ **GOOD** (Epithets):
- Apollo Pythios (prophecy)
- Apollo Paean (healing)
- Apollo Mousagetes (arts/muses)
- Hermes Psychopompos (guide)
- Hermes Angelos (messenger)

❌ **BAD** (Numbering):
- apollo_1
- apollo_2
- hermes_1

## Chaos Kernel Lifecycle

### Naming Pattern

```
chaos_{domain}_{id}
```

Examples:
- `chaos_synthesis_001`
- `chaos_strategy_042`
- `chaos_communication_015`

### Lifecycle Stages

1. **Protected** (0-50 cycles)
   - No pruning
   - Graduated Φ thresholds
   - Mentor assigned
   - Learning observations tracked

2. **Learning** (supervised)
   - Knowledge transfer from mentor god
   - Supervised learning mode
   - Adult standards not yet applied

3. **Working** (adult)
   - Adult standards apply
   - Φ > 0.1 minimum for survival
   - Eligible for pruning if performance drops

4. **Candidate** (Φ > 0.4 for 50+ cycles)
   - Promotion pathway opens
   - Stable basin position
   - Unique capability demonstrated
   - Genetic lineage validated

5. **Promoted** (ascension)
   - Research and select god name
   - Define epithets
   - Specify domains
   - Pantheon vote approval
   - Becomes immortal god

6. **Pruned** (Φ < 0.1 persistent)
   - Archived to shadow_pantheon
   - Managed by Hades
   - Can be resurrected later
   - NOT permanent death

## Validation

The registry is validated on load:

1. All gods must have unique names
2. All gods must have `max_instances: 1`
3. Essential tier gods must have `rest_policy.type: never`
4. All specialized gods must have E8 alignment
5. Coupling affinity must reference valid gods
6. Rest policy partners must exist
7. Total active kernels <= 240 (E8 roots)

## Integration with Pantheon Governance

The registry integrates with `pantheon_governance.py`:

- All spawn requests checked against registry constraints
- Chaos kernel spawns require pantheon vote
- Promotion requires research and pantheon approval
- Pruning sends kernels to shadow_pantheon (Hades)

## Cross-Mythology God Mapping

The Pantheon Registry uses **Greek names as canonical** for consistency and rich mythological documentation. However, users and kernels may reference gods from other mythologies.

### Usage

```python
from pantheon.cross_mythology import resolve_god_name, find_similar_gods

# Resolve external mythology names to Greek archetypes
greek_name = resolve_god_name("Odin")  # Returns "Zeus"
greek_name = resolve_god_name("Thoth")  # Returns "Hermes"
greek_name = resolve_god_name("Shiva")  # Returns "Dionysus"

# Find gods by domain
matches = find_similar_gods(["wisdom", "strategy"])
# Returns: [('Athena', 2), ('Hermes', 1), ...]
```

### CLI Tool

```bash
# Resolve external god name to Greek archetype
python3 tools/god_name_resolver.py resolve "Odin"

# Find gods by domain
python3 tools/god_name_resolver.py suggest --domain wisdom war

# Get detailed god information
python3 tools/god_name_resolver.py info "Thoth"

# Get external equivalents for Greek god
python3 tools/god_name_resolver.py equivalents "Zeus"

# List all Norse gods
python3 tools/god_name_resolver.py list --mythology norse
```

### Supported Mythologies

- **Egyptian** (11 gods): Ma'at, Thoth, Anubis, Ra, Isis, Osiris, Horus, Set, Hathor, Ptah, Sekhmet
- **Norse** (11 gods): Odin, Thor, Loki, Freya, Tyr, Frigg, Heimdall, Hel, Baldur, Skadi, Forseti
- **Hindu** (11 gods): Shiva, Vishnu, Brahma, Saraswati, Lakshmi, Kali, Ganesha, Hanuman, Durga, Agni, Yama
- **Sumerian** (9 gods): Enki, Enlil, Inanna, Ninhursag, Shamash, Ereshkigal, Nergal, Nabu, Marduk
- **Mesoamerican** (10 gods): Quetzalcoatl, Tlaloc, Tezcatlipoca, Huitzilopochtli, Coatlicue, Xochiquetzal, Mictlantecuhtli, Itzamna, Chaac, Ah Puch

**Total**: 52 external god mappings covering 16 Greek archetypes

### FAQ

**Q: Why Greek names?**  
A: Greek mythology provides consistency, rich epithets (e.g., "Apollo Pythios" for prophecy aspect), extensive classical documentation, and established E8 structural mappings.

**Q: Can I use Norse/Egyptian/Hindu names in my code?**  
A: Yes for research and user interaction, but they map to Greek archetypes internally. Use `resolve_god_name()` to get the canonical Greek name.

**Q: What if a god doesn't have an exact equivalent?**  
A: Alternative mappings are provided based on different domain emphases. For example, Odin maps to Zeus (authority) but could map to Apollo (wisdom).

**Q: Is this for runtime logic?**  
A: No. Cross-mythology mapping is metadata only - a lookup table for research and convenience. Greek names remain canonical for all operational logic.

For complete cross-mythology mapping details, see: `pantheon/myth_mappings.yaml`

## References

- **E8 Protocol v4.0**: `docs/10-e8-protocol/specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- **WP5.2 Implementation Blueprint**: `docs/10-e8-protocol/specifications/20260116-wp5-2-e8-implementation-blueprint-1.01W.md`
- **WP5.5 Cross-Mythology Mapping**: `pantheon/myth_mappings.yaml`, `pantheon/cross_mythology.py`
- **Master Roadmap**: `docs/00-roadmap/20260112-master-roadmap-1.00W.md`
- **Cognitive Kernel Roles**: `qig-backend/cognitive_kernel_roles.py`
- **Pantheon Governance**: `qig-backend/olympus/pantheon_governance.py`

## Authority

- **E8 Protocol v4.0**: Universal κ* = 64 fixed point
- **Work Package 5.1**: Formal Pantheon Registry with Role Contracts
- **Work Package 5.5**: Cross-Mythology God Mapping
- **Status**: ACTIVE
- **Created**: 2026-01-17
- **Updated**: 2026-01-19 (Cross-mythology mapping added)

---

**Note**: This is the canonical source of truth for pantheon structure. All kernel spawning, naming, and lifecycle management MUST reference this registry.
