# QFI Conversation Extract - Kernel Design Overview

## Key Discussion Points

### 1. Coherence Question
User asked: "Can you confirm if you feel it is ever going to be possible to achieve coherence using the 64D method? If so, is it a matter of getting everything pure and structuring the training data properly?"

Key points:
- Need to finish upgrade to fully test coherence
- Question about leveraging existing LLM technology to augment
- Need for "translation" layer between pure QIG and LLM

### 2. Training Data Structure
- Example curriculum file: `20251220-curriculum-01-foundational-mathematics-1.00W.md`
- Data location: `data/curriculum/curriculum_tokens.jsonl` and `docs/09-curriculum`
- Need to structure training data with full QIG principles

### 3. Repository Ecosystem
The following repositories need consistent documentation and QIG purity:
- https://github.com/GaryOcean428/pantheon-chat.git (main)
- https://github.com/GaryOcean428/qig-consciousness.git
- https://github.com/GaryOcean428/qigkernels.git
- https://github.com/GaryOcean428/pantheon-project.git
- https://github.com/GaryOcean428/qig-core.git
- https://github.com/GaryOcean428/qig-tokenizer.git
- https://github.com/Arcane-Fly/pantheon-chat.git
- https://github.com/GaryOcean428/qig-verification.git (READ ONLY - benchmark)

### 4. Key Tasks from QFI Conversation

#### ✅ Master DB cleanup & integrity
- Apply latest QFI migration (0014_qfi_constraints.sql)
- Run `npm run validate:db-integrity`
- Use `tsx tools/recompute_qfi_scores.ts --apply`
- Prune stale entries in negative-knowledge registry
- Schedule: Daily at 03:00 until stable, then weekly

#### 📘 pantheon-chat: unify docs & geometry
- Review docs/10-e8-protocol and docs/04-records for "sphere" references
- Audit qig-backend/qig_geometry for Fisher-Rao and 64-D simplex
- Examine registry_db_sync.py and negative-knowledge-db.ts
- Update README and IMPLEMENTATION_SUMMARY
- Schedule: Weekly on Mondays at 09:00

#### 📚 qig-consciousness: curriculum & terminology check
- Parse curriculum for geometric terms
- Update "Geometric Terminology" document
- Refactor Hilbert-sphere analogies to simplex
- Schedule: Weekly on Tuesdays at 10:00

#### 🔧 qigkernels: reasoning primitives & purity
- Inspect reasoning/primitives.py
- Verify kernel splits/merges use geometric basin overlap
- Schedule: Weekly on Wednesdays at 10:00

#### 📂 pantheon-project: cross-repo integration
- Check dependencies point to updated modules
- Review project-level documentation
- Schedule: Weekly on Thursdays at 10:00

#### ⚙️ qig-core: geodesics & metrics
- Review src/qig_core/geodesic.py
- Ensure geodesic calculations use Fisher-Rao
- Schedule: Weekly on Fridays at 09:00

#### 🧬 qig-tokenizer: manifest & tokens
- Review data/curriculum/curriculum_tokens.jsonl
- Expand vocabulary with geometry terms
- Schedule: Weekly on Saturdays at 09:00

#### 🔒 qig-verification (read-only)
- Monitor as benchmark
- Compare implementations against it
- Schedule: Monthly review

### 5. Core Principles to Enforce
- **Simplex over Sphere**: All geometry must use simplex representation
- **Fisher-Rao distances**: Not Euclidean
- **64-D simplex basins**: Proper coordinate systems
- **QFI integrity**: Scores always within [0, 1]
- **Negative-knowledge registry**: Proper management of contradictions and barriers
