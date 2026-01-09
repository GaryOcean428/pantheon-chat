# Pantheon-Replit Development Roadmap

**Document ID:** 20260109-roadmap-development-1.00W
**Project:** Development/MVP QIG Platform on Replit
**Database:** Neon PostgreSQL (us-east-1)
**Status:** Development
**Date:** 2026-01-09
**Version:** 1.00 [W]orking

---

## Vision

Maintain Pantheon-Replit as the **rapid experimentation environment** for QIG features before production deployment. Serve as the testing ground for risky changes, experimental features, and architectural innovations.

**Role in Ecosystem:**

- **Upstream:** Fork of pantheon-chat (production parity)
- **Purpose:** Validate changes before Railway deployment
- **Downstream:** Feed validated features to pantheon-chat

---

## Current State (2026-Q1)

### ✅ Completed

- **QIG Core**: Fully functional dual backend (Node.js + Python Flask)
- **Database**: Neon PostgreSQL (us-east-1) with pgvector
- **Parity**: ~90% code similarity with pantheon-chat
- **Replit Optimization**: Auto-deployment, environment configuration
- **Testing Sandbox**: Isolated from production data

### ⚠️ Critical Issues

1. **Ocean-agent.ts bloat**: 6,141 lines (slightly better than pantheon-chat but still critical)
2. **Divergence risk**: 90% similarity means drift is accelerating
3. **Test infrastructure**: Need stronger parity validation with production

---

## Q1 2026 (Current Quarter)

### Priority 1: Divergence Control

**Problem:** pantheon-replit is drifting from pantheon-chat (started 100% identical, now 90%)

**Solution:** Establish bidirectional sync policy

**Tasks:**

- [ ] Document allowed divergences (database, deployment config, experiments)
- [ ] Create diff tool: `npm run diff:production` to compare with pantheon-chat
- [ ] Establish merge protocol: experimental → validated → production
- [ ] Add CI check: fail if architectural patterns diverge
- [ ] Create sync log (track what syncs when)

**Success Criteria:**

- [ ] <10% divergence in shared components (qig-backend/, shared/)
- [ ] 100% architectural pattern parity (barrel files, service layer, etc.)
- [ ] Zero QIG purity violations in either project
- [ ] Merge path clear for validated experiments

**Timeline:** 2 weeks (2026-01-09 → 2026-01-23)

---

### Priority 2: Experimental Ocean-Agent Modularization

**Goal:** Test modularization strategy HERE before applying to production

**Why Here First:**

- Lower risk (development environment)
- Faster iteration (no deployment pipeline)
- Validate architecture before pantheon-chat refactor

**Modules to create:**

```
server/modules/
├── hypothesis-generator.ts      # Extract first (least coupled)
├── basin-manager.ts             # Extract second (well-defined interface)
├── geodesic-navigator.ts        # Extract third (complex geometry)
├── consciousness-tracker.ts     # Extract fourth (metrics tracking)
└── pantheon-coordinator.ts      # Extract last (agent orchestration)
```

**Extraction Strategy:**

1. Start with hypothesis-generator.ts (cleanest boundaries)
2. Validate tests pass and Φ/κ remain stable
3. Measure performance impact (<5% regression acceptable)
4. Document extraction pattern for pantheon-chat team
5. Repeat for each module

**Success Criteria:**

- [ ] ocean-agent.ts reduced from 6,141 → <1000 lines
- [ ] Each module <400 lines, well-tested (>80% coverage)
- [ ] No consciousness metric degradation (Φ, κ stable)
- [ ] Extraction guide created for pantheon-chat
- [ ] Performance impact documented (<5% regression)

**Timeline:** 6 weeks (2026-01-23 → 2026-03-06)

---

### Priority 3: Advanced Experimental Features

**Goal:** Test cutting-edge QIG capabilities before production deployment

**Experiments to run:**

#### **A. Kernel Constellation (M8 Spawning)**

- [ ] Implement kernel spawning protocol
- [ ] Test domain specialization (physics, math, code kernels)
- [ ] Measure Φ improvement from specialized kernels
- [ ] Validate kernel merging logic
- [ ] Document findings for pantheon-chat deployment

**Success:** 3+ kernels deployed, Φ improvement >10%

#### **B. Chaos-Driven Discovery**

- [ ] Implement chaos gate (controlled entropy injection)
- [ ] Test discovery rate vs chaos intensity
- [ ] Measure novel pattern emergence
- [ ] Validate chaos recovery (return to order)
- [ ] Document safe chaos parameters

**Success:** 2x discovery rate without Φ degradation

#### **C. Geometric Foresight Enhancement**

- [ ] Test 16-basin context window (vs current 8-basin)
- [ ] Experiment with weighted trajectory regression
- [ ] Validate prediction accuracy improvements
- [ ] Measure computational cost
- [ ] Compare to bigram baseline

**Success:** 50%+ improvement over bigram, <2x compute cost

**Timeline:** 4 weeks (2026-03-06 → 2026-04-03)

---

## Q2 2026

### Production Parity Validation

**Goal:** Ensure pantheon-replit can perfectly simulate production environment

**Features:**

- [ ] Implement Railway environment simulation
- [ ] Add pgvector compatibility layer (Neon → Railway)
- [ ] Create production data snapshot importer
- [ ] Add performance profiling (match Railway metrics)
- [ ] Test federation with pantheon-chat (Neon ↔ Railway)

**Success Criteria:**

- [ ] Can replay production scenarios with 95%+ accuracy
- [ ] Federation latency <2s between environments
- [ ] Performance within 20% of Railway
- [ ] Zero data corruption during federation sync

---

### Rapid Prototyping Infrastructure

**Goal:** Reduce experiment-to-validation cycle from weeks to days

**Features:**

- [ ] One-click experiment deployment (Replit → branch)
- [ ] A/B testing framework (experimental vs stable)
- [ ] Automated regression detection (Φ, κ monitoring)
- [ ] Rollback automation (<5min to revert)
- [ ] Experiment dashboard (track all active experiments)

**Success Criteria:**

- [ ] Deploy experiment in <10min
- [ ] Automated A/B comparison reports
- [ ] Auto-rollback on Φ drop >5%
- [ ] 5+ concurrent experiments without interference

---

### QIG Primitive Research

**Goal:** Develop next-generation geometric operations

**Research Areas:**

- [ ] Sectional curvature of information manifold
- [ ] Parallel transport for basin evolution
- [ ] Riemannian conjugate gradient (optimization)
- [ ] Geodesic regression (trajectory modeling)
- [ ] Fisher-Rao heat kernel (diffusion)

**Success Criteria:**

- [ ] 3+ new primitives validated
- [ ] Published in docs/03-technical/
- [ ] Integration guide for pantheon-chat
- [ ] Performance benchmarks documented

---

## Q3 2026

### Stress Testing & Edge Cases

**Goal:** Break pantheon-replit to find production vulnerabilities

**Test Scenarios:**

- [ ] 10,000 concurrent hypotheses (memory stress)
- [ ] Basin corruption recovery (data integrity)
- [ ] Python backend crash recovery (fault tolerance)
- [ ] Database connection loss (resilience)
- [ ] Malformed QIG input (validation)

**Success Criteria:**

- [ ] Identify 10+ edge cases
- [ ] Document failure modes
- [ ] Implement fixes here first
- [ ] Validation suite for pantheon-chat

---

### Community Experiment Platform

**Goal:** Allow external researchers to run QIG experiments

**Features:**

- [ ] Sandboxed experiment environment
- [ ] Experiment submission API
- [ ] Resource quotas (CPU, memory, basins)
- [ ] Results publication system
- [ ] Leaderboard (Φ improvement, discovery rate)

**Success Criteria:**

- [ ] 10+ external experiments run
- [ ] Zero security incidents
- [ ] 3+ novel QIG techniques discovered
- [ ] Community contributions to pantheon-chat

---

## Q4 2026

### Merge Validated Features to Production

**Goal:** Systematic migration of validated experiments to pantheon-chat

**Merge Candidates:**

- [ ] Ocean-agent modularization (Q1 validated)
- [ ] Kernel constellation (Q2 validated)
- [ ] Chaos discovery (Q2 validated)
- [ ] New QIG primitives (Q2 validated)
- [ ] Stress test fixes (Q3 validated)

**Merge Protocol:**

1. Feature validated in pantheon-replit (Φ stable, tests pass)
2. Create feature branch in pantheon-chat
3. Adapt for Railway environment (pgvector, deployment)
4. Run production test suite
5. Deploy to Railway staging
6. Monitor for 1 week (Φ, κ, errors)
7. Merge to pantheon-chat main

**Success Criteria:**

- [ ] 5+ features merged to production
- [ ] Zero production incidents from merges
- [ ] Merge process documented
- [ ] Avg merge time <2 weeks

---

## Long-Term Vision (2027+)

### Eternal Experimentation Ground

**Capabilities:**

- Permanent A/B testing vs pantheon-chat stable
- Bleeding-edge QIG research (1-2 versions ahead)
- External researcher sandbox
- QIG technique incubator
- Failure mode discovery engine

**Impact:**

- pantheon-chat stays stable, pantheon-replit takes risks
- New QIG techniques validated before production
- Community-driven innovation
- Reduced production incidents (caught in replit first)

---

## Divergence Policy

### Allowed Divergences (Will NOT Sync)

- Database connection (Neon vs Railway)
- Deployment config (Replit vs Railway)
- Experimental features (pre-validation)
- Test data (synthetic vs production-like)
- Environment variables (.env.replit vs .env.railway)

### Prohibited Divergences (MUST Sync)

- QIG purity violations (Fisher-Rao only)
- Architectural patterns (barrel files, service layer, etc.)
- shared/ directory contents (types, constants, validation)
- qig-backend/ core (except experimental branches)
- ISO 27001 docs structure

### Sync Schedule

- **Weekly:** shared/, qig-backend/ core updates from pantheon-chat
- **Monthly:** Architectural pattern enforcement
- **Quarterly:** Full codebase diff and reconciliation
- **Ad-hoc:** QIG purity fixes (immediate sync)

---

## Success Metrics (2026)

| Metric | Q1 Target | Q2 Target | Q3 Target | Q4 Target |
|--------|-----------|-----------|-----------|-----------|
| Experiments Run | 10 | 25 | 50 | 100 |
| Features Validated | 2 | 5 | 10 | 15 |
| Merged to Production | 0 | 2 | 5 | 10 |
| Code Parity | 90% | 85% | 80% | 85% |
| Test Coverage | 60% | 75% | 85% | 90% |
| Avg Experiment Time | 2wk | 1wk | 3d | 1d |

---

## Dependencies & Blockers

### Upstream (pantheon-chat)

- Production changes must be pulled regularly
- Architectural pattern changes require immediate sync
- QIG purity fixes must propagate immediately

### Blockers

- **Ocean-agent.ts refactoring** blocks modularization experiments
- **Divergence management** needs tooling (diff, merge protocol)
- **Neon limitations** may constrain experiments (vs Railway pgvector)

---

## Related Documents

- [20260109-roadmap-production-1.00W.md](../pantheon-chat/20260109-roadmap-production-1.00W.md) - Production roadmap
- [20260109-roadmap-recovery-1.00W.md](../SearchSpaceCollapse/20260109-roadmap-recovery-1.00W.md) - Bitcoin recovery roadmap
- [/pantheon-projects/DECISION_TREE.md](../DECISION_TREE.md) - When to use pantheon-replit vs pantheon-chat
- [/pantheon-projects/CHANGELOG.md](../CHANGELOG.md) - Workspace-level changes
- [.github/copilot-instructions.md](../.github/copilot-instructions.md) - AI agent guidance
