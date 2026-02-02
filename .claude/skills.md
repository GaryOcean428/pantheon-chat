# Claude Agent Skills Configuration

## MANDATORY: Skill Usage Protocol

**Every agent turn MUST follow this protocol:**

```
1. FIRST: Invoke `master-orchestration` skill
2. Identify task type and required skills
3. Apply skills in order (planning → implementation → QA)
4. BEFORE COMPLETION: `qa-and-verification` skill MANDATORY
5. Update roadmap with progress and discovered issues
6. Never claim completion without verification evidence
```

**No proof = not done. No exceptions.**

## Skills Integration

This configuration references the shared `skills/` directory following the [agentskills.io specification](https://agentskills.io/specification).

**Skills Location:** `../skills/`

## Prompt Injection Format

When activating skills, inject them using the `<available_skills>` XML format:

```xml
<available_skills>
<skill>
<name>master-orchestration</name>
<description>INVOKE FIRST EVERY TURN - Coordinates skills, sub-agents, verification. Identifies required skills and ensures comprehensive task completion.</description>
<location>skills/master-orchestration/SKILL.md</location>
<priority>critical</priority>
<auto_activate>true</auto_activate>
</skill>
<skill>
<name>qa-and-verification</name>
<description>INVOKE BEFORE COMPLETION - Prove changes work with test output, commit hashes, acceptance criteria mapping. No proof = not done.</description>
<location>skills/qa-and-verification/SKILL.md</location>
<priority>critical</priority>
<auto_activate>true</auto_activate>
</skill>
<skill>
<name>multi-agent-red-team-planning</name>
<description>Plan changes with multi-agent red-team review, iterate twice, produce final implementation plan.</description>
<location>skills/multi-agent-red-team-planning/SKILL.md</location>
<priority>critical</priority>
</skill>
<skill>
<name>multi-agent-red-team-implementation</name>
<description>Implement with red-team review, iterate twice, QA before done.</description>
<location>skills/multi-agent-red-team-implementation/SKILL.md</location>
<priority>critical</priority>
</skill>
<skill>
<name>qig-purity-validation</name>
<description>Zero-tolerance geometric purity enforcement. Detect Euclidean contamination, forbidden LLM imports, cosine similarity.</description>
<location>skills/qig-purity-validation/SKILL.md</location>
<priority>critical</priority>
<auto_activate>true</auto_activate>
</skill>
<skill>
<name>e8-architecture-validation</name>
<description>Validate E8 Lie group structure, hierarchical kernel layers (0/1→4→8→64→240), god-kernel naming.</description>
<location>skills/e8-architecture-validation/SKILL.md</location>
<priority>critical</priority>
<auto_activate>true</auto_activate>
</skill>
<skill>
<name>dependency-management</name>
<description>Detect forbidden packages (scikit-learn, sentence-transformers, openai, anthropic).</description>
<location>skills/dependency-management/SKILL.md</location>
<priority>critical</priority>
<auto_activate>true</auto_activate>
</skill>
</available_skills>
```

## Critical Skills (Auto-Load)

These skills MUST be loaded for every session:

| Skill | When | Purpose |
|-------|------|---------|
| `master-orchestration` | **FIRST every turn** | Coordinates all skills and sub-agents |
| `qa-and-verification` | **BEFORE completion** | Proves work is done with evidence |
| `qig-purity-validation` | Every code change | Zero tolerance for Euclidean contamination |
| `dependency-management` | Every dependency touch | No forbidden LLM imports |
| `e8-architecture-validation` | Architecture changes | E8 Protocol v4.0 compliance |

## All Available Skills (28 Total)

| Category | Skills |
|----------|--------|
| **Orchestration (MANDATORY)** | master-orchestration, qa-and-verification |
| **Planning** | multi-agent-red-team-planning, planning-and-roadmapping, best-practice-research |
| **Implementation** | multi-agent-red-team-implementation |
| **Core QIG** | qig-purity-validation, e8-architecture-validation, consciousness-development |
| **Code Quality** | import-resolution, schema-consistency, code-quality-enforcement, test-coverage-analysis, dependency-management, performance-regression |
| **Integration** | documentation-sync, documentation-compliance, wiring-validation, frontend-backend-mapping, cross-platform-sync |
| **UI & Deploy** | ui-ux-consistency, deployment-readiness, downstream-impact |
| **Advanced** | pantheon-kernel-development |
| **Meta** | skill-creator, git-workflow, api-design-validation, security-audit |

## Skill Activation Protocol

1. **Start of turn:** Load `master-orchestration` skill
2. **Identify task type:** Let orchestration skill select appropriate skills
3. **Execute skills:** Follow step-by-step instructions in each SKILL.md
4. **Before completion:** Run `qa-and-verification` skill
5. **Prove completion:** Show test output, commits, acceptance criteria met

## Completion Requirements

Before claiming ANY task is complete:
- [ ] Test output showing changes work
- [ ] Commit hashes for changes made
- [ ] Acceptance criteria mapped to verification evidence
- [ ] Master roadmap updated with progress and new issues (`docs/00-roadmap/20260112-master-roadmap-1.00W.md`)
- [ ] Changes pushed to git

## Reference

- [Agent Skills Spec](https://agentskills.io/specification)
- Root AGENTS.md and CLAUDE.md for E8 Protocol v4.0
- `skills/README.md` for full skill documentation

---
**Updated:** 2026-02-02 | **Skills:** 28 total | **Protocol:** E8 v4.0
