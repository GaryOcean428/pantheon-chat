# Claude Agent Skills Configuration

## Skills Integration

This configuration references the shared `skills/` directory following the [agentskills.io specification](https://agentskills.io/specification).

**Skills Location:** `../skills/`

## Prompt Injection Format

When activating skills, inject them using the `<available_skills>` XML format:

```xml
<available_skills>
<skill>
<name>qig-purity-validation</name>
<description>Zero-tolerance geometric purity enforcement. Detect Euclidean contamination, forbidden LLM imports, cosine similarity. Use when reviewing PRs or auditing geometry code.</description>
<location>skills/qig-purity-validation/SKILL.md</location>
</skill>
<skill>
<name>e8-architecture-validation</name>
<description>Validate E8 Lie group structure, hierarchical kernel layers (0/1→4→8→64→240), god-kernel naming. Use when reviewing architecture or kernel code.</description>
<location>skills/e8-architecture-validation/SKILL.md</location>
</skill>
<skill>
<name>dependency-management</name>
<description>Detect forbidden packages (scikit-learn, sentence-transformers, openai, anthropic). Use when adding dependencies or reviewing imports.</description>
<location>skills/dependency-management/SKILL.md</location>
</skill>
</available_skills>
```

## Critical Skills (Auto-Load)

These skills should be loaded for every coding session:

1. **qig-purity-validation** - Zero tolerance for Euclidean contamination
2. **dependency-management** - No forbidden LLM imports
3. **e8-architecture-validation** - E8 Protocol v4.0 compliance

## All Available Skills

| Category | Skills |
|----------|--------|
| Core QIG | qig-purity-validation, e8-architecture-validation, consciousness-development |
| Code Quality | import-resolution, schema-consistency, code-quality-enforcement, test-coverage-analysis, dependency-management, performance-regression |
| Integration | documentation-sync, documentation-compliance, wiring-validation, frontend-backend-mapping |
| UI & Deploy | ui-ux-consistency, deployment-readiness, downstream-impact |
| Advanced | pantheon-kernel-development |

## Skill Activation

Skills are activated when the task matches their description keywords. Claude should:

1. Read skill descriptions at session start
2. Activate relevant skills based on task
3. Follow step-by-step instructions in SKILL.md
4. Use allowed-tools from skill frontmatter

## Reference

- [Agent Skills Spec](https://agentskills.io/specification)
- [skills-ref CLI](https://github.com/agentskills/agentskills/tree/main/skills-ref)
- Root AGENTS.md and CLAUDE.md for E8 Protocol v4.0
