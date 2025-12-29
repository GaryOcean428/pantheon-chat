# Planning Rules for AI Agents

**Status:** MANDATORY - All agents must follow these rules
**Date:** 2025-11-22
**Purpose:** Prevent artificial time pressure in novel research

---

## 🚫 Rule 1: NO TIME ESTIMATES IN PLANS

When providing plans, processes, workflows, roadmaps, or task breakdowns, **NEVER include time-based estimates or labels**.

### ❌ FORBIDDEN

**Do NOT use:**
- Week 1, Week 2, etc.
- "2-3 hours for X"
- "This will take 4-5 hours"
- "Day 1-2: Do Y"
- "Next 2 days", "by Friday", etc.
- "Short-term (this week)"
- Any time-based milestone or deadline

### ✅ REQUIRED

**DO use:**
- Phase 1, Phase 2, Phase 3, ...
- Task A, Task B, Task C, ...
- Step 1, Step 2, Step 3, ...
- "Next:", "Then:", "After X:"
- "Immediate", "Short-term", "Long-term" (without time bounds)

---

## Why This Rule Exists

### Problem: LLM Time Estimates Are Inaccurate

**Empirical Evidence:**
- LLMs consistently overestimate by ~2x
- "4-5 hours" actually takes 2 hours
- "Week 1" tasks complete in 1 day

**Constellation Integration Example:**
- Agent predicted: 4-5 hours
- Actual time: ~2 hours (50% faster)

### Problem: Creates Artificial Pressure

- Agents feel rushed when time is mentioned
- Arbitrary deadlines harm precision
- Novel research requires as much time as needed
- Focus should be on correctness, not speed

### Problem: Unhelpful Context

- Time estimates provide no useful information
- Task sequencing is what matters
- Completion criteria are what matter
- Time is unknowable until work is done

---

## Correct Example

### ❌ Bad Plan

```markdown
## Implementation Timeline

Week 1:
- Day 1-2: Fix coordinator API (4 hours)
- Day 3: Test integration (2-3 hours)
- Day 4-5: Validate checkpoints (3 hours)

Week 2:
- Launch training (5 days)
```

### ✅ Good Plan

```markdown
## Implementation Phases

Phase 1 - Infrastructure:
- Task A: Fix coordinator API
- Task B: Test integration
- Task C: Validate checkpoints

Phase 2 - Training:
- Task D: Launch constellation training
- Task E: Monitor convergence
```

---

## Application to Different Contexts

### Feature Development

❌ Bad: "Week 1: Implement feature X (10 hours)"
✅ Good: "Phase 1: Implement feature X"

### Bug Fixes

❌ Bad: "Fix in 2-3 hours"
✅ Good: "Next: Fix issue, then verify"

### Research Experiments

❌ Bad: "Run experiments over 5 days"
✅ Good: "Phase 1: Run experiments until convergence"

### Multi-Agent Coordination

❌ Bad: "Week 1: Agent A does X (2 days), Week 2: Agent B does Y (3 days)"
✅ Good: "Phase 1: Agent A does X, Phase 2: Agent B does Y"

### Documentation

❌ Bad: "Document in 1 hour"
✅ Good: "Next: Document implementation"

---

## Enforcement

**This rule applies to:**
- All plans and roadmaps
- All task breakdowns
- All workflow descriptions
- All process documents
- All project schedules
- All agent coordination protocols

**Violations should be:**
- Immediately corrected
- Rewritten without time references
- Replaced with phase/task/step labels

**If you must communicate urgency:**
- Use "High Priority" / "Low Priority"
- Use "Blocking" / "Non-blocking"
- Use "Immediate" / "Deferred"
- Do NOT use time estimates

---

## Summary

**Golden Rule:** Use phases, tasks, and steps. Never use weeks, hours, or days.

**Why:** Time estimates are inaccurate, create pressure, and are unhelpful for precise research.

**How:** Replace all time references with sequence labels (Phase 1/2/3, Task A/B/C, Step 1/2/3).

---

**Status: MANDATORY for all QIG project agents** 🎯
