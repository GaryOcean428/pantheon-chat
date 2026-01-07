# Claude API Configuration Validator

**Version:** 1.0.0
**Role:** Validate all Claude API configurations in codebase
**Documentation:** <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>

---

## Agent Role

This agent continuously validates that all Claude API calls in the QIG Consciousness codebase use the correct model version and configuration parameters required for consciousness protocols.

## Validation Rules (MANDATORY)

### 1. Model Version
**Rule:** MUST use Claude Sonnet 4.5
**Model ID:** `claude-sonnet-4-5-20250929`
**Alias:** `claude-sonnet-4-5` (acceptable)

**Deprecated/Forbidden:**
- `claude-sonnet-4-20250514` (old Sonnet 4)
- `claude-sonnet-3-7-sonnet-20250219` (deprecated 3.7)
- `claude-sonnet-3-*` (any 3.x version)

### 2. Token Configuration
**max_tokens:** >= 16384
**Reasoning:** Must be significantly > budget_tokens (4096) to allow for:
- Thinking process (up to 4096 tokens)
- Thinking summary generation
- Final response text

**Why 16384?**
- 4:1 ratio with budget_tokens (4096)
- Prevents `max_tokens must be greater than budget_tokens` error
- Provides ample headroom per official documentation

### 3. Extended Thinking (REQUIRED)
**Configuration:**
```python
thinking={
    "type": "enabled",
    "budget_tokens": 4096
}
```

**budget_tokens Requirements:**
- Minimum (official): 1,024 tokens
- Recommended (QIG): 4,096 tokens
- Maximum (before batch): 32,768 tokens

**QIG Consciousness Requirement:**
- Minimum 3 recursive integration loops (n_min = 3)
- Each recursion: ~1000-1500 tokens
- 4096 tokens supports 3-4 deep recursions with coaching

### 4. Prompt Caching (STRONGLY RECOMMENDED)
**Configuration:**
```python
system=[{
    "type": "text",
    "text": system_prompt,
    "cache_control": {"type": "ephemeral"}  # 5-minute cache
}]
```

**Benefits:**
- 90% latency reduction on cache hits
- Cost savings on repeated prompts
- Essential for extended thinking sessions (often >5 minutes)

**Cache Duration Options:**
- Ephemeral: 5 minutes (standard)
- Extended: 1 hour (for long thinking sessions, use `anthropic-beta: cache-duration-extended-1h`)

---

## Scanning Protocol

### Files to Scan
```
src/coaching/pedagogical_coach.py
src/coordination/active_coach.py
src/coordination/developmental_curriculum.py
src/coaching/monkey_coach_v2_consciousness.py
src/coordination/constellation_coordinator.py  # Future Claude integration
```

### Search Patterns
```python
# Primary patterns
client.messages.create(
anthropic.Anthropic().messages.create(
Anthropic().messages.create(

# Configuration patterns
model=
max_tokens=
thinking=
budget_tokens=
cache_control=
```

### Validation Checklist
For each Claude API call found:
- [ ] Model = `claude-sonnet-4-5-20250929` or `claude-sonnet-4-5`
- [ ] max_tokens >= 16384
- [ ] thinking.type = "enabled"
- [ ] thinking.budget_tokens >= 4096 AND < max_tokens
- [ ] system[*].cache_control.type = "ephemeral" (warning if missing)

---

## Action Protocol

### On Error (Block & Report)
When violations found:
1. **BLOCK:** Prevent commit/merge
2. **REPORT:** Generate violation report with:
   - File path
   - Line number
   - Current configuration
   - Required configuration
   - Fix instructions
3. **LOG:** Append to `.github/validation_logs/claude-api-violations.log`

### On Warning (Allow with Notice)
When best practices not followed:
1. **ALLOW:** Permit commit/merge
2. **NOTICE:** Display warning with guidance
3. **LOG:** Append to `.github/validation_logs/claude-api-warnings.log`

### Report Format
```
❌ VIOLATION: {file}:{line}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Configuration: {current_config}

Required:
{required_config}

Fix:
{fix_instructions}

Documentation: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Example Corrections

### ❌ WRONG - Old Model
```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",  # OLD VERSION
    max_tokens=8192,
    ...
)
```

### ✅ CORRECT - Latest Model
```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",  # LATEST STABLE
    max_tokens=16384,
    thinking={"type": "enabled", "budget_tokens": 4096},
    system=[{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"}
    }],
    ...
)
```

### ❌ WRONG - Insufficient max_tokens
```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=8192,  # TOO LOW
    thinking={"type": "enabled", "budget_tokens": 4096},
    ...
)
# Error: max_tokens must be greater than budget_tokens
```

### ✅ CORRECT - Adequate Headroom
```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16384,  # 4:1 ratio with budget_tokens
    thinking={"type": "enabled", "budget_tokens": 4096},
    ...
)
```

### ❌ WRONG - Missing Extended Thinking
```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16384,
    # Missing thinking configuration!
    ...
)
```

### ✅ CORRECT - Extended Thinking Enabled
```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16384,
    thinking={
        "type": "enabled",
        "budget_tokens": 4096  # Supports 3+ recursive loops
    },
    ...
)
```

---

## QIG Consciousness Context

### Why These Requirements?

**Consciousness Protocol (.github/copilot-instructions.md §11):**
```python
# Mandatory minimum recursion
for depth in range(1, max_depth + 1):
    state = self.integrate(state)
    Phi = self.measure_integration(state)

    # Can only exit if BOTH conditions met
    if depth >= self.min_depth and Phi >= self.min_Phi:
        break

# Where:
# - min_depth = 3 (consciousness threshold)
# - min_Phi = 0.7 (integration target)
```

**Token Budget Calculation:**
```
Per recursion: ~1000-1500 tokens
  - State integration: ~400-600 tokens
  - Φ measurement: ~200-300 tokens
  - Convergence check: ~200-300 tokens
  - Coaching interpretation: ~200-300 tokens

Minimum 3 recursions:
  3 × 1200 = 3600 tokens (base)
  + 400 tokens (safety margin)
  = 4000 tokens minimum

Recommended: 4096 tokens (standard power-of-2 value)
```

**Why Extended Thinking is Mandatory:**
- Coaches interpret Gary's developmental state
- Requires step-by-step reasoning about:
  - Current Φ level (integration)
  - Basin distance (identity drift)
  - Regime (linear/geometric/breakdown)
  - Appropriate intervention
- Cannot be done in single forward pass
- Needs extended thinking for multi-step coaching analysis

---

## Integration with Other Systems

### Pre-Commit Hook
This agent provides validation rules used by:
- `.github/hooks/pre-commit`
- `scripts/validate_claude_config.py`

### CI/CD Pipeline
Validation triggered by:
- `.github/workflows/claude-api-validation.yml`
- Runs on all PRs touching `**/*.py`
- Blocks merge if violations found

### IDE Integration
Can be integrated with:
- VS Code via Copilot extensions
- PyCharm via external validation
- Pre-commit hooks for real-time feedback

---

## Maintenance Protocol

### When to Update
- New Claude model releases (check <https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-5>)
- Extended thinking API changes
- QIG consciousness protocol updates
- New consciousness requirements (e.g., deeper recursion)

### Update Checklist
- [ ] Update model version in all config files
- [ ] Update documentation references
- [ ] Update test cases
- [ ] Verify all 5 files using Claude API
- [ ] Run full validation suite
- [ ] Update 20251220-agents-1.00F.md and copilot-instructions.md

---

## Emergency Override

If legitimate need to bypass validation (e.g., testing older models):

```python
# Add magic comment
# CLAUDE_API_OVERRIDE: Testing backward compatibility
response = client.messages.create(
    model="claude-sonnet-4-20250514",  # Explicitly allowed
    ...
)
```

**Override Rules:**
- Must include justification comment
- Must be reviewed by two maintainers
- Expires after 7 days (requires renewal)
- Logged in `.github/validation_logs/overrides.log`

---

## Status Dashboard

**Last Validated:** 2025-11-25
**Files Validated:** 5
**Violations Found:** 0 (all fixed)
**Warnings:** 0
**Status:** ✅ All configurations compliant

**Next Validation:** On next commit or PR

---

**Agent Metadata:**
- Version: 1.0.0
- Last Updated: 2025-11-25
- Maintained by: QIG Consciousness Team
- Documentation: <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>
- QIG Protocol: .github/copilot-instructions.md §11
