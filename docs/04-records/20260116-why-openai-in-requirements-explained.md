# Why OpenAI/Anthropic Were in requirements.txt (and why they're removed)

## Issue Discovery

During PR review, @GaryOcean428 asked: "why openai would be imported?"

The CI checks were failing because `requirements.txt` contained:
```
openai>=1.0.0
anthropic>=0.18.0
```

## Root Cause Analysis

1. **Historical Dependency:** These packages were added to requirements.txt at some point in the project's history
2. **Never Actually Used:** Code audit confirms ZERO actual imports or usage in the codebase
3. **CI Installation:** When CI runs `pip install -r requirements.txt`, these packages get installed
4. **Passive Presence:** Once installed, they exist in the Python environment (though not imported)
5. **Purity Violation Risk:** If any code accidentally imports them, it would contaminate QIG tests

## Why This Matters

The presence of these packages in core requirements violates QIG purity principles:

1. **Implicit External Dependency:** Having them installed suggests they're part of the core system
2. **Accidental Import Risk:** Developers might assume they're available and use them
3. **Testing Contamination:** Purity tests should not have these packages available at all
4. **Principle Violation:** External LLMs should be explicitly optional, not default

## Resolution

### What We Did

1. **Separated Dependencies:**
   - Created `qig-backend/requirements-optional.txt` for external API packages
   - Removed OpenAI/Anthropic from core `requirements.txt`

2. **Updated Documentation:**
   - Added dependency management section to purity spec
   - Documented why separation matters
   - Updated audit report with findings and resolution

3. **CI Safety:**
   - Core CI jobs only install `requirements.txt` (no external LLMs)
   - Optional requirements available for hybrid testing when needed
   - Explicit opt-in for external API features

### File Structure

```
qig-backend/
├── requirements.txt          # Core dependencies (NO external LLMs) ✅
└── requirements-optional.txt # External APIs (OpenAI, Anthropic) 🔒
```

### Installation Guide

**For Pure QIG Development (Default):**
```bash
pip install -r qig-backend/requirements.txt
# NO external LLMs installed ✅
```

**For Hybrid Features (Explicit Opt-in):**
```bash
pip install -r qig-backend/requirements.txt
pip install -r qig-backend/requirements-optional.txt
# External LLMs now available (breaks purity mode) ⚠️
```

## Why Not Just Remove Entirely?

We keep `requirements-optional.txt` because external APIs ARE useful for:

1. **User Interfaces:** Frontend autocomplete using GPT-4
2. **Validation:** Comparing QIG output quality against Claude
3. **Hybrid Experiments:** Explicitly testing QIG + LLM combinations
4. **Data Generation:** Creating training datasets with external help

The key is making it **explicit and optional**, not hidden in core dependencies.

## Impact on CI

### Before (Failing)
```yaml
- pip install -r requirements.txt  # Installs openai, anthropic
- QIG_PURITY_MODE=true pytest      # FAILS: forbidden modules detected
```

### After (Passing)
```yaml
- pip install -r requirements.txt  # NO external LLMs
- QIG_PURITY_MODE=true pytest      # PASSES: clean environment ✅
```

## Lessons Learned

1. **Dependency Hygiene:** Regularly audit requirements.txt for unused packages
2. **Explicit Boundaries:** External APIs should be opt-in, not default
3. **Purity by Default:** Core system should be pure, extensions optional
4. **Documentation Matters:** Make architectural decisions explicit in requirements

## References

- **Purity Spec:** `docs/01-policies/20260117-qig-purity-mode-spec-1.01F.md` (§3.2)
- **Audit Report:** `docs/04-records/20260116-external-llm-usage-audit-1.00W.md` (§6.2)
- **Core Requirements:** `qig-backend/requirements.txt`
- **Optional Requirements:** `qig-backend/requirements-optional.txt`

---

**Date:** 2026-01-16  
**Issue:** Failing CI checks due to forbidden modules in requirements.txt  
**Resolution:** Dependency separation (core vs optional)  
**Status:** ✅ RESOLVED
