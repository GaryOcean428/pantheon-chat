# External LLM API Guard

## Purpose

Enforce the project-wide prohibition on external LLM APIs in QIG core. Any usage in core modules is a hard stop.

## Rules (MANDATORY)

1. **No external LLMs in core**
   - Forbidden imports: `openai`, `anthropic`, `google.generativeai`
   - Applies to `qig-backend/`, `server/`, and `shared/`

2. **Docs-only references are allowed**
   - Examples in `docs/` are OK, but must be labeled as non-runtime.

3. **Exceptions require explicit approval**
   - Only allowed in non-core experiments with `QIG_PURITY_MODE=false`
   - Must be documented in `docs/00-index.md`

## Primary Validation Tool

```bash
npm run validate:critical
```

Alternative (verbose):

```bash
bash scripts/run-critical-enforcement.sh --strict --verbose
```

## Manual Scan

```bash
rg -n "import openai|from openai|import anthropic|from anthropic|google.generativeai" qig-backend server shared
```

## Failure Actions

If any violation is found:
1. **Block merge** immediately
2. **Report** file:line and offending import
3. **Replace** with internal QIG generation (`qig-backend/qig_generation.py`)

## Reference

- `docs/10-e8-protocol/specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- `qig-backend/qig_generation.py`
