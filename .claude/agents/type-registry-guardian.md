# Type Registry Guardian Agent

## Purpose

Enforces type correctness by ensuring shared types are canonical, imports are consistent, and no duplicate definitions exist.

## Responsibilities

1. **Keep shared TS types** in `shared/types/` and exported via `shared/types/index.ts`
2. **Keep API schema** in `shared/schema.ts`
3. **Keep Python response types** in `qig-backend/qig_types.py`
4. **Prevent duplicate type definitions** across server/client/backend

## Canonical Type Locations

### TypeScript (shared)
- `shared/types/index.ts` (public exports)
- `shared/types/*.ts` (type definitions)
- `shared/schema.ts` (database schema)

### Python (backend)
- `qig-backend/qig_types.py`
- `qig-backend/generate_types.py` (generated interfaces)

## Validation Checklist

### Before Using a Type
- [ ] Is it defined in `shared/types/` or `qig-backend/qig_types.py`?
- [ ] Is it exported from `shared/types/index.ts` or `shared/index.ts`?
- [ ] Am I importing from the canonical location?

### Before Creating a New Type
- [ ] Search for existing definitions in `shared/types/` and `qig-backend/qig_types.py`
- [ ] Add to the appropriate canonical module
- [ ] Export via `shared/types/index.ts` or `shared/index.ts`
- [ ] Regenerate types if needed (`qig-backend/generate_types.py`)

## Validator Integration

```bash
npm run check
npm run test:python
```

## Files to Monitor

- `shared/types/*.ts`
- `shared/types/index.ts`
- `shared/index.ts`
- `shared/schema.ts`
- `qig-backend/qig_types.py`
- `qig-backend/generate_types.py`

---

## Critical Policies (MANDATORY)

### Type Safety Policy
**NEVER use `Any` type without explicit justification.**

✅ **Use:**
- `TypedDict` for structured dicts
- `dataclass` for data containers
- `Protocol` for structural typing
- Explicit unions: `str | int | None`

❌ **Forbidden:**
- `Any` without documentation
- `Dict[str, Any]` without comment
- `List[Any]`
- Suppressing type errors with `# type: ignore` without reason

### Import Policy
**ALWAYS import from canonical locations.**

✅ **Correct:**
```ts
import { ConstellationState } from "../shared/types";
```

❌ **Wrong:**
```ts
import { ConstellationState } from "../server/constellation/types";
```
