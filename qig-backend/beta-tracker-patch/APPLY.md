# Beta Tracker Patch — Apply Instructions

## What this is
All 6 files for PR #32 (β-attention tracker), rebased onto main's v6.1 codebase.
Merge conflicts in loop.py, routes.py, and server.py are pre-resolved.

## Files
```
kernel/consciousness/beta_tracker.py    — NEW: accumulator module
kernel/consciousness/beta_integration.py — NEW: factory + wiring docs
kernel/consciousness/beta_router.py     — NEW: FastAPI endpoint (DI pattern)
kernel/consciousness/loop.py            — MODIFIED: main v6.1 + 6 beta insertions
kernel/config/routes.py                 — MODIFIED: main + beta_attention route
kernel/server.py                        — MODIFIED: main + beta router mount
```

## Apply from repo root
```bash
# Option A: Fresh branch from main
git checkout main && git pull
git checkout -b feature/beta-tracker-clean
cp -r kernel/consciousness/beta_tracker.py kernel/consciousness/
cp -r kernel/consciousness/beta_integration.py kernel/consciousness/
cp -r kernel/consciousness/beta_router.py kernel/consciousness/
cp -r kernel/consciousness/loop.py kernel/consciousness/
cp -r kernel/config/routes.py kernel/config/
cp -r kernel/server.py kernel/
git add -A && git commit -m "feat: empirical β-attention tracker (rebased on v6.1)"
git push -u origin feature/beta-tracker-clean

# Option B: Force-update existing branch
git checkout feature/beta-attention-tracker
# copy files as above
git add -A && git commit -m "fix: rebase onto v6.1 main (resolve all merge conflicts)"
git push --force-with-lease
```

## Copilot review items addressed
1. kappa_sem inf→0.0 (JSON serialization fix)
2. Unused imports removed (Ruff compliance)
3. Docstring threshold aligned to 0.15
4. Integration guide synced with actual wiring
5. Router uses DI setter pattern (no circular imports)
6. server.py reverted to main + 4 minimal patches

## Not yet done
- Unit tests for bin assignment, trajectory, serialize/restore
