# CRITICAL: Restore qig_generative_service.py

## What happened
Commit `186fcfb` accidentally replaced the production `qig_generative_service.py`
(92KB, 2063 lines) with a simplified stub (27KB, 766 lines).

The original has Plan→Realize→Repair, StreamingCollapseMonitor, CoherenceTracker,
BasinVelocityMonitor, SelfObserver, DiagonalFisherNG, POS grammar, etc.

## Restoration commands (run in repo root)

```bash
# Step 1: Restore original from git history
git checkout 7e2b6772c196f0df443bc9afc37dc17ae6555d9 -- qig-backend/qig_generative_service.py

# Step 2: Apply the proxy_routed / proxy_kernels patch
git apply docs/04-records/qgs_proxy_restoration.patch

# Step 3: Verify AST clean
python3 -c "import ast; ast.parse(open('qig-backend/qig_generative_service.py').read()); print('OK')"

# Step 4: Commit
git add qig-backend/qig_generative_service.py
git commit -m 'fix: restore production qig_generative_service.py + add proxy_routed/proxy_kernels (TCP v6.1)'
git push
```

## Verification
After restore, file should:
- Be ~92KB / 2063 lines
- Contain `PRR_AVAILABLE`, `StreamingCollapseMonitor`, `BasinTrajectoryIntegrator`
- Contain `proxy_routed: bool = False` in `GenerationResult`
- Contain `proxy_routed=_proxy_routed` in `GenerationResult(...)` constructor

```bash
python3 -c "
with open('qig-backend/qig_generative_service.py') as f: c = f.read()
print('size OK:', len(c) > 85000)
print('PRR:', 'PRR_AVAILABLE' in c)
print('StreamingCollapse:', 'StreamingCollapseMonitor' in c)
print('proxy field:', 'proxy_routed: bool = False' in c)
print('proxy ctor:', 'proxy_routed=_proxy_routed' in c)
"
```
