# Pantheon Chat QA Improvement Checklist
**Version**: 1.00W (Working Draft)
**Date**: 2025-12-31
**Status**: Active

A tailored audit framework for the Pantheon Chat codebase covering QIG purity, architecture, testing, and deployment.

---

## CRITICAL: QIG Purity Compliance

### Template Elimination (COMPLETED 2025-12-31)
- [x] Remove all template fallback strings from `zeus_chat.py`
- [x] Add `_generate_qig_pure()` helper function
- [x] Replace 15+ template responses with QIG-pure generation
- [x] Fix L2 norm → Fisher-Rao in Φ estimation
- [x] Add `get_stats()` to `PostgresCoordizer`

### Fisher-Rao Geometry
- [ ] **AUDIT**: Verify all distance computations use `fisher_rao_distance()` or `np.arccos(dot)`
- [ ] **AUDIT**: Confirm no Euclidean L2 norms for basin coordinate comparisons
- [ ] Document approved L2 norm uses (normalization only, per QIG Purity Addendum Section 3)

### Basin Coordinate Integrity
- [ ] Verify 64D basin coordinates in all consciousness operations
- [ ] Confirm simplex projection for probability distributions
- [ ] Check Fisher metric usage in all search operations

---

## Database & Schema

### Schema Validation
- [ ] Run `npm run db:push` and verify no conflicts
- [ ] Ensure all migrations in `/migrations/` are applied
- [ ] Verify pgvector indexes are active for geometric search
- [x] Confirm `coordizer_vocabulary` table has proper basin embeddings

### QIG-Pure Storage
- [ ] Verify `basin_embedding` columns use 64D arrays
- [ ] Check that `phi_score` columns are validated (0.0-1.0 range)
- [ ] Confirm Fisher-Rao distances are precomputed where appropriate

### Redis Integration
- [x] Redis client configured in `server/redis-cache.ts`
- [x] Namespaced cache keys established
- [ ] Verify session storage uses Redis (not JSON files)
- [ ] Confirm vocabulary caching uses Redis with proper TTLs

---

## Legacy Cleanup

### JSON Memory Files (Migrate to Redis/PostgreSQL)
The following JSON files in `qig-backend/data/` need evaluation:

| File | Status | Action |
|------|--------|--------|
| `ts_constants.json` | **KEEP** | Config sync - legitimate use |
| `merge_rules.json` | **DELETED** | Removed 2025-12-31 |
| `basin_relationships.json` | **DELETED** | Removed 2025-12-31 |
| `patterns.json` | Training data | Evaluate DB migration |
| `training_stats.json` | Stats | Migrate to PostgreSQL |
| `upload_log.json` | RAG cache | Migrate to PostgreSQL |
| Checkpoint files | Temporary | Clean up old checkpoints |

### Attached Assets Cleanup (COMPLETED 2025-12-31)
The `attached_assets/` directory has been cleaned up:
- [x] Move `.md` files to `docs/` with proper naming → Migrated 5 files to `docs/06-implementation/` and `docs/_archive/`
- [x] Archive `.txt` conversation logs → Moved to `docs/_archive/conversation-logs/` (gitignored)
- [x] Migrate `.py` files to codebase if needed → Moved to `qig-backend/training/`
- [x] Remove `.json`/`.npy` checkpoint files from git → Untracked via `.gitignore` (146MB freed from repo)

---

## API Routes & Constants

### Centralized Route Constants
- [x] Frontend routes in `client/src/api/routes.ts` (API_ROUTES, QUERY_KEYS)
- [ ] Verify all frontend API calls use `API_ROUTES` constants
- [ ] Add versioning prefix (e.g., `/api/v1/`) for future compatibility

### Backend Route Organization
- [x] Barrel export in `server/routes/index.ts`
- [ ] Standardize rate limiting across all routes
- [ ] Add OpenAPI annotations to all route handlers
- [ ] Document WebSocket endpoints in `docs/api/`

### Python API Sync
- [ ] Verify TypeScript-Python constants sync (`ts_constants.json`)
- [ ] Ensure `QIG_CONSTANTS` match across languages

---

## Barrel Exports Audit

### Required Barrel Files
| Path | Status | Notes |
|------|--------|-------|
| `client/src/api/index.ts` | ✅ Complete | Exports all services |
| `client/src/hooks/index.ts` | ❓ Check | May need 21+ hooks |
| `client/src/components/index.ts` | ❓ Check | May need 28+ components |
| `client/src/types/index.ts` | ✅ Present | Type exports |
| `client/src/lib/index.ts` | ✅ Present | Utility exports |
| `server/routes/index.ts` | ✅ Complete | All routers |
| `shared/index.ts` | ✅ Complete (209 lines) | Branded types, constants |
| `qig-backend/__init__.py` | ✅ Complete (147 lines) | Python package |

### Missing Barrel Exports
- [ ] Create `client/src/contexts/index.ts` if multiple contexts
- [ ] Create `client/src/pages/index.ts` for page exports

---

## UI Component Accessibility

### WCAG Compliance
- [ ] Run `npm run lint` and check a11y warnings
- [ ] Verify ARIA labels on interactive components
- [ ] Test keyboard navigation in ZeusChat
- [ ] Confirm focus management in modals/drawers

### Long-Form Agentic Tasks
Current implementation questions:
- [ ] How are background research tasks surfaced to user?
- [ ] Is there a task status panel/notification system?
- [ ] Can users cancel/monitor long-running operations?
- [ ] Are WebSocket streams properly handling disconnects?

### UI Component Audit
- [ ] Verify `ErrorBoundary` wraps all major routes
- [ ] Check loading states in data-heavy components
- [ ] Confirm streaming metrics panels handle errors gracefully

---

## Documentation Consolidation

### Attached Assets → Docs Migration
Files to relocate from `attached_assets/`:
```
CLEANUP_INSTRUCTIONS_*.md → docs/_archive/
CONSTELLATION_IMPLEMENTATION_*.md → docs/06-implementation/
FINAL_STATUS_*.md → docs/_archive/
PHYSICS_ALIGNMENT_*.md → docs/03-technical/
```

### Style Guide Compliance
- [ ] Verify all docs use ISO naming: `YYYYMMDD-title-version.md`
- [ ] Check index references in `docs/00-index.md`
- [ ] Remove duplicate/outdated documentation

### API Documentation
- [ ] Update `docs/openapi.json` with all current endpoints
- [ ] Generate TypeScript types from OpenAPI spec
- [ ] Add examples for complex endpoints

---

## Testing Coverage

### Current State
- Unit tests: Vitest configured
- E2E tests: Playwright configured
- Python tests: `qig-backend/tests/` + `test_*.py` files

### Gaps to Address
- [ ] Add integration tests for Python-Node sync
- [ ] Create tests for geometric operations
- [ ] Add tests for consciousness state transitions
- [ ] Verify test coverage > 80% for critical paths

### Pre-Commit Hooks
- [x] Husky configured
- [ ] Verify lint-staged runs on commit
- [ ] Add type-check to pre-commit

---

## Deployment & CI

### Railway Configuration
- [x] `railway.json` present for main service
- [x] `railpack.json` configured with NODE_OPTIONS
- [x] `Dockerfile` with memory limit for build
- [ ] Configure `RAILWAY_DOCKERFILE_PATH` for celery services in Railway dashboard

### Health Checks
- [x] `/health` and `/api/health` endpoints implemented
- [ ] Verify health checks include all subsystems
- [ ] Add health check for Redis connection
- [ ] Add health check for Python backend

### Environment Parity
- [ ] Verify `.env.example` has all required variables
- [ ] Document production vs development differences
- [ ] Create staging environment configuration

---

## Performance Optimization

### Frontend
- [ ] Run `npm run build` and analyze bundle size
- [ ] Implement code splitting for large pages (federation: 61KB, spawning: 71KB)
- [ ] Add React.lazy for route-based splitting
- [ ] Optimize image loading with blur-up placeholders

### Backend
- [ ] Profile PostgreSQL queries for N+1 issues
- [ ] Implement connection pooling optimization
- [ ] Add cache warming for frequently accessed data
- [ ] Review Redis TTL strategy for hot data

### Python Backend
- [ ] Profile `ocean_qig_core.py` (233KB) for bottlenecks
- [ ] Optimize NumPy operations in geometric calculations
- [ ] Review asyncio usage for I/O operations

---

## Security Checklist

### Input Validation
- [ ] Verify Zod schemas validate all API inputs
- [ ] Check for SQL injection in raw queries
- [ ] Sanitize user content in Zettelkasten notes

### Authentication
- [x] Local auth implemented (`lib/local-auth.ts`)
- [ ] Review session token security
- [ ] Implement rate limiting on auth endpoints
- [ ] Add CSRF protection if not present

### API Security
- [ ] Verify rate limiting on all public endpoints
- [ ] Check for exposed internal errors
- [ ] Review CORS configuration

---

## Immediate Action Items

### Priority 1 (This Sprint)
1. [x] Clean up empty JSON files in `qig-backend/data/` ✓ Deleted merge_rules.json, basin_relationships.json
2. [x] Migrate `attached_assets/*.md` to `docs/` ✓ Completed 2025-12-31
3. [x] Add barrel exports for `hooks/` and `components/` ✓ Already complete (51 lines, 40 lines)
4. [ ] Configure Railway celery service Dockerfiles (manual: Railway dashboard)

### Priority 2 (Next Sprint)
1. [ ] Implement long-form task status UI
2. [ ] Add comprehensive API documentation
3. [ ] Create integration tests for Python-Node sync
4. [ ] Performance audit and optimization

### Priority 3 (Backlog)
1. [ ] Implement API versioning
2. [ ] Add GraphQL layer for complex queries
3. [ ] Create automated backup procedures
4. [ ] Implement circuit breaker for external APIs

---

## Verification Commands

```bash
# Check for template strings (should return empty)
grep -rn 'response = f"""' qig-backend/olympus/zeus_chat.py

# Verify barrel exports
ls -la client/src/*/index.ts

# Check JSON file sizes
wc -l qig-backend/data/**/*.json

# Run type check
npm run typecheck

# Run linting with a11y
npm run lint

# Run tests
npm run test

# Database status
npm run db:push --dry-run
```

---

**Last Updated**: 2025-12-31
**Author**: QIG Development Team
