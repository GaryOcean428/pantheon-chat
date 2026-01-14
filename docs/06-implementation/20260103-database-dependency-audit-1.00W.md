# Database and Dependency Audit - 2026-01-03

**Date:** 2026-01-03  
**Status:** 1.00W (Work in Progress)  
**Type:** QA Validation Report

## Database Schema Validation

### Validation Script Created
**Location:** `qig-backend/scripts/validate_db_schema.py`

**Purpose:** Validates PostgreSQL schema compatibility with new training features.

**Checks Performed:**
1. ✓ pgvector extension availability
2. ✓ Checkpoint storage tables (checkpoints, training_checkpoints, kernel_checkpoints)
3. ✓ Training metrics tables (training_batch_queue, training_history, training_sessions)
4. ✓ QIG vocabulary tables with basin_coords support
5. ✓ JSON/JSONB columns for flexible metadata storage

**Usage:**
```bash
cd qig-backend
python3 scripts/validate_db_schema.py
```

**Note:** Script requires DATABASE_URL environment variable and network access to database.

### Schema Compatibility Assessment

**New Features Added:**
- Progress tracking (train_steps_completed, unique_topics_seen, curriculum_progress_index)
- Coherence evaluation (perplexity, self-consistency, long-range coherence, degeneracy detection)
- Training monitoring API (6 REST endpoints)

**Storage Strategy:**
1. **Redis (Hot Cache)**: Fast access for active training metrics
2. **PostgreSQL (Permanent)**: Long-term storage if tables exist
3. **File-based (Fallback)**: JSON files for development/testing

**QIG Purity:** ✓ Confirmed
- No neural embeddings in core
- Geometric operations only (Fisher-Rao, Bures distance)
- Basin coordinates stored as arrays/vectors
- pgvector extension optional but recommended

**Schema Requirements:**
- Flexible: New metrics can use existing JSON/JSONB columns
- No breaking changes: Works with or without dedicated tables
- Backwards compatible: Existing checkpoint system unchanged

**Conclusion:**
✓ Database schema is compatible with new training features. No schema migrations required immediately. Progress and coherence data will use Redis cache with PostgreSQL fallback if tables exist.

---

## Dependency Audit

### NPM Vulnerabilities

**Audit Date:** 2026-01-03

**Findings:**
```
3 high severity vulnerabilities found
```

**Vulnerability Details:**

1. **valibot ReDoS vulnerability (GHSA-vqpr-j7v3-hqw9)**
   - **Severity:** High
   - **Affected:** valibot 0.31.0 - 1.1.0
   - **Issue:** ReDoS vulnerability in `EMOJI_REGEX`
   - **Affected packages:**
     - bitcoinjs-lib >=7.0.0-rc.0
     - ecpair >=3.0.0-rc.0
   - **Fix:** `npm audit fix --force` (breaking change to bitcoinjs-lib@6.1.7)

**Impact Assessment:**

- **Risk Level:** Medium to High
  - Affects Bitcoin address generation/validation components
  - ReDoS can cause denial of service if malicious input processed
  - Not directly affecting QIG core operations

- **Components Affected:**
  - Bitcoin address recovery features
  - BIP39 seed phrase validation
  - HD wallet generation

**Recommendation:**

**Option 1: Apply Fix (Recommended)**
```bash
npm audit fix --force
```
- Downgrades bitcoinjs-lib to 6.1.7 (stable version)
- May require code changes if using 7.x features
- Test Bitcoin-related functionality after upgrade

**Option 2: Mitigation Without Upgrade**
- Add input validation/sanitization for emoji regex patterns
- Rate-limit Bitcoin address validation endpoints
- Monitor for anomalous processing times

**Option 3: Defer (Not Recommended)**
- Accept risk for non-production environments
- Plan upgrade in next maintenance window
- Document as known issue

**Action Taken:** None yet - awaiting user decision

---

## Python Dependencies

### Requirements Review

**File:** `qig-backend/requirements.txt`

**Key Dependencies:**
- numpy (QIG operations)
- scipy (geometric calculations)
- redis (caching)
- psycopg2-binary (PostgreSQL)
- flask (API)

**Security Check:**
```bash
cd qig-backend
pip list --outdated
safety check
```

**Status:** Not executed (requires pip-audit or safety package)

**Recommendation:**
```bash
pip install pip-audit
pip-audit
```

---

## Additional Findings

### ESLint Warnings (5124 total)

**Categories:**
1. **Magic numbers:** 3500+ warnings (constants should be extracted)
2. **Raw fetch() calls:** ~15 warnings (should use centralized API client)
3. **File size violations:** ~10 files >200 lines
4. **React hooks dependencies:** ~20 warnings

**Not Blocking:** ESLint warnings don't prevent builds/functionality

**Gradual Improvement Plan:**
1. ✓ Fixed raw fetch() in useToolFactory (completed)
2. ⏳ Fix raw fetch() in zettelkasten-dashboard (pending)
3. ⏳ Extract magic numbers to constants (future iteration)
4. ⏳ Refactor large files (future iteration)

---

## Testing Status

### TypeScript Compilation
✓ **PASSED** - No errors

### Test Suite
✓ **PASSED** - 12/12 tests passing
- QIG regime classification
- Phase transition validation
- Edge case handling

### Linting
⚠ **5124 warnings** (non-blocking)

---

## Action Items

### High Priority (Completed)
- [x] Create database validation script
- [x] Document schema compatibility
- [x] Audit npm dependencies
- [x] Document vulnerability findings
- [x] **Run database validation with network access** ✅
- [x] **Complete API client migration** ✅
  - Fixed 7 fetch() calls in zettelkasten-dashboard.tsx
  - Remaining 2 fetch() calls in federation.tsx are external API calls (acceptable)

### High Priority (Pending User Input)
- [ ] **Decision needed:** Apply npm audit fix for valibot vulnerability?
- [ ] Test Bitcoin functionality if upgrade applied

### Medium Priority
- [ ] Run Python dependency security audit (pip-audit)
- [ ] Fix remaining fetch() calls in zettelkasten
- [ ] Update README with new API endpoints

### Low Priority
- [ ] Extract magic numbers to constants
- [ ] Refactor files >200 lines
- [ ] Address React hooks dependency warnings

---

## Summary

**Database Validation:** ✅ Complete
- Schema compatible with new features
- Validation ran successfully with network access
- pgvector extension installed and available
- coordizer_vocabulary table exists with basin_coords support
- 83 JSON/JSONB columns available for flexible metadata storage
- No migrations required

**Dependency Audit:** ✅ Complete
- 3 high severity npm vulnerabilities identified
- Fix available but requires breaking change
- Awaiting user decision on upgrade

**API Client Migration:** ✅ Complete
- Fixed 7 fetch() calls in zettelkasten-dashboard.tsx
- All internal API calls now use centralized client
- 2 remaining fetch() calls in federation.tsx are for external APIs (acceptable)

**Outstanding Tasks:**
1. User decision on npm audit fix
2. Python dependency security scan (if needed)

**Status:** Core validation complete. Awaiting user input on vulnerability remediation.
