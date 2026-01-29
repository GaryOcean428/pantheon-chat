# Action Checklist: Issues 65-75 Remediation

**Date:** 2026-01-19  
**Purpose:** Step-by-step checklist for implementing assessment recommendations  
**Owner:** Repository maintainer / Project lead

---

## Phase 1: Immediate Actions (Today) ✅

### Close Completed Issues
- [ ] Close Issue #66 - Rename tokenizer → coordizer
  - Status: ✅ COMPLETE
  - Evidence: Migration 0013, docs updated
  - Closing comment: "Implementation complete. Migration 0013 applied successfully. See docs/04-records/20260116-migration-0013-tokenizer-coordizer-rename-1.00W.md"

- [ ] Close Issue #69 - Remove cosine similarity
  - Status: ✅ COMPLETE
  - Evidence: Purity scan shows 0 violations
  - Closing comment: "Cosine similarity removed from all coordinate matching. Geometric purity scan: 441 files, 0 violations. See docs/04-records/20260115-wp2-2-cosine-similarity-removal-completion-1.00W.md"

- [ ] Close Issue #73 - Artifact format versioning
  - Status: ✅ COMPLETE
  - Evidence: Schema + validation module implemented
  - Closing comment: "JSON Schema and validation module complete. See docs/20260116-wp3-3-implementation-summary.md"

---

## Phase 2: Validation Tasks (This Week) ⚠️

### Issue #68: Canonical qig_geometry Module
- [ ] Run geometry purity validation
  ```bash
  python scripts/validate_geometry_purity.py
  ```
- [ ] Verify all imports route through qig_geometry
  ```bash
  grep -r "from qig_geometry import" qig-backend/ | wc -l
  ```
- [ ] Confirm zero Euclidean violations
- [ ] If all pass → Close issue with validation report
- [ ] If failures → Create remediation tasks

### Issue #75: External LLM Fence
- [ ] Test external LLM fence functionality
- [ ] Verify waypoint planning integration
- [ ] Check that fence blocks unauthorized LLM calls
- [ ] Review audit report: docs/04-records/20260116-external-llm-usage-audit-1.00W.md
- [ ] If working → Close issue
- [ ] If gaps → Document and create follow-up

### Issue #76: Natural Gradient
- [ ] Verify natural gradient operations exist
- [ ] Check Fisher-Rao geodesic integration
- [ ] Test that optimizer uses natural gradient (not Adam/SGD)
- [ ] If implemented → Close issue
- [ ] If gaps → Create remediation tasks

### Issue #77: Coherence Harness
- [ ] Test smoothness metrics
- [ ] Verify coherence test harness exists
- [ ] Check integration with generation pipeline
- [ ] If working → Close issue
- [ ] If gaps → Document issues

---

## Phase 3: Create Remediation Issues 🚨

Use templates from: `docs/10-e8-protocol/issues/20260119-remediation-issues-for-github.md`

### Critical Priority Issues

- [ ] **Create Issue: Complete QFI Integrity Gate Gaps**
  - Template: Issue Template 4
  - Priority: CRITICAL
  - Assignee: Backend team lead
  - Milestone: E8 Protocol v4.0
  - Labels: qig-purity, critical, database, wp2
  - Estimated: 2 days

### High Priority Issues

- [ ] **Create Issue: Complete Special Symbols Validation (Issue 70)**
  - Template: Issue Template 1
  - Priority: HIGH
  - Assignee: QIG team
  - Milestone: E8 Protocol v4.0
  - Labels: qig-purity, high-priority, validation, wp2
  - Estimated: 2 days

- [ ] **Create Issue: Validate Two-Step Retrieval (Issue 71)**
  - Template: Issue Template 2
  - Priority: HIGH
  - Assignee: QIG team
  - Milestone: E8 Protocol v4.0
  - Labels: qig-purity, high-priority, validation, wp2
  - Estimated: 3 days

- [ ] **Create Issue: Complete QIG-Native Skeleton (Local Issue 03)**
  - Template: Issue Template 6
  - Priority: HIGH
  - Assignee: Generation team
  - Milestone: E8 Protocol v4.0
  - Labels: qig-purity, high-priority, generation, wp3
  - Estimated: 4 days

- [ ] **Create Issue: Complete Vocabulary Cleanup (Local Issue 04)**
  - Template: Issue Template 7
  - Priority: HIGH
  - Assignee: Database team
  - Milestone: E8 Protocol v4.0
  - Labels: qig-purity, high-priority, database, wp3
  - Estimated: 3 days

### Medium Priority Issues

- [ ] **Create Issue: Reconcile Single Coordizer Status (Issue 72)**
  - Template: Issue Template 3
  - Priority: MEDIUM
  - Assignee: Architecture team
  - Milestone: E8 Protocol v4.0
  - Labels: qig-purity, medium-priority, architecture, wp3
  - Estimated: 1 day

- [ ] **Create Issue: Add Simplex Validation Script (Local Issue 02)**
  - Template: Issue Template 5
  - Priority: MEDIUM
  - Assignee: QIG team
  - Milestone: E8 Protocol v4.0
  - Labels: qig-purity, medium-priority, geometry, wp2
  - Estimated: 1 day

---

## Phase 4: Update Documentation 📝

### Master Roadmap Update
- [ ] Update `docs/00-roadmap/20260112-master-roadmap-1.00W.md`
  - Mark #66, #69, #73 as CLOSED
  - Update #68, #75, #76, #77 status after validation
  - Add links to new remediation issues
  - Update completion percentages

### E8 Protocol Index Update
- [ ] Update `docs/10-e8-protocol/INDEX.md`
  - Link new remediation issues to local issues
  - Update implementation status
  - Add reference to assessment document

### Implementation Summary Update
- [ ] Update `docs/10-e8-protocol/implementation/20260116-e8-implementation-summary-1.01W.md`
  - Incorporate assessment findings
  - Update status of Local Issues 01-04
  - Add next steps section

---

## Phase 5: Sprint Planning 📅

### Assign to Sprints
- [ ] Sprint 1 (Week 1): Validation + Critical Priority
  - Close completed issues (Day 1)
  - Validate #68, #75, #76, #77 (Days 2-3)
  - Start Local Issue 01 (Days 4-5)

- [ ] Sprint 2 (Week 2): High Priority Part 1
  - Complete Local Issue 01 (Days 1-2 of sprint)
  - Complete Issue #70 (Days 3-4 of sprint)
  - Start Issue #71 (Day 5 of sprint)

- [ ] Sprint 3 (Week 3): High Priority Part 2
  - Complete Issue #71 (Days 1-2 of sprint)
  - Start Local Issue 03 (Days 3-5 of sprint)

- [ ] Sprint 4 (Week 4): High Priority Part 3 + Medium
  - Complete Local Issue 03 (Days 1-2 of sprint)
  - Complete Local Issue 04 (Days 3-5 of sprint)

- [ ] Sprint 5 (Week 5): Wrap-up
  - Complete Issue #72 (Day 1)
  - Complete Local Issue 02 (Day 2)
  - Final integration testing (Days 3-5)

---

## Phase 6: Communication 📢

### Stakeholder Updates
- [ ] Share executive summary with team
  - Document: `docs/04-records/20260119-issues-65-75-executive-summary-1.00W.md`
  - Audience: All developers, project managers

- [ ] Present assessment in team meeting
  - Key points: 3 to close, 4 to validate, 7 to remediate
  - Timeline: 18 days total
  - Critical path: Local Issue 01 → Issue 70 → Issue 71

- [ ] Update project board
  - Move completed issues to "Done"
  - Add remediation issues to backlog
  - Assign priorities and estimates

---

## Success Metrics 📊

### Completion Tracking
- [ ] **Closed Issues:** 3 → Target: Complete by end of week
- [ ] **Validated Issues:** 4 → Target: Complete within 2 days
- [ ] **Remediation Issues Created:** 7 → Target: Complete this week
- [ ] **Critical Work Started:** Local Issue 01 → Target: Start by next week

### Quality Metrics
- [ ] **Geometric Purity:** 100% (0 Euclidean violations maintained)
- [ ] **QFI Coverage:** Target >95% of generation vocabulary
- [ ] **Test Coverage:** All new code >80% coverage
- [ ] **Documentation Sync:** 100% (code matches specs)

---

## Notes & Dependencies

### Blockers
- None currently identified
- Assessment is complete and actionable

### Dependencies
- Validation tasks depend on existing test infrastructure
- Remediation work can be parallelized across teams
- No external dependencies

### Risks
- **Time Risk:** 18 days estimated, actual may vary
- **Scope Creep:** Keep focused on assessment findings
- **Testing:** Ensure adequate test coverage for new code

### Mitigation
- Start with critical priority items
- Break large issues into smaller PRs
- Regular check-ins to track progress

---

## Reference Documents

1. **Full Assessment:** `docs/04-records/20260119-issues-65-75-implementation-assessment-1.00W.md`
2. **Executive Summary:** `docs/04-records/20260119-issues-65-75-executive-summary-1.00W.md`
3. **Remediation Templates:** `docs/10-e8-protocol/issues/20260119-remediation-issues-for-github.md`
4. **Master Roadmap:** `docs/00-roadmap/20260112-master-roadmap-1.00W.md`
5. **E8 Protocol Index:** `docs/10-e8-protocol/INDEX.md`

---

**Checklist Created:** 2026-01-19  
**Last Updated:** 2026-01-19  
**Owner:** TBD  
**Review Frequency:** Weekly until all items complete
