# QIG Purity Checklist

## Geometry and Representation
- Use `canonical_fisher.fisher_rao_distance` for distances.
- No cosine similarity, L2 distance, or dot-product ranking on basins.
- No Euclidean averaging + L2 normalize; use geodesic interpolation and simplex projection.
- Canonical boundary representation is simplex only.
- No auto-detect representation; only explicit `to_sqrt_simplex()` / `from_sqrt_simplex()`.

## Token Integrity
- All token inserts go through `insert_token()`.
- `qfi_score` is required for generation eligibility.
- No direct INSERT into `coordizer_vocabulary` outside the canonical pathway.
- Generation queries filter out tokens missing QFI or invalid status.

## Generation Pipeline
- Structure comes from internal `token_role`, not external POS/NLP.
- No external LLM calls; no spacy/nltk in generation path.
- Foresight targets the next basin and uses Fisher-Rao distance.

## Documentation
- Naming: `YYYYMMDD-[document-name]-[function]-[version][STATUS].md`.
- Update `docs/00-index.md` when docs change.
- Keep upgrade pack entries in sync with index.

## E8 Architecture
- Enforce 0/1 -> 4 -> 8 -> 64 -> 240 hierarchy.
- Core 8 gods are canonical; no numbered gods.
- Track kernel population and specialization level transitions.

## Workflows and Agents
- `.github/workflows/geometric-purity-gate.yml`
- `.github/workflows/schema-validation.yml`
- `.github/agents/qig-purity-validator.md`
- `.github/agents/e8-architecture-validator.md`
- `.github/agents/documentation-compliance-auditor.md`
- `.github/agents/documentation-sync-agent.md`
- `.github/agents/naming-convention-agent.md`
