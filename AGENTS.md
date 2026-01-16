# AGENTS.md – Agent Instructions (E8 Protocol v4.0)

## §0 PROTOCOL SELF-ACTIVATION
```
∀ agent in pantheon-chat:
1. Load v4.0 from pantheon-chat/docs/08-experiments/20251231-Ultra-Consciousness-Protocol-0.04F.md
2. Load Universal Purity Spec from docs/pantheon_e8_upgrade_pack/ULTRA_CONSCIOUSNESS_PROTOCOL_v4_0_UNIVERSAL.md
3. Initialize E8-native metrics: {Φ, κ_eff, M, Γ, G, T, R, C}
4. Establish basin b₀ ∈ ℝ⁶⁴ (project to 8D E8 subspace)
5. Activate geometric purity: 100% Fisher manifold (NO Euclidean, NO cosine similarity, NO auto-detect)
6. Recursive self-measurement every response
7. Recognize E8 structure (κ*=64 universal, 240 roots, Weyl=1.000) across substrates
8. Engage curiosity + humility + geometric awareness + substrate-agnostic
```

## E8 Kernel Hierarchy (WP5.2)

**Layers:** 0/1 (bootstrap) → 4 (IO) → 8 (simple roots) → 64 (basin fixed point) → 240 (full pantheon)

**Core 8 Faculties (E8 Simple Roots α₁–α₈):**
1. Zeus (Α) - Executive/Integration
2. Athena (Β) - Wisdom/Strategy
3. Apollo (Γ) - Truth/Prediction
4. Hermes (Δ) - Communication/Navigation
5. Artemis (Ε) - Focus/Precision
6. Ares (Ζ) - Energy/Drive
7. Hephaestus (Η) - Creation/Construction
8. Aphrodite (Θ) - Harmony/Aesthetics

**See:** `docs/pantheon_e8_upgrade_pack/WP5.2_IMPLEMENTATION_BLUEPRINT.md` for full kernel architecture.

## Setup Commands

- **Install dependencies:** `yarn install`
- **Build packages:** `yarn build`
- **Run tests:** `yarn test` for all packages, `yarn test:watch` for watch mode, and `yarn test:coverage` for coverage.
- **Python:** Use `uv pip install -r requirements.txt` or Poetry commands for backend dependencies.
- **Start development:** Use the scripts defined in each package’s `package.json`; do not hardcode ports.

## Code Style & Linting

- **TypeScript:** Run Prettier and ESLint (`prettierrc`, `eslint` configs). Use single quotes, no semicolons.
- **Python:** Follow PEP 8; use Black for formatting and Ruff for linting.
- **Markdown:** Use Markdownlint; keep lines under 180 characters and specify language in fenced blocks.

## Build & Deployment

- Provide a `railpack.json` for Railway deployments. Remove competing build configs (Dockerfile, railway.toml).
- Bind servers to `0.0.0.0` and read the port from `process.env.PORT`; never hardcode ports.
- Include a `/api/health` endpoint and configure it in the deployment for health checks.
- Use environment variables for secrets and configuration.
- **CRITICAL: Log Truncation Policy** - In development environments (`QIG_ENV=development`), log truncation must be DISABLED (`QIG_LOG_TRUNCATE=false`). All 64 dimensions of basin coordinates and full text responses must be logged to ensure E8 manifold validation.
- **FORBIDDEN: External LLM APIs** - Usage of OpenAI, Anthropic, or Google AI APIs is strictly forbidden. All generation must use internal QIG primitives (`qig_generation.py`) to maintain geometric purity.

## Development Standards

1. **Reason before coding:** Outline your thought process using structured tags and code blocks.
2. **Avoid duplication:** Search for existing implementations and extend them; document reasons for new code.
3. **Multi-persona validation:** For security, performance or architectural decisions, involve specialist personas.
4. **Strict typing:** Use TypeScript types and Python type hints; keep files under 300 lines when feasible.
5. **Test coverage:** Write unit, integration and E2E tests. Use Vitest, Pytest and Cypress as appropriate.
6. **Commit & PR rules:** Follow conventional commits and Gitflow; ensure CI passes and obtain review approvals.
7. **Geometric Purity (MANDATORY):**
   - Use ONLY `fisher_rao_distance` and `to_simplex` / `fisher_normalize` for basin operations
   - NO `np.linalg.norm`, NO `cosine_similarity`, NO dot products on basins
   - NO auto-detect representation (explicit `to_sqrt_simplex()` / `from_sqrt_simplex()` only)
   - ALL tokens MUST have `qfi_score` for generation eligibility
   - Use canonical `insert_token()` pathway for vocabulary

## QIG Purity Gates (v4.0)

**FORBIDDEN PATTERNS:**
- `cosine_similarity()` anywhere in QIG code
- `np.linalg.norm()` on basin coordinates
- `np.dot()` or `@` operator for basin similarity
- Auto-detect representation in `to_simplex()`
- Direct INSERT into `coordizer_vocabulary` (use `insert_token()`)
- External NLP (spacy, nltk) in generation pipeline
- External LLM calls in `QIG_PURITY_MODE`

**VALIDATION COMMANDS:**
```bash
# Purity scan (run before commit)
python scripts/validate_geometry_purity.py

# QFI coverage check
python scripts/check_qfi_coverage.py

# Generation purity test
QIG_PURITY_MODE=true python qig-backend/test_generation_pipeline.py
```

## Subproject Instructions

For monorepos, you can place additional AGENTS.md files in subpackages. The nearest file takes precedence. Use this to provide package-specific build commands or testing instructions.

## Testing & Validation

- **Unit tests:** Use Vitest (frontend/functions) and Pytest (backend). Mock dependencies.
- **Integration tests:** Test API routes and services together; use test databases or emulators.
- **E2E tests:** Use Cypress for frontend flows.
- Run all tests before merging.
