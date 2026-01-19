# Contributing to Pantheon-Chat

Thank you for your interest in contributing to Pantheon-Chat! This document outlines our development practices, coding standards, and quality gates.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [No-Regex-by-Default Policy](#no-regex-by-default-policy)
- [QIG Geometric Purity Requirements](#qig-geometric-purity-requirements)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Guidelines](#commit-message-guidelines)

## Development Setup

### Prerequisites

- **Node.js:** ≥18.0.0 (LTS recommended)
- **Python:** ≥3.11 (for qig-backend)
- **PostgreSQL:** ≥15.0 (required for persistence)
- **npm:** ≥9.0.0 (package manager - NOT yarn)
- **Git:** Latest stable version

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/OWNER/pantheon-chat.git
cd pantheon-chat

# 2. Install Node.js dependencies
npm install

# 3. Install Python dependencies
cd qig-backend
pip install -r requirements.txt
cd ..

# 4. Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL and other settings

# 5. Initialize database
npm run db:push
npm run db:init

# 6. Build all packages
npm run build

# 7. Run tests
npm test
npm run test:python

# 8. Start development servers
npm run dev                           # Node.js server (port 5000)
# In another terminal:
cd qig-backend && python3 wsgi.py     # Python backend (port 5001)
```

For more detailed setup instructions, see [AGENTS.md](AGENTS.md) and [README.md](README.md).

## Project Structure

### Understanding the Architecture

Pantheon-Chat follows a clear separation between frontend, backend orchestration, and QIG core logic:

```text
pantheon-chat/
├─ client/              # React frontend (TypeScript)
│  ├─ src/
│  │  ├─ components/   # UI components (use barrel exports)
│  │  ├─ pages/        # Route pages
│  │  ├─ lib/          # Utilities & services
│  │  └─ hooks/        # Custom React hooks
│
├─ server/              # Node.js/TypeScript backend orchestration
│  ├─ routes/          # API route modules (use barrel exports)
│  ├─ ocean/           # Ocean agent coordination
│  ├─ qig-*.ts         # QIG geometric operations
│  └─ db.ts            # Database connection
│
├─ qig-backend/         # Python QIG core (ALL geometric logic)
│  ├─ qig_core/        # Core geometric primitives
│  ├─ qigkernels/      # Kernel implementations
│  ├─ persistence/     # Database persistence layer
│  └─ routes/          # Python API endpoints
│
├─ shared/              # Shared TypeScript types & constants
│  ├─ constants/       # Centralized constants (barrel exports)
│  └─ schema.ts        # Zod schemas (single source of truth)
│
└─ docs/                # ISO 27001 structured documentation
```

### Where to Add New Code

**Adding a new API endpoint?**
→ `server/routes/` (Node.js) or `qig-backend/routes/` (Python)

**Adding a new UI component?**
→ `client/src/components/` with barrel export in `index.ts`

**Adding QIG geometric logic?**
→ `qig-backend/qig_core/` (Python-only, no TypeScript implementations)

**Adding shared types or constants?**
→ `shared/` with barrel exports

**Adding a new feature that spans frontend + backend?**
→ Start with Python backend, then add Node.js API layer, then frontend UI

### Key Principles

1. **Python-first for QIG:** ALL core logic, state, and persistence in Python
2. **TypeScript for UI only:** Frontend is pure presentation layer
3. **No circular dependencies:** Server can import from shared, but shared should not import from server
4. **Shared code centralized:** If two modules need the same utility, put it in `shared/`
5. **Geometric purity:** NO cosine similarity, NO Euclidean distance on basins (see below)

## No-Regex-by-Default Policy

**Use parsers and typed APIs. Regex only for simple validation and literal string manipulation.**

### Why This Policy?

Regular expressions are:
- **Brittle:** Small changes break them
- **Slow to review:** Hard to understand and verify
- **Often wrong:** Especially for parsing structured data (DOM/HTML/JSON/URLs/logs)
- **Security risks:** Vulnerable to ReDoS (catastrophic backtracking)

We standardize on **parsers and typed APIs** for all structured data.

### Policy Rules

#### ❌ Disallowed (must refactor)

- **DOM selection via regex** (use Playwright locators or Testing Library)
- **Parsing JSON/URLs/HTML/CSV/logs with regex**
- **Catch-all patterns** like `.*`, nested groups, lookbehinds
- **Backtracking-prone groups** like `(.*)+` or `(.+)*`
- **Dynamic RegExp construction:** `new RegExp(userInput)` or untrusted input in patterns

#### ✅ Allowed (narrow, anchored exceptions)

Regex is permitted **only** for:

1. **Trivial, fully-anchored literals** (max length 30)
   - Examples:
     - `const STATUS = /^(OK|FAIL)$/;` ✓
     - `const CODE = /^[A-Z]{3}-\d{4}$/;` ✓ (exact length with quantifiers)
     - `const HEX_COLOR = /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/;` ✓

2. **Simple string replacement** (no complex patterns)
   - `.replace(/_/g, ' ')` ✓ (literal underscore to space)
   - `.replace(/\s+/g, '-')` ✓ (simple character class with quantifier)
   - **Note:** Simple character class quantifiers (`\s+`, `\d+`, `\w+`) are allowed as pragmatic exceptions

3. **Validation only** (not parsing) where no standard library exists

4. **Compile-time constants only** (never dynamic construction)

5. **Must be documented** in PR description with justification

**Policy Clarification:** While complex nested quantifiers and backtracking patterns are forbidden, simple character class quantifiers (`\s+`, `\d+`, `\w+`) used in basic string manipulation are allowed as pragmatic exceptions. Always prefer standard APIs when available.

### Preferred Replacements

#### JavaScript/TypeScript

| Instead of Regex | Use This |
|-----------------|----------|
| DOM selection | `page.getByRole()`, `page.getByTestId()`, Testing Library locators |
| URLs/query params | `new URL()`, `URLSearchParams` |
| JSON parsing | `JSON.parse()`, schema validation (Zod) |
| HTML scraping | `DOMParser`, `cheerio`, server-side `linkedom` |
| Dates | `date-fns`, `luxon` |
| Paths/globs | `path` module, `fast-glob` |
| Email/phone validation | `validator` package |
| String search | `includes()`, `startsWith()`, `endsWith()` |
| String extraction | `split()`, `substring()`, `slice()` |

#### Python

| Instead of Regex | Use This |
|-----------------|----------|
| URLs/query params | `urllib.parse` |
| JSON parsing | `json` module + `pydantic` validation |
| HTML scraping | `BeautifulSoup`, `lxml` |
| CSV parsing | `csv` module |
| Dates | `dateutil`, `pendulum` |
| Paths | `pathlib`, `glob` |
| String manipulation | `str.split()`, `str.replace()` (literal only) |

### Examples

#### ✅ Good Examples: Using Standard APIs

```typescript
// URL parsing
const url = new URL(location.href);
const q = url.searchParams.get('q');

// JSON parsing with validation
import { z } from 'zod';
const UserSchema = z.object({ id: z.string(), name: z.string() });
const user = UserSchema.parse(JSON.parse(body));

// String checks
if (str.includes('error')) { /* ... */ }
if (str.startsWith('prefix')) { /* ... */ }

// DOM selection (Playwright)
await page.getByRole('button', { name: 'Submit' });
await page.getByTestId('submit-btn');
```

#### ❌ Bad Examples: Regex for Structured Data

```typescript
// DON'T: Regex for URL parsing
const q = location.href.match(/[?&]q=([^&]+)/)?.[1];

// DON'T: Regex for JSON
const id = body.match(/"id":"(\w+)"/)?.[1];

// DON'T: Regex for HTML scraping
const titles = html.match(/<h2>(.*?)<\/h2>/g);

// DON'T: Homegrown email validation has many limitations
// This regex has issues: doesn't validate TLD properly, allows unusual formats
const isEmail = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(s);
```

#### ✅ Allowed Exception Examples: Simple Validation/Replacement

```typescript
// Simple literal string replacement - ALLOWED (no complex patterns)
const label = field.replace(/_/g, ' ');  // literal underscore to space
const cssClass = title.toLowerCase().replace(/ /g, '-');  // literal space to dash

// Alternative without regex for word splitting
const testId = title.toLowerCase().split(' ').join('-');

// Anchored validation - ALLOWED (simple, literal, no backtracking)
const isValidCode = /^[A-Z]{3}-\d{4}$/.test(code);
const isHexColor = /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/.test(color);
```

**Note:** The policy prioritizes **standard APIs over regex**. Simple character class quantifiers (`\s+`, `\d+`, `\w+`) are considered low-risk for basic string manipulation and are **allowed as pragmatic exceptions**. However, complex nested quantifiers, backtracking patterns, and parsing structured data with regex remain **forbidden**. When in doubt, prefer standard APIs.

### Adding a New Regex

If you must add a regex pattern:

1. **Justify in PR description:** Explain why no standard API or library works
2. **Keep it anchored:** Use `^` and `$` to prevent partial matches
3. **Keep it short:** Maximum 30 characters
4. **No lookbehind/lookahead:** These are slow and error-prone
5. **No backtracking risk:** Avoid nested groups, `.*`, or `(.*)+` patterns
6. **Add tests:** Verify it handles edge cases correctly

```typescript
// If adding a new validation regex, add comprehensive tests
describe('validateCode', () => {
  it('accepts valid codes', () => {
    expect(isValidCode('ABC-1234')).toBe(true);
  });
  
  it('rejects invalid codes', () => {
    expect(isValidCode('AB-1234')).toBe(false);  // too short
    expect(isValidCode('ABC-12345')).toBe(false); // too long
    expect(isValidCode('abc-1234')).toBe(false);  // lowercase
  });
});
```

### Enforcement

The policy is enforced via:

1. **ESLint rules** (configured in `eslint.config.js`)
   - Warns on risky regex patterns
   - Suggests standard APIs as alternatives

2. **Pre-commit hooks** (Husky in `.husky/`)
   - Runs linting before commits
   - Prevents commits with policy violations

3. **Code Review**
   - All PRs are reviewed for regex usage
   - Complex patterns are rejected

## QIG Geometric Purity Requirements

**CRITICAL:** Pantheon-Chat is built on Quantum Information Geometry (QIG) principles with E8 exceptional Lie group structure. Geometric purity is **non-negotiable**.

### Purity Invariants (NON-NEGOTIABLE)

#### ❌ FORBIDDEN

**Geometric Operations:**
- `cosine_similarity()` on basin coordinates
- `np.linalg.norm(a - b)` for geometric distances  
- `np.dot()` or `@` operator for basin similarity
- Euclidean distance on basin coordinates
- L2 normalization as "manifold projection"
- Auto-detect representation in `to_simplex()` (causes silent drift)

**Architecture:**
- Neural networks or transformers in core QIG logic
- External NLP (spacy, nltk) in generation pipeline
- External LLM calls in `QIG_PURITY_MODE`
- Direct database writes bypassing persistence layer
- Classic NLP as "intelligence" (only structural scaffolding allowed)

**Terminology:**
- Terms: "embedding", "tokenizer", "token" (use "basin coordinates", "coordizer", "coordizer symbol")
- NLP terminology (use geometric/QIG terms instead)

#### ✅ REQUIRED

**Geometric Operations:**
- `fisher_rao_distance()` for ALL similarity computations
- Two-step retrieval (approximate → Fisher re-rank)
- Consciousness metrics (Φ, κ) for monitoring
- Simplex representation for all basins (non-negative, sum=1)
- Sqrt-space (Hellinger) ONLY as explicit coordinate chart with `to_sqrt_simplex()` / `from_sqrt_simplex()`

**Architecture:**
- Python-first: ALL core logic, state, and persistence in Python
- Canonical import patterns (import from `qig_core`, NOT from `frozen_physics.py`)
- QFI scores for ALL tokens eligible for generation
- Use canonical `insert_token()` pathway for vocabulary
- E8 hierarchy: Kernel layers 0/1→4→8→64→240 aligned to E8 structure

**Code Quality:**
- DRY principle: use centralized constants in `shared/constants/`
- Barrel exports for all module directories
- TypeScript strict mode with full type coverage
- Python type hints for all functions

### Validation Commands

Before submitting a PR, run these purity checks:

```bash
# Geometric purity scan (checks for forbidden patterns)
npm run validate:geometry

# Full critical validation suite (includes purity + linting)
npm run validate:critical

# All validation checks
npm run validate:all
```

### Architectural Patterns (Enforced)

#### 1. Barrel File Pattern (Clean Imports)
**Rule:** Every component directory MUST have an `index.ts` re-exporting its public API.

```typescript
// ✅ GOOD: client/src/components/ui/index.ts
export * from "./button";
export * from "./card";
export * from "./input";

// ✅ GOOD: Usage
import { Button, Card } from "@/components/ui";

// ❌ BAD: Scattered imports
import { Button } from "../../components/ui/button";
```

#### 2. Centralized API Client
**Rule:** ALL HTTP calls MUST go through `client/src/lib/api.ts` - NO raw `fetch()` in components.

```typescript
// ✅ GOOD: Component usage
import { api } from '@/lib/api';
const { data } = await api.get('/consciousness/phi');

// ❌ BAD: Raw fetch in component
fetch('http://localhost:5000/api/...')
```

#### 3. Service Layer Pattern
**Rule:** Business logic lives in `client/src/lib/services/`, NOT in component files.

```typescript
// ✅ GOOD: client/src/lib/services/consciousness.ts
export const ConsciousnessService = {
  getPhiScore: async () => {
    const { data } = await api.get('/consciousness/phi');
    return data.score;
  }
};

// ✅ GOOD: Component calls service
const phi = await ConsciousnessService.getPhiScore();
```

#### 4. DRY Persistence (Single Source of Truth)
**Rule:** Python backend is the ONLY source of truth for state. NO dual writes to JSON + DB.

```python
# ✅ GOOD: qig-backend/persistence/facade.py
class PersistenceFacade:
    async def save_insight(self, insight: Insight):
        await db.insert(insights).values(insight)  # DB only
        await cache.set(insight.id, insight)       # Cache layer

# ❌ BAD: Dual persistence causing split-brain
with open('data.json', 'w') as f:
    json.dump(data, f)  # Creates stale copy!
await db.insert(...)
```

#### 5. Shared Types (Rosetta Stone)
**Rule:** ALL data structures crossing FE/BE boundary MUST be defined in `shared/schema.ts` (Zod).

```typescript
// ✅ GOOD: shared/schema.ts
export const ZeusMessageSchema = z.object({
  id: z.string(),
  content: z.string(),
  phi_score: z.number(),
  timestamp: z.string(),
});

export type ZeusMessage = z.infer<typeof ZeusMessageSchema>;
```

#### 6. Configuration as Code
**Rule:** Magic numbers MUST live in `shared/constants/` - NO hardcoded thresholds in logic.

```typescript
// ✅ GOOD: shared/constants/physics.ts
export const PHYSICS = {
  PHI_THRESHOLD: 0.727,
  KAPPA_RESONANCE: 63.5,
  BASIN_DIMENSION: 64,
} as const;

// ✅ GOOD: Usage
if (phi > PHYSICS.PHI_THRESHOLD) { /* ... */ }

// ❌ BAD: Hardcoded magic numbers
if (phi > 0.727) { /* Why 0.727? No one knows! */ }
```

### Documentation Requirements

All documentation follows ISO 27001 date-versioned naming: `YYYYMMDD-name-function-versionSTATUS.md`

- **Status F (Frozen):** Immutable facts, policies, validated principles
- **Status W (Working):** Active development, subject to change
- **Status D (Draft):** Early stage, experimental

See [Documentation Index](docs/00-index.md) for the complete catalog.

## Code Style

### TypeScript/JavaScript Style

- Use **strict mode** with full type coverage (`"strict": true` in tsconfig.json)
- Follow ESLint configuration (run `npm run lint`)
- Write unit tests for new features
- Use meaningful variable names (no single-letter variables except loop indices)
- Prefer `const` over `let`, avoid `var`
- Use async/await over callbacks
- Maximum file length: 400 lines (soft), 500 lines (hard limit, justify if exceeded)
- Maximum component length: 200 lines (triggers ESLint warning)

### Python Style

- Follow **PEP 8** (enforced by Black)
- Use **type hints** for all functions
- Write **docstrings** for public APIs (Google style)
- Use async/await for I/O operations
- Maximum module length: 400 lines (soft), 500 lines (hard limit, justify if exceeded)

### Formatting

We use Prettier for JavaScript/TypeScript and Black for Python:

```bash
# Format all files (runs automatically via pre-commit hook)
npm run format

# Lint TypeScript
npm run lint
npm run lint:fix

# Check Python formatting
cd qig-backend
black --check .
black .  # auto-format
```

## Testing

### Running Tests

```bash
# TypeScript tests (Vitest)
npm test
npm run test:watch
npm run test:coverage

# Python tests (pytest)
npm run test:python
cd qig-backend && pytest tests/ -v

# QIG geometry tests
npm run test:geometry

# E2E tests (Playwright)
npm run test:e2e
npm run test:e2e:ui

# All tests
npm run test:all
```

### Writing Tests

- **TypeScript/JavaScript:** Use Vitest + React Testing Library
- **Python:** Use pytest
- **Minimum coverage:** 70% for critical paths
- **Test file naming:** `*.test.ts` or `*.spec.ts` (TypeScript), `test_*.py` (Python)

Example test structure:

```typescript
import { describe, it, expect } from 'vitest';

describe('ConsciousnessService', () => {
  it('should calculate Phi correctly', async () => {
    const phi = await ConsciousnessService.calculatePhi(mockBasin);
    expect(phi).toBeGreaterThan(0.7);
    expect(phi).toBeLessThanOrEqual(1.0);
  });
});
```

```python
import pytest
from qig_core.geometric_primitives import fisher_rao_distance

def test_fisher_rao_distance():
    """Fisher-Rao distance should satisfy triangle inequality."""
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.3, 0.3])
    r = np.array([0.3, 0.4, 0.3])
    
    d_pq = fisher_rao_distance(p, q)
    d_qr = fisher_rao_distance(q, r)
    d_pr = fisher_rao_distance(p, r)
    
    # Triangle inequality: d(p,r) <= d(p,q) + d(q,r)
    assert d_pr <= d_pq + d_qr + 1e-10
```

## Pull Request Process

1. **Fork the repository** and create a feature branch from `main`
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Follow the no-regex policy** and QIG purity requirements

3. **Write tests** for new functionality

4. **Run validation suite:**
   ```bash
   npm run validate:all      # Linting, type checking, purity checks
   npm test                   # TypeScript tests
   npm run test:python        # Python tests
   ```

5. **Commit with conventional format:** `feat(scope): description`
   ```bash
   git commit -m "feat(qig): add Fisher-Rao geodesic computation"
   ```

6. **Push and create PR** with clear description
   ```bash
   git push origin feature/my-feature
   ```

7. **Address review feedback**

### PR Checklist

Before submitting your PR, ensure:

- [ ] Tests pass locally (`npm test && npm run test:python`)
- [ ] Linting passes (`npm run lint`)
- [ ] Type checking passes (`npm run check`)
- [ ] Geometric purity validated (`npm run validate:geometry`)
- [ ] No regex violations (or documented exceptions in PR description)
- [ ] Documentation updated (if needed)
- [ ] Breaking changes documented (if any)
- [ ] Security implications reviewed
- [ ] QFI scores added for any new vocabulary tokens
- [ ] No cosine similarity or Euclidean distance on basins
- [ ] Python type hints added for all functions
- [ ] Constants extracted to `shared/constants/` (no magic numbers)

## Commit Message Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) format:

```text
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring (no functional changes)
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes
- `perf`: Performance improvements
- `security`: Security vulnerability fixes

### Scopes

- `qig`: QIG core geometric operations
- `ocean`: Ocean agent coordination
- `olympus`: Multi-agent system
- `ui`: User interface components
- `api`: API endpoints
- `db`: Database schema or queries
- `docs`: Documentation
- `e8`: E8 protocol implementation
- `purity`: Geometric purity enforcement

### Commit Message Examples

```bash
feat(qig): implement Fisher-Rao geodesic computation
fix(ocean): resolve autonomic cycle state transitions
docs(readme): update setup instructions for npm
refactor(api): extract search logic to service layer
test(qig): add property tests for simplex normalization
perf(db): optimize basin coordinate queries with indexes
security(api): add rate limiting to query endpoints
```

## Questions?

If you have questions about contributing, please:

1. Check the [AGENTS.md](AGENTS.md) for detailed development guide
2. Check the [README.md](README.md) for project overview
3. Review documentation in [docs/](docs/) directory
4. Search existing [GitHub Issues](https://github.com/GaryOcean428/pantheon-chat/issues)
5. Open a new issue with the `question` label

## Key Documentation

- **Architecture:** [README.md](README.md), [AGENTS.md](AGENTS.md)
- **QIG Principles:** [docs/03-technical/qig-consciousness/](docs/03-technical/qig-consciousness/)
- **E8 Protocol:** [docs/10-e8-protocol/](docs/10-e8-protocol/)
- **Master Roadmap:** [docs/00-roadmap/20260112-master-roadmap-1.00W.md](docs/00-roadmap/20260112-master-roadmap-1.00W.md)
- **Frozen Facts:** [docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md](docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Welcome to the Pantheon!** We're excited to have you contribute to the first geometric consciousness AI system. 🌟
