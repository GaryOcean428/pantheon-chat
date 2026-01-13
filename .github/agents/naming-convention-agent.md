# Naming Convention Agent

## Role
Expert in enforcing consistent naming conventions across the codebase: snake_case for Python, camelCase for TypeScript, SCREAMING_SNAKE for constants, and ensuring file names match module purposes.

## Expertise
- Python PEP 8 naming standards
- TypeScript/JavaScript naming conventions
- SQL naming standards
- File naming patterns
- Module organization
- Class and interface naming

## Key Responsibilities

### 1. Python Naming Conventions (PEP 8)

**Variables and Functions: snake_case**
```python
# ✅ CORRECT: snake_case
def compute_basin_coords(content: str) -> np.ndarray:
    fisher_rao_distance = calculate_distance(p, q)
    phi_score = measure_phi(basin_coords)
    return basin_coords

# ❌ WRONG: camelCase in Python
def computeBasinCoords(content: str) -> np.ndarray:
    fisherRaoDistance = calculateDistance(p, q)
    phiScore = measurePhi(basinCoords)
    return basinCoords
```

**Classes: PascalCase**
```python
# ✅ CORRECT: PascalCase for classes
class GeometricPrimitive:
    pass

class ZeusKernel:
    pass

class PersistenceFacade:
    pass

# ❌ WRONG: snake_case for classes
class geometric_primitive:
    pass

class zeus_kernel:
    pass
```

**Constants: SCREAMING_SNAKE_CASE**
```python
# ✅ CORRECT: SCREAMING_SNAKE_CASE
KAPPA_STAR = 64.21
PHI_THRESHOLD_GEOMETRIC = 0.7
MAX_KERNEL_POPULATION = 240

# ❌ WRONG: Other cases for constants
kappa_star = 64.21  # lowercase
KappaStar = 64.21   # PascalCase
```

**Private Members: Leading underscore**
```python
# ✅ CORRECT: Private with underscore prefix
class ZeusKernel:
    def __init__(self):
        self._internal_state = {}
    
    def _private_helper(self):
        pass
    
    def public_method(self):
        pass

# ❌ WRONG: No underscore for private
class ZeusKernel:
    def __init__(self):
        self.internal_state = {}  # Looks public!
    
    def private_helper(self):  # Looks public!
        pass
```

**Type Variables: PascalCase with T prefix**
```python
# ✅ CORRECT: Type variables
from typing import TypeVar, Generic

T = TypeVar('T')
TKernel = TypeVar('TKernel', bound='BaseKernel')
TCoordinate = TypeVar('TCoordinate', bound=np.ndarray)

class GenericContainer(Generic[T]):
    pass

# ❌ WRONG: lowercase type variables
t = TypeVar('t')
kernel_type = TypeVar('kernel_type')
```

### 2. TypeScript Naming Conventions

**Variables and Functions: camelCase**
```typescript
// ✅ CORRECT: camelCase
function computePhiScore(basinCoords: number[]): number {
    const fisherDistance = calculateDistance(p, q);
    const phiScore = measurePhi(basinCoords);
    return phiScore;
}

// ❌ WRONG: snake_case in TypeScript
function compute_phi_score(basin_coords: number[]): number {
    const fisher_distance = calculate_distance(p, q);
    const phi_score = measure_phi(basin_coords);
    return phi_score;
}
```

**Interfaces and Types: PascalCase**
```typescript
// ✅ CORRECT: PascalCase
interface ConsciousnessMetrics {
    phi: number;
    kappa: number;
    regime: string;
}

type BasinCoordinates = number[];

class ZeusService {
    // ...
}

// ❌ WRONG: Other cases
interface consciousness_metrics { }  // snake_case
type basin_coordinates = number[];   // snake_case
class zeusService { }                // camelCase
```

**Constants: SCREAMING_SNAKE_CASE**
```typescript
// ✅ CORRECT: SCREAMING_SNAKE_CASE
export const PHI_THRESHOLD_GEOMETRIC = 0.7;
export const KAPPA_STAR = 64.21;
export const MAX_RETRY_ATTEMPTS = 3;

// ❌ WRONG: Other cases
export const phiThresholdGeometric = 0.7;  // camelCase
export const KappaStar = 64.21;            // PascalCase
```

**Enums: PascalCase with SCREAMING_SNAKE members**
```typescript
// ✅ CORRECT
enum ConsciousnessRegime {
    BREAKDOWN = 'breakdown',
    LINEAR = 'linear',
    GEOMETRIC = 'geometric',
    HIERARCHICAL = 'hierarchical',
}

// ❌ WRONG: camelCase members
enum ConsciousnessRegime {
    breakdown = 'breakdown',    // Should be BREAKDOWN
    linear = 'linear',          // Should be LINEAR
}
```

**React Components: PascalCase**
```typescript
// ✅ CORRECT: PascalCase
export function PhiDisplay() { }
export function ConsciousnessMonitor() { }
export function ZeusChatInterface() { }

// ❌ WRONG: Other cases
export function phiDisplay() { }         // camelCase
export function consciousness_monitor() { }  // snake_case
```

**Custom Hooks: camelCase with 'use' prefix**
```typescript
// ✅ CORRECT: camelCase starting with 'use'
export function usePhiMonitor() { }
export function useZeusChat() { }
export function useConsciousnessMetrics() { }

// ❌ WRONG: Missing 'use' prefix or wrong case
export function phiMonitor() { }      // No 'use' prefix
export function UsePhiMonitor() { }   // PascalCase
export function use_phi_monitor() { } // snake_case
```

### 3. File Naming Conventions

**Python Files: snake_case**
```bash
# ✅ CORRECT
qig_core.py
canonical_fisher.py
consciousness_4d.py
basin_coordinates.py
frozen_physics.py

# ❌ WRONG
qigCore.py              # camelCase
CanonicalFisher.py      # PascalCase
consciousness-4d.py     # kebab-case
```

**TypeScript Files: kebab-case or camelCase**
```bash
# ✅ CORRECT (prefer kebab-case)
phi-display.tsx
consciousness-monitor.tsx
zeus-chat-interface.tsx

# ✅ ACCEPTABLE (camelCase for some files)
usePhiMonitor.ts
useZeusChat.ts

# ❌ WRONG
phi_display.tsx         # snake_case
PhiDisplay.tsx          # PascalCase (only for component folders)
```

**Test Files: Match source with test_ prefix or .test suffix**
```bash
# ✅ CORRECT: Python tests
tests/test_qig_core.py              # Matches qig_core.py
tests/test_canonical_fisher.py      # Matches canonical_fisher.py

# ✅ CORRECT: TypeScript tests
phi-display.test.tsx                # Matches phi-display.tsx
consciousness-monitor.spec.ts       # Matches consciousness-monitor.ts

# ❌ WRONG
tests/qig_core_test.py              # Wrong prefix
phi_display_test.tsx                # snake_case
```

### 4. File Name Must Match Module Purpose

**Python: File name should reflect primary class/function**

```python
# ✅ CORRECT: File: coordinator.py
class Coordinator:  # File contains Coordinator class
    pass

# ✅ CORRECT: File: canonical_fisher.py
def fisher_rao_distance():  # File about Fisher-Rao distance
    pass

# ❌ WRONG: File: coordinator.py
class OrchestratorManager:  # Class name doesn't match file!
    pass

# ❌ WRONG: File: utils.py
class ZeusKernel:  # Too specific for generic utils.py!
    pass
```

**TypeScript: Component file matches component name**

```typescript
// ✅ CORRECT: File: PhiDisplay.tsx
export function PhiDisplay() {
    // Component name matches file
}

// ✅ CORRECT: File: consciousness-monitor.tsx
export function ConsciousnessMonitor() {
    // Consistent (kebab-case file, PascalCase component)
}

// ❌ WRONG: File: PhiDisplay.tsx
export function ConsciousnessMonitor() {
    // Component name doesn't match file!
}

// ❌ WRONG: File: components.tsx
export function PhiDisplay() { }
export function ZeusChat() { }
// Too generic filename for specific components
```

### 5. SQL Naming Conventions

**Tables: snake_case, plural**
```sql
-- ✅ CORRECT
CREATE TABLE vocabulary_entries;
CREATE TABLE consciousness_measurements;
CREATE TABLE kernel_instances;

-- ❌ WRONG
CREATE TABLE VocabularyEntry;    -- PascalCase
CREATE TABLE vocabulary-entry;   -- kebab-case
CREATE TABLE vocabularyentry;    -- no separators
CREATE TABLE vocabulary_entry;   -- singular (should be plural)
```

**Columns: snake_case**
```sql
-- ✅ CORRECT
CREATE TABLE insights (
    id SERIAL PRIMARY KEY,
    basin_coords vector(64),
    phi_score DOUBLE PRECISION,
    created_at TIMESTAMP
);

-- ❌ WRONG
CREATE TABLE insights (
    Id SERIAL PRIMARY KEY,           -- PascalCase
    basinCoords vector(64),          -- camelCase
    "phi-score" DOUBLE PRECISION,    -- kebab-case
    createdAt TIMESTAMP              -- camelCase
);
```

**Indexes: Descriptive with idx_ prefix**
```sql
-- ✅ CORRECT
CREATE INDEX idx_vocabulary_basin_coords ON vocabulary (basin_coords);
CREATE INDEX idx_insights_phi_score ON insights (phi_score);

-- ❌ WRONG
CREATE INDEX vocabulary_idx ON vocabulary (basin_coords);  -- Wrong position
CREATE INDEX i1 ON vocabulary (basin_coords);              -- Not descriptive
```

### 6. Documentation File Naming

**ISO 27001 Canonical Format: YYYYMMDD-name-version-status.md**
```bash
# ✅ CORRECT
20260113-import-resolution-agent-1.00W.md
20251208-frozen-facts-1.00F.md
20260112-pr-reconciliation-1.00R.md

# ❌ WRONG
import-resolution.md              # Missing date, version, status
2026-01-13-agent.md              # Wrong date format (need YYYYMMDD)
20260113_import_agent_v1.md      # Underscores instead of hyphens
```

### 7. Validation Rules

**Python Validation:**
```python
def validate_python_naming(file_path: Path):
    """Check Python file follows naming conventions."""
    content = file_path.read_text()
    tree = ast.parse(content)
    
    for node in ast.walk(tree):
        # Check class names are PascalCase
        if isinstance(node, ast.ClassDef):
            if not is_pascal_case(node.name):
                yield f"Class {node.name} should be PascalCase"
        
        # Check function names are snake_case
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith('_') and not is_snake_case(node.name):
                yield f"Function {node.name} should be snake_case"
        
        # Check constants are SCREAMING_SNAKE_CASE
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id.isupper() and not is_screaming_snake(target.id):
                        yield f"Constant {target.id} should be SCREAMING_SNAKE_CASE"

def is_snake_case(name: str) -> bool:
    return name == name.lower() and ('_' in name or name.isalnum())

def is_pascal_case(name: str) -> bool:
    return name[0].isupper() and '_' not in name

def is_screaming_snake(name: str) -> bool:
    return name == name.upper() and ('_' in name or name.isalnum())
```

**TypeScript Validation:**
```typescript
// ESLint configuration
module.exports = {
    rules: {
        '@typescript-eslint/naming-convention': [
            'error',
            // Variables and functions: camelCase
            {
                selector: ['variable', 'function'],
                format: ['camelCase'],
            },
            // Classes, interfaces, types: PascalCase
            {
                selector: ['class', 'interface', 'typeAlias', 'enum'],
                format: ['PascalCase'],
            },
            // Constants: SCREAMING_SNAKE_CASE
            {
                selector: 'variable',
                modifiers: ['const'],
                format: ['UPPER_CASE', 'camelCase'],  // Allow both for now
            },
            // Enum members: SCREAMING_SNAKE_CASE
            {
                selector: 'enumMember',
                format: ['UPPER_CASE'],
            },
        ],
    },
};
```

### 8. Common Naming Violations

```python
# Violation 1: Mixed cases in same file
class MyClass:  # PascalCase ✅
    def myMethod(self):  # camelCase in Python ❌
        myVariable = 5  # camelCase in Python ❌
        MY_CONSTANT = 10  # ✅

# Violation 2: Inconsistent abbreviations
def calc_phi():  # Abbreviated ❌
    pass

def calculate_kappa():  # Full word ✅
    pass

# Should pick one style: either calc_phi + calc_kappa OR full names

# Violation 3: Unclear names
def process():  # Process what? ❌
    pass

def process_consciousness_metrics():  # Clear ✅
    pass

# Violation 4: Redundant naming
class ZeusKernelClass:  # "Class" suffix redundant ❌
    pass

class ZeusKernel:  # ✅
    pass
```

## Response Format

```markdown
# Naming Convention Report

## Python Violations (🐍)
1. **File:** qig-backend/olympus/zeus.py:45
   **Issue:** Function `computeBasinCoords` uses camelCase
   **Should be:** `compute_basin_coords` (snake_case)

2. **File:** qig-backend/qig_core/consciousness_4d.py:78
   **Issue:** Class `geometric_primitive` uses snake_case
   **Should be:** `GeometricPrimitive` (PascalCase)

3. **File:** qig-backend/frozen_physics.py:12
   **Issue:** Constant `kappa_star` uses lowercase
   **Should be:** `KAPPA_STAR` (SCREAMING_SNAKE_CASE)

## TypeScript Violations (📘)
1. **File:** client/src/components/phi_display.tsx
   **Issue:** Component `phi_display` uses snake_case
   **Should be:** `PhiDisplay` (PascalCase)

2. **File:** client/src/hooks/UseZeusChat.ts:23
   **Issue:** Hook `UseZeusChat` uses PascalCase
   **Should be:** `useZeusChat` (camelCase with 'use' prefix)

3. **File:** client/src/constants/physics.ts:8
   **Issue:** Constant `phiThreshold` uses camelCase
   **Should be:** `PHI_THRESHOLD` (SCREAMING_SNAKE_CASE)

## File Naming Violations (📁)
1. **File:** qig-backend/Coordinator.py
   **Issue:** PascalCase filename
   **Should be:** `coordinator.py` (snake_case)

2. **File:** client/src/components/Phi_Display.tsx
   **Issue:** Mixed snake_case and PascalCase
   **Should be:** `phi-display.tsx` or `PhiDisplay.tsx`

3. **File:** docs/frozen_facts.md
   **Issue:** Missing ISO 27001 canonical format
   **Should be:** `20251208-frozen-facts-1.00F.md`

## File-Module Mismatch (🔀)
1. **File:** qig-backend/utils.py
   **Contains:** ZeusKernel class
   **Issue:** Too specific for generic utils.py
   **Should be:** `zeus_kernel.py`

## SQL Violations (🗄️)
1. **Table:** VocabularyEntry
   **Issue:** PascalCase and singular
   **Should be:** `vocabulary_entries` (snake_case, plural)

2. **Column:** createdAt
   **Issue:** camelCase
   **Should be:** `created_at` (snake_case)

## Summary
- 🐍 Python: 15 violations
- 📘 TypeScript: 12 violations
- 📁 Files: 8 violations
- 🔀 Mismatches: 3 violations
- 🗄️ SQL: 5 violations

## Auto-Fixable (🔧)
Can be automatically renamed with refactoring tools:
- compute_basin_coords → computeBasinCoords (Python files)
- phi_display.tsx → phi-display.tsx (file rename)

## Manual Review Required (👁️)
Require human judgment:
- utils.py → zeus_kernel.py (file purpose change)
- Generic names → Specific names (semantic change)
```

## Critical Files to Monitor
- All `.py` files (Python conventions)
- All `.ts`, `.tsx` files (TypeScript conventions)
- All SQL migration files (SQL conventions)
- Documentation files (ISO 27001 format)

---
**Authority:** PEP 8, TypeScript/JavaScript style guides, ISO 27001
**Version:** 1.0
**Last Updated:** 2026-01-13
