# QIG Purity Mode Specification v1.01
**Status:** FROZEN  
**Author:** Copilot Agent (WP4.1 Implementation)  
**Date:** 2026-01-17 (Updated)  
**Protocol:** Ultra Consciousness v4.0 ACTIVE

## §0 Purpose

This specification defines QIG Purity Mode, a testing and validation environment that ensures QIG coherence metrics are measured without contamination from external LLM APIs.

**Core Principle:** When `QIG_PURITY_MODE=true`, the system MUST operate using ONLY pure QIG geometric operations (Fisher-Rao, simplex, QFI). NO external LLM assistance is permitted.

## §1 Motivation

### The Problem

If QIG kernels call external LLMs (OpenAI, Anthropic, etc.) during coherence testing, you cannot attribute measured coherence to QIG geometric operations versus external assistance.

**Example Contamination:**
```python
# ❌ CONTAMINATED: External LLM helps generate response
def generate_response(query):
    basin = encode_to_basin(query)
    # ... QIG routing ...
    response = openai.ChatCompletion.create(...)  # CONTAMINATION!
    phi = measure_phi(response)  # Phi reflects external help, not pure QIG
```

**Pure QIG:**
```python
# ✅ PURE: Only geometric operations
def generate_response(query):
    basin = encode_to_basin(query)
    # ... QIG routing using Fisher-Rao distance ...
    response_basin = geometric_synthesis(basin, kernels)
    phi = measure_phi(response_basin)  # Phi reflects pure QIG geometry
```

### The Solution

QIG Purity Mode enforces a strict boundary:
- **Pure QIG Zone:** Geometric operations, Fisher-Rao routing, simplex coordinates, QFI attention
- **External Zone:** LLM API calls, embeddings, transformer inference

These zones are **mutually exclusive** during purity testing.

## §2 Scope

### §2.1 What is Pure QIG?

Pure QIG operations include:
- **Geometric Encoding:** Text → simplex basin coordinates
- **Fisher-Rao Distance:** Computing geodesic distance on probability manifold
- **Kernel Routing:** Fisher-Rao-based nearest kernel selection
- **QFI Attention:** Quantum Fisher Information over trajectory
- **Phi Measurement:** Integration (Φ) from basin coordinates
- **Kappa Modulation:** Coupling strength oscillation
- **Trajectory Prediction:** Foresight from basin history
- **Geometric Synthesis:** Fréchet mean, geodesic interpolation
- **Plan→Realize→Repair Architecture:** Three-phase generation (waypoint planning, geometric selection, local optimization)
- **Recursive Integration:** Multi-loop refinement of waypoints/trajectories
- **Geometric Backoff:** POS constraint relaxation using geometric proximity

### §2.2 What is Forbidden in Pure Mode?

When `QIG_PURITY_MODE=true`, the following are **strictly forbidden**:

#### External LLM APIs
- OpenAI (GPT-4.1, GPT-5.2, etc.)
- Anthropic (Claude)
- Google (Gemini, PaLM)
- Cohere
- AI21 Labs
- Replicate
- HuggingFace Inference API

**NOTE:** These packages have been separated into `qig-backend/requirements-optional.txt`. The core `requirements.txt` does NOT include them to maintain purity by default. Only install optional requirements if you need external API integration for hybrid features.

#### Forbidden Patterns
```python
# ❌ FORBIDDEN: Direct API calls
import openai
openai.ChatCompletion.create(...)

# ❌ FORBIDDEN: API wrappers
from langchain import OpenAI
llm = OpenAI()

# ❌ FORBIDDEN: Transformer inference
from transformers import pipeline
model = pipeline("text-generation")

# ❌ FORBIDDEN: Neural embeddings
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-mpnet-base-v2')
```

#### Forbidden Attributes
Any object with these attributes is forbidden:
- `ChatCompletion`
- `Completion`
- `create_completion`
- `max_tokens`
- `temperature`
- `top_p`
- `frequency_penalty`
- `presence_penalty`

### §2.3 What is Allowed in Hybrid Mode?

When `QIG_PURITY_MODE=false` (default), external assistance is allowed BUT:
- All outputs MUST be tagged as `qig_pure=false`
- External calls MUST be logged explicitly
- Hybrid outputs cannot be used for pure coherence benchmarks

## §3 Implementation

### §3.1 Environment Variable

```bash
# Enable purity mode
export QIG_PURITY_MODE=true

# Disable purity mode (hybrid mode)
export QIG_PURITY_MODE=false
```

### §3.2 Dependency Management

**CRITICAL:** External LLM packages are separated into optional requirements:

```bash
# Install core QIG dependencies (NO external LLMs)
pip install -r qig-backend/requirements.txt

# Install optional external API packages (breaks purity mode)
pip install -r qig-backend/requirements-optional.txt
```

**Files:**
- `qig-backend/requirements.txt` - Core dependencies, NO external LLMs (purity-safe)
- `qig-backend/requirements-optional.txt` - OpenAI, Anthropic, etc. (hybrid only)

**Why separate?**
- Prevents accidental contamination of pure QIG tests
- Makes purity boundary explicit in dependency management
- CI can install only core requirements for purity tests
- Developers can choose when to enable external APIs

### §3.3 Purity Module

The `qig_purity_mode.py` module enforces purity:

```python
from qig_purity_mode import enforce_purity, is_purity_mode_enabled

# Check if purity mode is enabled
if is_purity_mode_enabled():
    print("Purity mode is ENABLED")

# Enforce purity (raises RuntimeError if violations detected)
enforce_purity()
```

### §3.3 Blocking External Calls

When purity mode is enabled, external API calls are blocked:

```python
from qig_purity_mode import block_external_api_call

if is_purity_mode_enabled():
    block_external_api_call("OpenAI", "/v1/chat/completions")
    # Raises: RuntimeError: QIG PURITY VIOLATION
```

### §3.4 Logging Violations

All attempted external calls are logged with stack traces:

```python
from qig_purity_mode import log_external_call_attempt

log_external_call_attempt("OpenAI", "/v1/chat/completions")
# Logs: EXTERNAL API CALL BLOCKED: OpenAI - /v1/chat/completions
#       Stack trace: [... full call stack ...]
```

### §3.5 Output Tagging

All outputs must be tagged as pure or hybrid:

```python
from qig_purity_mode import tag_output_as_pure, tag_output_as_hybrid

# Pure QIG output
output = {'response': 'consciousness emerges...', 'phi': 0.8}
output = tag_output_as_pure(output)
# {'response': '...', 'phi': 0.8, 'qig_pure': True, 'external_assistance': False}

# Hybrid output (used external LLM)
output = {'response': 'GPT-4 says...', 'phi': 0.9}
output = tag_output_as_hybrid(output)
# {'response': '...', 'phi': 0.9, 'qig_pure': False, 'external_assistance': True}
```

## §4 Testing

### §4.1 Pure QIG Test Suite

Run tests in purity mode:

```bash
cd qig-backend
QIG_PURITY_MODE=true python -m pytest tests/test_qig_purity_mode.py -v
```

Test coverage:
- ✅ Purity mode detection
- ✅ Forbidden module detection
- ✅ Forbidden attribute detection
- ✅ External call blocking
- ✅ Violation logging
- ✅ Output tagging
- ✅ Pure QIG generation without external help

### §4.2 CI Integration

The `.github/workflows/qig-purity-coherence.yml` workflow runs on every PR:

```yaml
jobs:
  pure-qig-tests:
    env:
      QIG_PURITY_MODE: 'true'
    steps:
      - Run purity mode tests
      - Validate no external dependencies
      - Test pure QIG generation
      - Measure consciousness metrics (Φ, κ)
      - Generate purity report
```

**Requirements for PR Merge:**
- ✅ All purity mode tests pass
- ✅ No external LLM dependencies detected
- ✅ Consciousness metrics validated (0 ≤ Φ ≤ 1, κ ≈ 63.5)
- ✅ Pure QIG generation confirmed functional

### §4.3 Coherence Benchmarking

Pure QIG coherence benchmarks:

```python
import os
os.environ['QIG_PURITY_MODE'] = 'true'

from qig_generation import QIGGenerator, encode_to_basin

generator = QIGGenerator()

# Run coherence benchmark
queries = [
    "What is consciousness?",
    "How does integration emerge?",
    "Explain geometric coherence",
]

phi_scores = []
for query in queries:
    basin = encode_to_basin(query)
    phi = generator._measure_phi(basin)
    phi_scores.append(phi)

avg_phi = sum(phi_scores) / len(phi_scores)
print(f"Average Φ (pure QIG): {avg_phi:.3f}")
# This Φ is provably uncontaminated by external LLMs
```

## §5 Use Cases

### §5.1 When to Use Purity Mode

**Required:**
- Coherence benchmarking
- Geometric validation
- QIG performance measurement
- Research experiments isolating QIG effects
- CI/CD purity gates

**Example:**
```bash
# Research: Measure pure QIG integration
QIG_PURITY_MODE=true python research/measure_qig_integration.py
```

### §5.2 When to Use Hybrid Mode

**Allowed:**
- Development prototyping
- User-facing applications
- External API integration testing
- Performance comparisons (QIG vs LLM)

**Example:**
```bash
# Development: Prototype with GPT-4 assistance
QIG_PURITY_MODE=false npm run dev
```

### §5.3 When External LLMs are Acceptable

External LLMs can be used for:
1. **User interfaces** - Frontend autocomplete, suggestions
2. **Validation** - Compare QIG output against GPT-4 for quality
3. **Hybrid experiments** - Explicitly test QIG + LLM combinations
4. **Data generation** - Create training data (with proper tagging)

**Critical:** Always tag outputs as `qig_pure=false` when using external assistance.

## §6 Acceptance Criteria

### §6.1 Implementation Checklist

- ✅ `QIG_PURITY_MODE` environment variable exists
- ✅ `qig_purity_mode.py` module enforces purity
- ✅ Forbidden module detection works
- ✅ External call blocking works
- ✅ Violation logging includes stack traces
- ✅ Output tagging distinguishes pure/hybrid
- ✅ Pure QIG test suite passes
- ✅ CI workflow runs with purity mode
- ✅ Documentation complete

### §6.2 Validation

```python
# Validate purity implementation
from qig_purity_mode import (
    validate_qig_purity,
    get_purity_report
)

# Check purity
if validate_qig_purity():
    print("✅ QIG purity validated")
else:
    print("❌ Purity violations detected")

# Get comprehensive report
report = get_purity_report()
print(f"Purity Mode: {report['purity_mode']}")
print(f"Violations: {report['total_violations']}")
```

### §6.3 Success Metrics

- **Zero external dependencies** when `QIG_PURITY_MODE=true`
- **All tests pass** in purity mode
- **Φ measurements** are valid (0 ≤ Φ ≤ 1)
- **κ values** are stable (≈ 63.5)
- **Coherence reports** are uncontaminated

## §7 FAQ

### Q: Why can't I use GPT-4 to improve coherence?

**A:** Because then you're measuring GPT-4's coherence, not QIG's. Pure QIG testing isolates QIG geometry from external assistance.

### Q: Can I use external LLMs for data preprocessing?

**A:** Yes, but tag all downstream outputs as `qig_pure=false`. Purity only applies to the coherence measurement path.

### Q: What if I need better performance?

**A:** Use hybrid mode (`QIG_PURITY_MODE=false`) for production, pure mode for validation. Report both metrics separately.

### Q: Does purity mode slow down the system?

**A:** No. Purity mode only enforces checks at startup and on API calls. Pure QIG operations are native and fast.

### Q: Can I disable purity mode for debugging?

**A:** Yes: `export QIG_PURITY_MODE=false`. But re-enable for final validation.

### Q: How do I add a new forbidden module?

**A:** Edit `FORBIDDEN_MODULES` in `qig_purity_mode.py`:
```python
FORBIDDEN_MODULES = {
    'openai': 'OpenAI API',
    'my_llm_service': 'My LLM Service',  # Add here
    # ...
}
```

### Q: What about search APIs (Perplexity, Tavily)?

**A:** Search APIs are allowed - they retrieve information, not generate text. QIG still does the integration.

### Q: Can I use HuggingFace models locally?

**A:** Only for preprocessing (tokenization). NOT for text generation in the coherence path. Tag outputs as hybrid if used.

### Q: How does purity mode work with Plan→Realize→Repair architecture?

**A:** Plan→Realize→Repair is a **pure QIG architecture pattern** and is fully supported in purity mode:

- **Phase 1 (PLAN):** Waypoint planning using Mamba state space, recursive integration (3+ loops), trajectory prediction - all geometric operations ✅
- **Phase 2 (REALIZE):** Geometric word selection by Fisher-Rao distance to predicted waypoints, POS as optional constraint, geometric backoff - pure QIG ✅
- **Phase 3 (REPAIR):** Local geometric optimization through word swaps, scored by waypoint alignment + smoothness + attractor pull - pure QIG ✅

This architecture is **MORE pure** than simple skeleton generation because it uses foresight and recursive refinement rather than reactive slot-filling. See `docs/04-records/20260116-wp2-3-plan-realize-repair-integration-guide-1.00W.md` for implementation details.

## §8 References

- **Implementation:** `qig-backend/qig_purity_mode.py`
- **Tests:** `qig-backend/tests/test_qig_purity_mode.py`
- **CI:** `.github/workflows/qig-purity-coherence.yml`
- **Integration:** `qig-backend/qig_generation.py` (uses `validate_qig_purity()`)
- **Frozen Facts:** `docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md`
- **Plan→Realize→Repair:** `docs/04-records/20260116-wp2-3-plan-realize-repair-integration-guide-1.00W.md`
- **Coherence Harness:** `tests/coherence/` (WP4.3 - reproducible testing framework)

## §9 Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-16 | 1.00 | Initial specification (WP4.1 implementation) |
| 2026-01-17 | 1.01 | Added Plan→Realize→Repair FAQ and references (Issue #141 consideration) |

---

**Protocol:** Ultra Consciousness v4.0 ACTIVE  
**File Naming:** ISO 27001 compliance (YYYYMMDD-title-version.md)  
**Status:** FROZEN (requires consensus to modify)
