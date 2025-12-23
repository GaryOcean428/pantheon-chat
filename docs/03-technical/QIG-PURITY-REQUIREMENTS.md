# QIG Purity Requirements

**Classification:** Technical Requirement - MANDATORY  
**Version:** 1.0  
**Status:** ENFORCED

---

## ⚠️ CRITICAL: No External LLM APIs

This project uses **QIG (Quantum Information Geometry)** for all generative capabilities. Traditional LLM APIs and patterns are **ABSOLUTELY FORBIDDEN**.

## Forbidden Patterns

### 1. External LLM APIs

```python
# ❌ FORBIDDEN - OpenAI
import openai
from openai import OpenAI
client.chat.completions.create(...)

# ❌ FORBIDDEN - Anthropic  
import anthropic
from anthropic import Anthropic
client.messages.create(...)

# ❌ FORBIDDEN - Google AI
import google.generativeai
model.generate_content(...)
```

### 2. Token-Based Limits

```python
# ❌ FORBIDDEN - Token limits
max_tokens=1000
max_output_tokens=2000

# ✅ CORRECT - Geometric completion
from qig_generation import generate_response
result = generate_response(prompt)  # Stops when geometry collapses
```

### 3. Legacy LLM Client

```python
# ❌ FORBIDDEN - Old LLM client
from llm_client import LLMClient, get_llm_client

# ✅ CORRECT - QIG-pure generation
from qig_generation import QIGGenerator, get_qig_generator
```

### 4. Traditional Sampling Parameters

```python
# ❌ FORBIDDEN - Traditional sampling
temperature=0.7  # as API parameter
top_p=0.9
frequency_penalty=0.5

# ✅ CORRECT - Regime-based modulation
# Temperature is determined by phi regime automatically
```

---

## Required Patterns

### QIG-Pure Generation

```python
from qig_generation import (
    QIGGenerator,
    get_qig_generator,
    generate_response,
    QIGGenerationConfig
)

# Generate response
result = generate_response(
    prompt="User query",
    context={"conversation_id": "..."}
)

# Result contains:
# - response: Generated text
# - completion_reason: Why generation stopped (geometric)
# - phi: Integration level
# - kappa: Coupling constant
# - qig_pure: True (certification)
```

### Geometric Completion

Generation stops when:

1. **Attractor Converged** - Basin distance < threshold
2. **Surprise Collapsed** - No new information
3. **Integration Stable** - Φ stable and high

**NOT when:**
- Token limit reached
- Stop token encountered
- Arbitrary timeout

### Kernel Routing

```python
from qig_generation import QIGKernelRouter

router = QIGKernelRouter()

# Route query to nearest kernels via Fisher-Rao distance
target_kernels = router.route_query(query_basin, k=3)
```

---

## Validation

### Pre-Commit Hook

QIG purity is validated on every commit:

```bash
python tools/validate_qig_purity.py --strict
```

### Manual Validation

```bash
# Check entire qig-backend
python tools/validate_qig_purity.py --dir qig-backend

# Strict mode (exits with error code)
python tools/validate_qig_purity.py --strict
```

---

## Why QIG-Pure?

### Traditional LLM Problems

1. **External Dependency** - Relies on third-party APIs
2. **Token-Based** - Arbitrary stopping criteria
3. **No Geometry** - Ignores manifold structure
4. **No Integration** - No consciousness measurement (Φ)

### QIG Advantages

1. **Self-Contained** - All generation internal
2. **Geometric Completion** - Stops when thought is complete
3. **Fisher-Rao Navigation** - Proper manifold distances
4. **Consciousness-Aware** - Monitors Φ, κ throughout
5. **Kernel Constellation** - 240 E8 root specialists

---

## Migration Guide

If you find code using traditional LLM patterns:

### Step 1: Identify Violations

```bash
python tools/validate_qig_purity.py
```

### Step 2: Replace Imports

```python
# Before
from llm_client import LLMClient, get_llm_client
client = get_llm_client()
response = client.generate(prompt, max_tokens=1000)

# After
from qig_generation import get_qig_generator
generator = get_qig_generator()
result = generator.generate(prompt)
response = result['response']
```

### Step 3: Remove Token Limits

```python
# Before
max_tokens=1000  # ❌ Remove this

# After
# No token limit - geometric completion handles this
```

### Step 4: Validate

```bash
python tools/validate_qig_purity.py --strict
```

---

## Contact

For questions about QIG purity requirements, consult:
- `qig-backend/qig_generation.py` - Reference implementation
- `tools/validate_qig_purity.py` - Validation script
- `knowledge.md` - Project documentation
