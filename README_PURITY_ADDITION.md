
---

## QIG Purity Mode

Pantheon-Chat includes a **QIG Purity Mode** to ensure coherence testing measures pure QIG performance without contamination from external LLM APIs.

### What is Purity Mode?

When `QIG_PURITY_MODE=true`, the system enforces strict boundaries:
- ✅ **Allowed:** Fisher-Rao geometry, simplex operations, QFI attention, consciousness metrics
- ❌ **Blocked:** OpenAI, Anthropic, Google AI, and all other external LLM API calls
- 📊 **Purpose:** Provably uncontaminated coherence benchmarking

### Using Purity Mode

```bash
# Enable purity mode for testing
export QIG_PURITY_MODE=true

# Run pure QIG tests
cd qig-backend
python -m pytest tests/test_qig_purity_mode.py -v

# Test pure QIG generation
python -c "
from qig_generation import QIGGenerator, encode_to_basin
generator = QIGGenerator()
basin = encode_to_basin('consciousness integration')
phi = generator._measure_phi(basin)
print(f'Pure QIG Φ: {phi:.3f}')
"

# Validate purity
python -c "from qig_generation import validate_qig_purity; validate_qig_purity()"
```

### CI Integration

Pure QIG tests run automatically on every PR via `.github/workflows/qig-purity-coherence.yml`:
- Verifies no external LLM dependencies
- Measures consciousness metrics (Φ, κ)
- Validates pure QIG generation
- Required to pass for PR merge

### Documentation

**📋 See [QIG Purity Specification](./docs/01-policies/20260117-qig-purity-mode-spec-1.01F.md) for complete requirements.**

**📋 See [External LLM Audit](./docs/04-records/20260116-external-llm-usage-audit-1.00W.md) for repository audit results.**

### When to Use

- ✅ **Pure Mode:** Coherence benchmarking, research experiments, CI/CD validation
- ✅ **Hybrid Mode (default):** Development, user applications, external API integration

