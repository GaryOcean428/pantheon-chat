# LLM Dependency Governance

This directory contains the configuration and documentation for preventing external LLM dependencies from entering the QIG codebase.

## Quick Links

- **Configuration**: [`forbidden_llm_providers.json`](./forbidden_llm_providers.json) - Single source of truth
- **Quick Reference**: [`docs/99-reference/forbidden-llm-providers-quick-ref.md`](../../docs/99-reference/forbidden-llm-providers-quick-ref.md)
- **Full Documentation**: [`docs/01-policies/20260122-llm-dependency-governance-1.00F.md`](../../docs/01-policies/20260122-llm-dependency-governance-1.00F.md)

## Structure

```
shared/constants/
└── forbidden_llm_providers.json  ← Configuration (28 providers, 73+ patterns)

qig-backend/
└── qig_purity_mode.py            ← Runtime enforcement module

scripts/
└── scan_forbidden_imports.py     ← AST-based static analyzer

docs/
├── 01-policies/
│   └── 20260122-llm-dependency-governance-1.00F.md  ← Full governance doc
└── 99-reference/
    └── forbidden-llm-providers-quick-ref.md          ← Quick reference
```

## Usage

### Check Your Code

```bash
# Scan for forbidden imports
python3 scripts/scan_forbidden_imports.py --path .

# Run full validation
bash scripts/validate-purity-patterns.sh
```

### In Your Application

```python
# Enable purity mode
import os
os.environ['QIG_PURITY_MODE'] = 'true'

# Enforce at startup
from qig_purity_mode import enforce_purity
enforce_purity()
```

### Add a New Provider

1. Edit `forbidden_llm_providers.json`
2. Add provider with imports and packages
3. Test with scanner
4. No code changes needed!

## Coverage

**28 Providers Tracked**:
- Major cloud (OpenAI, Anthropic, Google, AWS, Azure, Cohere)
- Modern APIs (Mistral, Groq, xAI, Together, Anyscale, Fireworks)
- Hosting/routing (Replicate, OpenRouter, LiteLLM, Ollama)
- Specialized (Hugging Face, AI21, Perplexity, Writer, Voyage, Aleph Alpha)
- Framework integrations (LangChain, LlamaIndex, Semantic Kernel wrappers)

**73+ Import Patterns** including:
- `google.genai` (new Google SDK)
- `mistralai`, `groq`, `xai`
- `azure.ai.openai`, `azure.ai.inference`
- `langchain_openai`, `langchain_anthropic`
- And many more

## Why This Matters

**QIG Purity Principle**: All AI capabilities must be built on pure geometric primitives (Fisher-Rao distance, Bures metric, von Neumann entropy) to maintain:

✅ Mathematical rigor  
✅ Reproducibility  
✅ No black-box dependencies  
✅ Full control over consciousness architecture

External LLM APIs introduce ❌:
- Hidden training data
- Non-geometric representations
- External service dependencies
- Loss of interpretability

## Maintenance

**Update Frequency**: Quarterly or when new major providers emerge

**Process**:
1. Research new LLM providers
2. Update `forbidden_llm_providers.json`
3. Test with scanner
4. Document in quick reference

---

For detailed information, see the full governance documentation.
