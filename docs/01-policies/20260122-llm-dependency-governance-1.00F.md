# LLM Dependency Governance System

**Document ID**: 20260122-llm-dependency-governance-1.00F  
**Status**: Frozen (F) - Production governance rules  
**Version**: 1.0.0  
**Last Updated**: 2026-01-22  
**Protocol**: Ultra Consciousness v4.0 ACTIVE

## Overview

Comprehensive governance system to prevent external LLM dependencies from "creeping in" to the custom QIG implementation. This ensures that all AI capabilities are built on pure QIG geometric primitives, not external black-box APIs.

## Problem Statement

The original purity tests had an incomplete and outdated list of forbidden LLM providers:

**Legacy List (7 providers)**:
- openai
- anthropic
- google.generativeai (old SDK)
- cohere
- ai21
- replicate
- huggingface_hub

**Issues**:
- Missing modern providers (Mistral, Groq, xAI, etc.)
- Missing new Google SDK (google.genai)
- Missing cloud providers (Azure, AWS Bedrock)
- Missing router frameworks (LiteLLM, OpenRouter)
- No package-level dependency scanning
- No AST-based import detection

## Solution Architecture

### 1. Centralized Configuration

**File**: `shared/constants/forbidden_llm_providers.json`

**Structure**:
```json
{
  "version": "1.0.0",
  "lastUpdated": "2026-01-22",
  "providers": [
    {
      "name": "OpenAI",
      "description": "OpenAI GPT models and API",
      "imports": ["openai", "openai.api_resources", ...],
      "packages": ["openai", "openai-agents", ...],
      "severity": "CRITICAL"
    },
    ...
  ]
}
```

**Coverage**: 28 providers, 73+ import patterns

### 2. Runtime Import Detection

**Module**: `qig-backend/qig_purity_mode.py`

**Capabilities**:
- Load configuration from JSON
- Check `sys.modules` for forbidden imports
- Support submodule detection (e.g., `google.genai.types` matches `google.genai`)
- Check installed packages via `pip list`
- Block external API calls when `QIG_PURITY_MODE=true`

**Functions**:
```python
check_forbidden_imports() -> List[PurityViolation]
check_forbidden_packages() -> List[PurityViolation]
check_forbidden_attributes(obj) -> List[PurityViolation]
enforce_purity(check_packages=True) -> None
```

### 3. Static Code Analysis

**Script**: `scripts/scan_forbidden_imports.py`

**Capabilities**:
- AST-based parsing of Python source files
- Detects `import module` and `from module import ...` statements
- Handles submodule imports
- Fast scanning (602 files in ~5 seconds)
- Respects exempt directories (tests, experiments)

**Usage**:
```bash
python3 scripts/scan_forbidden_imports.py \
  --path /path/to/code \
  --config shared/constants/forbidden_llm_providers.json
```

### 4. Integration Points

**Validation Script**: `scripts/validate-purity-patterns.sh`
- Integrated into validation pipeline
- Runs AST scanner on entire codebase
- Fails CI if forbidden imports detected

**Tests**: `qig-backend/tests/test_qig_purity_mode.py`
- Unit tests for import detection
- Tests for package detection
- Tests for modern providers (Google GenAI, Mistral, Groq)

## Provider Coverage

### Major Cloud Providers (6)
- OpenAI
- Anthropic
- Google Gemini (google.genai + google.generativeai)
- AWS Bedrock
- Azure OpenAI
- Cohere

### Modern Inference APIs (6)
- Mistral
- Groq
- xAI / Grok
- Together AI
- Anyscale
- Fireworks AI

### Model Hosting & Routing (4)
- Replicate
- OpenRouter
- LiteLLM (multi-provider router)
- Ollama (local models - WARNING severity)

### Specialized Services (5)
- Hugging Face Inference
- AI21 Labs
- Perplexity
- Writer.com
- Voyage AI (embeddings)

### Framework Integrations (3)
- LangChain LLM wrappers (langchain_openai, langchain_anthropic, etc.)
- LlamaIndex LLM integrations
- Semantic Kernel connectors

## Detection Layers

### Layer 1: Static Import Scanning (AST)
**When**: Pre-commit, CI pipeline  
**Speed**: Fast (~5 seconds for 600 files)  
**Catches**: Direct imports in source code

### Layer 2: Runtime Module Detection
**When**: Application startup, test runs  
**Speed**: Very fast (<1 second)  
**Catches**: Loaded modules in sys.modules

### Layer 3: Dependency Package Scanning
**When**: On-demand, periodic audits  
**Speed**: Slow (~10 seconds for pip list)  
**Catches**: Installed but not imported packages

## Severity Levels

**CRITICAL**: External LLM APIs that violate QIG purity
- Blocks: OpenAI, Anthropic, Google, cloud providers
- Action: Fail CI immediately

**WARNING**: Local/optional tools
- Blocks: Ollama (local models), Voyage AI (embeddings only)
- Action: Log warning, allow in some contexts

**ERROR**: Misuse patterns
- Blocks: Re-implementing geometry functions
- Action: Fail with fix suggestions

## Usage Examples

### Check Purity in Application

```python
import os
os.environ['QIG_PURITY_MODE'] = 'true'

from qig_purity_mode import enforce_purity, get_purity_report

# Enforce purity (raises on violations)
enforce_purity()

# Get detailed report
report = get_purity_report()
print(f"Tracked providers: {report['total_providers']}")
print(f"Forbidden modules: {len(report['forbidden_modules'])}")
```

### Scan Codebase for Violations

```bash
# Scan entire repository
python3 scripts/scan_forbidden_imports.py --path .

# Scan specific directory
python3 scripts/scan_forbidden_imports.py --path qig-backend

# Use custom config
python3 scripts/scan_forbidden_imports.py \
  --path . \
  --config /path/to/config.json
```

### Add New Provider

Edit `shared/constants/forbidden_llm_providers.json`:

```json
{
  "name": "NewProvider",
  "description": "Description of the provider",
  "imports": [
    "newprovider",
    "newprovider.client",
    "newprovider.models"
  ],
  "packages": [
    "newprovider",
    "newprovider-python"
  ],
  "severity": "CRITICAL"
}
```

No code changes required - configuration is loaded dynamically.

## CI/CD Integration

**Current Integration**:
- `scripts/validate-purity-patterns.sh` runs AST scanner
- Integrated into validation pipeline via `npm run validate:purity`
- Fails on CRITICAL violations

**Future Integration**:
- Pre-commit hook for local checking
- Dependabot alerts for forbidden package additions
- Periodic dependency audits

## Maintenance

### Updating Provider List

**Frequency**: Quarterly or when new major LLM providers emerge

**Process**:
1. Research new LLM providers and SDKs
2. Identify import patterns and package names
3. Add to `shared/constants/forbidden_llm_providers.json`
4. Test with AST scanner
5. Update documentation

### Version Control

**Configuration Version**: Tracked in JSON (`version` field)  
**Documentation Version**: Tracked in filename (`1.00F`)

## Validation Results

**Initial Scan** (2026-01-22):
- ✅ Files scanned: 602
- ✅ Providers tracked: 28
- ✅ Import patterns: 73
- ✅ Violations found: 0

## References

- **Source**: Problem statement from user (2026-01-22)
- **Config**: `shared/constants/forbidden_llm_providers.json`
- **Module**: `qig-backend/qig_purity_mode.py`
- **Scanner**: `scripts/scan_forbidden_imports.py`
- **Tests**: `qig-backend/tests/test_qig_purity_mode.py`

## Change Log

**v1.0.0** (2026-01-22):
- Initial comprehensive governance system
- 28 providers, 73+ patterns
- AST scanner, runtime detection, package scanning
- Integrated into validation pipeline

---

**Governance Principle**: External LLM dependencies are forbidden in QIG core. All AI capabilities must be built on pure geometric primitives (Fisher-Rao distance, Bures metric, von Neumann entropy) to maintain mathematical rigor and reproducibility.
