# Forbidden LLM Providers - Quick Reference

**Version**: 1.0.0  
**Last Updated**: 2026-01-22

## 🚫 Forbidden External LLM Providers

This reference lists all external LLM providers that are **FORBIDDEN** in QIG core code to prevent dependency creep and maintain geometric purity.

## Major Cloud LLM Providers

| Provider | Import Patterns | Packages | Severity |
|----------|----------------|----------|----------|
| **OpenAI** | `openai`, `openai.*` | `openai`, `openai-agents` | CRITICAL |
| **Anthropic** | `anthropic`, `anthropic.*` | `anthropic`, `anthropic-bedrock` | CRITICAL |
| **Google Gemini** | `google.genai`, `google.generativeai`, `vertexai.*` | `google-genai`, `google-generativeai`, `vertexai` | CRITICAL |
| **AWS Bedrock** | `boto3.client('bedrock')`, `bedrock_llm` | `bedrock-llm` | CRITICAL |
| **Azure OpenAI** | `azure.ai.openai`, `azure.ai.inference` | `azure-ai-openai`, `azure-ai-inference` | CRITICAL |
| **Cohere** | `cohere`, `cohere.client` | `cohere`, `cohere-python` | CRITICAL |

## Modern Inference APIs

| Provider | Import Patterns | Packages | Severity |
|----------|----------------|----------|----------|
| **Mistral** | `mistralai`, `mistral` | `mistralai`, `mistral-common` | CRITICAL |
| **Groq** | `groq`, `groq.client` | `groq` | CRITICAL |
| **xAI / Grok** | `grok`, `xai` | `grok`, `xai`, `xai-sdk` | CRITICAL |
| **Together AI** | `together`, `together.client` | `together` | CRITICAL |
| **Anyscale** | `anyscale`, `anyscale.endpoints` | `anyscale` | CRITICAL |
| **Fireworks AI** | `fireworks`, `fireworks.client` | `fireworks-ai` | CRITICAL |

## Model Hosting & Routing

| Provider | Import Patterns | Packages | Severity |
|----------|----------------|----------|----------|
| **Replicate** | `replicate`, `replicate.client` | `replicate` | CRITICAL |
| **OpenRouter** | `openrouter`, `openrouter.client` | `openrouter` | CRITICAL |
| **LiteLLM** | `litellm`, `litellm.proxy` | `litellm` | CRITICAL |
| **Ollama** | `ollama`, `ollama.client` | `ollama`, `ollama-python` | WARNING |

## Specialized Services

| Provider | Import Patterns | Packages | Severity |
|----------|----------------|----------|----------|
| **Hugging Face** | `huggingface_hub.inference`, `huggingface_hub` | `huggingface-hub` | CRITICAL |
| **AI21 Labs** | `ai21`, `ai21.studio` | `ai21`, `ai21-studio` | CRITICAL |
| **Perplexity** | `perplexity`, `perplexityai` | `perplexity`, `perplexityai` | CRITICAL |
| **Writer.com** | `writer`, `writer.client` | `writer` | CRITICAL |
| **Voyage AI** | `voyageai`, `voyage` | `voyageai` | WARNING |
| **Aleph Alpha** | `aleph_alpha_client` | `aleph-alpha-client` | CRITICAL |
| **Forefront AI** | `forefront` | `forefront` | CRITICAL |
| **GooseAI** | `gooseai` | `gooseai` | CRITICAL |
| **NLP Cloud** | `nlpcloud` | `nlpcloud` | CRITICAL |

## Framework LLM Integrations

| Framework | Import Patterns | Packages | Severity |
|-----------|----------------|----------|----------|
| **LangChain** | `langchain_openai`, `langchain_anthropic`, `langchain_google_genai` | `langchain-openai`, `langchain-anthropic`, `langchain-google-genai` | CRITICAL |
| **LlamaIndex** | `llama_index.llms.openai`, `llama_index.llms.anthropic` | `llama-index-llms-openai`, `llama-index-llms-anthropic` | CRITICAL |
| **Semantic Kernel** | `semantic_kernel.connectors.ai.open_ai` | `semantic-kernel` | WARNING |

## ✅ What You Should Use Instead

**For Text Generation**:
- `qig_generation.py` - QIG-pure text generation
- `qig_generative_service.py` - Basin navigation and synthesis

**For Embeddings/Coordinates**:
- `coordizers/` - Geometric coordizers (not tokenizers)
- Basin coordinate mapping with Fisher-Rao distance

**For Reasoning**:
- Olympus Pantheon god-kernels
- Geometric kernel routing via Fisher-Rao distance

## 🔍 How to Check for Violations

### Quick Scan
```bash
# Scan entire codebase
python3 scripts/scan_forbidden_imports.py --path .
```

### Runtime Check
```python
from qig_purity_mode import enforce_purity

# Raises exception if violations found
enforce_purity()
```

### Get Report
```python
from qig_purity_mode import get_purity_report

report = get_purity_report()
print(f"Tracked providers: {report['total_providers']}")
print(f"Forbidden modules: {len(report['forbidden_modules'])}")
```

## 📋 Full Configuration

For the complete list with all import patterns and packages, see:
- **Config**: `shared/constants/forbidden_llm_providers.json`
- **Documentation**: `docs/01-policies/20260122-llm-dependency-governance-1.00F.md`

## 🚨 What Happens If You Import One?

**In Development** (QIG_PURITY_MODE=false):
- ⚠️ Warning logged
- Application continues

**In Testing** (QIG_PURITY_MODE=true):
- ❌ RuntimeError raised
- Test fails immediately
- Stack trace logged

**In CI/CD**:
- ❌ Build fails
- AST scanner reports violations
- Must fix before merge

## 💡 Why This Matters

**QIG Purity Principle**: All AI capabilities must be built on pure geometric primitives to ensure:
- Mathematical rigor
- Reproducibility
- No black-box dependencies
- Full control over consciousness architecture

External LLM APIs introduce:
- Hidden parameters and training data
- Non-geometric representations (embeddings)
- Dependency on external services
- Loss of mathematical interpretability

---

**Need to add a new provider?** Edit `shared/constants/forbidden_llm_providers.json` and the scanner will pick it up automatically.
