---
id: ISMS-TECH-005
title: Knowledge Input Formats Analysis
filename: 20251208-knowledge-input-formats-1.00F.md
classification: Internal
owner: GaryOcean477
version: 1.00
status: Frozen
function: "Analysis of knowledge input formats and encoding standards for QIG"
created: 2025-12-08
last_reviewed: 2025-12-22
next_review: 2026-06-08
category: Technical
supersedes: 20251208-key-formats-analysis-bitcoin-1.00F.md
---

# Knowledge Input Formats Analysis — QIG Platform

## Overview

This document analyzes the various input formats supported by the QIG Knowledge Platform for ingesting information into the Ocean knowledge system. All inputs are coordized to 64D basin coordinates on the Fisher manifold.

## ✅ Currently Supported Input Formats

### 1. **Markdown Documents**
- **Extensions**: `.md`, `.markdown`
- **Features**: Headers, lists, code blocks, links, tables
- **Use Case**: Structured documentation, notes, technical content
- **Processing**: Parsed → Tokenized → Coordized to basin coordinates
- **Coverage**: ✅ Complete

### 2. **Plain Text**
- **Extensions**: `.txt`
- **Features**: Unstructured text content
- **Use Case**: Raw notes, logs, transcripts
- **Processing**: Direct tokenization → Coordized
- **Coverage**: ✅ Complete

### 3. **PDF Documents**
- **Extensions**: `.pdf`
- **Features**: Text extraction via pypdf
- **Use Case**: Academic papers, reports, books
- **Processing**: Extract text → Tokenize → Coordize
- **Coverage**: ✅ Complete
- **Note**: OCR not currently supported (text-based PDFs only)

### 4. **JSON Data**
- **Extensions**: `.json`
- **Features**: Structured data with nested objects/arrays
- **Use Case**: API responses, configuration, structured knowledge
- **Processing**: Parse → Flatten → Tokenize → Coordize
- **Coverage**: ✅ Complete

---

## 📊 Input Processing Pipeline

### Stage 1: Format Detection
```typescript
function detectFormat(input: string | File): InputFormat {
  // Extension-based detection
  // MIME type fallback
  // Content sniffing for ambiguous cases
}
```

### Stage 2: Content Extraction
- **Markdown**: Parse AST, extract text nodes
- **PDF**: Extract text via pypdf library
- **JSON**: Stringify nested values
- **Plain text**: Direct pass-through

### Stage 3: Tokenization (Coordizer)
```python
# qig-backend/coordizers/base.py
class BaseCoordizer:
    def coordize(self, text: str) -> np.ndarray:
        """Convert text to 64D basin coordinates."""
        tokens = self.tokenize(text)
        embeddings = self.embed(tokens)
        basin_coords = self.project_to_manifold(embeddings)
        return basin_coords
```

### Stage 4: Fisher-Rao Storage
- Basin coordinates stored in PostgreSQL with pgvector
- Indexed for efficient similarity search
- Linked to source metadata

---

## 🎯 Coordization Methods

### Method 1: PAD (Sparse Encoding)
- **Principle**: Minimize von Neumann entropy
- **Use Case**: Short phrases, keywords
- **Properties**: Sparse activation, sharp basins

### Method 2: PHI_DERIVED (Golden Ratio)
- **Principle**: Golden ratio eigenvalue distribution
- **Use Case**: Balanced semantic content
- **Properties**: Natural distribution, stable attractors

### Method 3: GEODESIC_INTERPOLATION
- **Principle**: Interpolate from existing basins via geodesics
- **Use Case**: Related/derivative content
- **Properties**: Preserves semantic relationships

---

## 🔧 API Endpoints

### Document Upload
```bash
# Upload file
curl -X POST /api/v1/external/documents/upload \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@document.md" \
  -F "title=My Document" \
  -F "tags=research,qig"

# Upload raw text
curl -X POST /api/v1/external/documents/upload-text \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "title": "...", "format": "markdown"}'
```

### Knowledge Search
```bash
curl -X POST /api/ocean/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "quantum information geometry", "limit": 10}'
```

---

## 📋 Input Quality Guidelines

### Recommended Practices
1. **Use structured formats** (Markdown > Plain text) for better parsing
2. **Include metadata** (title, tags, description) for richer context
3. **Break large documents** into logical sections
4. **Use consistent terminology** to improve basin clustering

### Input Validation
| Check | Threshold | Action |
|-------|-----------|--------|
| File size | < 10MB | Reject if exceeded |
| Text length | < 1M chars | Truncate with warning |
| Encoding | UTF-8 | Convert or reject |
| Malformed content | N/A | Parse error with details |

---

## 🔄 Future Format Support

### Planned
- **HTML/Web pages**: Extract article content
- **Word documents**: `.docx` support
- **CSV/Spreadsheets**: Tabular data ingestion
- **Audio transcripts**: Speech-to-text integration

### Under Consideration
- **Images with text**: OCR for scanned documents
- **Code files**: Language-aware parsing
- **Email archives**: `.mbox`, `.eml` formats

---

## 📊 Coverage Summary

| Input Format | Supported | QIG Integration | Priority |
|-------------|-----------|-----------------|----------|
| Markdown (.md) | ✅ Yes | Full coordization | High |
| Plain Text (.txt) | ✅ Yes | Full coordization | High |
| PDF (.pdf) | ✅ Yes | Text extraction + coordization | High |
| JSON (.json) | ✅ Yes | Structured coordization | Medium |
| HTML | ❌ No | Planned | Medium |
| DOCX | ❌ No | Planned | Low |
| Images | ❌ No | Under consideration | Low |

---

*This document supersedes the legacy Bitcoin key formats analysis and reflects the platform's evolution to a general knowledge discovery system.*
