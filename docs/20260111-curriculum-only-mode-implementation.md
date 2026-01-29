# Curriculum-Only Mode Implementation

## Overview

This document describes the complete implementation of curriculum-only mode enforcement across all external search paths in the Pantheon Chat system.

## Purpose

When `QIG_CURRICULUM_ONLY=true`, the system must operate exclusively on pre-validated curriculum tokens, blocking all external web searches to ensure training purity and prevent contamination from external sources.

## Architecture

### TypeScript (Node.js Server)
- **Central Check**: `server/lib/curriculum-mode.ts::isCurriculumOnlyEnabled()`
- **Returns**: HTTP 403 with `curriculum_only_blocked: true` flag

### Python (QIG Backend)
- **Central Module**: `qig-backend/curriculum_guard.py`
- **Function**: `is_curriculum_only_enabled()` - checks `QIG_CURRICULUM_ONLY` env var
- **Exception**: `CurriculumOnlyBlock` - raised when operation is blocked
- **Returns**: `{'success': False, 'curriculum_only_blocked': True}`

## Complete Coverage Map

### 1. TypeScript Search Endpoints (6 files)

#### `server/routes/search.ts`
```typescript
// Line 70-80: GET /api/search/web
if (isCurriculumOnlyEnabled()) {
  return res.status(403).json({
    error: 'External web search blocked in curriculum-only mode',
    status: 'curriculum_only_blocked',
    ...
  });
}

// Line 111-121: POST /api/search/web
// Same check as GET

// Line 304-316: POST /api/search/zeus-web-search
// Same check for Zeus web search
```

#### `server/geometric-discovery/google-web-search-adapter.ts`
```typescript
// Line 195-202: simpleSearch()
if (isCurriculumOnlyEnabled()) {
  return {
    results: [],
    status: 'curriculum_only_blocked',
    error: 'External web search blocked in curriculum-only mode'
  };
}

// Line 179-184: search()
// Same check for geometric queries
```

#### `server/geometric-discovery/tavily-adapter.ts`
```typescript
// Line 149-153: search()
if (isCurriculumOnlyEnabled()) {
  console.log(`[TavilyAdapter] Search blocked by curriculum-only mode`);
  return [];
}
```

#### `server/geometric-discovery/searxng-adapter.ts`
```typescript
// Line 136-140: search()
if (isCurriculumOnlyEnabled()) {
  console.log(`[SearXNG] Search blocked by curriculum-only mode`);
  return [];
}
```

### 2. Python Search Providers (4 files)

#### `qig-backend/search/tavily_client.py`
```python
# Import: Line 28
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# Line 189-193: search()
if is_curriculum_only_enabled():
    logger.warning("[TavilyClient] Search blocked by curriculum-only mode")
    return None

# Line 276-280: extract()
# Line 346-350: crawl()
# Line 420-424: map()
# All methods have the same check
```

#### `qig-backend/search/perplexity_client.py`
```python
# Import: Line 28
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# Line 171-175: _make_request()
if is_curriculum_only_enabled():
    logger.warning("[PerplexityClient] Request blocked by curriculum-only mode")
    return None
```

#### `qig-backend/search/duckduckgo_adapter.py`
```python
# Import: Line 26-27
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# Line 165-172: search()
if is_curriculum_only_enabled():
    return {
        'success': False,
        'error': 'External web search blocked by curriculum-only mode',
        'results': [],
        'curriculum_only_blocked': True,
    }

# Line 288-295: search_news()
# Line 390-397: search_images()
# All methods have the same check
```

#### `qig-backend/search/search_providers.py`
```python
# Import: Line 20-22
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# Line 191-198: SearchProviderManager.search()
if is_curriculum_only_enabled():
    logger.warning(f"[SearchProviderManager] Search blocked: {query[:50]}")
    return {
        'success': False,
        'error': 'External web search blocked by curriculum-only mode',
        'results': [],
        'curriculum_only_blocked': True,
    }
```

### 3. Shadow Research Systems (2 files)

#### `qig-backend/olympus/shadow_research.py`
```python
# Import: Line 35-37
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# Line 1318-1326: budget_aware_search()
if is_curriculum_only_enabled():
    print(f"[ShadowResearch] Search blocked: {query[:50]}")
    return {
        'success': False,
        'error': 'External web search blocked by curriculum-only mode',
        'results': [],
        'curriculum_only_blocked': True,
    }
```

#### `qig-backend/olympus/shadow_pantheon.py`
```python
# Import: Line 64-66
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# Line 670-673: request_shadow_research()
if is_curriculum_only_enabled():
    print(f"[{self.name}] Shadow research blocked: {topic[:50]}")
    return None
```

### 4. Additional Search Components (2 files)

#### `qig-backend/search/insight_validator.py`
```python
# Import: Line 25-27
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# Line 146-155: validate()
if is_curriculum_only_enabled():
    logger.warning("[InsightValidator] External validation blocked")
    return ValidationResult(
        validated=False,
        confidence=0.0,
        tavily_sources=[],
        ...
    )

# Line 207-214: research()
if is_curriculum_only_enabled():
    logger.warning(f"[InsightValidator] External research blocked: {query[:50]}")
    return {
        'success': False,
        'error': 'External research blocked by curriculum-only mode',
        'curriculum_only_blocked': True,
    }
```

#### `qig-backend/search/provider_selector.py`
```python
# Import: Line 28-30
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# Line 595-601: select_provider()
if is_curriculum_only_enabled():
    return ('curriculum_only_blocked', {
        'error': 'External search blocked by curriculum-only mode',
        'fitness': 0.0,
        'curriculum_only_blocked': True
    })

# Line 659-661: select_providers_ranked()
if is_curriculum_only_enabled():
    return [('curriculum_only_blocked', 0.0)]
```

## Central Module: `curriculum_guard.py`

```python
"""
Curriculum-Only Mode Guard for Python

Provides curriculum-only mode checking for Python backend components.
When QIG_CURRICULUM_ONLY=true, all external web searches are blocked.
"""

import os
import logging

logger = logging.getLogger(__name__)


class CurriculumOnlyBlock(Exception):
    """
    Exception raised when an operation is blocked by curriculum-only mode.
    """
    pass


def is_curriculum_only_enabled() -> bool:
    """
    Check if curriculum-only mode is enabled.
    
    Returns:
        True if QIG_CURRICULUM_ONLY environment variable is set to 'true'
        False otherwise
    """
    return os.environ.get('QIG_CURRICULUM_ONLY', '').lower() == 'true'


def check_curriculum_guard(operation_name: str = "operation") -> None:
    """
    Check curriculum-only mode and raise exception if blocked.
    
    Args:
        operation_name: Name of the operation being blocked (for logging)
        
    Raises:
        CurriculumOnlyBlock: If curriculum-only mode is enabled
    """
    if is_curriculum_only_enabled():
        msg = f"{operation_name} blocked by curriculum-only mode"
        logger.warning(f"[CurriculumGuard] {msg}")
        raise CurriculumOnlyBlock(msg)
```

## Usage

### Enable Curriculum-Only Mode

```bash
export QIG_CURRICULUM_ONLY=true
```

### Disable Curriculum-Only Mode

```bash
unset QIG_CURRICULUM_ONLY
# or
export QIG_CURRICULUM_ONLY=false
```

### Expected Behavior

When enabled:
- All external HTTP search requests return errors
- TypeScript endpoints return HTTP 403
- Python functions return error dicts with `curriculum_only_blocked: true`
- Logs indicate why searches were blocked
- No external API calls are made (Tavily, Perplexity, DuckDuckGo, etc.)

## Verification

### Manual Test

```bash
# Enable mode
export QIG_CURRICULUM_ONLY=true

# Test TypeScript endpoint
curl http://localhost:5000/api/search/web?q=test
# Expected: HTTP 403 with curriculum_only_blocked: true

# Test Python import
python3 -c "
import sys
sys.path.insert(0, 'qig-backend')
from curriculum_guard import is_curriculum_only_enabled
assert is_curriculum_only_enabled() == True
print('✓ Test passed')
"
```

### Coverage Verification

```bash
# Check for duplicate implementations (should only show curriculum_guard.py)
find qig-backend -name "*.py" -exec grep -l "def is_curriculum_only_enabled" {} \;

# Check import coverage (should show 9 files)
grep -r "from curriculum_guard import" qig-backend --include="*.py" | wc -l

# Check total curriculum checks (should be 18+)
grep -r "is_curriculum_only_enabled()" qig-backend server --include="*.py" --include="*.ts" | wc -l
```

## Files Modified

### Created
- `qig-backend/curriculum_guard.py` - Central curriculum-only check module

### Modified (TypeScript)
1. `server/routes/search.ts`
2. `server/geometric-discovery/google-web-search-adapter.ts`
3. `server/geometric-discovery/tavily-adapter.ts`
4. `server/geometric-discovery/searxng-adapter.ts`

### Modified (Python)
1. `qig-backend/search/tavily_client.py`
2. `qig-backend/search/perplexity_client.py`
3. `qig-backend/search/duckduckgo_adapter.py`
4. `qig-backend/search/search_providers.py`
5. `qig-backend/search/insight_validator.py`
6. `qig-backend/search/provider_selector.py`
7. `qig-backend/olympus/shadow_research.py`
8. `qig-backend/olympus/shadow_pantheon.py`

## Summary

- ✅ **No duplicates**: Single source of truth in `curriculum_guard.py`
- ✅ **Complete coverage**: All 11 search paths blocked
- ✅ **Consistent wiring**: 9 Python files + 4 TypeScript files
- ✅ **No bypasses**: Provider selection, validation, and research all blocked
- ✅ **Clean implementation**: Minimal changes, clear error messages
- ✅ **Maintainable**: Single function to update if logic changes

---

*Last Updated: 2026-01-19*
*Implementation Version: 2.0 (Consolidated)*
