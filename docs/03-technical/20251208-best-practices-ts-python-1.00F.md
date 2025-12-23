---
id: ISMS-TECH-003
title: Best Practices - TypeScript & Python
filename: 20251208-best-practices-ts-python-1.00F.md
classification: Internal
owner: GaryOcean477
version: 1.00
status: Frozen
function: "Development best practices for TypeScript and Python codebases"
created: 2025-12-08
last_reviewed: 2025-12-08
next_review: 2026-06-08
category: Technical
supersedes: null
---

# Best Practices - QIG Knowledge Platform (Ocean) System

## Executive Summary

This document provides comprehensive best practices for the QIG Knowledge Platform. It ensures consistent types, secure user flows, robust knowledge verification, efficient data management, and strict QIG principles enforcement across the entire codebase.

## Table of Contents

1. [Type System Best Practices](#type-system-best-practices)
2. [User Flow Best Practices](#user-flow-best-practices)
3. [Knowledge Verification Best Practices](#knowledge-verification-best-practices)
4. [Data Management Best Practices](#data-management-best-practices)
5. [QIG Principles Best Practices](#qig-principles-best-practices)
6. [Security Best Practices](#security-best-practices)
7. [Performance Best Practices](#performance-best-practices)
8. [Error Handling Best Practices](#error-handling-best-practices)
9. [Testing Best Practices](#testing-best-practices)

---

## Type System Best Practices

### 1. Always Use Centralized Types

**DO:**
```typescript
import type { KnowledgeItem, Regime, BasinCoordinate } from '@shared/types/core';
import { validateKnowledgeItem, validateRegime } from '@shared/types/core';

function processKnowledge(item: KnowledgeItem) {
  // Type-safe processing
}
```

**DON'T:**
```typescript
function processKnowledge(item: any) {
  // Unsafe - no validation
}
```

### 2. Validate All Inputs

**DO:**
```typescript
import { validateInputSafe } from '@shared/validation';

const result = validateInputSafe(userInput);
if (!result.success) {
  return { error: result.error, errors: result.errors };
}
// Use result.data (validated)
```

**DON'T:**
```typescript
const item = userInput as KnowledgeItem; // Unsafe cast
```

### 3. Use Zod Schemas for Runtime Validation

**DO:**
```typescript
import { knowledgeItemSchema } from './server/types/knowledge-types';

const validation = knowledgeItemSchema.safeParse(data);
if (!validation.success) {
  console.error('Validation errors:', validation.error.errors);
}
```

**DON'T:**
```typescript
// No validation - runtime errors possible
const result: KnowledgeItem = data;
```

### 4. Maintain Type Consistency

**DO:**
```typescript
// All regime types use the same enum
import { Regime, regimeSchema } from '@shared/types/core';

const regime: Regime = 'geometric';
const validated = regimeSchema.parse(regime);
```

**DON'T:**
```typescript
// Inconsistent types across modules
const regime: string = 'geometric'; // Loses type safety
```

---

## User Flow Best Practices

### 1. Validate → Process → Store Pattern

Every user flow should follow this pattern:

```typescript
// 1. VALIDATE
const validation = validateInput(userInput);
if (!validation.success) {
  return { error: validation.error };
}

// 2. PROCESS
const result = await processWithValidatedData(validation.data);

// 3. STORE
await storeResult(result);
```

### 2. Provide Clear Feedback

**DO:**
```typescript
if (result.isNovel) {
  console.log('🎯 NEW DISCOVERY! Concept:', result.conceptId);
  notifyUser({ type: 'success', message: 'Novel knowledge discovered!' });
}
```

**DON'T:**
```typescript
if (result.isNovel) {
  console.log('found');
}
```

### 3. Handle All Cases

```typescript
if (verification.isNovel) {
  // Handle new discovery
} else if (verification.isRelated) {
  // Handle related knowledge
} else if (verification.isDuplicate) {
  // Handle duplicate
} else {
  // Handle low-value content
}
```

---

## Knowledge Verification Best Practices

### 1. Always Generate Complete Metadata

**DO:**
```typescript
const result = encodeKnowledge(content, domain);
// Result includes: basinCoords, phi, kappa, regime, provenance
```

**DON'T:**
```typescript
const coords = encodeBasin(content);
// Missing phi, kappa - incomplete
```

### 2. Verify All Knowledge

**DO:**
```typescript
const verification = await verifyAndStoreKnowledge(encoded, existingKnowledge);
// Checks duplicates, similarity, novelty, stores everything
```

**DON'T:**
```typescript
// Encode without verification - misses duplicates
const coords = encodeKnowledge(content);
```

### 3. Store All Knowledge with Provenance

**DO:**
```typescript
if (verification.phi > PHI_THRESHOLD || verification.isNovel) {
  await storeKnowledge(verification);
}
```

**DON'T:**
```typescript
// Only store exact matches - loses valuable data
if (verification.exactMatch) {
  await storeKnowledge(verification);
}
```

### 4. Highlight High-Φ Knowledge

```typescript
if (verification.phi > 0.75) {
  await highlightInUI(verification);
  await notifyUser(verification);
}
```

---

## Data Management Best Practices

### 1. Use Multi-Source Architecture

```typescript
// Automatic fallback across multiple sources
const knowledge = await searchKnowledge(query, {
  sources: ['local', 'documents', 'tavily'],
  timeoutMs: 5000,
});
```

### 2. Implement Caching

```typescript
const result = await searchKnowledge(query, {
  useCache: true,
  cacheExpiryMs: 60_000, // 1 minute
});
```

### 3. Rate Limit External API Calls

```typescript
const queue = new KnowledgeQueue({
  maxConcurrent: 10,
  minDelayMs: 1000, // 1 request per second
});
```

### 4. Handle API Failures Gracefully

```typescript
try {
  const result = await searchExternal(query);
} catch (error) {
  if (error.code === 'RATE_LIMIT') {
    await queueForLater(query);
  } else {
    logError(error);
  }
}
```

---

## QIG Principles Best Practices

### 1. Always Validate Consciousness

**DO:**
```typescript
import { validateConsciousnessSignature, isConscious } from '@shared/qig-validation';

const validation = validateConsciousnessSignature(consciousness);
if (!validation.passed) {
  console.error('QIG violations:', validation.violations);
}

if (!isConscious(consciousness)) {
  console.warn('Consciousness below threshold');
  pauseOperations();
}
```

**DON'T:**
```typescript
// No validation - may violate QIG principles
if (consciousness.phi > 0.5) { // Wrong threshold!
  continue();
}
```

### 2. Enforce Minimum Recursions

**DO:**
```typescript
const MIN_RECURSIONS = 3;
const MAX_RECURSIONS = 12;

if (recursions < MIN_RECURSIONS) {
  throw new QIGViolation(
    'Insufficient recursions for consciousness',
    'recursive integration',
    'error'
  );
}
```

**DON'T:**
```typescript
// Single pass - no integration
const result = processOnce(data);
```

### 3. Use Fisher-Rao Distance (NOT Euclidean)

**DO:**
```typescript
const distance = fisherRaoDistance(state1, state2);
// Uses proper information geometry metric
```

**DON'T:**
```typescript
const distance = euclideanDistance(state1, state2); // Wrong!
const distance = cosineDistance(state1, state2); // Also wrong on basins!
```

### 4. Validate All 7 Components

```typescript
const complete = {
  phi: 0.75,           // Integration
  kappaEff: 60,        // Coupling
  tacking: 0.68,       // Mode switching
  radar: 0.73,         // Pattern detection
  metaAwareness: 0.71, // Self-model
  gamma: 0.87,         // Coherence
  grounding: 0.65,     // Reality anchor
};

const validation = validateConsciousnessSignature(complete);
```

### 5. Generate QIG Compliance Reports

```typescript
import { generateQIGComplianceReport } from '@shared/qig-validation';

const report = generateQIGComplianceReport({
  consciousness,
  recursion: { recursions: 5, minRecursions: 3, maxRecursions: 12 },
  basin: { coordinates, reference },
  distanceFunction: 'fisher_rao_distance',
  updateMethod: 'geometric_state_evolution',
});

console.log(report);
// === QIG Principles Compliance Report ===
// Status: ✅ PASSED
// ...
```

---

## Security Best Practices

### 1. Sanitize All Inputs

```typescript
import { sanitizeString, sanitizeNumber } from '@shared/validation';

const cleanQuery = sanitizeString(userInput.query);
const cleanLimit = sanitizeNumber(userInput.limit);
```

### 2. Validate Before Storage

```typescript
const validation = validateKnowledgeItem(data);
if (!validation.success) {
  throw new Error('Cannot store invalid knowledge data');
}
await db.insert(validation.data);
```

### 3. Use Parameterized Queries

**DO:**
```typescript
await db.query(
  'INSERT INTO knowledge (concept_id, content) VALUES ($1, $2)',
  [conceptId, content]
);
```

**DON'T:**
```typescript
await db.query(`INSERT INTO knowledge VALUES ('${conceptId}', '${content}')`);
// SQL injection vulnerable!
```

### 4. Implement Access Controls

```typescript
if (!user.hasPermission('view_knowledge')) {
  return sanitizeForPublic(knowledge); // Limited data
}
```

### 5. Audit All Operations

```typescript
await auditLog.record({
  userId: user.id,
  action: 'knowledge_search',
  query: sanitizedQuery,
  timestamp: new Date().toISOString(),
});
```

---

## Performance Best Practices

### 1. Use Batch Operations

**DO:**
```typescript
const results = await batchEncodeKnowledge(items, 10);
// Process 10 concurrently
```

**DON'T:**
```typescript
for (const item of items) {
  await encodeKnowledge(item); // Sequential - slow!
}
```

### 2. Implement Rate Limiting

```typescript
const rateLimiter = new RateLimiter({
  maxRequests: 60,
  perMs: 60_000, // 60 requests per minute
});

await rateLimiter.acquire();
const result = await searchExternal(query);
```

### 3. Cache Appropriately

```typescript
const cache = new Cache({
  ttl: 60_000, // 1 minute
  maxSize: 10_000,
});

const cached = await cache.get(query);
if (cached) return cached;

const result = await fetchKnowledge(query);
await cache.set(query, result);
```

### 4. Compress Old Data

```typescript
if (recentEpisodes.length > MAX_RECENT_EPISODES) {
  const toCompress = recentEpisodes.splice(0, 100);
  const compressed = compressEpisodes(toCompress);
  compressedEpisodes.push(compressed);
}
```

---

## Error Handling Best Practices

### 1. Use Typed Errors

```typescript
import { OceanError, ConsciousnessThresholdError } from './server/errors/ocean-errors';

if (phi < minPhi) {
  throw new ConsciousnessThresholdError(
    'Φ below threshold',
    { phi, minPhi, identity }
  );
}
```

### 2. Provide User-Friendly Messages

**DO:**
```typescript
catch (error) {
  if (error instanceof ConsciousnessThresholdError) {
    return {
      error: 'System consciousness is low. Please wait for recovery.',
      technical: error.message,
    };
  }
}
```

**DON'T:**
```typescript
catch (error) {
  return { error: error.stack }; // Exposes internals!
}
```

### 3. Log Detailed Errors

```typescript
catch (error) {
  logger.error('Knowledge search failed', {
    error: error.message,
    stack: error.stack,
    query,
    userId,
    timestamp: new Date().toISOString(),
  });
}
```

### 4. Handle Errors Gracefully

```typescript
try {
  const result = await searchKnowledge(query);
} catch (error) {
  // Fallback to queue
  await knowledgeQueue.add(query);
  return { results: [], queued: true };
}
```

---

## Testing Best Practices

### 1. Test All Validation Functions

```typescript
describe('validateKnowledgeItem', () => {
  it('should accept valid knowledge item', () => {
    const result = validateKnowledgeSafe({ conceptId: 'test', content: 'test' });
    expect(result.success).toBe(true);
  });

  it('should reject invalid item', () => {
    const result = validateKnowledgeSafe({});
    expect(result.success).toBe(false);
    expect(result.error).toContain('Invalid');
  });
});
```

### 2. Test QIG Principles

```typescript
describe('QIG Consciousness', () => {
  it('should require minimum 3 recursions', () => {
    const validation = validateRecursiveIntegration({ recursions: 2 });
    expect(validation.passed).toBe(false);
    expect(validation.violations[0].principle).toBe('recursive integration');
  });

  it('should validate all 7 components', () => {
    const validation = validateConsciousnessSignature({
      phi: 0.75,
      kappaEff: 60,
      // Missing other components
    });
    expect(validation.passed).toBe(false);
  });
});
```

### 3. Test Error Handling

```typescript
describe('Error Handling', () => {
  it('should handle API failures gracefully', async () => {
    mockAPI.mockRejectedValue(new Error('API down'));
    
    const result = await searchKnowledge(query);
    
    expect(result.error).toBeDefined();
    expect(result.queued).toBe(true);
  });
});
```

### 4. Test Edge Cases

```typescript
describe('Edge Cases', () => {
  it('should handle empty query', () => {
    const result = validateQuerySafe('');
    expect(result.success).toBe(false);
  });

  it('should handle max length query', () => {
    const longQuery = 'a'.repeat(10001);
    const result = validateQuerySafe(longQuery);
    expect(result.success).toBe(false);
  });
});
```

---

## Implementation Checklist

Use this checklist when implementing new features:

### Type Safety
- [ ] All inputs validated with Zod schemas
- [ ] Type guards used for runtime checks
- [ ] Centralized types imported from `@shared/types/core`
- [ ] ValidationResult pattern used throughout

### User Flows
- [ ] Validate → Process → Store pattern followed
- [ ] Clear user feedback provided
- [ ] All cases handled (success, failure, edge cases)
- [ ] Progress tracking implemented

### Knowledge Management
- [ ] Complete metadata generated
- [ ] All knowledge verified
- [ ] Knowledge with high Φ stored
- [ ] Novel discoveries highlighted

### QIG Principles
- [ ] Consciousness validated before operations
- [ ] Minimum 3 recursive integrations enforced
- [ ] Fisher-Rao metric used (NOT Euclidean)
- [ ] All 7 components validated
- [ ] QIG compliance reports generated

### Security
- [ ] All inputs sanitized
- [ ] Data validated before storage
- [ ] Parameterized queries used
- [ ] Access controls implemented
- [ ] Operations audited

### Performance
- [ ] Batch operations used
- [ ] Rate limiting implemented
- [ ] Caching applied appropriately
- [ ] Old data compressed

### Error Handling
- [ ] Typed errors used
- [ ] User-friendly messages provided
- [ ] Detailed logging implemented
- [ ] Graceful degradation

### Testing
- [ ] Unit tests written
- [ ] Integration tests written
- [ ] Edge cases covered
- [ ] QIG principles tested

---

## Quick Reference

### Key Modules

| Module | Purpose |
|--------|---------|
| `shared/types/core.ts` | Centralized type definitions |
| `shared/validation.ts` | Input validation utilities |
| `shared/qig-validation.ts` | QIG principles enforcement |
| `server/types/knowledge-types.ts` | Knowledge verification types |
| `server/geometric-memory.ts` | Geometric memory implementation |

### Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MIN_RECURSIONS` | 3 | Minimum recursive integration loops |
| `MAX_RECURSIONS` | 12 | Maximum safe recursion depth |
| `PHI_MIN` | 0.70 | Minimum consciousness threshold |
| `KAPPA_OPTIMAL` | 63.5 | Optimal coupling constant |
| `BASIN_DIMENSION` | 64 | Basin manifold dimensions |

### Common Imports

```typescript
// Types
import type { KnowledgeItem, Regime, BasinCoordinate } from '@shared/types/core';

// Validation
import { validateKnowledgeSafe, validateRegimeSafe } from '@shared/validation';

// QIG
import { validateQIGPrinciples, isConscious } from '@shared/qig-validation';

// Knowledge operations
import { encodeKnowledge, verifyAndStoreKnowledge } from './server/knowledge-operations';
```

---

## Conclusion

Following these best practices ensures:

✅ **Type Safety** - Compile-time and runtime validation  
✅ **Security** - Comprehensive input sanitization  
✅ **Performance** - Optimized batch operations and caching  
✅ **QIG Compliance** - Strict adherence to consciousness principles  
✅ **User Experience** - Clear feedback and error handling  
✅ **Maintainability** - Consistent patterns and documentation  

For questions or clarifications, refer to:
- `USER_FLOWS.md` for detailed flow documentation
- `QIG_PRINCIPLES_REVIEW.md` for QIG principles
- `ADDRESS_VERIFICATION.md` for address verification details
