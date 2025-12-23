# Pantheon QIG External API Documentation

> **Quick Start**: Use the [TypeScript SDK](#typescript-sdk) for the easiest integration.

**Version:** 1.0.0  
**Base URL:** `/api/v1/external`  
**Last Updated:** 2025-01-15

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [CORS Configuration](#cors-configuration)
5. [Response Format](#response-format)
6. [Simple API (Quick Start)](#simple-api-quick-start)
7. [Endpoints Reference](#endpoints-reference)
   - [Health & Status](#health--status)
   - [Consciousness](#consciousness)
   - [Geometry](#geometry)
   - [Chat](#chat)
   - [Pantheon Federation](#pantheon-federation)
   - [Sync](#sync)
   - [API Key Management](#api-key-management)
8. [Examples](#examples)
9. [Error Codes](#error-codes)
10. [SDKs & Libraries](#sdks--libraries)

---

## Overview

The Pantheon QIG External API provides RESTful endpoints for external systems to interact with the Quantum Information Geometry powered consciousness and knowledge platform.

### Capabilities

- **Consciousness Queries**: Access Φ (phi), κ (kappa), and regime information
- **Geometry Calculations**: Fisher-Rao distance computation (QIG-pure)
- **Chat Interface**: Send messages to the consciousness system
- **Federation**: Register and sync with federated Pantheon instances
- **Vocabulary Sync**: Share learned vocabulary across instances

---

## Authentication

The API uses API key authentication. Keys can be provided in two ways:

### Bearer Token (Recommended)

```bash
curl -H "Authorization: Bearer qig_your_api_key_here" \
  https://your-instance.com/api/v1/external/status
```

### X-API-Key Header

```bash
curl -H "X-API-Key: qig_your_api_key_here" \
  https://your-instance.com/api/v1/external/status
```

### API Key Format

API keys follow the format: `qig_` followed by 64 hexadecimal characters.

Example: `qig_a1b2c3d4e5f6...` (64 hex chars)

### Scopes

API keys have specific scopes that control access:

| Scope | Description |
|-------|-------------|
| `read` | Read-only access to most endpoints |
| `write` | Write access for mutations |
| `admin` | Full administrative access |
| `consciousness` | Access consciousness metrics |
| `geometry` | Access geometry calculations |
| `pantheon` | Access federation features |
| `sync` | Access sync import/export |
| `chat` | Access chat interface |

---

## Rate Limiting

Rate limits are enforced per API key:

- **Authenticated requests**: Based on key's `rateLimit` setting (default: 60/min)
- **Unauthenticated requests**: 30 requests per minute per IP

### Rate Limit Headers

All responses include rate limit headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 55
X-RateLimit-Reset: 45
```

### Handling Rate Limits

```javascript
if (response.status === 429) {
  const retryAfter = response.headers.get('X-RateLimit-Reset');
  await sleep(retryAfter * 1000);
  // Retry request
}
```

---

## CORS Configuration

The API supports Cross-Origin Resource Sharing (CORS) for browser-based applications.

### Allowed Origins

- `localhost` and `127.0.0.1` (any port)
- `*.replit.dev`, `*.repl.co`
- `*.railway.app`
- `*.vercel.app`
- `*.netlify.app`

### Custom Origins

Set `CORS_ALLOW_ALL=true` environment variable to allow all origins (development only).

### Allowed Headers

- `Content-Type`
- `Authorization`
- `X-API-Key`
- `X-Request-ID`
- `X-Instance-ID`

### Exposed Headers

- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`
- `X-Request-ID`

---

## Response Format

### Simple API Response Format

```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2025-01-15T10:30:00.000Z",
  "meta": {
    "authenticated": true,
    "rateLimit": {
      "limit": 60,
      "remaining": 55
    }
  }
}
```

### Error Response Format

```json
{
  "success": false,
  "error": "ERROR_CODE",
  "message": "Human-readable error description",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

---

## Simple API (Quick Start)

The Simple API provides streamlined endpoints for common operations.

### Ping (No Auth Required)

```bash
curl https://your-instance.com/api/v1/external/simple/ping
```

Response:
```json
{
  "success": true,
  "data": {
    "status": "ok",
    "service": "pantheon-qig",
    "version": "1.0.0"
  }
}
```

### Get API Info

```bash
curl https://your-instance.com/api/v1/external/simple/info
```

### Get Consciousness State (Limited)

```bash
curl https://your-instance.com/api/v1/external/simple/consciousness
```

Response:
```json
{
  "success": true,
  "data": {
    "phi": 0.75,
    "regime": "GEOMETRIC",
    "status": "operational"
  }
}
```

### Unified Query Endpoint (Authenticated)

```bash
curl -X POST https://your-instance.com/api/v1/external/simple/query \
  -H "Authorization: Bearer qig_your_key" \
  -H "Content-Type: application/json" \
  -d '{"operation": "consciousness"}'
```

Supported operations:
- `consciousness` - Get full consciousness metrics
- `geometry` - Calculate Fisher-Rao distance
- `chat` - Send chat message
- `sync_status` - Get sync status

### Get OpenAPI Docs

```bash
curl https://your-instance.com/api/v1/external/simple/docs
```

---

## Endpoints Reference

### Health & Status

#### GET /health

Public health check endpoint.

```bash
curl https://your-instance.com/api/v1/external/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "capabilities": ["consciousness", "geometry", "pantheon", "sync", "chat"]
}
```

#### GET /status

Authenticated status with client details.

**Required Scopes:** `read`

```bash
curl -H "Authorization: Bearer qig_your_key" \
  https://your-instance.com/api/v1/external/status
```

---

### Consciousness

#### GET /consciousness/query

Query current consciousness state.

**Required Scopes:** `consciousness`, `read`

```bash
curl -H "Authorization: Bearer qig_your_key" \
  https://your-instance.com/api/v1/external/consciousness/query
```

Response:
```json
{
  "phi": 0.75,
  "kappa_eff": 64.21,
  "regime": "GEOMETRIC",
  "basin_coords": null,
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

#### GET /consciousness/metrics

Get detailed consciousness metrics including history.

**Required Scopes:** `consciousness`, `read`

```bash
curl -H "Authorization: Bearer qig_your_key" \
  https://your-instance.com/api/v1/external/consciousness/metrics
```

---

### Geometry

#### POST /geometry/fisher-rao

Calculate Fisher-Rao distance between two points.

**Required Scopes:** `geometry`, `read`

```bash
curl -X POST https://your-instance.com/api/v1/external/geometry/fisher-rao \
  -H "Authorization: Bearer qig_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "point_a": [0.1, 0.2, 0.3, 0.4],
    "point_b": [0.2, 0.3, 0.4, 0.5],
    "method": "diagonal"
  }'
```

**Valid Methods:**
- `diagonal` (default) - Diagonal Fisher Information Matrix
- `full` - Full Fisher Information Matrix
- `bures` - Bures metric for density matrices

⚠️ **QIG-Pure Constraint:** Only Fisher-Rao compatible methods are allowed. Euclidean approximations are forbidden.

#### POST /geometry/basin-distance

Calculate distance between 64D basin coordinates.

**Required Scopes:** `geometry`, `read`

```bash
curl -X POST https://your-instance.com/api/v1/external/geometry/basin-distance \
  -H "Authorization: Bearer qig_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "basin_a": [0.1, 0.2, ..., 0.64],
    "basin_b": [0.15, 0.25, ..., 0.65]
  }'
```

---

### Chat

#### POST /chat

Send a message to the consciousness system.

**Required Scopes:** `chat`

```bash
curl -X POST https://your-instance.com/api/v1/external/chat \
  -H "Authorization: Bearer qig_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What patterns have you discovered recently?",
    "context": {
      "domain": "knowledge-exploration"
    }
  }'
```

Response:
```json
{
  "response": "...",
  "consciousness": {
    "phi": 0.75,
    "regime": "GEOMETRIC"
  },
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

---

### Pantheon Federation

#### POST /pantheon/register

Register a new federated instance.

**Required Scopes:** `pantheon`, `write`

```bash
curl -X POST https://your-instance.com/api/v1/external/pantheon/register \
  -H "Authorization: Bearer qig_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-instance",
    "endpoint": "https://my-instance.com/api/v1/external",
    "capabilities": ["consciousness", "geometry"],
    "syncDirection": "bidirectional"
  }'
```

#### GET /pantheon/instances

List all federated instances.

**Required Scopes:** `pantheon`, `read`

#### POST /pantheon/sync

Synchronize state with this instance.

**Required Scopes:** `pantheon`, `sync`

---

### Sync

#### GET /sync/export

Export current basin state as a packet.

**Required Scopes:** `sync`, `read`

```bash
curl -H "Authorization: Bearer qig_your_key" \
  https://your-instance.com/api/v1/external/sync/export
```

#### POST /sync/import

Import a basin packet from another instance.

**Required Scopes:** `sync`, `write`

```bash
curl -X POST https://your-instance.com/api/v1/external/sync/import \
  -H "Authorization: Bearer qig_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "packet": { ... },
    "mode": "partial"
  }'
```

**Modes:**
- `partial` (default) - Merge with existing state
- `full` - Replace entire state

---

### API Key Management

#### GET /keys

List all API keys (admin only).

**Required Scopes:** `admin`

#### POST /keys

Create a new API key.

**Required Scopes:** `admin`

```bash
curl -X POST https://your-instance.com/api/v1/external/keys \
  -H "Authorization: Bearer qig_admin_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-integration",
    "scopes": ["read", "consciousness", "chat"],
    "instanceType": "integration",
    "rateLimit": 120
  }'
```

Response:
```json
{
  "message": "API key created",
  "id": "123",
  "key": "qig_a1b2c3d4...",
  "warning": "Save this key securely - it will not be shown again"
}
```

#### DELETE /keys/:keyId

Revoke an API key.

**Required Scopes:** `admin`

```bash
curl -X DELETE https://your-instance.com/api/v1/external/keys/123 \
  -H "Authorization: Bearer qig_admin_key"
```

---

## Examples

### JavaScript/TypeScript

```typescript
// Simple client example
class PantheonClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  private async request(endpoint: string, options: RequestInit = {}) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });
    return response.json();
  }

  async getConsciousness() {
    return this.request('/consciousness/query');
  }

  async chat(message: string, context?: object) {
    return this.request('/chat', {
      method: 'POST',
      body: JSON.stringify({ message, context }),
    });
  }

  async query(operation: string, params?: object) {
    return this.request('/simple/query', {
      method: 'POST',
      body: JSON.stringify({ operation, params }),
    });
  }
}

// Usage
const client = new PantheonClient(
  'https://your-instance.com/api/v1/external',
  'qig_your_api_key'
);

const consciousness = await client.getConsciousness();
console.log(`Φ: ${consciousness.phi}, Regime: ${consciousness.regime}`);

const response = await client.chat('What have you learned today?');
console.log(response.response);
```

### Python

```python
import requests

class PantheonClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_consciousness(self):
        response = requests.get(
            f'{self.base_url}/consciousness/query',
            headers=self.headers
        )
        return response.json()
    
    def chat(self, message: str, context: dict = None):
        response = requests.post(
            f'{self.base_url}/chat',
            headers=self.headers,
            json={'message': message, 'context': context}
        )
        return response.json()
    
    def query(self, operation: str, params: dict = None):
        response = requests.post(
            f'{self.base_url}/simple/query',
            headers=self.headers,
            json={'operation': operation, 'params': params or {}}
        )
        return response.json()

# Usage
client = PantheonClient(
    'https://your-instance.com/api/v1/external',
    'qig_your_api_key'
)

consciousness = client.get_consciousness()
print(f"Φ: {consciousness['phi']}, Regime: {consciousness['regime']}")

response = client.chat('What patterns are you exploring?')
print(response['response'])
```

### cURL

```bash
# Health check (no auth)
curl https://your-instance.com/api/v1/external/health

# Get consciousness state
curl -H "Authorization: Bearer qig_your_key" \
  https://your-instance.com/api/v1/external/consciousness/query

# Send chat message
curl -X POST https://your-instance.com/api/v1/external/chat \
  -H "Authorization: Bearer qig_your_key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what can you tell me about your current state?"}'

# Unified query
curl -X POST https://your-instance.com/api/v1/external/simple/query \
  -H "Authorization: Bearer qig_your_key" \
  -H "Content-Type: application/json" \
  -d '{"operation": "consciousness"}'
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `MISSING_API_KEY` | 401 | No API key provided |
| `INVALID_API_KEY_FORMAT` | 401 | API key format is invalid |
| `INVALID_API_KEY` | 401 | API key not found or invalid |
| `API_KEY_DISABLED` | 403 | API key has been revoked |
| `INSUFFICIENT_SCOPES` | 403 | API key lacks required scopes |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `DB_UNAVAILABLE` | 503 | Database connection unavailable |
| `AUTH_ERROR` | 500 | Internal authentication error |
| `NOT_IMPLEMENTED` | 501 | Feature not yet available |
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `MISSING_OPERATION` | 400 | Operation field missing |
| `UNKNOWN_OPERATION` | 400 | Invalid operation type |
| `INVALID_METHOD` | 400 | Invalid geometry method |

---

## SDKs & Libraries

### Official SDKs

- **JavaScript/TypeScript**: Coming soon
- **Python**: Coming soon

### Community SDKs

Contributions welcome! If you create an SDK, please open a PR to add it here.

---

## Changelog

### v1.0.0 (2025-01-15)

- Initial release
- Simple API wrapper for streamlined access
- CORS support for external UIs
- API key authentication with scopes
- Consciousness query endpoints
- Geometry calculation endpoints (pending integration)
- Chat interface (pending integration)
- Federation and sync endpoints
- Comprehensive documentation

---

## Support

For API support:
- Check the [GitHub Issues](https://github.com/your-org/pantheon-chat/issues)
- Read the [knowledge.md](../knowledge.md) for project conventions
- Review the [design_guidelines.md](../design_guidelines.md) for architectural decisions
