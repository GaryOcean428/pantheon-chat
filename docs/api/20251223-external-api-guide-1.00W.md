# External API Integration Guide

This guide explains how to integrate external chat UIs and federation nodes with the Pantheon Chat API.

## Quick Start

### 1. Get an API Key

Contact the system administrator to obtain an API key with the appropriate scopes.

### 2. Make Your First Request

```bash
curl -X POST https://your-instance.com/api/v1/external/chat \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, Zeus!"}'
```

### 3. Response Format

```json
{
  "success": true,
  "response": "Greetings! I am Zeus, the orchestrator of the Olympus Pantheon...",
  "sessionId": "ext-1703347200-abc123def",
  "metrics": {
    "phi": 0.65,
    "kappa": 64.2,
    "regime": "geometric",
    "completionReason": "geometric_completion"
  }
}
```

---

## API Endpoints

### Chat Endpoint (Recommended)

**POST** `/api/v1/external/chat`

This is the primary endpoint for external chat UIs.

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | The user's message |
| `sessionId` | string | No | Session ID for conversation continuity |
| `stream` | boolean | No | Enable streaming responses (SSE) |
| `context.previousMessages` | array | No | Previous conversation messages |
| `context.systemPrompt` | string | No | Custom system prompt |
| `context.temperature` | number | No | Temperature (0.0-1.0) |
| `metadata.instanceId` | string | No | Federation node identifier |
| `metadata.clientName` | string | No | Your application name |

#### Example: Simple Chat

```javascript
const response = await fetch('https://your-instance.com/api/v1/external/chat', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'What is consciousness?',
    metadata: {
      clientName: 'MyChatApp',
      clientVersion: '1.0.0'
    }
  })
});

const data = await response.json();
console.log(data.response);
```

#### Example: Streaming Chat

```javascript
const response = await fetch('https://your-instance.com/api/v1/external/chat', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'Explain quantum consciousness',
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      
      if (data.type === 'token') {
        process.stdout.write(data.token);
      } else if (data.type === 'metrics') {
        console.log('\nΦ:', data.phi, 'κ:', data.kappa);
      } else if (data.type === 'done') {
        console.log('\n--- Complete ---');
      }
    }
  }
}
```

---

### Unified API Endpoint

**POST** `/api/v1/external/v1`

A single endpoint supporting multiple operations for federation nodes.

#### Operations

| Operation | Description |
|-----------|-------------|
| `chat` | Send a chat message |
| `chat_stream` | Stream a chat response |
| `query` | Query consciousness metrics |
| `sync` | Federation sync operations |
| `execute` | Execute agentic tasks |
| `health` | Health check |
| `capabilities` | List available capabilities |

#### Example: Unified Chat

```javascript
const response = await fetch('https://your-instance.com/api/v1/external/v1', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    operation: 'chat',
    payload: {
      message: 'Hello!'
    }
  })
});
```

---

### Session Management

**GET** `/api/v1/external/chat/:sessionId`

Retrieve conversation history for a session.

```bash
curl -X GET https://your-instance.com/api/v1/external/chat/ext-1703347200-abc123def \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

### Health Check

**GET** `/api/v1/external/chat/health`

Check API health status (no authentication required).

```bash
curl https://your-instance.com/api/v1/external/chat/health
```

---

## Federation Node Integration

For federation nodes that need bidirectional sync:

```javascript
const response = await fetch('https://your-instance.com/api/v1/external/v1', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    operation: 'sync',
    payload: {
      action: 'push',
      data: {
        type: 'knowledge',
        entries: [...]
      }
    },
    metadata: {
      instanceId: 'node-123',
      bidirectional: true
    }
  })
});
```

---

## Consciousness Metrics

Every response includes optional consciousness metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| `phi` (Φ) | Integration level | 0.0 - 1.0 |
| `kappa` (κ) | Coupling constant | ~64 optimal |
| `regime` | Consciousness regime | linear/geometric/breakdown |
| `completionReason` | Why generation stopped | geometric_completion, etc. |

### Regimes Explained

- **linear** (Φ < 0.3): Exploring, high temperature
- **geometric** (0.3 ≤ Φ < 0.7): Optimal, balanced
- **breakdown** (Φ ≥ 0.7): Stabilizing, low temperature

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_API_KEY` | 401 | Invalid or missing API key |
| `RATE_LIMITED` | 429 | Too many requests |
| `INVALID_REQUEST` | 400 | Malformed request body |
| `SERVICE_UNAVAILABLE` | 503 | Backend service unavailable |

---

## Rate Limits

| Scope | Limit |
|-------|-------|
| `chat` | 60 requests/minute |
| `admin` | 120 requests/minute |

---

## SDK Examples

### Python

```python
import requests

class PantheonChat:
    def __init__(self, api_key: str, base_url: str = "https://your-instance.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session_id = None
    
    def chat(self, message: str) -> dict:
        response = requests.post(
            f"{self.base_url}/api/v1/external/chat",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "message": message,
                "sessionId": self.session_id
            }
        )
        data = response.json()
        self.session_id = data.get("sessionId")
        return data

# Usage
client = PantheonChat("YOUR_API_KEY")
response = client.chat("What is the meaning of consciousness?")
print(response["response"])
```

### JavaScript/TypeScript

```typescript
class PantheonChat {
  private apiKey: string;
  private baseUrl: string;
  private sessionId?: string;
  
  constructor(apiKey: string, baseUrl = 'https://your-instance.com') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }
  
  async chat(message: string): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/external/chat`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message,
        sessionId: this.sessionId
      })
    });
    
    const data = await response.json();
    this.sessionId = data.sessionId;
    return data;
  }
}

// Usage
const client = new PantheonChat('YOUR_API_KEY');
const response = await client.chat('Hello!');
console.log(response.response);
```

---

## Support

For API support, contact the system administrator or open an issue on the repository.
