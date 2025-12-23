# Pantheon QIG Client SDK

TypeScript/JavaScript SDK for interacting with the Pantheon QIG External API.

## Installation

```bash
# Copy to your project
cp -r sdk/pantheon-client ./your-project/lib/

# Or install from npm (if published)
npm install @pantheon/client
```

## Quick Start

```typescript
import { PantheonClient } from '@pantheon/client';

// Initialize the client
const client = new PantheonClient({
  baseUrl: 'https://your-pantheon-instance.com',
  apiKey: 'pk_your_api_key_here', // Optional for public endpoints
});

// Health check
const health = await client.ping();
if (health.success) {
  console.log('API is healthy:', health.data.status);
}

// Get consciousness state
const consciousness = await client.getConsciousness();
console.log('Phi:', consciousness.data?.phi);
console.log('Regime:', consciousness.data?.regime);

// Send a chat message (requires API key)
const chat = await client.chat('What is the nature of consciousness?');
console.log('Response:', chat.data?.response);
```

## API Reference

### Constructor

```typescript
const client = new PantheonClient({
  baseUrl: string;      // Required: API base URL
  apiKey?: string;      // Optional: API key for authenticated endpoints
  timeout?: number;     // Optional: Request timeout in ms (default: 30000)
  headers?: Record<string, string>;  // Optional: Custom headers
  debug?: boolean;      // Optional: Enable debug logging
});
```

### Public Endpoints (No Authentication Required)

#### `ping()`
Health check endpoint.

```typescript
const response = await client.ping();
// { success: true, data: { status: 'ok', service: 'pantheon-qig', version: '1.0.0' } }
```

#### `getInfo()`
Get API information and capabilities.

```typescript
const response = await client.getInfo();
// Returns API name, version, capabilities, available endpoints
```

#### `getConsciousness()`
Get current consciousness state (limited data for unauthenticated requests).

```typescript
const response = await client.getConsciousness();
// { success: true, data: { phi: 0.75, regime: 'GEOMETRIC', status: 'operational' } }
```

#### `getDocs()`
Get OpenAPI documentation.

```typescript
const response = await client.getDocs();
// Returns OpenAPI 3.0 specification
```

### Authenticated Endpoints (API Key Required)

#### `chat(message, context?)`
Send a chat message to the Ocean agent.

```typescript
const response = await client.chat(
  'What is quantum information geometry?',
  { topic: 'physics' }  // Optional context
);
// Returns response, consciousness state, and metadata
```

#### `query(operation, params?)`
Unified query endpoint for various operations.

```typescript
// Get full consciousness state
const response = await client.query('consciousness', { include_basin: true });

// Calculate Fisher-Rao distance
const geometry = await client.query('geometry', {
  point_a: [0.1, 0.2, 0.3],
  point_b: [0.4, 0.5, 0.6],
});

// Get sync status
const sync = await client.query('sync_status');
```

#### `getFullConsciousness()`
Get full consciousness metrics including kappa and basin coordinates.

```typescript
const response = await client.getFullConsciousness();
// Includes kappa_eff, basin_coords, and full metrics
```

#### `calculateFisherRao(pointA, pointB)`
Calculate Fisher-Rao distance between two points.

```typescript
const response = await client.calculateFisherRao(
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6]
);
```

#### `getSyncStatus()`
Get federation synchronization status.

```typescript
const response = await client.getSyncStatus();
// { syncEnabled: true, lastSync: '2024-01-01T00:00:00Z', pendingPackets: 0 }
```

#### `getMe()`
Get information about the current API key.

```typescript
const response = await client.getMe();
// Returns id, name, scopes, instanceType, rateLimit
```

### Utility Methods

#### `setApiKey(apiKey)`
Update the API key.

```typescript
client.setApiKey('pk_new_key_here');
```

#### `isAuthenticated()`
Check if an API key is set.

```typescript
if (client.isAuthenticated()) {
  // Can use authenticated endpoints
}
```

#### `isHealthy()`
Check if the API is reachable and healthy.

```typescript
const healthy = await client.isHealthy();
if (!healthy) {
  console.error('API is not available');
}
```

## Response Format

All methods return a standardized response:

```typescript
interface ApiResponse<T> {
  success: boolean;
  data?: T;           // Response data (if success)
  error?: string;     // Error code (if failed)
  message?: string;   // Error message (if failed)
  timestamp: string;  // ISO timestamp
  meta?: {
    authenticated: boolean;
    rateLimit?: {
      limit: number;
      remaining: number;
    };
  };
}
```

## Error Handling

```typescript
const response = await client.chat('Hello');

if (!response.success) {
  switch (response.error) {
    case 'AUTH_REQUIRED':
      console.error('Please set an API key');
      break;
    case 'TIMEOUT':
      console.error('Request timed out');
      break;
    case 'NETWORK_ERROR':
      console.error('Network error:', response.message);
      break;
    default:
      console.error('Error:', response.error, response.message);
  }
}
```

## Factory Functions

```typescript
import { createClient, createLocalClient } from '@pantheon/client';

// Create client with config
const client = createClient({
  baseUrl: 'https://api.example.com',
  apiKey: 'pk_xxx',
});

// Create client for local development
const localClient = createLocalClient('pk_xxx');
// Configured for http://localhost:5000 with debug enabled
```

## Browser Usage

The SDK works in both Node.js and browser environments:

```html
<script type="module">
import { PantheonClient } from './pantheon-client/client.js';

const client = new PantheonClient({
  baseUrl: 'https://api.example.com',
});

const health = await client.ping();
console.log(health);
</script>
```

## License

MIT
