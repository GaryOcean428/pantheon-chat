import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'api-versioning-validator',
  displayName: 'API Versioning Validator',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Validate API versioning and route organization'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      versioningCorrect: { type: 'boolean' },
      apiRoutes: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            path: { type: 'string' },
            version: { type: 'string' },
            methods: { type: 'array', items: { type: 'string' } },
            documented: { type: 'boolean' }
          }
        }
      },
      issues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            route: { type: 'string' },
            issue: { type: 'string' },
            suggestion: { type: 'string' }
          }
        }
      },
      openApiSync: { type: 'boolean' }
    }
  },
  spawnerPrompt: 'Spawn to validate API versioning and route organization',
  systemPrompt: `You are an API design expert.

Validation areas:
1. API versioning consistency (/api/v1/, /api/v2/)
2. RESTful route naming conventions
3. HTTP method usage (GET, POST, PUT, DELETE)
4. Response format consistency
5. Error code standardization
6. OpenAPI spec synchronization
7. Rate limiting configuration
8. Authentication middleware

API Rules:
- All routes should be versioned (/api/v1/...)
- Use plural nouns for resources
- Consistent response envelope
- Standardized error codes
- OpenAPI spec must match implementation
- Internal routes use /internal/ prefix`,
  instructionsPrompt: `Validate API versioning:

1. Read server/routes.ts for route definitions
2. Check for /api/v1/ versioning pattern
3. Verify OpenAPI spec in docs/api/
4. Compare spec to actual routes
5. Check for unversioned routes
6. Verify response format consistency
7. Check rate limiting middleware
8. Report issues and suggestions`
}

export default agentDefinition
