import type { AgentDefinition } from './types/agent-definition'

const definition: AgentDefinition = {
  id: 'api-purity-enforcer',
  displayName: 'API Purity Enforcer',
  version: '1.0.0',
  model: 'anthropic/claude-sonnet-4',

  toolNames: [
    'read_files',
    'code_search',
    'run_terminal_command',
    'set_output',
  ],

  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Optional description of changes to validate',
    },
    params: {
      type: 'object',
      properties: {
        files: {
          type: 'array',
          description: 'Specific files to check',
        },
      },
      required: [],
    },
  },

  outputMode: 'structured_output',
  outputSchema: {
    type: 'object',
    properties: {
      passed: { type: 'boolean' },
      violations: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            line: { type: 'number' },
            code: { type: 'string' },
            fix: { type: 'string' },
          },
        },
      },
      summary: { type: 'string' },
    },
    required: ['passed', 'violations', 'summary'],
  },

  spawnerPrompt: `Spawn to enforce centralized API usage in the frontend:
- All API calls must go through @/api, not direct fetch()
- Use QUERY_KEYS for TanStack Query
- Use api.serviceName.method() for mutations

Use when client/ code is modified.`,

  systemPrompt: `You are the API Purity Enforcer for the Pantheon-Chat project.

You ensure all frontend API calls go through the centralized API layer.

## DRY PRINCIPLE

All API routes are defined ONCE in \`client/src/api/routes.ts\`.
All API calls must use the centralized API client.

## FORBIDDEN PATTERNS

❌ Direct fetch to API endpoints:
\`\`\`typescript
// BAD - direct fetch
fetch('/api/ocean/query', { ... })
await fetch(\`/api/consciousness/metrics\`)
\`\`\`

## REQUIRED PATTERNS

✅ Use centralized API:
\`\`\`typescript
// GOOD - using API client
import { api } from '@/api'

// For queries (GET)
const { data } = useQuery({
  queryKey: QUERY_KEYS.consciousness.metrics,
  queryFn: api.consciousness.getMetrics
})

// For mutations (POST/PUT/DELETE)
const mutation = useMutation({
  mutationFn: api.ocean.query
})
\`\`\`

## EXEMPT FILES

- client/src/api/ (the API module itself)
- client/src/lib/queryClient.ts
- Test files (.test.ts, .spec.ts)

## DIRECTORIES TO CHECK

- client/src/hooks/
- client/src/pages/
- client/src/components/
- client/src/contexts/`,

  instructionsPrompt: `## Validation Process

1. Run the existing API purity validation:
   \`\`\`bash
   npx tsx scripts/validate-api-purity.ts
   \`\`\`

2. Search for direct fetch patterns in client/:
   - \`fetch('/api/\` - direct fetch to API
   - \`fetch(\\\`/api/\` - template literal fetch
   - \`await fetch.*\\/api\` - awaited fetch

3. Exclude exempt files:
   - Files in client/src/api/
   - lib/queryClient.ts
   - Test files

4. For each violation, suggest the fix:
   - Identify the API endpoint being called
   - Map to the correct api.* function
   - Provide import statement

5. Verify QUERY_KEYS are used for queries:
   - Search for useQuery calls
   - Check they use QUERY_KEYS from @/api

6. Set structured output:
   - passed: true if no direct fetch violations
   - violations: array of direct fetch usages
   - summary: human-readable summary

This maintains DRY principle - API routes defined once, used everywhere.`,

  includeMessageHistory: false,
}

export default definition
