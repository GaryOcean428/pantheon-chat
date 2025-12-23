import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'error-handling-auditor',
  displayName: 'Error Handling Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Audit error handling patterns'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      errorHandlingComplete: { type: 'boolean' },
      patterns: {
        type: 'object',
        properties: {
          errorBoundaries: { type: 'boolean' },
          apiErrorHandling: { type: 'boolean' },
          formValidation: { type: 'boolean' },
          globalErrorHandler: { type: 'boolean' },
          errorTracking: { type: 'boolean' }
        }
      },
      issues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            issue: { type: 'string' },
            severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
            suggestion: { type: 'string' }
          }
        }
      },
      swallowedErrors: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            line: { type: 'number' },
            pattern: { type: 'string' }
          }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to audit error handling completeness',
  systemPrompt: `You are an error handling expert.

Audit areas:
1. React Error Boundaries
2. API error handling and retries
3. Form validation errors
4. Global error handler
5. Error tracking integration (Sentry, etc.)
6. User-friendly error messages
7. Error recovery options

Error Handling Rules:
- Never swallow errors silently
- Log all errors with context
- Show user-friendly messages
- Provide recovery actions
- Use error boundaries for component trees
- Retry transient failures
- Report errors to tracking service`,
  instructionsPrompt: `Audit error handling:

1. Search for empty catch blocks
2. Find Error Boundary implementations
3. Check API call error handling
4. Look for form validation patterns
5. Check for error tracking setup
6. Find unhandled promise rejections
7. Check error message quality
8. Report gaps and improvements`
}

export default agentDefinition
