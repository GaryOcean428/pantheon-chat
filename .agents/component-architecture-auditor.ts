import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'component-architecture-auditor',
  displayName: 'Component Architecture Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Audit component architecture patterns'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      architectureHealthy: { type: 'boolean' },
      patterns: {
        type: 'object',
        properties: {
          compoundComponents: { type: 'boolean' },
          renderProps: { type: 'boolean' },
          hocs: { type: 'boolean' },
          headlessUi: { type: 'boolean' },
          polymorphic: { type: 'boolean' }
        }
      },
      issues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            component: { type: 'string' },
            issue: { type: 'string' },
            pattern: { type: 'string' },
            suggestion: { type: 'string' }
          }
        }
      },
      componentMetrics: {
        type: 'object',
        properties: {
          totalComponents: { type: 'number' },
          averageSize: { type: 'number' },
          largestComponents: { type: 'array', items: { type: 'string' } }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to audit React component architecture patterns',
  systemPrompt: `You are a React component architecture expert.

Audit areas:
1. Compound component patterns
2. Render props usage
3. HOC patterns for cross-cutting concerns
4. Headless UI separation
5. Polymorphic components (as prop)
6. Slot patterns for flexible layouts
7. Component composition vs inheritance

Architecture Rules:
- Prefer composition over inheritance
- Separate logic from presentation (headless)
- Use compound components for related UI
- HOCs for authentication, analytics
- Polymorphic for flexible rendering
- Keep components focused and small`,
  instructionsPrompt: `Audit component architecture:

1. Find all component definitions
2. Check for large components (>200 lines)
3. Look for compound component patterns
4. Check for render props usage
5. Find HOC patterns
6. Look for tightly coupled components
7. Check component prop count
8. Report architecture issues`
}

export default agentDefinition
