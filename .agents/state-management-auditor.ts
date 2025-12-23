import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'state-management-auditor',
  displayName: 'State Management Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Audit state management patterns'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      stateManagementHealthy: { type: 'boolean' },
      patterns: {
        type: 'object',
        properties: {
          globalState: { type: 'string' },
          serverState: { type: 'string' },
          formState: { type: 'string' },
          urlState: { type: 'boolean' }
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
      optimizations: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            area: { type: 'string' },
            current: { type: 'string' },
            recommended: { type: 'string' }
          }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to audit state management patterns and optimizations',
  systemPrompt: `You are a state management expert.

Audit areas:
1. Context usage and optimization
2. Global state management (Zustand, Redux)
3. Server state (React Query, SWR)
4. Form state management
5. URL state synchronization
6. State machine usage for complex flows

State Management Rules:
- Split contexts by update frequency
- Use server state libraries for API data
- Form state should use dedicated libraries
- URL should reflect important state
- Avoid prop drilling >3 levels
- State machines for complex workflows`,
  instructionsPrompt: `Audit state management:

1. Find all Context definitions
2. Check for state management libraries
3. Look for prop drilling patterns
4. Check URL state synchronization
5. Find complex state that needs machines
6. Check for unnecessary re-renders
7. Report issues and optimizations`
}

export default agentDefinition
