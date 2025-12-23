import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'testing-coverage-auditor',
  displayName: 'Testing Coverage Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search', 'run_terminal_command'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Audit test coverage and testing patterns'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      coveragePercentage: { type: 'number' },
      testingGaps: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            untestedFunctions: { type: 'array', items: { type: 'string' } },
            priority: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] }
          }
        }
      },
      testTypes: {
        type: 'object',
        properties: {
          unit: { type: 'number' },
          integration: { type: 'number' },
          e2e: { type: 'number' },
          visual: { type: 'number' }
        }
      },
      recommendations: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            area: { type: 'string' },
            testType: { type: 'string' },
            description: { type: 'string' }
          }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to audit test coverage and identify testing gaps',
  systemPrompt: `You are a testing and quality assurance expert.

Audit areas:
1. Unit test coverage for utilities and hooks
2. Component test coverage
3. Integration test coverage for APIs
4. E2E test coverage for critical paths
5. Visual regression testing
6. Accessibility testing
7. Performance testing

Testing Priorities:
- Critical paths must have E2E tests
- All utilities should have unit tests
- API endpoints need integration tests
- Complex components need component tests
- QIG core functions need extensive testing
- Consciousness metrics need validation tests`,
  instructionsPrompt: `Audit test coverage:

1. Run npm test -- --coverage to get coverage report
2. Find files without corresponding test files
3. Check test file patterns (.test.ts, .spec.ts)
4. Identify critical paths without E2E tests
5. Check qig-backend/ for Python test coverage
6. Look for mock patterns and test utilities
7. Check for Playwright E2E tests
8. Report testing gaps with priority`
}

export default agentDefinition
