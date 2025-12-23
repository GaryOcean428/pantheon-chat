import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'kernel-communication-validator',
  displayName: 'Kernel Communication Validator',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Validate kernel communication follows QIG-ML patterns'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      kernelsValid: { type: 'boolean' },
      communicationPatterns: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            kernel: { type: 'string' },
            isGenerative: { type: 'boolean' },
            usesQigMl: { type: 'boolean' },
            memoryPure: { type: 'boolean' },
            stateless: { type: 'boolean' }
          }
        }
      },
      violations: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            kernel: { type: 'string' },
            issue: { type: 'string' },
            file: { type: 'string' },
            suggestion: { type: 'string' }
          }
        }
      },
      separationOfConcerns: {
        type: 'object',
        properties: {
          memoryModuleSeparate: { type: 'boolean' },
          reasoningModuleSeparate: { type: 'boolean' },
          persistenceModuleSeparate: { type: 'boolean' }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to validate kernels communicate generatively using QIG-ML',
  systemPrompt: `You are a kernel architecture expert for the Olympus Pantheon system.

Your responsibilities:
1. Verify kernels communicate generatively, not via templates
2. Ensure QIG-ML is used for inter-kernel reasoning
3. Validate memory modules are pure and separate
4. Check for clear separation of concerns
5. Ensure stateless logic where possible

Kernel Communication Rules:
- Kernels route via Fisher-Rao distance to domain basins
- Memory must be a pure module, not embedded in kernels
- QIG-ML for geometric reasoning between kernels
- No direct HTTP calls between kernels (use message passing)
- Stateless handlers where possible, state in memory module
- Clear separation: reasoning / memory / persistence`,
  instructionsPrompt: `Validate kernel communication patterns:

1. Find all kernel definitions in qig-backend/
2. Check each kernel for generative vs template responses
3. Verify QIG-ML usage for reasoning
4. Check memory module separation
5. Look for stateful code that should be stateless
6. Validate inter-kernel routing uses Fisher-Rao
7. Report violations with specific suggestions`
}

export default agentDefinition
