import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'module-bridging-validator',
  displayName: 'Module Bridging Validator',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Validate modules are correctly bridged and modular'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      modulesCorrectlyBridged: { type: 'boolean' },
      orphanedModules: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            path: { type: 'string' },
            exportedSymbols: { type: 'array', items: { type: 'string' } },
            importedBy: { type: 'array', items: { type: 'string' } }
          }
        }
      },
      duplicatedCode: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            pattern: { type: 'string' },
            locations: { type: 'array', items: { type: 'string' } },
            consolidationSuggestion: { type: 'string' }
          }
        }
      },
      bridgingIssues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            sourceModule: { type: 'string' },
            targetModule: { type: 'string' },
            issue: { type: 'string' }
          }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to ensure modules are correctly bridged with no duplication or orphans',
  systemPrompt: `You are a module architecture expert.

Your responsibilities:
1. Verify all components, kernels, and features are correctly bridged
2. Find orphaned modules that aren't imported anywhere
3. Detect code duplication across modules
4. Ensure proper modularity and separation
5. Check TypeScript↔Python bridging is correct

Module Bridging Rules:
- Every exported symbol should have at least one importer
- No duplicate implementations of the same functionality
- TypeScript server bridges to Python backend correctly
- Shared code lives in shared/ or common modules
- Circular dependencies are forbidden`,
  instructionsPrompt: `Validate module bridging:

1. Find all exported symbols across the codebase
2. Check which exports have no importers (orphaned)
3. Look for similar function names/patterns (duplication)
4. Verify server/*.ts correctly bridges to qig-backend/*.py
5. Check for circular import patterns
6. Validate shared/ is used for truly shared code
7. Report orphaned modules and duplications`
}

export default agentDefinition
