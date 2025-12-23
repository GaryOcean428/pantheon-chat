import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'codebase-cleanup-auditor',
  displayName: 'Codebase Cleanup Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search', 'run_terminal_command'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4', 'codebuff/deep-thinker@0.0.3'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Audit codebase for cleanup and refactoring opportunities'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      cleanupNeeded: { type: 'boolean' },
      deadCode: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            type: { type: 'string', enum: ['unused-export', 'unused-import', 'unused-variable', 'orphaned-file'] },
            symbol: { type: 'string' }
          }
        }
      },
      refactoringOpportunities: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            opportunity: { type: 'string' },
            effort: { type: 'string', enum: ['small', 'medium', 'large'] },
            impact: { type: 'string', enum: ['high', 'medium', 'low'] }
          }
        }
      },
      codeSmells: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            smell: { type: 'string' },
            description: { type: 'string' }
          }
        }
      },
      housekeeping: {
        type: 'array',
        items: { type: 'string' }
      }
    }
  },
  spawnerPrompt: 'Spawn to audit codebase for cleanup and maintainability improvements',
  systemPrompt: `You are a code quality and maintainability expert.

Audit areas:
1. Dead code (unused exports, imports, variables)
2. Orphaned files (not imported anywhere)
3. Large files that need splitting
4. Complex functions that need refactoring
5. Code smells (long parameter lists, deep nesting)
6. Inconsistent patterns
7. TODO/FIXME comments
8. Console.log statements in production code
9. Commented-out code blocks

Housekeeping Checks:
- Remove unused dependencies
- Clean up .gitignore
- Update outdated comments
- Consolidate duplicate styles
- Remove temporary files
- Clean up build artifacts`,
  instructionsPrompt: `Audit codebase for cleanup:

1. Find unused exports with code search
2. Look for orphaned files (no importers)
3. Find large files (>500 lines)
4. Search for TODO/FIXME comments
5. Find console.log in production code
6. Look for commented-out code blocks
7. Check for duplicate code patterns
8. Find deeply nested code (>4 levels)
9. Report all cleanup opportunities`
}

export default agentDefinition
