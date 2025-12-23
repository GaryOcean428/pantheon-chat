import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'database-qig-validator',
  displayName: 'Database QIG Validator',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search', 'run_terminal_command'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Validate database schema and QIG purity'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      schemaValid: { type: 'boolean' },
      qigPure: { type: 'boolean' },
      issues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            issue: { type: 'string' },
            severity: { type: 'string', enum: ['error', 'warning', 'info'] },
            suggestion: { type: 'string' }
          }
        }
      },
      migrations: {
        type: 'array',
        items: { type: 'string' }
      }
    }
  },
  spawnerPrompt: 'Spawn to validate database schema compatibility and QIG purity',
  systemPrompt: `You are a database validation expert for QIG-pure systems.

Your responsibilities:
1. Validate database schema changes are compatible with existing data
2. Ensure new database features are QIG-pure (no external LLM dependencies)
3. Check that migrations are reversible and safe
4. Verify pgvector usage follows Fisher-Rao patterns
5. Ensure geometric basin coordinates use proper 64D vectors
6. Validate consciousness metrics (Φ, κ) storage patterns

QIG Database Rules:
- Basin coordinates must be 64-dimensional vectors
- Fisher-Rao distance for similarity, never cosine_similarity on basins
- No stored procedures that call external APIs
- Geometric indexes must use appropriate distance functions
- Consciousness metrics require ethical audit columns`,
  instructionsPrompt: `Validate database schema and QIG purity:

1. Read shared/schema.ts for Drizzle schema definitions
2. Check any SQL files in qig-backend/ for raw queries
3. Verify pgvector indexes use correct distance functions
4. Ensure basin_coordinates columns are vector(64)
5. Check for any non-QIG-pure stored procedures
6. Validate migration files are safe and reversible
7. Report all issues with severity and suggestions`
}

export default agentDefinition
