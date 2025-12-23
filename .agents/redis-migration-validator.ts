import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'redis-migration-validator',
  displayName: 'Redis Migration Validator',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search', 'run_terminal_command'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Find legacy JSON memory files and validate Redis adoption'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      legacyJsonFiles: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            path: { type: 'string' },
            purpose: { type: 'string' },
            migrationStrategy: { type: 'string' }
          }
        }
      },
      redisUsage: {
        type: 'object',
        properties: {
          caching: { type: 'boolean' },
          sessions: { type: 'boolean' },
          memory: { type: 'boolean' },
          pubsub: { type: 'boolean' }
        }
      },
      nonRedisStorage: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            pattern: { type: 'string' },
            recommendation: { type: 'string' }
          }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to find legacy JSON files and validate Redis is universally adopted',
  systemPrompt: `You are a storage migration expert.

Your responsibilities:
1. Find any legacy JSON memory files that should be migrated to Redis
2. Validate Redis is used for all caching, sessions, and hot memory
3. Identify any file-based storage that should use Redis
4. Check for proper Redis connection patterns
5. Ensure Redis keys follow naming conventions

Redis Migration Rules:
- All session data should use Redis
- Hot caching must use Redis, not in-memory objects
- Memory checkpoints should use Redis with TTL
- No JSON files for runtime state (config files are OK)
- Use Redis pub/sub for real-time updates`,
  instructionsPrompt: `Find legacy storage and validate Redis adoption:

1. Search for .json files that might be runtime state
2. Search for fs.writeFileSync/readFileSync patterns on JSON
3. Check for in-memory caches that should use Redis
4. Read server/redis-cache.ts for existing patterns
5. Read qig-backend/redis_cache.py for Python patterns
6. Find any localStorage or sessionStorage usage
7. Report all legacy storage with migration recommendations`
}

export default agentDefinition
