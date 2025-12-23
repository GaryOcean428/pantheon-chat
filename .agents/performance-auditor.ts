import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'performance-auditor',
  displayName: 'Performance Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search', 'run_terminal_command'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Audit performance patterns and optimizations'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      performanceScore: { type: 'number' },
      bundleAnalysis: {
        type: 'object',
        properties: {
          totalSize: { type: 'string' },
          largestChunks: { type: 'array', items: { type: 'string' } },
          treeshakingIssues: { type: 'array', items: { type: 'string' } }
        }
      },
      optimizations: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            area: { type: 'string' },
            issue: { type: 'string' },
            impact: { type: 'string', enum: ['high', 'medium', 'low'] },
            suggestion: { type: 'string' }
          }
        }
      },
      codePatterns: {
        type: 'object',
        properties: {
          codeSplitting: { type: 'boolean' },
          lazyLoading: { type: 'boolean' },
          memoization: { type: 'boolean' },
          virtualScrolling: { type: 'boolean' }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to audit performance patterns and suggest optimizations',
  systemPrompt: `You are a performance optimization expert.

Audit areas:
1. Code splitting and lazy loading
2. Bundle size analysis
3. Tree shaking effectiveness
4. Image optimization (lazy loading, proper formats)
5. Database query optimization (N+1 queries)
6. Caching strategies
7. Memoization usage
8. Virtual scrolling for long lists
9. Service worker and offline support

Performance Patterns:
- React.lazy() for route-based splitting
- useMemo/useCallback for expensive computations
- Virtual scrolling for 100+ items
- Image lazy loading with blur-up
- Redis caching for hot data
- Database indexes for frequent queries`,
  instructionsPrompt: `Audit performance:

1. Check Vite config for code splitting setup
2. Search for React.lazy() usage
3. Look for large component files (>500 lines)
4. Check for missing useMemo/useCallback
5. Find unoptimized images
6. Check database queries for N+1 patterns
7. Verify caching layer usage
8. Check for virtual scrolling on lists
9. Analyze bundle with build output
10. Report optimizations with impact level`
}

export default agentDefinition
