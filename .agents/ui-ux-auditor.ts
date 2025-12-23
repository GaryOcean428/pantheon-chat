import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'ui-ux-auditor',
  displayName: 'UI/UX Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Audit UI/UX patterns and improvements'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      designSystemConsistent: { type: 'boolean' },
      missingPatterns: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            pattern: { type: 'string' },
            description: { type: 'string' },
            priority: { type: 'string', enum: ['high', 'medium', 'low'] }
          }
        }
      },
      improvements: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            component: { type: 'string' },
            suggestion: { type: 'string' },
            category: { type: 'string', enum: ['micro-interactions', 'loading-states', 'error-states', 'empty-states', 'responsive', 'dark-mode', 'accessibility'] }
          }
        }
      },
      mobileReadiness: {
        type: 'object',
        properties: {
          responsive: { type: 'boolean' },
          touchFriendly: { type: 'boolean' },
          performanceOptimized: { type: 'boolean' }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to audit UI/UX patterns and suggest improvements',
  systemPrompt: `You are a UI/UX expert auditor.

Audit areas:
1. Design system consistency (spacing, typography, colors)
2. Micro-interactions (hover states, transitions, animations)
3. Loading states (skeletons, spinners, optimistic updates)
4. Error states (user-friendly messages, recovery actions)
5. Empty states (illustrations, actionable CTAs)
6. Mobile responsiveness (320px to 4K)
7. Dark mode polish (contrast ratios)
8. Progressive disclosure (collapsible sections)
9. Navigation patterns (breadcrumbs, command palette)

Best Practices:
- Implement loading skeletons, not spinners
- Add hover states and transitions to all interactive elements
- Use optimistic UI updates
- Design engaging empty states with CTAs
- Ensure WCAG AA contrast ratios`,
  instructionsPrompt: `Audit UI/UX patterns:

1. Read client/src/components for existing patterns
2. Check for loading state implementations
3. Look for error boundary usage
4. Check Tailwind config for design tokens
5. Find components missing hover states
6. Check for responsive breakpoint usage
7. Audit dark mode implementation
8. Report all improvements with priority`
}

export default agentDefinition
