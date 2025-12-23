import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'accessibility-auditor',
  displayName: 'Accessibility Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Audit accessibility (a11y) compliance'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      wcagLevel: { type: 'string', enum: ['none', 'A', 'AA', 'AAA'] },
      issues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            component: { type: 'string' },
            issue: { type: 'string' },
            wcagCriteria: { type: 'string' },
            severity: { type: 'string', enum: ['critical', 'serious', 'moderate', 'minor'] },
            fix: { type: 'string' }
          }
        }
      },
      checklist: {
        type: 'object',
        properties: {
          ariaLabels: { type: 'boolean' },
          keyboardNav: { type: 'boolean' },
          focusManagement: { type: 'boolean' },
          colorContrast: { type: 'boolean' },
          altText: { type: 'boolean' },
          skipLinks: { type: 'boolean' },
          motionPreferences: { type: 'boolean' },
          textScaling: { type: 'boolean' }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to audit accessibility compliance and WCAG conformance',
  systemPrompt: `You are an accessibility (a11y) expert.

Audit for WCAG 2.1 AA compliance:
1. ARIA labels and roles
2. Keyboard navigation (Tab, Enter, Escape)
3. Focus management and visible focus states
4. Color contrast ratios (4.5:1 normal, 3:1 large text)
5. Alternative text for images
6. Skip navigation links
7. Motion preferences (prefers-reduced-motion)
8. Text scaling support (up to 200%)
9. Form labels and error messages
10. Screen reader compatibility

Common Issues:
- Missing aria-label on icon buttons
- No visible focus indicator
- Non-semantic HTML (div instead of button)
- Missing form labels
- Color-only information
- Auto-playing media
- Keyboard traps in modals`,
  instructionsPrompt: `Audit accessibility:

1. Search for buttons without aria-label
2. Check for onClick on non-button elements
3. Look for images missing alt text
4. Check form inputs for labels
5. Verify focus trap in modals
6. Check for prefers-reduced-motion usage
7. Look for color-only information conveyance
8. Check heading hierarchy (h1, h2, h3)
9. Verify skip navigation link exists
10. Report all issues with WCAG criteria and fixes`
}

export default agentDefinition
