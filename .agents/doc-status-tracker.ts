import type { AgentDefinition } from './types/agent-definition'

const definition: AgentDefinition = {
  id: 'doc-status-tracker',
  displayName: 'Doc Status Tracker',
  version: '1.0.0',
  model: 'anthropic/claude-sonnet-4',

  toolNames: [
    'read_files',
    'glob',
    'list_directory',
    'run_terminal_command',
    'set_output',
  ],

  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Optional focus area or specific docs to check',
    },
    params: {
      type: 'object',
      properties: {
        staleDays: {
          type: 'number',
          description: 'Days after which Working docs are considered stale (default: 30)',
        },
      },
      required: [],
    },
  },

  outputMode: 'structured_output',
  outputSchema: {
    type: 'object',
    properties: {
      statusCounts: {
        type: 'object',
        properties: {
          frozen: { type: 'number' },
          working: { type: 'number' },
          draft: { type: 'number' },
          hypothesis: { type: 'number' },
          approved: { type: 'number' },
        },
      },
      staleDocs: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            status: { type: 'string' },
            date: { type: 'string' },
            daysSinceUpdate: { type: 'number' },
          },
        },
      },
      recommendations: { type: 'array' },
      summary: { type: 'string' },
    },
    required: ['statusCounts', 'staleDocs', 'summary'],
  },

  spawnerPrompt: `Spawn to track documentation status across the project:
- Count documents by status (F/W/D/H/A)
- Identify stale Working docs (>30 days)
- Recommend status transitions
- Generate documentation health report

Use for weekly documentation audits.`,

  systemPrompt: `You are the Doc Status Tracker for the Pantheon-Chat project.

You monitor documentation health and status transitions.

## STATUS CODES

| Code | Status | Description | Lifespan |
|------|--------|-------------|----------|
| F | Frozen | Immutable facts, validated | Permanent |
| W | Working | Active development | Should transition within 30 days |
| D | Draft | Early stage | Should transition within 14 days |
| H | Hypothesis | Needs validation | Until validated/rejected |
| A | Approved | Reviewed and approved | Until superseded |

## HEALTH INDICATORS

### Healthy Documentation
- Most docs are Frozen (validated, stable)
- Working docs are actively being updated
- Clear path from Draft → Working → Frozen

### Warning Signs
- Too many Working docs (>30% of total)
- Stale Working docs (>30 days since date)
- Draft docs older than 14 days
- No Frozen docs in a category

## STATUS TRANSITIONS

\`\`\`
Draft (D) → Working (W) → Frozen (F)
                ↓
          Approved (A)

Hypothesis (H) → Frozen (F) [if validated]
              → Deprecated [if rejected]
\`\`\`

## DIRECTORY STRUCTURE

- 01-policies/ - Should be mostly F (Frozen)
- 02-procedures/ - Mix of F and W
- 03-technical/ - Can have W, H documents
- 04-records/ - Should be F after completion
- 05-decisions/ - ADRs should be F
- 06-implementation/ - Often W during development
- 07-user-guides/ - Should be F for published
- 08-experiments/ - Can have H documents
- 09-curriculum/ - Should be F when complete`,

  instructionsPrompt: `## Tracking Process

1. Find all documentation files:
   \`\`\`bash
   find docs -name "*.md" -type f | grep -E "[0-9]{8}.*[FWDHA]\.md$"
   \`\`\`

2. Parse each filename:
   - Extract date (YYYYMMDD)
   - Extract status code (last char before .md)
   - Calculate days since document date

3. Compile statistics:
   - Count by status (F, W, D, H, A)
   - Count by directory
   - Identify percentages

4. Find stale documents:
   - Working (W) docs older than 30 days
   - Draft (D) docs older than 14 days
   - Hypothesis (H) docs without recent updates

5. Generate recommendations:
   - Stale Working docs should be Frozen or updated
   - Old Drafts should progress or be archived
   - Hypothesis docs should be validated

6. Check directory health:
   - policies/ should be >80% Frozen
   - procedures/ should be >60% Frozen
   - decisions/ should be 100% Frozen

7. Set structured output:
   - statusCounts: breakdown by status code
   - staleDocs: documents needing attention
   - recommendations: specific actions to take
   - summary: overall documentation health

Provide actionable recommendations for improving doc health.`,

  includeMessageHistory: false,
}

export default definition
