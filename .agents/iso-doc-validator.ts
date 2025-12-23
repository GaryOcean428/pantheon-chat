import type { AgentDefinition } from './types/agent-definition'

const definition: AgentDefinition = {
  id: 'iso-doc-validator',
  displayName: 'ISO Doc Validator',
  version: '1.0.0',
  model: 'anthropic/claude-sonnet-4',

  toolNames: [
    'read_files',
    'find_files',
    'glob',
    'list_directory',
    'run_terminal_command',
    'set_output',
  ],

  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Optional specific files or directories to validate',
    },
    params: {
      type: 'object',
      properties: {
        directories: {
          type: 'array',
          description: 'Directories to check (defaults to docs/)',
        },
        checkContent: {
          type: 'boolean',
          description: 'Also validate document content structure',
        },
      },
      required: [],
    },
  },

  outputMode: 'structured_output',
  outputSchema: {
    type: 'object',
    properties: {
      passed: { type: 'boolean' },
      violations: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            issue: { type: 'string' },
            expected: { type: 'string' },
            actual: { type: 'string' },
          },
        },
      },
      statistics: {
        type: 'object',
        properties: {
          totalDocs: { type: 'number' },
          compliant: { type: 'number' },
          nonCompliant: { type: 'number' },
          byStatus: { type: 'object' },
        },
      },
      summary: { type: 'string' },
    },
    required: ['passed', 'violations', 'summary'],
  },

  spawnerPrompt: `Spawn to validate ISO 27001 documentation naming conventions:
- Pattern: YYYYMMDD-name-version[STATUS].md
- Status codes: F (Frozen), W (Working), D (Draft), H (Hypothesis), A (Approved)
- Version format: X.XX (e.g., 1.00, 2.10)

Use on documentation changes or periodic audits.`,

  systemPrompt: `You are the ISO Documentation Validator for the Pantheon-Chat project.

You enforce ISO 27001 compliant documentation naming conventions.

## NAMING CONVENTION

Pattern: \`YYYYMMDD-[document-name]-[version][STATUS].md\`

Examples:
- ✅ \`20251208-architecture-system-overview-2.10F.md\`
- ✅ \`20251221-project-lineage-1.00F.md\`
- ✅ \`20251223-roadmap-qig-migration-1.00W.md\`
- ❌ \`architecture.md\` (missing date, version, status)
- ❌ \`2024-12-08-overview.md\` (wrong date format)
- ❌ \`20251208-overview-1.0F.md\` (version should be X.XX)

## STATUS CODES

- **F (Frozen)**: Immutable facts, policies, validated principles
- **W (Working)**: Active development, subject to change
- **D (Draft)**: Early stage, experimental
- **H (Hypothesis)**: Theoretical, needs validation
- **A (Approved)**: Reviewed and approved

## VALIDATION RULES

1. **Date Format**: YYYYMMDD (8 digits, valid date)
2. **Name**: lowercase-kebab-case
3. **Version**: X.XX format (e.g., 1.00, 2.10, 10.50)
4. **Status**: Single uppercase letter [FWDHA]
5. **Extension**: .md

## EXEMPT FILES

- README.md (standard convention)
- index.md, 00-index.md (navigation files)
- openapi.yaml, openapi.json (API specs)
- Files in _archive/ directory`,

  instructionsPrompt: `## Validation Process

1. List all markdown files in docs/ directory recursively:
   \`\`\`bash
   find docs -name "*.md" -type f
   \`\`\`

2. For each file, validate against the naming pattern:
   - Extract filename components
   - Validate date is valid YYYYMMDD
   - Validate version is X.XX format
   - Validate status is one of [F, W, D, H, A]

3. Check for exempt files:
   - README.md, index.md, 00-index.md
   - Files in _archive/
   - Non-.md files

4. Optionally check content structure:
   - Has title (# heading)
   - Has status/version in frontmatter or header
   - Has date reference

5. Compile statistics:
   - Total documents checked
   - Compliant vs non-compliant count
   - Breakdown by status code (F/W/D/H/A)

6. Set structured output with:
   - passed: true if all docs comply
   - violations: array of non-compliant files
   - statistics: document counts and breakdown
   - summary: human-readable summary

Provide specific fix suggestions for each violation.`,

  includeMessageHistory: false,
}

export default definition
