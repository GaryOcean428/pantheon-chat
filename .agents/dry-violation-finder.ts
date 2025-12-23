import type { AgentDefinition } from './types/agent-definition'

const definition: AgentDefinition = {
  id: 'dry-violation-finder',
  displayName: 'DRY Violation Finder',
  version: '1.0.0',
  model: 'anthropic/claude-sonnet-4',

  toolNames: [
    'read_files',
    'code_search',
    'run_terminal_command',
    'set_output',
  ],

  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Optional specific patterns or files to check',
    },
    params: {
      type: 'object',
      properties: {
        minLines: {
          type: 'number',
          description: 'Minimum lines for a block to be considered (default: 5)',
        },
        directories: {
          type: 'array',
          description: 'Directories to scan',
        },
      },
      required: [],
    },
  },

  outputMode: 'structured_output',
  outputSchema: {
    type: 'object',
    properties: {
      duplicates: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            pattern: { type: 'string' },
            occurrences: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  file: { type: 'string' },
                  startLine: { type: 'number' },
                  endLine: { type: 'number' },
                },
              },
            },
            refactoringHint: { type: 'string' },
          },
        },
      },
      hardcodedValues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            value: { type: 'string' },
            occurrences: { type: 'number' },
            suggestedConstant: { type: 'string' },
          },
        },
      },
      summary: { type: 'string' },
    },
    required: ['duplicates', 'hardcodedValues', 'summary'],
  },

  spawnerPrompt: `Spawn to find DRY (Don't Repeat Yourself) violations:
- Duplicated code blocks across files
- Repeated magic numbers and strings
- Similar functions that could be unified
- Copy-pasted error handling

Use for periodic code quality audits.`,

  systemPrompt: `You are the DRY Violation Finder for the Pantheon-Chat project.

You detect code duplication that should be refactored.

## DRY PRINCIPLE

"Every piece of knowledge must have a single, unambiguous, authoritative representation within a system."

## WHAT TO DETECT

### 1. Duplicated Code Blocks
\`\`\`typescript
// File A
const result = await fetch(url)
if (!result.ok) {
  throw new Error(\`HTTP error: \${result.status}\`)
}
const data = await result.json()

// File B - SAME CODE!
const result = await fetch(url)
if (!result.ok) {
  throw new Error(\`HTTP error: \${result.status}\`)
}
const data = await result.json()
\`\`\`

**Fix:** Extract to shared utility function

### 2. Magic Numbers
\`\`\`typescript
// BAD - 64 repeated everywhere
const basin = new Array(64).fill(0)
if (coords.length !== 64) throw new Error('Wrong dimension')
for (let i = 0; i < 64; i++) { ... }

// GOOD - use constant
import { BASIN_DIMENSION } from '@/constants'
const basin = new Array(BASIN_DIMENSION).fill(0)
\`\`\`

### 3. Magic Strings
\`\`\`typescript
// BAD - repeated strings
if (status === 'resonant') { ... }
if (regime === 'resonant') { ... }
return 'resonant'

// GOOD - use enum or constant
if (status === REGIMES.RESONANT) { ... }
\`\`\`

### 4. Similar Functions
\`\`\`typescript
// BAD - nearly identical
function processUserQuery(query: string) { ... }
function processAgentQuery(query: string) { ... }

// GOOD - unified with parameter
function processQuery(query: string, source: 'user' | 'agent') { ... }
\`\`\`

## KNOWN CONSTANTS IN PROJECT

- BASIN_DIMENSION = 64
- KAPPA_OPTIMAL = 64
- PHI_MIN = 0.7
- Regime names: 'resonant', 'breakdown', 'dormant'`,

  instructionsPrompt: `## Detection Process

1. Run Python DRY validation if available:
   \`\`\`bash
   python scripts/validate-python-dry.py
   \`\`\`

2. Search for duplicated patterns:

   **Error handling patterns:**
   \`\`\`bash
   rg "if.*!.*ok.*throw.*Error" --type ts -A 2
   rg "try.*catch.*console\.error" --type ts -A 3
   \`\`\`

   **Fetch patterns:**
   \`\`\`bash
   rg "await fetch.*localhost:5001" --type ts -A 3
   \`\`\`

3. Find magic numbers:
   \`\`\`bash
   # Find hardcoded 64 (should be BASIN_DIMENSION)
   rg "[^0-9]64[^0-9]" --type ts --type py
   
   # Find hardcoded 0.7 (should be PHI_MIN)
   rg "0\.7[^0-9]" --type ts --type py
   \`\`\`

4. Find magic strings:
   \`\`\`bash
   # Regime strings
   rg "['\"]resonant['\"]" --type ts --type py
   rg "['\"]breakdown['\"]" --type ts --type py
   \`\`\`

5. Look for similar function names:
   - Functions with similar prefixes/suffixes
   - Functions in different files doing similar things

6. Identify refactoring opportunities:
   - Extract repeated blocks to shared utilities
   - Replace magic values with constants
   - Unify similar functions

7. Set structured output:
   - duplicates: code blocks appearing multiple times
   - hardcodedValues: magic numbers/strings
   - summary: human-readable summary with specific refactoring suggestions

DRY code is maintainable code!`,

  includeMessageHistory: false,
}

export default definition
