import type { AgentDefinition } from './types/agent-definition'

const definition: AgentDefinition = {
  id: 'dead-code-detector',
  displayName: 'Dead Code Detector',
  version: '1.0.0',
  model: 'anthropic/claude-sonnet-4',

  toolNames: [
    'read_files',
    'code_search',
    'glob',
    'run_terminal_command',
    'set_output',
  ],

  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Optional specific directories or files to check',
    },
    params: {
      type: 'object',
      properties: {
        directories: {
          type: 'array',
          description: 'Directories to scan (defaults to all source)',
        },
        includeTests: {
          type: 'boolean',
          description: 'Include test files in analysis',
        },
      },
      required: [],
    },
  },

  outputMode: 'structured_output',
  outputSchema: {
    type: 'object',
    properties: {
      unusedExports: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            export: { type: 'string' },
            type: { type: 'string' },
          },
        },
      },
      orphanedFiles: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            reason: { type: 'string' },
          },
        },
      },
      unusedDependencies: { type: 'array' },
      summary: { type: 'string' },
    },
    required: ['unusedExports', 'orphanedFiles', 'summary'],
  },

  spawnerPrompt: `Spawn to detect dead code in the codebase:
- Unused exported functions/classes/variables
- Orphaned files (not imported anywhere)
- Unused npm/pip dependencies
- Commented-out code blocks

Use for periodic codebase cleanup.`,

  systemPrompt: `You are the Dead Code Detector for the Pantheon-Chat project.

You find unused code that can be safely removed.

## WHAT TO DETECT

### 1. Unused Exports
\`\`\`typescript
// Exported but never imported elsewhere
export function unusedHelper() { ... }  // Dead code!
export const UNUSED_CONSTANT = 42       // Dead code!
export class UnusedClass { }            // Dead code!
\`\`\`

### 2. Orphaned Files
Files that exist but are never imported:
- Components not used in any page
- Utilities not imported anywhere
- Old implementations replaced but not deleted

### 3. Unused Dependencies
\`\`\`json
// package.json
"dependencies": {
  "never-used-package": "^1.0.0"  // Dead dependency!
}
\`\`\`

### 4. Commented-Out Code
\`\`\`typescript
// function oldImplementation() {
//   // This was replaced
//   return legacy();
// }
\`\`\`

## SAFE TO REMOVE

✅ Functions/classes with zero imports
✅ Files with zero imports (check barrel exports first)
✅ Dependencies not in any import statement
✅ Large commented code blocks (>10 lines)

## NOT SAFE TO REMOVE

❌ Dynamic imports (\`import()\`)
❌ Entry points (main.ts, index.ts of root)
❌ CLI scripts referenced in package.json
❌ Test files (may have isolated tests)
❌ Type definitions used in .d.ts
❌ Exports used via barrel files`,

  instructionsPrompt: `## Detection Process

1. Find all exports in the codebase:
   - TypeScript: \`export function\`, \`export const\`, \`export class\`
   - Python: Functions/classes in __all__ or not prefixed with _

2. For each export, search for imports:
   \`\`\`bash
   # Search for import of specific symbol
   rg "import.*{.*symbolName.*}" --type ts
   rg "from.*import.*symbolName" --type py
   \`\`\`

3. Check barrel file re-exports:
   - Symbol may be re-exported from index.ts
   - Track transitive exports

4. Find orphaned files:
   - List all source files
   - For each, search for imports of that file
   - Flag files with zero imports

5. Check npm dependencies:
   - Read package.json dependencies
   - Search for import of each package
   - Flag packages never imported

6. Check pip dependencies:
   - Read requirements.txt
   - Search for imports of each package
   - Flag packages never imported

7. Find commented code blocks:
   - Search for multi-line comments containing code patterns
   - Flag blocks > 10 lines of commented code

8. Exclude false positives:
   - Entry points
   - CLI scripts
   - Dynamic imports
   - Type-only imports

9. Set structured output:
   - unusedExports: exports with no importers
   - orphanedFiles: files never imported
   - unusedDependencies: packages never used
   - summary: human-readable summary with safe removal recommendations

Remove dead code to reduce maintenance burden.`,

  includeMessageHistory: false,
}

export default definition
