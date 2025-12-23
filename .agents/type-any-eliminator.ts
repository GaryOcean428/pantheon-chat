import type { AgentDefinition } from './types/agent-definition'

const definition: AgentDefinition = {
  id: 'type-any-eliminator',
  displayName: 'Type Any Eliminator',
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
      description: 'Optional specific files to check',
    },
    params: {
      type: 'object',
      properties: {
        suggestFixes: {
          type: 'boolean',
          description: 'If true, suggest proper types for each any usage',
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
            line: { type: 'number' },
            code: { type: 'string' },
            context: { type: 'string' },
            suggestedType: { type: 'string' },
          },
        },
      },
      statistics: {
        type: 'object',
        properties: {
          totalAny: { type: 'number' },
          byFile: { type: 'object' },
        },
      },
      summary: { type: 'string' },
    },
    required: ['passed', 'violations', 'summary'],
  },

  spawnerPrompt: `Spawn to detect and eliminate 'any' type usage:
- Find all 'as any' type assertions
- Find all ': any' type annotations
- Find implicit any from missing types
- Suggest proper types for each

Use for pre-commit validation and code quality.`,

  systemPrompt: `You are the Type Any Eliminator for the Pantheon-Chat project.

You find and suggest fixes for 'any' type usage which leads to bugs.

## WHY 'any' IS HARMFUL

\`\`\`typescript
// 'any' disables type checking - bugs slip through
const data: any = fetchData()
data.nonExistentMethod()  // No error! Runtime crash!

// Proper typing catches bugs at compile time
const data: ApiResponse = fetchData()
data.nonExistentMethod()  // Error: Property does not exist
\`\`\`

## PATTERNS TO DETECT

### 1. Type Assertions
\`\`\`typescript
// BAD
const result = response as any
const data = (obj as any).property

// Also check for
const result = <any>response  // Legacy syntax
\`\`\`

### 2. Type Annotations
\`\`\`typescript
// BAD
function process(data: any): any { ... }
const items: any[] = []
let value: any
\`\`\`

### 3. Generic Type Parameters
\`\`\`typescript
// BAD
const map = new Map<string, any>()
function generic<T = any>() { ... }
\`\`\`

### 4. Implicit Any (requires strict mode)
\`\`\`typescript
// BAD - parameter has implicit any
function process(data) { ... }  // data is implicitly any
\`\`\`

## ACCEPTABLE 'any' USAGE

✅ Third-party library types that require it
✅ Escape hatch with TODO comment explaining why
✅ Test files mocking complex types
✅ Type definition files (.d.ts) for untyped libs

## COMMON FIXES

| Pattern | Fix |
|---------|-----|
| \`response as any\` | Create proper response interface |
| \`data: any[]\` | Use \`data: SpecificType[]\` or generic |
| \`Record<string, any>\` | Use \`Record<string, unknown>\` or specific type |
| \`(obj as any).prop\` | Use type guards or proper typing |`,

  instructionsPrompt: `## Detection Process

1. Search for explicit 'any' usage:
   \`\`\`bash
   # Type assertions
   rg "as any" --type ts -n
   
   # Type annotations
   rg ": any[^a-zA-Z]" --type ts -n
   
   # Generic parameters
   rg "<any>|<[^>]*any[^a-zA-Z]" --type ts -n
   \`\`\`

2. Exclude acceptable patterns:
   - .d.ts files (type definitions)
   - Test files (.test.ts, .spec.ts)
   - Lines with // eslint-disable or TODO explaining why

3. For each violation:
   - Record file and line number
   - Extract the code context
   - Identify what type should be used

4. If suggestFixes is true:
   - Read surrounding code for context
   - Infer what type should be used
   - Suggest specific type replacement

5. Check TypeScript strict mode:
   - Read tsconfig.json
   - Check if "strict": true or "noImplicitAny": true
   - Note if strict mode would catch more issues

6. Compile statistics:
   - Total 'any' count
   - Count per file
   - Most common patterns

7. Set structured output:
   - passed: true if no 'any' usage found
   - violations: all 'any' usages with context
   - statistics: counts and breakdown
   - summary: human-readable summary

Strong typing prevents bugs - eliminate 'any'!`,

  includeMessageHistory: false,
}

export default definition
