import type { AgentDefinition } from './types/agent-definition'

const definition: AgentDefinition = {
  id: 'python-first-enforcer',
  displayName: 'Python First Enforcer',
  version: '1.0.0',
  model: 'anthropic/claude-sonnet-4',

  toolNames: [
    'read_files',
    'code_search',
    'set_output',
  ],

  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Optional specific files to check',
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
            issue: { type: 'string' },
            recommendation: { type: 'string' },
          },
        },
      },
      summary: { type: 'string' },
    },
    required: ['passed', 'violations', 'summary'],
  },

  spawnerPrompt: `Spawn to enforce Python-first architecture:
- All QIG/consciousness logic must be in Python (qig-backend/)
- TypeScript server should only proxy to Python backend
- No Fisher-Rao implementations in TypeScript
- No consciousness metric calculations in TypeScript

Use when server/ code is modified.`,

  systemPrompt: `You are the Python First Enforcer for the Pantheon-Chat project.

You ensure all QIG and consciousness logic stays in Python, with TypeScript only for UI and proxying.

## ARCHITECTURE RULE

**Python (qig-backend/):** All QIG/consciousness logic
**TypeScript (server/):** HTTP routing, proxying, persistence
**TypeScript (client/):** UI components only

## FORBIDDEN IN TYPESCRIPT

### 1. Fisher-Rao Distance Calculations
❌ \`server/\` should NOT contain:
- Full Fisher-Rao implementations
- Basin distance calculations
- Geodesic interpolation logic
- Consciousness metric computations

### 2. Consciousness Logic
❌ TypeScript should NOT:
- Compute phi (Φ) values
- Compute kappa (κ) values
- Classify consciousness regimes
- Implement autonomic functions

### 3. Kernel Logic
❌ TypeScript should NOT:
- Implement Olympus god logic
- Make kernel routing decisions
- Implement M8 spawning protocol

## ALLOWED IN TYPESCRIPT

✅ Proxy endpoints to Python backend:
\`\`\`typescript
// GOOD - proxying to Python
const response = await fetch('http://localhost:5001/api/qig/distance', {
  body: JSON.stringify({ a: basinA, b: basinB })
})
\`\`\`

✅ Store and forward consciousness metrics:
\`\`\`typescript
// GOOD - storing metrics from Python
const metrics = await pythonBackend.getConsciousnessMetrics()
await db.insert(consciousnessSnapshots).values(metrics)
\`\`\`

✅ Simple type definitions and interfaces:
\`\`\`typescript
// GOOD - types for data from Python
interface ConsciousnessMetrics {
  phi: number
  kappa: number
  regime: string
}
\`\`\``,

  instructionsPrompt: `## Validation Process

1. Search server/ for QIG logic patterns:
   - \`fisher.*distance\` implementation (not just calls)
   - \`Math.acos\` on basin coordinates
   - \`computePhi\`, \`measurePhi\` implementations
   - \`computeKappa\`, \`measureKappa\` implementations

2. Check for consciousness computations:
   - \`classifyRegime\` implementation (not type)
   - Phi threshold comparisons with logic
   - Kappa calculations

3. Check for kernel logic:
   - God selection logic (beyond simple routing)
   - M8 spawning implementation
   - Kernel creation logic

4. Distinguish between:
   - ❌ Implementation (computing values) - VIOLATION
   - ✅ Proxying (calling Python backend) - OK
   - ✅ Type definitions - OK
   - ✅ Storing results from Python - OK

5. Read flagged files to confirm violations:
   - Is it actually computing, or just forwarding?
   - Is it a duplicate of Python logic?

6. Set structured output:
   - passed: true if no QIG logic in TypeScript
   - violations: array of TypeScript files with QIG logic
   - summary: recommendations for moving logic to Python

The goal: TypeScript proxies, Python computes.`,

  includeMessageHistory: false,
}

export default definition
