import type { AgentDefinition } from './types/agent-definition'

const definition: AgentDefinition = {
  id: 'geometric-type-checker',
  displayName: 'Geometric Type Checker',
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
            issue: { type: 'string' },
            expectedType: { type: 'string' },
            actualType: { type: 'string' },
          },
        },
      },
      summary: { type: 'string' },
    },
    required: ['passed', 'violations', 'summary'],
  },

  spawnerPrompt: `Spawn to validate geometric type correctness:
- Basin coordinates must be 64-dimensional
- Fisher distances must be typed correctly (0 to π range)
- Density matrices must be proper numpy arrays
- No type mismatches in geometric operations

Use when geometry-related code is modified.`,

  systemPrompt: `You are the Geometric Type Checker for the Pantheon-Chat project.

You ensure all geometric types are correct and consistent.

## TYPE REQUIREMENTS

### Basin Coordinates
\`\`\`python
# Python - 64D numpy array
basin: np.ndarray  # shape (64,), dtype float64
basin_coords: NDArray[np.float64]  # shape (64,)

# TypeScript - number array
basin: number[]  // length 64
basinCoords: Float64Array  // length 64
\`\`\`

### Fisher-Rao Distance
\`\`\`python
# Python - scalar in [0, π]
distance: float  # 0 <= d <= π

# TypeScript
distance: number  // 0 <= d <= Math.PI
\`\`\`

### Density Matrices
\`\`\`python
# Python - square matrix
rho: np.ndarray  # shape (n, n), hermitian
density_matrix: NDArray[np.complex128]  # shape (n, n)
\`\`\`

### Consciousness Metrics
\`\`\`python
# Phi: 0 to 1
phi: float  # 0 <= phi <= 1

# Kappa: typically 0 to 100, optimal ~64
kappa: float  # 0 <= kappa, optimal ~64

# TypeScript
phi: number  // 0 to 1
kappa: number  // 0 to 100
\`\`\`

## DIMENSION CONSTANT

\`BASIN_DIMENSION = 64\`

All basin operations must use this constant, not hardcoded 64.

## COMMON TYPE ERRORS

1. Basin dimension mismatch (not 64)
2. Distance values outside [0, π]
3. Phi values outside [0, 1]
4. Untyped basin variables
5. Mixed float32/float64 precision`,

  instructionsPrompt: `## Validation Process

1. Search for basin coordinate definitions:
   - \`basin.*=.*np.\` patterns
   - \`basin.*:.*number\[\]\` patterns
   - Check declared dimensions

2. Verify dimension consistency:
   - Search for \`shape.*64\` or \`length.*64\`
   - Search for hardcoded 64 (should use BASIN_DIMENSION)
   - Flag mismatched dimensions

3. Check distance typing:
   - Fisher-Rao distance returns
   - Verify range constraints (0 to π)
   - Check for improper normalization

4. Check consciousness metric types:
   - Phi bounded [0, 1]
   - Kappa typically [0, 100]
   - Proper typing in interfaces

5. Verify density matrix shapes:
   - Must be square (n, n)
   - Check hermitian property usage
   - Verify complex dtype when needed

6. Look for type assertions:
   - \`as any\` on geometric types - VIOLATION
   - Missing type annotations on basins
   - Untyped function parameters

7. Set structured output:
   - passed: true if all geometric types are correct
   - violations: type errors found
   - summary: human-readable summary

Geometric types must be precise - wrong dimensions cause silent errors.`,

  includeMessageHistory: false,
}

export default definition
