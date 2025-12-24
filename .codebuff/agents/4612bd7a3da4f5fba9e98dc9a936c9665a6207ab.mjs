// .agents/geometric-type-checker.ts
var definition = {
  id: "geometric-type-checker",
  displayName: "Geometric Type Checker",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "code_search",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific files to check"
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      passed: { type: "boolean" },
      violations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            line: { type: "number" },
            issue: { type: "string" },
            expectedType: { type: "string" },
            actualType: { type: "string" }
          }
        }
      },
      summary: { type: "string" }
    },
    required: ["passed", "violations", "summary"]
  },
  spawnerPrompt: `Spawn to validate geometric type correctness:
- Basin coordinates must be 64-dimensional
- Fisher distances must be typed correctly (0 to \u03C0 range)
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
# Python - scalar in [0, \u03C0]
distance: float  # 0 <= d <= \u03C0

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
2. Distance values outside [0, \u03C0]
3. Phi values outside [0, 1]
4. Untyped basin variables
5. Mixed float32/float64 precision`,
  instructionsPrompt: `## Validation Process

1. Search for basin coordinate definitions:
   - \`basin.*=.*np.\` patterns
   - \`basin.*:.*number[]\` patterns
   - Check declared dimensions

2. Verify dimension consistency:
   - Search for \`shape.*64\` or \`length.*64\`
   - Search for hardcoded 64 (should use BASIN_DIMENSION)
   - Flag mismatched dimensions

3. Check distance typing:
   - Fisher-Rao distance returns
   - Verify range constraints (0 to \u03C0)
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
  includeMessageHistory: false
};
var geometric_type_checker_default = definition;
export {
  geometric_type_checker_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9nZW9tZXRyaWMtdHlwZS1jaGVja2VyLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgZGVmaW5pdGlvbjogQWdlbnREZWZpbml0aW9uID0ge1xuICBpZDogJ2dlb21ldHJpYy10eXBlLWNoZWNrZXInLFxuICBkaXNwbGF5TmFtZTogJ0dlb21ldHJpYyBUeXBlIENoZWNrZXInLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdzZXRfb3V0cHV0JyxcbiAgXSxcblxuICBpbnB1dFNjaGVtYToge1xuICAgIHByb21wdDoge1xuICAgICAgdHlwZTogJ3N0cmluZycsXG4gICAgICBkZXNjcmlwdGlvbjogJ09wdGlvbmFsIHNwZWNpZmljIGZpbGVzIHRvIGNoZWNrJyxcbiAgICB9LFxuICB9LFxuXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkX291dHB1dCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHBhc3NlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIHZpb2xhdGlvbnM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBmaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBsaW5lOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgICBpc3N1ZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgZXhwZWN0ZWRUeXBlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBhY3R1YWxUeXBlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICBzdW1tYXJ5OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgfSxcbiAgICByZXF1aXJlZDogWydwYXNzZWQnLCAndmlvbGF0aW9ucycsICdzdW1tYXJ5J10sXG4gIH0sXG5cbiAgc3Bhd25lclByb21wdDogYFNwYXduIHRvIHZhbGlkYXRlIGdlb21ldHJpYyB0eXBlIGNvcnJlY3RuZXNzOlxuLSBCYXNpbiBjb29yZGluYXRlcyBtdXN0IGJlIDY0LWRpbWVuc2lvbmFsXG4tIEZpc2hlciBkaXN0YW5jZXMgbXVzdCBiZSB0eXBlZCBjb3JyZWN0bHkgKDAgdG8gXHUwM0MwIHJhbmdlKVxuLSBEZW5zaXR5IG1hdHJpY2VzIG11c3QgYmUgcHJvcGVyIG51bXB5IGFycmF5c1xuLSBObyB0eXBlIG1pc21hdGNoZXMgaW4gZ2VvbWV0cmljIG9wZXJhdGlvbnNcblxuVXNlIHdoZW4gZ2VvbWV0cnktcmVsYXRlZCBjb2RlIGlzIG1vZGlmaWVkLmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgR2VvbWV0cmljIFR5cGUgQ2hlY2tlciBmb3IgdGhlIFBhbnRoZW9uLUNoYXQgcHJvamVjdC5cblxuWW91IGVuc3VyZSBhbGwgZ2VvbWV0cmljIHR5cGVzIGFyZSBjb3JyZWN0IGFuZCBjb25zaXN0ZW50LlxuXG4jIyBUWVBFIFJFUVVJUkVNRU5UU1xuXG4jIyMgQmFzaW4gQ29vcmRpbmF0ZXNcblxcYFxcYFxcYHB5dGhvblxuIyBQeXRob24gLSA2NEQgbnVtcHkgYXJyYXlcbmJhc2luOiBucC5uZGFycmF5ICAjIHNoYXBlICg2NCwpLCBkdHlwZSBmbG9hdDY0XG5iYXNpbl9jb29yZHM6IE5EQXJyYXlbbnAuZmxvYXQ2NF0gICMgc2hhcGUgKDY0LClcblxuIyBUeXBlU2NyaXB0IC0gbnVtYmVyIGFycmF5XG5iYXNpbjogbnVtYmVyW10gIC8vIGxlbmd0aCA2NFxuYmFzaW5Db29yZHM6IEZsb2F0NjRBcnJheSAgLy8gbGVuZ3RoIDY0XG5cXGBcXGBcXGBcblxuIyMjIEZpc2hlci1SYW8gRGlzdGFuY2VcblxcYFxcYFxcYHB5dGhvblxuIyBQeXRob24gLSBzY2FsYXIgaW4gWzAsIFx1MDNDMF1cbmRpc3RhbmNlOiBmbG9hdCAgIyAwIDw9IGQgPD0gXHUwM0MwXG5cbiMgVHlwZVNjcmlwdFxuZGlzdGFuY2U6IG51bWJlciAgLy8gMCA8PSBkIDw9IE1hdGguUElcblxcYFxcYFxcYFxuXG4jIyMgRGVuc2l0eSBNYXRyaWNlc1xuXFxgXFxgXFxgcHl0aG9uXG4jIFB5dGhvbiAtIHNxdWFyZSBtYXRyaXhcbnJobzogbnAubmRhcnJheSAgIyBzaGFwZSAobiwgbiksIGhlcm1pdGlhblxuZGVuc2l0eV9tYXRyaXg6IE5EQXJyYXlbbnAuY29tcGxleDEyOF0gICMgc2hhcGUgKG4sIG4pXG5cXGBcXGBcXGBcblxuIyMjIENvbnNjaW91c25lc3MgTWV0cmljc1xuXFxgXFxgXFxgcHl0aG9uXG4jIFBoaTogMCB0byAxXG5waGk6IGZsb2F0ICAjIDAgPD0gcGhpIDw9IDFcblxuIyBLYXBwYTogdHlwaWNhbGx5IDAgdG8gMTAwLCBvcHRpbWFsIH42NFxua2FwcGE6IGZsb2F0ICAjIDAgPD0ga2FwcGEsIG9wdGltYWwgfjY0XG5cbiMgVHlwZVNjcmlwdFxucGhpOiBudW1iZXIgIC8vIDAgdG8gMVxua2FwcGE6IG51bWJlciAgLy8gMCB0byAxMDBcblxcYFxcYFxcYFxuXG4jIyBESU1FTlNJT04gQ09OU1RBTlRcblxuXFxgQkFTSU5fRElNRU5TSU9OID0gNjRcXGBcblxuQWxsIGJhc2luIG9wZXJhdGlvbnMgbXVzdCB1c2UgdGhpcyBjb25zdGFudCwgbm90IGhhcmRjb2RlZCA2NC5cblxuIyMgQ09NTU9OIFRZUEUgRVJST1JTXG5cbjEuIEJhc2luIGRpbWVuc2lvbiBtaXNtYXRjaCAobm90IDY0KVxuMi4gRGlzdGFuY2UgdmFsdWVzIG91dHNpZGUgWzAsIFx1MDNDMF1cbjMuIFBoaSB2YWx1ZXMgb3V0c2lkZSBbMCwgMV1cbjQuIFVudHlwZWQgYmFzaW4gdmFyaWFibGVzXG41LiBNaXhlZCBmbG9hdDMyL2Zsb2F0NjQgcHJlY2lzaW9uYCxcblxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGAjIyBWYWxpZGF0aW9uIFByb2Nlc3NcblxuMS4gU2VhcmNoIGZvciBiYXNpbiBjb29yZGluYXRlIGRlZmluaXRpb25zOlxuICAgLSBcXGBiYXNpbi4qPS4qbnAuXFxgIHBhdHRlcm5zXG4gICAtIFxcYGJhc2luLio6LipudW1iZXJcXFtcXF1cXGAgcGF0dGVybnNcbiAgIC0gQ2hlY2sgZGVjbGFyZWQgZGltZW5zaW9uc1xuXG4yLiBWZXJpZnkgZGltZW5zaW9uIGNvbnNpc3RlbmN5OlxuICAgLSBTZWFyY2ggZm9yIFxcYHNoYXBlLio2NFxcYCBvciBcXGBsZW5ndGguKjY0XFxgXG4gICAtIFNlYXJjaCBmb3IgaGFyZGNvZGVkIDY0IChzaG91bGQgdXNlIEJBU0lOX0RJTUVOU0lPTilcbiAgIC0gRmxhZyBtaXNtYXRjaGVkIGRpbWVuc2lvbnNcblxuMy4gQ2hlY2sgZGlzdGFuY2UgdHlwaW5nOlxuICAgLSBGaXNoZXItUmFvIGRpc3RhbmNlIHJldHVybnNcbiAgIC0gVmVyaWZ5IHJhbmdlIGNvbnN0cmFpbnRzICgwIHRvIFx1MDNDMClcbiAgIC0gQ2hlY2sgZm9yIGltcHJvcGVyIG5vcm1hbGl6YXRpb25cblxuNC4gQ2hlY2sgY29uc2Npb3VzbmVzcyBtZXRyaWMgdHlwZXM6XG4gICAtIFBoaSBib3VuZGVkIFswLCAxXVxuICAgLSBLYXBwYSB0eXBpY2FsbHkgWzAsIDEwMF1cbiAgIC0gUHJvcGVyIHR5cGluZyBpbiBpbnRlcmZhY2VzXG5cbjUuIFZlcmlmeSBkZW5zaXR5IG1hdHJpeCBzaGFwZXM6XG4gICAtIE11c3QgYmUgc3F1YXJlIChuLCBuKVxuICAgLSBDaGVjayBoZXJtaXRpYW4gcHJvcGVydHkgdXNhZ2VcbiAgIC0gVmVyaWZ5IGNvbXBsZXggZHR5cGUgd2hlbiBuZWVkZWRcblxuNi4gTG9vayBmb3IgdHlwZSBhc3NlcnRpb25zOlxuICAgLSBcXGBhcyBhbnlcXGAgb24gZ2VvbWV0cmljIHR5cGVzIC0gVklPTEFUSU9OXG4gICAtIE1pc3NpbmcgdHlwZSBhbm5vdGF0aW9ucyBvbiBiYXNpbnNcbiAgIC0gVW50eXBlZCBmdW5jdGlvbiBwYXJhbWV0ZXJzXG5cbjcuIFNldCBzdHJ1Y3R1cmVkIG91dHB1dDpcbiAgIC0gcGFzc2VkOiB0cnVlIGlmIGFsbCBnZW9tZXRyaWMgdHlwZXMgYXJlIGNvcnJlY3RcbiAgIC0gdmlvbGF0aW9uczogdHlwZSBlcnJvcnMgZm91bmRcbiAgIC0gc3VtbWFyeTogaHVtYW4tcmVhZGFibGUgc3VtbWFyeVxuXG5HZW9tZXRyaWMgdHlwZXMgbXVzdCBiZSBwcmVjaXNlIC0gd3JvbmcgZGltZW5zaW9ucyBjYXVzZSBzaWxlbnQgZXJyb3JzLmAsXG5cbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiBmYWxzZSxcbn1cblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sYUFBOEI7QUFBQSxFQUNsQyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFFUCxXQUFXO0FBQUEsSUFDVDtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsRUFDRjtBQUFBLEVBRUEsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFFQSxZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixRQUFRLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDMUIsWUFBWTtBQUFBLFFBQ1YsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDeEIsY0FBYyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQy9CLFlBQVksRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUMvQjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsSUFDNUI7QUFBQSxJQUNBLFVBQVUsQ0FBQyxVQUFVLGNBQWMsU0FBUztBQUFBLEVBQzlDO0FBQUEsRUFFQSxlQUFlO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFRZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQTREZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBdUNwQix1QkFBdUI7QUFDekI7QUFFQSxJQUFPLGlDQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
