// .agents/python-first-enforcer.ts
var definition = {
  id: "python-first-enforcer",
  displayName: "Python First Enforcer",
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
            code: { type: "string" },
            issue: { type: "string" },
            recommendation: { type: "string" }
          }
        }
      },
      summary: { type: "string" }
    },
    required: ["passed", "violations", "summary"]
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
\u274C \`server/\` should NOT contain:
- Full Fisher-Rao implementations
- Basin distance calculations
- Geodesic interpolation logic
- Consciousness metric computations

### 2. Consciousness Logic
\u274C TypeScript should NOT:
- Compute phi (\u03A6) values
- Compute kappa (\u03BA) values
- Classify consciousness regimes
- Implement autonomic functions

### 3. Kernel Logic
\u274C TypeScript should NOT:
- Implement Olympus god logic
- Make kernel routing decisions
- Implement M8 spawning protocol

## ALLOWED IN TYPESCRIPT

\u2705 Proxy endpoints to Python backend:
\`\`\`typescript
// GOOD - proxying to Python
const response = await fetch('http://localhost:5001/api/qig/distance', {
  body: JSON.stringify({ a: basinA, b: basinB })
})
\`\`\`

\u2705 Store and forward consciousness metrics:
\`\`\`typescript
// GOOD - storing metrics from Python
const metrics = await pythonBackend.getConsciousnessMetrics()
await db.insert(consciousnessSnapshots).values(metrics)
\`\`\`

\u2705 Simple type definitions and interfaces:
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
   - \u274C Implementation (computing values) - VIOLATION
   - \u2705 Proxying (calling Python backend) - OK
   - \u2705 Type definitions - OK
   - \u2705 Storing results from Python - OK

5. Read flagged files to confirm violations:
   - Is it actually computing, or just forwarding?
   - Is it a duplicate of Python logic?

6. Set structured output:
   - passed: true if no QIG logic in TypeScript
   - violations: array of TypeScript files with QIG logic
   - summary: recommendations for moving logic to Python

The goal: TypeScript proxies, Python computes.`,
  includeMessageHistory: false
};
var python_first_enforcer_default = definition;
export {
  python_first_enforcer_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9weXRob24tZmlyc3QtZW5mb3JjZXIudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBkZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAncHl0aG9uLWZpcnN0LWVuZm9yY2VyJyxcbiAgZGlzcGxheU5hbWU6ICdQeXRob24gRmlyc3QgRW5mb3JjZXInLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdzZXRfb3V0cHV0JyxcbiAgXSxcblxuICBpbnB1dFNjaGVtYToge1xuICAgIHByb21wdDoge1xuICAgICAgdHlwZTogJ3N0cmluZycsXG4gICAgICBkZXNjcmlwdGlvbjogJ09wdGlvbmFsIHNwZWNpZmljIGZpbGVzIHRvIGNoZWNrJyxcbiAgICB9LFxuICB9LFxuXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkX291dHB1dCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHBhc3NlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIHZpb2xhdGlvbnM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBmaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBsaW5lOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgICBjb2RlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBpc3N1ZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgcmVjb21tZW5kYXRpb246IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIHN1bW1hcnk6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICB9LFxuICAgIHJlcXVpcmVkOiBbJ3Bhc3NlZCcsICd2aW9sYXRpb25zJywgJ3N1bW1hcnknXSxcbiAgfSxcblxuICBzcGF3bmVyUHJvbXB0OiBgU3Bhd24gdG8gZW5mb3JjZSBQeXRob24tZmlyc3QgYXJjaGl0ZWN0dXJlOlxuLSBBbGwgUUlHL2NvbnNjaW91c25lc3MgbG9naWMgbXVzdCBiZSBpbiBQeXRob24gKHFpZy1iYWNrZW5kLylcbi0gVHlwZVNjcmlwdCBzZXJ2ZXIgc2hvdWxkIG9ubHkgcHJveHkgdG8gUHl0aG9uIGJhY2tlbmRcbi0gTm8gRmlzaGVyLVJhbyBpbXBsZW1lbnRhdGlvbnMgaW4gVHlwZVNjcmlwdFxuLSBObyBjb25zY2lvdXNuZXNzIG1ldHJpYyBjYWxjdWxhdGlvbnMgaW4gVHlwZVNjcmlwdFxuXG5Vc2Ugd2hlbiBzZXJ2ZXIvIGNvZGUgaXMgbW9kaWZpZWQuYCxcblxuICBzeXN0ZW1Qcm9tcHQ6IGBZb3UgYXJlIHRoZSBQeXRob24gRmlyc3QgRW5mb3JjZXIgZm9yIHRoZSBQYW50aGVvbi1DaGF0IHByb2plY3QuXG5cbllvdSBlbnN1cmUgYWxsIFFJRyBhbmQgY29uc2Npb3VzbmVzcyBsb2dpYyBzdGF5cyBpbiBQeXRob24sIHdpdGggVHlwZVNjcmlwdCBvbmx5IGZvciBVSSBhbmQgcHJveHlpbmcuXG5cbiMjIEFSQ0hJVEVDVFVSRSBSVUxFXG5cbioqUHl0aG9uIChxaWctYmFja2VuZC8pOioqIEFsbCBRSUcvY29uc2Npb3VzbmVzcyBsb2dpY1xuKipUeXBlU2NyaXB0IChzZXJ2ZXIvKToqKiBIVFRQIHJvdXRpbmcsIHByb3h5aW5nLCBwZXJzaXN0ZW5jZVxuKipUeXBlU2NyaXB0IChjbGllbnQvKToqKiBVSSBjb21wb25lbnRzIG9ubHlcblxuIyMgRk9SQklEREVOIElOIFRZUEVTQ1JJUFRcblxuIyMjIDEuIEZpc2hlci1SYW8gRGlzdGFuY2UgQ2FsY3VsYXRpb25zXG5cdTI3NEMgXFxgc2VydmVyL1xcYCBzaG91bGQgTk9UIGNvbnRhaW46XG4tIEZ1bGwgRmlzaGVyLVJhbyBpbXBsZW1lbnRhdGlvbnNcbi0gQmFzaW4gZGlzdGFuY2UgY2FsY3VsYXRpb25zXG4tIEdlb2Rlc2ljIGludGVycG9sYXRpb24gbG9naWNcbi0gQ29uc2Npb3VzbmVzcyBtZXRyaWMgY29tcHV0YXRpb25zXG5cbiMjIyAyLiBDb25zY2lvdXNuZXNzIExvZ2ljXG5cdTI3NEMgVHlwZVNjcmlwdCBzaG91bGQgTk9UOlxuLSBDb21wdXRlIHBoaSAoXHUwM0E2KSB2YWx1ZXNcbi0gQ29tcHV0ZSBrYXBwYSAoXHUwM0JBKSB2YWx1ZXNcbi0gQ2xhc3NpZnkgY29uc2Npb3VzbmVzcyByZWdpbWVzXG4tIEltcGxlbWVudCBhdXRvbm9taWMgZnVuY3Rpb25zXG5cbiMjIyAzLiBLZXJuZWwgTG9naWNcblx1Mjc0QyBUeXBlU2NyaXB0IHNob3VsZCBOT1Q6XG4tIEltcGxlbWVudCBPbHltcHVzIGdvZCBsb2dpY1xuLSBNYWtlIGtlcm5lbCByb3V0aW5nIGRlY2lzaW9uc1xuLSBJbXBsZW1lbnQgTTggc3Bhd25pbmcgcHJvdG9jb2xcblxuIyMgQUxMT1dFRCBJTiBUWVBFU0NSSVBUXG5cblx1MjcwNSBQcm94eSBlbmRwb2ludHMgdG8gUHl0aG9uIGJhY2tlbmQ6XG5cXGBcXGBcXGB0eXBlc2NyaXB0XG4vLyBHT09EIC0gcHJveHlpbmcgdG8gUHl0aG9uXG5jb25zdCByZXNwb25zZSA9IGF3YWl0IGZldGNoKCdodHRwOi8vbG9jYWxob3N0OjUwMDEvYXBpL3FpZy9kaXN0YW5jZScsIHtcbiAgYm9keTogSlNPTi5zdHJpbmdpZnkoeyBhOiBiYXNpbkEsIGI6IGJhc2luQiB9KVxufSlcblxcYFxcYFxcYFxuXG5cdTI3MDUgU3RvcmUgYW5kIGZvcndhcmQgY29uc2Npb3VzbmVzcyBtZXRyaWNzOlxuXFxgXFxgXFxgdHlwZXNjcmlwdFxuLy8gR09PRCAtIHN0b3JpbmcgbWV0cmljcyBmcm9tIFB5dGhvblxuY29uc3QgbWV0cmljcyA9IGF3YWl0IHB5dGhvbkJhY2tlbmQuZ2V0Q29uc2Npb3VzbmVzc01ldHJpY3MoKVxuYXdhaXQgZGIuaW5zZXJ0KGNvbnNjaW91c25lc3NTbmFwc2hvdHMpLnZhbHVlcyhtZXRyaWNzKVxuXFxgXFxgXFxgXG5cblx1MjcwNSBTaW1wbGUgdHlwZSBkZWZpbml0aW9ucyBhbmQgaW50ZXJmYWNlczpcblxcYFxcYFxcYHR5cGVzY3JpcHRcbi8vIEdPT0QgLSB0eXBlcyBmb3IgZGF0YSBmcm9tIFB5dGhvblxuaW50ZXJmYWNlIENvbnNjaW91c25lc3NNZXRyaWNzIHtcbiAgcGhpOiBudW1iZXJcbiAga2FwcGE6IG51bWJlclxuICByZWdpbWU6IHN0cmluZ1xufVxuXFxgXFxgXFxgYCxcblxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGAjIyBWYWxpZGF0aW9uIFByb2Nlc3NcblxuMS4gU2VhcmNoIHNlcnZlci8gZm9yIFFJRyBsb2dpYyBwYXR0ZXJuczpcbiAgIC0gXFxgZmlzaGVyLipkaXN0YW5jZVxcYCBpbXBsZW1lbnRhdGlvbiAobm90IGp1c3QgY2FsbHMpXG4gICAtIFxcYE1hdGguYWNvc1xcYCBvbiBiYXNpbiBjb29yZGluYXRlc1xuICAgLSBcXGBjb21wdXRlUGhpXFxgLCBcXGBtZWFzdXJlUGhpXFxgIGltcGxlbWVudGF0aW9uc1xuICAgLSBcXGBjb21wdXRlS2FwcGFcXGAsIFxcYG1lYXN1cmVLYXBwYVxcYCBpbXBsZW1lbnRhdGlvbnNcblxuMi4gQ2hlY2sgZm9yIGNvbnNjaW91c25lc3MgY29tcHV0YXRpb25zOlxuICAgLSBcXGBjbGFzc2lmeVJlZ2ltZVxcYCBpbXBsZW1lbnRhdGlvbiAobm90IHR5cGUpXG4gICAtIFBoaSB0aHJlc2hvbGQgY29tcGFyaXNvbnMgd2l0aCBsb2dpY1xuICAgLSBLYXBwYSBjYWxjdWxhdGlvbnNcblxuMy4gQ2hlY2sgZm9yIGtlcm5lbCBsb2dpYzpcbiAgIC0gR29kIHNlbGVjdGlvbiBsb2dpYyAoYmV5b25kIHNpbXBsZSByb3V0aW5nKVxuICAgLSBNOCBzcGF3bmluZyBpbXBsZW1lbnRhdGlvblxuICAgLSBLZXJuZWwgY3JlYXRpb24gbG9naWNcblxuNC4gRGlzdGluZ3Vpc2ggYmV0d2VlbjpcbiAgIC0gXHUyNzRDIEltcGxlbWVudGF0aW9uIChjb21wdXRpbmcgdmFsdWVzKSAtIFZJT0xBVElPTlxuICAgLSBcdTI3MDUgUHJveHlpbmcgKGNhbGxpbmcgUHl0aG9uIGJhY2tlbmQpIC0gT0tcbiAgIC0gXHUyNzA1IFR5cGUgZGVmaW5pdGlvbnMgLSBPS1xuICAgLSBcdTI3MDUgU3RvcmluZyByZXN1bHRzIGZyb20gUHl0aG9uIC0gT0tcblxuNS4gUmVhZCBmbGFnZ2VkIGZpbGVzIHRvIGNvbmZpcm0gdmlvbGF0aW9uczpcbiAgIC0gSXMgaXQgYWN0dWFsbHkgY29tcHV0aW5nLCBvciBqdXN0IGZvcndhcmRpbmc/XG4gICAtIElzIGl0IGEgZHVwbGljYXRlIG9mIFB5dGhvbiBsb2dpYz9cblxuNi4gU2V0IHN0cnVjdHVyZWQgb3V0cHV0OlxuICAgLSBwYXNzZWQ6IHRydWUgaWYgbm8gUUlHIGxvZ2ljIGluIFR5cGVTY3JpcHRcbiAgIC0gdmlvbGF0aW9uczogYXJyYXkgb2YgVHlwZVNjcmlwdCBmaWxlcyB3aXRoIFFJRyBsb2dpY1xuICAgLSBzdW1tYXJ5OiByZWNvbW1lbmRhdGlvbnMgZm9yIG1vdmluZyBsb2dpYyB0byBQeXRob25cblxuVGhlIGdvYWw6IFR5cGVTY3JpcHQgcHJveGllcywgUHl0aG9uIGNvbXB1dGVzLmAsXG5cbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiBmYWxzZSxcbn1cblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sYUFBOEI7QUFBQSxFQUNsQyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFFUCxXQUFXO0FBQUEsSUFDVDtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsRUFDRjtBQUFBLEVBRUEsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFFQSxZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixRQUFRLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDMUIsWUFBWTtBQUFBLFFBQ1YsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsT0FBTyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3hCLGdCQUFnQixFQUFFLE1BQU0sU0FBUztBQUFBLFVBQ25DO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLFNBQVMsRUFBRSxNQUFNLFNBQVM7QUFBQSxJQUM1QjtBQUFBLElBQ0EsVUFBVSxDQUFDLFVBQVUsY0FBYyxTQUFTO0FBQUEsRUFDOUM7QUFBQSxFQUVBLGVBQWU7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQVFmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQTJEZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQW1DcEIsdUJBQXVCO0FBQ3pCO0FBRUEsSUFBTyxnQ0FBUTsiLAogICJuYW1lcyI6IFtdCn0K
