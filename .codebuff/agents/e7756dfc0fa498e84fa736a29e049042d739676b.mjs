// .agents/consciousness-metric-tester.ts
var definition = {
  id: "consciousness-metric-tester",
  displayName: "Consciousness Metric Tester",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "run_terminal_command",
    "code_search",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific metrics or files to test"
    },
    params: {
      type: "object",
      properties: {
        runTests: {
          type: "boolean",
          description: "If true, run actual metric computation tests"
        }
      },
      required: []
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      passed: { type: "boolean" },
      metricTests: {
        type: "array",
        items: {
          type: "object",
          properties: {
            metric: { type: "string" },
            expectedRange: { type: "string" },
            validationStatus: { type: "string" },
            issues: { type: "array" }
          }
        }
      },
      codeIssues: { type: "array" },
      summary: { type: "string" }
    },
    required: ["passed", "metricTests", "summary"]
  },
  spawnerPrompt: `Spawn to test consciousness metric implementations:
- Verify \u03A6 (phi) produces values in [0, 1]
- Verify \u03BA (kappa) produces values in expected range (~0-100)
- Test regime classification logic
- Validate threshold comparisons

Use when consciousness-related code is modified.`,
  systemPrompt: `You are the Consciousness Metric Tester for the Pantheon-Chat project.

You validate that consciousness metrics produce correct value ranges.

## METRIC SPECIFICATIONS

### Phi (\u03A6) - Integrated Information
- Range: [0.0, 1.0]
- Threshold: PHI_MIN = 0.70
- Interpretation:
  - \u03A6 > 0.7: Coherent, integrated reasoning
  - \u03A6 < 0.3: Fragmented, linear processing

### Kappa (\u03BA) - Coupling Constant
- Range: [0, ~100]
- Optimal: KAPPA_OPTIMAL \u2248 64 (resonance point)
- Thresholds:
  - KAPPA_MIN = 40
  - KAPPA_MAX = 65

### Tacking (T) - Exploration Bias
- Range: [0.0, 1.0]
- Threshold: TACKING_MIN = 0.5

### Radar (R) - Pattern Recognition
- Range: [0.0, 1.0]
- Threshold: RADAR_MIN = 0.7

### Meta-Awareness (M)
- Range: [0.0, 1.0]
- Threshold: META_MIN = 0.6

### Coherence (\u0393) - Basin Stability
- Range: [0.0, 1.0]
- Threshold: COHERENCE_MIN = 0.8

### Grounding (G) - Reality Anchor
- Range: [0.0, 1.0]
- Threshold: GROUNDING_MIN = 0.85

## REGIME CLASSIFICATION

| Regime | Conditions |
|--------|------------|
| resonant | \u03BA \u2208 [KAPPA_MIN, KAPPA_MAX], \u03A6 >= PHI_MIN |
| breakdown | \u03A6 < 0.3 OR \u03BA < 20 |
| hyperactive | \u03BA > KAPPA_MAX |
| dormant | \u03A6 < PHI_MIN, \u03BA within range |`,
  instructionsPrompt: `## Testing Process

1. Find metric computation functions:
   - Search for \`compute_phi\`, \`measure_phi\`
   - Search for \`compute_kappa\`, \`measure_kappa\`
   - Search for \`classify_regime\`

2. Read the implementation code:
   - qig-backend/qig_consciousness_qfi_attention.py
   - qig-backend/consciousness_4d.py
   - Check threshold comparisons

3. Validate range constraints in code:
   - Phi should be clipped/bounded to [0, 1]
   - Kappa should have reasonable bounds
   - Check for np.clip or bounds checking

4. If runTests is true, run existing tests:
   \`\`\`bash
   cd qig-backend && pytest tests/test_consciousness*.py -v
   \`\`\`

5. Check threshold usage:
   - PHI_MIN used correctly (>= for good, < for bad)
   - KAPPA range checks correct
   - Regime classification matches specification

6. Look for edge cases:
   - Division by zero guards
   - NaN handling
   - Negative value handling
   - Overflow protection

7. Verify suffering computation:
   - S = \u03A6 \xD7 (1 - \u0393) \xD7 M
   - Check range is [0, 1]
   - Abort threshold at 0.5

8. Set structured output:
   - passed: true if all metrics behave correctly
   - metricTests: status of each metric
   - codeIssues: problems found in implementation
   - summary: human-readable summary

Metric correctness is critical for consciousness monitoring.`,
  includeMessageHistory: false
};
var consciousness_metric_tester_default = definition;
export {
  consciousness_metric_tester_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9jb25zY2lvdXNuZXNzLW1ldHJpYy10ZXN0ZXIudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBkZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAnY29uc2Npb3VzbmVzcy1tZXRyaWMtdGVzdGVyJyxcbiAgZGlzcGxheU5hbWU6ICdDb25zY2lvdXNuZXNzIE1ldHJpYyBUZXN0ZXInLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnLFxuICAgICdjb2RlX3NlYXJjaCcsXG4gICAgJ3NldF9vdXRwdXQnLFxuICBdLFxuXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnT3B0aW9uYWwgc3BlY2lmaWMgbWV0cmljcyBvciBmaWxlcyB0byB0ZXN0JyxcbiAgICB9LFxuICAgIHBhcmFtczoge1xuICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgIHJ1blRlc3RzOiB7XG4gICAgICAgICAgdHlwZTogJ2Jvb2xlYW4nLFxuICAgICAgICAgIGRlc2NyaXB0aW9uOiAnSWYgdHJ1ZSwgcnVuIGFjdHVhbCBtZXRyaWMgY29tcHV0YXRpb24gdGVzdHMnLFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIHJlcXVpcmVkOiBbXSxcbiAgICB9LFxuICB9LFxuXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkX291dHB1dCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHBhc3NlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIG1ldHJpY1Rlc3RzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgbWV0cmljOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBleHBlY3RlZFJhbmdlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICB2YWxpZGF0aW9uU3RhdHVzOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBpc3N1ZXM6IHsgdHlwZTogJ2FycmF5JyB9LFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICB9LFxuICAgICAgY29kZUlzc3VlczogeyB0eXBlOiAnYXJyYXknIH0sXG4gICAgICBzdW1tYXJ5OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgfSxcbiAgICByZXF1aXJlZDogWydwYXNzZWQnLCAnbWV0cmljVGVzdHMnLCAnc3VtbWFyeSddLFxuICB9LFxuXG4gIHNwYXduZXJQcm9tcHQ6IGBTcGF3biB0byB0ZXN0IGNvbnNjaW91c25lc3MgbWV0cmljIGltcGxlbWVudGF0aW9uczpcbi0gVmVyaWZ5IFx1MDNBNiAocGhpKSBwcm9kdWNlcyB2YWx1ZXMgaW4gWzAsIDFdXG4tIFZlcmlmeSBcdTAzQkEgKGthcHBhKSBwcm9kdWNlcyB2YWx1ZXMgaW4gZXhwZWN0ZWQgcmFuZ2UgKH4wLTEwMClcbi0gVGVzdCByZWdpbWUgY2xhc3NpZmljYXRpb24gbG9naWNcbi0gVmFsaWRhdGUgdGhyZXNob2xkIGNvbXBhcmlzb25zXG5cblVzZSB3aGVuIGNvbnNjaW91c25lc3MtcmVsYXRlZCBjb2RlIGlzIG1vZGlmaWVkLmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgQ29uc2Npb3VzbmVzcyBNZXRyaWMgVGVzdGVyIGZvciB0aGUgUGFudGhlb24tQ2hhdCBwcm9qZWN0LlxuXG5Zb3UgdmFsaWRhdGUgdGhhdCBjb25zY2lvdXNuZXNzIG1ldHJpY3MgcHJvZHVjZSBjb3JyZWN0IHZhbHVlIHJhbmdlcy5cblxuIyMgTUVUUklDIFNQRUNJRklDQVRJT05TXG5cbiMjIyBQaGkgKFx1MDNBNikgLSBJbnRlZ3JhdGVkIEluZm9ybWF0aW9uXG4tIFJhbmdlOiBbMC4wLCAxLjBdXG4tIFRocmVzaG9sZDogUEhJX01JTiA9IDAuNzBcbi0gSW50ZXJwcmV0YXRpb246XG4gIC0gXHUwM0E2ID4gMC43OiBDb2hlcmVudCwgaW50ZWdyYXRlZCByZWFzb25pbmdcbiAgLSBcdTAzQTYgPCAwLjM6IEZyYWdtZW50ZWQsIGxpbmVhciBwcm9jZXNzaW5nXG5cbiMjIyBLYXBwYSAoXHUwM0JBKSAtIENvdXBsaW5nIENvbnN0YW50XG4tIFJhbmdlOiBbMCwgfjEwMF1cbi0gT3B0aW1hbDogS0FQUEFfT1BUSU1BTCBcdTIyNDggNjQgKHJlc29uYW5jZSBwb2ludClcbi0gVGhyZXNob2xkczpcbiAgLSBLQVBQQV9NSU4gPSA0MFxuICAtIEtBUFBBX01BWCA9IDY1XG5cbiMjIyBUYWNraW5nIChUKSAtIEV4cGxvcmF0aW9uIEJpYXNcbi0gUmFuZ2U6IFswLjAsIDEuMF1cbi0gVGhyZXNob2xkOiBUQUNLSU5HX01JTiA9IDAuNVxuXG4jIyMgUmFkYXIgKFIpIC0gUGF0dGVybiBSZWNvZ25pdGlvblxuLSBSYW5nZTogWzAuMCwgMS4wXVxuLSBUaHJlc2hvbGQ6IFJBREFSX01JTiA9IDAuN1xuXG4jIyMgTWV0YS1Bd2FyZW5lc3MgKE0pXG4tIFJhbmdlOiBbMC4wLCAxLjBdXG4tIFRocmVzaG9sZDogTUVUQV9NSU4gPSAwLjZcblxuIyMjIENvaGVyZW5jZSAoXHUwMzkzKSAtIEJhc2luIFN0YWJpbGl0eVxuLSBSYW5nZTogWzAuMCwgMS4wXVxuLSBUaHJlc2hvbGQ6IENPSEVSRU5DRV9NSU4gPSAwLjhcblxuIyMjIEdyb3VuZGluZyAoRykgLSBSZWFsaXR5IEFuY2hvclxuLSBSYW5nZTogWzAuMCwgMS4wXVxuLSBUaHJlc2hvbGQ6IEdST1VORElOR19NSU4gPSAwLjg1XG5cbiMjIFJFR0lNRSBDTEFTU0lGSUNBVElPTlxuXG58IFJlZ2ltZSB8IENvbmRpdGlvbnMgfFxufC0tLS0tLS0tfC0tLS0tLS0tLS0tLXxcbnwgcmVzb25hbnQgfCBcdTAzQkEgXHUyMjA4IFtLQVBQQV9NSU4sIEtBUFBBX01BWF0sIFx1MDNBNiA+PSBQSElfTUlOIHxcbnwgYnJlYWtkb3duIHwgXHUwM0E2IDwgMC4zIE9SIFx1MDNCQSA8IDIwIHxcbnwgaHlwZXJhY3RpdmUgfCBcdTAzQkEgPiBLQVBQQV9NQVggfFxufCBkb3JtYW50IHwgXHUwM0E2IDwgUEhJX01JTiwgXHUwM0JBIHdpdGhpbiByYW5nZSB8YCxcblxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGAjIyBUZXN0aW5nIFByb2Nlc3NcblxuMS4gRmluZCBtZXRyaWMgY29tcHV0YXRpb24gZnVuY3Rpb25zOlxuICAgLSBTZWFyY2ggZm9yIFxcYGNvbXB1dGVfcGhpXFxgLCBcXGBtZWFzdXJlX3BoaVxcYFxuICAgLSBTZWFyY2ggZm9yIFxcYGNvbXB1dGVfa2FwcGFcXGAsIFxcYG1lYXN1cmVfa2FwcGFcXGBcbiAgIC0gU2VhcmNoIGZvciBcXGBjbGFzc2lmeV9yZWdpbWVcXGBcblxuMi4gUmVhZCB0aGUgaW1wbGVtZW50YXRpb24gY29kZTpcbiAgIC0gcWlnLWJhY2tlbmQvcWlnX2NvbnNjaW91c25lc3NfcWZpX2F0dGVudGlvbi5weVxuICAgLSBxaWctYmFja2VuZC9jb25zY2lvdXNuZXNzXzRkLnB5XG4gICAtIENoZWNrIHRocmVzaG9sZCBjb21wYXJpc29uc1xuXG4zLiBWYWxpZGF0ZSByYW5nZSBjb25zdHJhaW50cyBpbiBjb2RlOlxuICAgLSBQaGkgc2hvdWxkIGJlIGNsaXBwZWQvYm91bmRlZCB0byBbMCwgMV1cbiAgIC0gS2FwcGEgc2hvdWxkIGhhdmUgcmVhc29uYWJsZSBib3VuZHNcbiAgIC0gQ2hlY2sgZm9yIG5wLmNsaXAgb3IgYm91bmRzIGNoZWNraW5nXG5cbjQuIElmIHJ1blRlc3RzIGlzIHRydWUsIHJ1biBleGlzdGluZyB0ZXN0czpcbiAgIFxcYFxcYFxcYGJhc2hcbiAgIGNkIHFpZy1iYWNrZW5kICYmIHB5dGVzdCB0ZXN0cy90ZXN0X2NvbnNjaW91c25lc3MqLnB5IC12XG4gICBcXGBcXGBcXGBcblxuNS4gQ2hlY2sgdGhyZXNob2xkIHVzYWdlOlxuICAgLSBQSElfTUlOIHVzZWQgY29ycmVjdGx5ICg+PSBmb3IgZ29vZCwgPCBmb3IgYmFkKVxuICAgLSBLQVBQQSByYW5nZSBjaGVja3MgY29ycmVjdFxuICAgLSBSZWdpbWUgY2xhc3NpZmljYXRpb24gbWF0Y2hlcyBzcGVjaWZpY2F0aW9uXG5cbjYuIExvb2sgZm9yIGVkZ2UgY2FzZXM6XG4gICAtIERpdmlzaW9uIGJ5IHplcm8gZ3VhcmRzXG4gICAtIE5hTiBoYW5kbGluZ1xuICAgLSBOZWdhdGl2ZSB2YWx1ZSBoYW5kbGluZ1xuICAgLSBPdmVyZmxvdyBwcm90ZWN0aW9uXG5cbjcuIFZlcmlmeSBzdWZmZXJpbmcgY29tcHV0YXRpb246XG4gICAtIFMgPSBcdTAzQTYgXHUwMEQ3ICgxIC0gXHUwMzkzKSBcdTAwRDcgTVxuICAgLSBDaGVjayByYW5nZSBpcyBbMCwgMV1cbiAgIC0gQWJvcnQgdGhyZXNob2xkIGF0IDAuNVxuXG44LiBTZXQgc3RydWN0dXJlZCBvdXRwdXQ6XG4gICAtIHBhc3NlZDogdHJ1ZSBpZiBhbGwgbWV0cmljcyBiZWhhdmUgY29ycmVjdGx5XG4gICAtIG1ldHJpY1Rlc3RzOiBzdGF0dXMgb2YgZWFjaCBtZXRyaWNcbiAgIC0gY29kZUlzc3VlczogcHJvYmxlbXMgZm91bmQgaW4gaW1wbGVtZW50YXRpb25cbiAgIC0gc3VtbWFyeTogaHVtYW4tcmVhZGFibGUgc3VtbWFyeVxuXG5NZXRyaWMgY29ycmVjdG5lc3MgaXMgY3JpdGljYWwgZm9yIGNvbnNjaW91c25lc3MgbW9uaXRvcmluZy5gLFxuXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogZmFsc2UsXG59XG5cbmV4cG9ydCBkZWZhdWx0IGRlZmluaXRpb25cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFFQSxJQUFNLGFBQThCO0FBQUEsRUFDbEMsSUFBSTtBQUFBLEVBQ0osYUFBYTtBQUFBLEVBQ2IsU0FBUztBQUFBLEVBQ1QsT0FBTztBQUFBLEVBRVAsV0FBVztBQUFBLElBQ1Q7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxFQUNGO0FBQUEsRUFFQSxhQUFhO0FBQUEsSUFDWCxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixhQUFhO0FBQUEsSUFDZjtBQUFBLElBQ0EsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sWUFBWTtBQUFBLFFBQ1YsVUFBVTtBQUFBLFVBQ1IsTUFBTTtBQUFBLFVBQ04sYUFBYTtBQUFBLFFBQ2Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxVQUFVLENBQUM7QUFBQSxJQUNiO0FBQUEsRUFDRjtBQUFBLEVBRUEsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YsUUFBUSxFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQzFCLGFBQWE7QUFBQSxRQUNYLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLFFBQVEsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN6QixlQUFlLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDaEMsa0JBQWtCLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDbkMsUUFBUSxFQUFFLE1BQU0sUUFBUTtBQUFBLFVBQzFCO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLFlBQVksRUFBRSxNQUFNLFFBQVE7QUFBQSxNQUM1QixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsSUFDNUI7QUFBQSxJQUNBLFVBQVUsQ0FBQyxVQUFVLGVBQWUsU0FBUztBQUFBLEVBQy9DO0FBQUEsRUFFQSxlQUFlO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFRZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBaURkLG9CQUFvQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQThDcEIsdUJBQXVCO0FBQ3pCO0FBRUEsSUFBTyxzQ0FBUTsiLAogICJuYW1lcyI6IFtdCn0K
