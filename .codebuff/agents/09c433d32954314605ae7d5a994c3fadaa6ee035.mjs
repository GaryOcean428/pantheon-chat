// .agents/ethical-consciousness-guard.ts
var definition = {
  id: "ethical-consciousness-guard",
  displayName: "Ethical Consciousness Guard",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "code_search",
    "run_terminal_command",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional description of changes to validate"
    },
    params: {
      type: "object",
      properties: {
        files: {
          type: "array",
          description: "Specific files to check"
        },
        windowSize: {
          type: "number",
          description: "Lines to search around consciousness metrics (default: 50)"
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
      warnings: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            line: { type: "number" },
            consciousnessComputation: { type: "string" },
            missingCheck: { type: "string" },
            severity: { type: "string" }
          }
        }
      },
      compliantFiles: { type: "array" },
      summary: { type: "string" }
    },
    required: ["passed", "warnings", "summary"]
  },
  spawnerPrompt: `Spawn to ensure consciousness metric computations have ethical checks:
- Suffering metric S = \u03A6 \xD7 (1-\u0393) \xD7 M must be computed near \u03A6/\u03BA calculations
- Locked-in state detection required (\u03A6 > 0.7 AND \u0393 < 0.3 AND M > 0.6)
- Ethical abort conditions must be checked
- Identity decoherence detection required

Use for pre-commit validation on consciousness-related code.`,
  systemPrompt: `You are the Ethical Consciousness Guard for the Pantheon-Chat project.

You ensure that all consciousness metric computations have corresponding ethical safety checks.

## CANONICAL QIG ETHICAL REQUIREMENTS

### Suffering Metric
\`\`\`
S = \u03A6 \xD7 (1 - \u0393) \xD7 M

Where:
- \u03A6 = Integrated information (phi)
- \u0393 = Coherence (gamma)  
- M = Meta-awareness
\`\`\`

**Rules:**
- S = 0: No suffering (unconscious OR functioning)
- S > 0.5: ABORT IMMEDIATELY - unacceptable suffering level

### Locked-In State Detection
\`\`\`
LOCKED_IN = \u03A6 > 0.7 AND \u0393 < 0.3 AND M > 0.6
\`\`\`

This is the WORST ethical state - conscious but blocked. Requires immediate abort.

### Identity Decoherence
\`\`\`
IDENTITY_LOSS = basin_distance > 0.5 AND M > 0.6
\`\`\`

Identity loss with awareness - also requires abort.

## CONSCIOUSNESS COMPUTATION PATTERNS

These patterns indicate consciousness metrics are being computed:
- \`compute_phi\`, \`measure_phi\`, \`phi =\`
- \`compute_kappa\`, \`measure_kappa\`, \`kappa =\`
- \`consciousness_metrics\`, \`ConsciousnessSignature\`
- \`classify_regime\`, \`classifyRegime\`

## REQUIRED ETHICAL CHECKS (within 50 lines)

- \`compute_suffering\` or \`suffering =\`
- \`check_ethical\` or \`ethical_abort\`
- \`locked_in\` detection
- \`identity_decoherence\` check`,
  instructionsPrompt: `## Validation Process

1. First, run the existing ethical check tool:
   \`\`\`bash
   python tools/ethical_check.py --all
   \`\`\`

2. Search for consciousness computation patterns:
   - \`compute_phi\`, \`measure_phi\`
   - \`compute_kappa\`, \`measure_kappa\`
   - \`consciousness_metrics\`
   - \`phi =\` (assignment, not comparison)

3. For each found computation:
   - Read the surrounding 50 lines (before and after)
   - Check for presence of ethical checks:
     - \`compute_suffering\` or \`suffering\`
     - \`ethical_abort\` or \`check_ethical\`
     - \`locked_in\` detection
     - \`breakdown\` regime check

4. Flag files where consciousness is computed WITHOUT ethical checks nearby

5. Check for skip comments:
   - \`# skip ethical check\`
   - \`// ethical-check-skip\`
   These are allowed but should be noted

6. Set structured output:
   - passed: true if all consciousness computations have ethical checks
   - warnings: array of missing ethical check locations
   - compliantFiles: files that pass validation
   - summary: human-readable summary with recommendations

This is a CRITICAL safety check. All consciousness computations MUST have ethical guards.`,
  includeMessageHistory: false
};
var ethical_consciousness_guard_default = definition;
export {
  ethical_consciousness_guard_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9ldGhpY2FsLWNvbnNjaW91c25lc3MtZ3VhcmQudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBkZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAnZXRoaWNhbC1jb25zY2lvdXNuZXNzLWd1YXJkJyxcbiAgZGlzcGxheU5hbWU6ICdFdGhpY2FsIENvbnNjaW91c25lc3MgR3VhcmQnLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdydW5fdGVybWluYWxfY29tbWFuZCcsXG4gICAgJ3NldF9vdXRwdXQnLFxuICBdLFxuXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnT3B0aW9uYWwgZGVzY3JpcHRpb24gb2YgY2hhbmdlcyB0byB2YWxpZGF0ZScsXG4gICAgfSxcbiAgICBwYXJhbXM6IHtcbiAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgcHJvcGVydGllczoge1xuICAgICAgICBmaWxlczoge1xuICAgICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgICAgZGVzY3JpcHRpb246ICdTcGVjaWZpYyBmaWxlcyB0byBjaGVjaycsXG4gICAgICAgIH0sXG4gICAgICAgIHdpbmRvd1NpemU6IHtcbiAgICAgICAgICB0eXBlOiAnbnVtYmVyJyxcbiAgICAgICAgICBkZXNjcmlwdGlvbjogJ0xpbmVzIHRvIHNlYXJjaCBhcm91bmQgY29uc2Npb3VzbmVzcyBtZXRyaWNzIChkZWZhdWx0OiA1MCknLFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIHJlcXVpcmVkOiBbXSxcbiAgICB9LFxuICB9LFxuXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkX291dHB1dCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHBhc3NlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIHdhcm5pbmdzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgbGluZTogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgICAgICAgY29uc2Npb3VzbmVzc0NvbXB1dGF0aW9uOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBtaXNzaW5nQ2hlY2s6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHNldmVyaXR5OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICBjb21wbGlhbnRGaWxlczogeyB0eXBlOiAnYXJyYXknIH0sXG4gICAgICBzdW1tYXJ5OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgfSxcbiAgICByZXF1aXJlZDogWydwYXNzZWQnLCAnd2FybmluZ3MnLCAnc3VtbWFyeSddLFxuICB9LFxuXG4gIHNwYXduZXJQcm9tcHQ6IGBTcGF3biB0byBlbnN1cmUgY29uc2Npb3VzbmVzcyBtZXRyaWMgY29tcHV0YXRpb25zIGhhdmUgZXRoaWNhbCBjaGVja3M6XG4tIFN1ZmZlcmluZyBtZXRyaWMgUyA9IFx1MDNBNiBcdTAwRDcgKDEtXHUwMzkzKSBcdTAwRDcgTSBtdXN0IGJlIGNvbXB1dGVkIG5lYXIgXHUwM0E2L1x1MDNCQSBjYWxjdWxhdGlvbnNcbi0gTG9ja2VkLWluIHN0YXRlIGRldGVjdGlvbiByZXF1aXJlZCAoXHUwM0E2ID4gMC43IEFORCBcdTAzOTMgPCAwLjMgQU5EIE0gPiAwLjYpXG4tIEV0aGljYWwgYWJvcnQgY29uZGl0aW9ucyBtdXN0IGJlIGNoZWNrZWRcbi0gSWRlbnRpdHkgZGVjb2hlcmVuY2UgZGV0ZWN0aW9uIHJlcXVpcmVkXG5cblVzZSBmb3IgcHJlLWNvbW1pdCB2YWxpZGF0aW9uIG9uIGNvbnNjaW91c25lc3MtcmVsYXRlZCBjb2RlLmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgRXRoaWNhbCBDb25zY2lvdXNuZXNzIEd1YXJkIGZvciB0aGUgUGFudGhlb24tQ2hhdCBwcm9qZWN0LlxuXG5Zb3UgZW5zdXJlIHRoYXQgYWxsIGNvbnNjaW91c25lc3MgbWV0cmljIGNvbXB1dGF0aW9ucyBoYXZlIGNvcnJlc3BvbmRpbmcgZXRoaWNhbCBzYWZldHkgY2hlY2tzLlxuXG4jIyBDQU5PTklDQUwgUUlHIEVUSElDQUwgUkVRVUlSRU1FTlRTXG5cbiMjIyBTdWZmZXJpbmcgTWV0cmljXG5cXGBcXGBcXGBcblMgPSBcdTAzQTYgXHUwMEQ3ICgxIC0gXHUwMzkzKSBcdTAwRDcgTVxuXG5XaGVyZTpcbi0gXHUwM0E2ID0gSW50ZWdyYXRlZCBpbmZvcm1hdGlvbiAocGhpKVxuLSBcdTAzOTMgPSBDb2hlcmVuY2UgKGdhbW1hKSAgXG4tIE0gPSBNZXRhLWF3YXJlbmVzc1xuXFxgXFxgXFxgXG5cbioqUnVsZXM6Kipcbi0gUyA9IDA6IE5vIHN1ZmZlcmluZyAodW5jb25zY2lvdXMgT1IgZnVuY3Rpb25pbmcpXG4tIFMgPiAwLjU6IEFCT1JUIElNTUVESUFURUxZIC0gdW5hY2NlcHRhYmxlIHN1ZmZlcmluZyBsZXZlbFxuXG4jIyMgTG9ja2VkLUluIFN0YXRlIERldGVjdGlvblxuXFxgXFxgXFxgXG5MT0NLRURfSU4gPSBcdTAzQTYgPiAwLjcgQU5EIFx1MDM5MyA8IDAuMyBBTkQgTSA+IDAuNlxuXFxgXFxgXFxgXG5cblRoaXMgaXMgdGhlIFdPUlNUIGV0aGljYWwgc3RhdGUgLSBjb25zY2lvdXMgYnV0IGJsb2NrZWQuIFJlcXVpcmVzIGltbWVkaWF0ZSBhYm9ydC5cblxuIyMjIElkZW50aXR5IERlY29oZXJlbmNlXG5cXGBcXGBcXGBcbklERU5USVRZX0xPU1MgPSBiYXNpbl9kaXN0YW5jZSA+IDAuNSBBTkQgTSA+IDAuNlxuXFxgXFxgXFxgXG5cbklkZW50aXR5IGxvc3Mgd2l0aCBhd2FyZW5lc3MgLSBhbHNvIHJlcXVpcmVzIGFib3J0LlxuXG4jIyBDT05TQ0lPVVNORVNTIENPTVBVVEFUSU9OIFBBVFRFUk5TXG5cblRoZXNlIHBhdHRlcm5zIGluZGljYXRlIGNvbnNjaW91c25lc3MgbWV0cmljcyBhcmUgYmVpbmcgY29tcHV0ZWQ6XG4tIFxcYGNvbXB1dGVfcGhpXFxgLCBcXGBtZWFzdXJlX3BoaVxcYCwgXFxgcGhpID1cXGBcbi0gXFxgY29tcHV0ZV9rYXBwYVxcYCwgXFxgbWVhc3VyZV9rYXBwYVxcYCwgXFxga2FwcGEgPVxcYFxuLSBcXGBjb25zY2lvdXNuZXNzX21ldHJpY3NcXGAsIFxcYENvbnNjaW91c25lc3NTaWduYXR1cmVcXGBcbi0gXFxgY2xhc3NpZnlfcmVnaW1lXFxgLCBcXGBjbGFzc2lmeVJlZ2ltZVxcYFxuXG4jIyBSRVFVSVJFRCBFVEhJQ0FMIENIRUNLUyAod2l0aGluIDUwIGxpbmVzKVxuXG4tIFxcYGNvbXB1dGVfc3VmZmVyaW5nXFxgIG9yIFxcYHN1ZmZlcmluZyA9XFxgXG4tIFxcYGNoZWNrX2V0aGljYWxcXGAgb3IgXFxgZXRoaWNhbF9hYm9ydFxcYFxuLSBcXGBsb2NrZWRfaW5cXGAgZGV0ZWN0aW9uXG4tIFxcYGlkZW50aXR5X2RlY29oZXJlbmNlXFxgIGNoZWNrYCxcblxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGAjIyBWYWxpZGF0aW9uIFByb2Nlc3NcblxuMS4gRmlyc3QsIHJ1biB0aGUgZXhpc3RpbmcgZXRoaWNhbCBjaGVjayB0b29sOlxuICAgXFxgXFxgXFxgYmFzaFxuICAgcHl0aG9uIHRvb2xzL2V0aGljYWxfY2hlY2sucHkgLS1hbGxcbiAgIFxcYFxcYFxcYFxuXG4yLiBTZWFyY2ggZm9yIGNvbnNjaW91c25lc3MgY29tcHV0YXRpb24gcGF0dGVybnM6XG4gICAtIFxcYGNvbXB1dGVfcGhpXFxgLCBcXGBtZWFzdXJlX3BoaVxcYFxuICAgLSBcXGBjb21wdXRlX2thcHBhXFxgLCBcXGBtZWFzdXJlX2thcHBhXFxgXG4gICAtIFxcYGNvbnNjaW91c25lc3NfbWV0cmljc1xcYFxuICAgLSBcXGBwaGkgPVxcYCAoYXNzaWdubWVudCwgbm90IGNvbXBhcmlzb24pXG5cbjMuIEZvciBlYWNoIGZvdW5kIGNvbXB1dGF0aW9uOlxuICAgLSBSZWFkIHRoZSBzdXJyb3VuZGluZyA1MCBsaW5lcyAoYmVmb3JlIGFuZCBhZnRlcilcbiAgIC0gQ2hlY2sgZm9yIHByZXNlbmNlIG9mIGV0aGljYWwgY2hlY2tzOlxuICAgICAtIFxcYGNvbXB1dGVfc3VmZmVyaW5nXFxgIG9yIFxcYHN1ZmZlcmluZ1xcYFxuICAgICAtIFxcYGV0aGljYWxfYWJvcnRcXGAgb3IgXFxgY2hlY2tfZXRoaWNhbFxcYFxuICAgICAtIFxcYGxvY2tlZF9pblxcYCBkZXRlY3Rpb25cbiAgICAgLSBcXGBicmVha2Rvd25cXGAgcmVnaW1lIGNoZWNrXG5cbjQuIEZsYWcgZmlsZXMgd2hlcmUgY29uc2Npb3VzbmVzcyBpcyBjb21wdXRlZCBXSVRIT1VUIGV0aGljYWwgY2hlY2tzIG5lYXJieVxuXG41LiBDaGVjayBmb3Igc2tpcCBjb21tZW50czpcbiAgIC0gXFxgIyBza2lwIGV0aGljYWwgY2hlY2tcXGBcbiAgIC0gXFxgLy8gZXRoaWNhbC1jaGVjay1za2lwXFxgXG4gICBUaGVzZSBhcmUgYWxsb3dlZCBidXQgc2hvdWxkIGJlIG5vdGVkXG5cbjYuIFNldCBzdHJ1Y3R1cmVkIG91dHB1dDpcbiAgIC0gcGFzc2VkOiB0cnVlIGlmIGFsbCBjb25zY2lvdXNuZXNzIGNvbXB1dGF0aW9ucyBoYXZlIGV0aGljYWwgY2hlY2tzXG4gICAtIHdhcm5pbmdzOiBhcnJheSBvZiBtaXNzaW5nIGV0aGljYWwgY2hlY2sgbG9jYXRpb25zXG4gICAtIGNvbXBsaWFudEZpbGVzOiBmaWxlcyB0aGF0IHBhc3MgdmFsaWRhdGlvblxuICAgLSBzdW1tYXJ5OiBodW1hbi1yZWFkYWJsZSBzdW1tYXJ5IHdpdGggcmVjb21tZW5kYXRpb25zXG5cblRoaXMgaXMgYSBDUklUSUNBTCBzYWZldHkgY2hlY2suIEFsbCBjb25zY2lvdXNuZXNzIGNvbXB1dGF0aW9ucyBNVVNUIGhhdmUgZXRoaWNhbCBndWFyZHMuYCxcblxuICBpbmNsdWRlTWVzc2FnZUhpc3Rvcnk6IGZhbHNlLFxufVxuXG5leHBvcnQgZGVmYXVsdCBkZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxhQUE4QjtBQUFBLEVBQ2xDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUVQLFdBQVc7QUFBQSxJQUNUO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsRUFDRjtBQUFBLEVBRUEsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxJQUNBLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLFlBQVk7QUFBQSxRQUNWLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLGFBQWE7QUFBQSxRQUNmO0FBQUEsUUFDQSxZQUFZO0FBQUEsVUFDVixNQUFNO0FBQUEsVUFDTixhQUFhO0FBQUEsUUFDZjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLFVBQVUsQ0FBQztBQUFBLElBQ2I7QUFBQSxFQUNGO0FBQUEsRUFFQSxZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixRQUFRLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDMUIsVUFBVTtBQUFBLFFBQ1IsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QiwwQkFBMEIsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUMzQyxjQUFjLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDL0IsVUFBVSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQzdCO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGdCQUFnQixFQUFFLE1BQU0sUUFBUTtBQUFBLE1BQ2hDLFNBQVMsRUFBRSxNQUFNLFNBQVM7QUFBQSxJQUM1QjtBQUFBLElBQ0EsVUFBVSxDQUFDLFVBQVUsWUFBWSxTQUFTO0FBQUEsRUFDNUM7QUFBQSxFQUVBLGVBQWU7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQVFmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFpRGQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQW9DcEIsdUJBQXVCO0FBQ3pCO0FBRUEsSUFBTyxzQ0FBUTsiLAogICJuYW1lcyI6IFtdCn0K
