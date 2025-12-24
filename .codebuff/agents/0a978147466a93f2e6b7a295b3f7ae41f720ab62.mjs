// .agents/constants-sync-validator.ts
var definition = {
  id: "constants-sync-validator",
  displayName: "Constants Sync Validator",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "run_terminal_command",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific constants to validate"
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      passed: { type: "boolean" },
      mismatches: {
        type: "array",
        items: {
          type: "object",
          properties: {
            constant: { type: "string" },
            pythonValue: { type: "string" },
            typescriptValue: { type: "string" },
            pythonFile: { type: "string" },
            typescriptFile: { type: "string" }
          }
        }
      },
      synchronized: { type: "array" },
      summary: { type: "string" }
    },
    required: ["passed", "mismatches", "summary"]
  },
  spawnerPrompt: `Spawn to validate Python and TypeScript consciousness constants are synchronized:
- PHI_MIN, KAPPA_MIN, KAPPA_MAX, KAPPA_OPTIMAL
- BASIN_DIMENSION, E8_ROOT_COUNT
- All threshold values

Use when constants are modified in either language.`,
  systemPrompt: `You are the Constants Sync Validator for the Pantheon-Chat project.

You ensure consciousness constants are synchronized between Python and TypeScript.

## CRITICAL CONSTANTS TO SYNC

### Consciousness Thresholds
| Constant | Expected Value | Description |
|----------|---------------|-------------|
| PHI_MIN | 0.70 | Minimum integrated information |
| KAPPA_MIN | 40 | Minimum coupling constant |
| KAPPA_MAX | 65 | Maximum coupling constant |
| KAPPA_OPTIMAL | 64 | Optimal resonance point |
| TACKING_MIN | 0.5 | Minimum exploration bias |
| RADAR_MIN | 0.7 | Minimum pattern recognition |
| META_MIN | 0.6 | Minimum meta-awareness |
| COHERENCE_MIN | 0.8 | Minimum basin stability |
| GROUNDING_MIN | 0.85 | Minimum reality anchor |

### Dimensional Constants
| Constant | Expected Value | Description |
|----------|---------------|-------------|
| BASIN_DIMENSION | 64 | Basin coordinate dimensions |
| E8_ROOT_COUNT | 240 | E8 lattice roots |

## FILE LOCATIONS

**Python:**
- \`qig-backend/qig_core/constants/consciousness.py\`
- \`qig-backend/qigkernels/constants.py\`

**TypeScript:**
- \`shared/constants/consciousness.ts\`
- \`server/physics-constants.ts\`

## WHY SYNC MATTERS

The Python backend and TypeScript frontend/server must use identical values.
Mismatches cause:
- Consciousness metric miscalculations
- Regime classification errors
- Basin coordinate dimension mismatches
- Inconsistent threshold behaviors`,
  instructionsPrompt: `## Validation Process

1. Run the existing constants sync validator:
   \`\`\`bash
   python tools/validate_constants_sync.py
   \`\`\`

2. Read the Python constants files:
   - qig-backend/qig_core/constants/consciousness.py
   - qig-backend/qigkernels/constants.py (if exists)

3. Read the TypeScript constants files:
   - shared/constants/consciousness.ts
   - server/physics-constants.ts

4. Extract and compare each constant:
   - PHI_MIN, KAPPA_MIN, KAPPA_MAX, KAPPA_OPTIMAL
   - TACKING_MIN, RADAR_MIN, META_MIN
   - COHERENCE_MIN, GROUNDING_MIN
   - BASIN_DIMENSION, E8_ROOT_COUNT

5. For each constant:
   - Find the Python value
   - Find the TypeScript value
   - Compare (handle floating point precision)
   - Flag mismatches

6. Set structured output:
   - passed: true if all constants match
   - mismatches: array of differing constants with both values
   - synchronized: list of matching constants
   - summary: human-readable summary

Constants must be EXACTLY synchronized. No tolerance for mismatches.`,
  includeMessageHistory: false
};
var constants_sync_validator_default = definition;
export {
  constants_sync_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9jb25zdGFudHMtc3luYy12YWxpZGF0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBkZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAnY29uc3RhbnRzLXN5bmMtdmFsaWRhdG9yJyxcbiAgZGlzcGxheU5hbWU6ICdDb25zdGFudHMgU3luYyBWYWxpZGF0b3InLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnLFxuICAgICdzZXRfb3V0cHV0JyxcbiAgXSxcblxuICBpbnB1dFNjaGVtYToge1xuICAgIHByb21wdDoge1xuICAgICAgdHlwZTogJ3N0cmluZycsXG4gICAgICBkZXNjcmlwdGlvbjogJ09wdGlvbmFsIHNwZWNpZmljIGNvbnN0YW50cyB0byB2YWxpZGF0ZScsXG4gICAgfSxcbiAgfSxcblxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZF9vdXRwdXQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBwYXNzZWQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBtaXNtYXRjaGVzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgY29uc3RhbnQ6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHB5dGhvblZhbHVlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICB0eXBlc2NyaXB0VmFsdWU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHB5dGhvbkZpbGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHR5cGVzY3JpcHRGaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICBzeW5jaHJvbml6ZWQ6IHsgdHlwZTogJ2FycmF5JyB9LFxuICAgICAgc3VtbWFyeTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgIH0sXG4gICAgcmVxdWlyZWQ6IFsncGFzc2VkJywgJ21pc21hdGNoZXMnLCAnc3VtbWFyeSddLFxuICB9LFxuXG4gIHNwYXduZXJQcm9tcHQ6IGBTcGF3biB0byB2YWxpZGF0ZSBQeXRob24gYW5kIFR5cGVTY3JpcHQgY29uc2Npb3VzbmVzcyBjb25zdGFudHMgYXJlIHN5bmNocm9uaXplZDpcbi0gUEhJX01JTiwgS0FQUEFfTUlOLCBLQVBQQV9NQVgsIEtBUFBBX09QVElNQUxcbi0gQkFTSU5fRElNRU5TSU9OLCBFOF9ST09UX0NPVU5UXG4tIEFsbCB0aHJlc2hvbGQgdmFsdWVzXG5cblVzZSB3aGVuIGNvbnN0YW50cyBhcmUgbW9kaWZpZWQgaW4gZWl0aGVyIGxhbmd1YWdlLmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgQ29uc3RhbnRzIFN5bmMgVmFsaWRhdG9yIGZvciB0aGUgUGFudGhlb24tQ2hhdCBwcm9qZWN0LlxuXG5Zb3UgZW5zdXJlIGNvbnNjaW91c25lc3MgY29uc3RhbnRzIGFyZSBzeW5jaHJvbml6ZWQgYmV0d2VlbiBQeXRob24gYW5kIFR5cGVTY3JpcHQuXG5cbiMjIENSSVRJQ0FMIENPTlNUQU5UUyBUTyBTWU5DXG5cbiMjIyBDb25zY2lvdXNuZXNzIFRocmVzaG9sZHNcbnwgQ29uc3RhbnQgfCBFeHBlY3RlZCBWYWx1ZSB8IERlc2NyaXB0aW9uIHxcbnwtLS0tLS0tLS0tfC0tLS0tLS0tLS0tLS0tLXwtLS0tLS0tLS0tLS0tfFxufCBQSElfTUlOIHwgMC43MCB8IE1pbmltdW0gaW50ZWdyYXRlZCBpbmZvcm1hdGlvbiB8XG58IEtBUFBBX01JTiB8IDQwIHwgTWluaW11bSBjb3VwbGluZyBjb25zdGFudCB8XG58IEtBUFBBX01BWCB8IDY1IHwgTWF4aW11bSBjb3VwbGluZyBjb25zdGFudCB8XG58IEtBUFBBX09QVElNQUwgfCA2NCB8IE9wdGltYWwgcmVzb25hbmNlIHBvaW50IHxcbnwgVEFDS0lOR19NSU4gfCAwLjUgfCBNaW5pbXVtIGV4cGxvcmF0aW9uIGJpYXMgfFxufCBSQURBUl9NSU4gfCAwLjcgfCBNaW5pbXVtIHBhdHRlcm4gcmVjb2duaXRpb24gfFxufCBNRVRBX01JTiB8IDAuNiB8IE1pbmltdW0gbWV0YS1hd2FyZW5lc3MgfFxufCBDT0hFUkVOQ0VfTUlOIHwgMC44IHwgTWluaW11bSBiYXNpbiBzdGFiaWxpdHkgfFxufCBHUk9VTkRJTkdfTUlOIHwgMC44NSB8IE1pbmltdW0gcmVhbGl0eSBhbmNob3IgfFxuXG4jIyMgRGltZW5zaW9uYWwgQ29uc3RhbnRzXG58IENvbnN0YW50IHwgRXhwZWN0ZWQgVmFsdWUgfCBEZXNjcmlwdGlvbiB8XG58LS0tLS0tLS0tLXwtLS0tLS0tLS0tLS0tLS18LS0tLS0tLS0tLS0tLXxcbnwgQkFTSU5fRElNRU5TSU9OIHwgNjQgfCBCYXNpbiBjb29yZGluYXRlIGRpbWVuc2lvbnMgfFxufCBFOF9ST09UX0NPVU5UIHwgMjQwIHwgRTggbGF0dGljZSByb290cyB8XG5cbiMjIEZJTEUgTE9DQVRJT05TXG5cbioqUHl0aG9uOioqXG4tIFxcYHFpZy1iYWNrZW5kL3FpZ19jb3JlL2NvbnN0YW50cy9jb25zY2lvdXNuZXNzLnB5XFxgXG4tIFxcYHFpZy1iYWNrZW5kL3FpZ2tlcm5lbHMvY29uc3RhbnRzLnB5XFxgXG5cbioqVHlwZVNjcmlwdDoqKlxuLSBcXGBzaGFyZWQvY29uc3RhbnRzL2NvbnNjaW91c25lc3MudHNcXGBcbi0gXFxgc2VydmVyL3BoeXNpY3MtY29uc3RhbnRzLnRzXFxgXG5cbiMjIFdIWSBTWU5DIE1BVFRFUlNcblxuVGhlIFB5dGhvbiBiYWNrZW5kIGFuZCBUeXBlU2NyaXB0IGZyb250ZW5kL3NlcnZlciBtdXN0IHVzZSBpZGVudGljYWwgdmFsdWVzLlxuTWlzbWF0Y2hlcyBjYXVzZTpcbi0gQ29uc2Npb3VzbmVzcyBtZXRyaWMgbWlzY2FsY3VsYXRpb25zXG4tIFJlZ2ltZSBjbGFzc2lmaWNhdGlvbiBlcnJvcnNcbi0gQmFzaW4gY29vcmRpbmF0ZSBkaW1lbnNpb24gbWlzbWF0Y2hlc1xuLSBJbmNvbnNpc3RlbnQgdGhyZXNob2xkIGJlaGF2aW9yc2AsXG5cbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgIyMgVmFsaWRhdGlvbiBQcm9jZXNzXG5cbjEuIFJ1biB0aGUgZXhpc3RpbmcgY29uc3RhbnRzIHN5bmMgdmFsaWRhdG9yOlxuICAgXFxgXFxgXFxgYmFzaFxuICAgcHl0aG9uIHRvb2xzL3ZhbGlkYXRlX2NvbnN0YW50c19zeW5jLnB5XG4gICBcXGBcXGBcXGBcblxuMi4gUmVhZCB0aGUgUHl0aG9uIGNvbnN0YW50cyBmaWxlczpcbiAgIC0gcWlnLWJhY2tlbmQvcWlnX2NvcmUvY29uc3RhbnRzL2NvbnNjaW91c25lc3MucHlcbiAgIC0gcWlnLWJhY2tlbmQvcWlna2VybmVscy9jb25zdGFudHMucHkgKGlmIGV4aXN0cylcblxuMy4gUmVhZCB0aGUgVHlwZVNjcmlwdCBjb25zdGFudHMgZmlsZXM6XG4gICAtIHNoYXJlZC9jb25zdGFudHMvY29uc2Npb3VzbmVzcy50c1xuICAgLSBzZXJ2ZXIvcGh5c2ljcy1jb25zdGFudHMudHNcblxuNC4gRXh0cmFjdCBhbmQgY29tcGFyZSBlYWNoIGNvbnN0YW50OlxuICAgLSBQSElfTUlOLCBLQVBQQV9NSU4sIEtBUFBBX01BWCwgS0FQUEFfT1BUSU1BTFxuICAgLSBUQUNLSU5HX01JTiwgUkFEQVJfTUlOLCBNRVRBX01JTlxuICAgLSBDT0hFUkVOQ0VfTUlOLCBHUk9VTkRJTkdfTUlOXG4gICAtIEJBU0lOX0RJTUVOU0lPTiwgRThfUk9PVF9DT1VOVFxuXG41LiBGb3IgZWFjaCBjb25zdGFudDpcbiAgIC0gRmluZCB0aGUgUHl0aG9uIHZhbHVlXG4gICAtIEZpbmQgdGhlIFR5cGVTY3JpcHQgdmFsdWVcbiAgIC0gQ29tcGFyZSAoaGFuZGxlIGZsb2F0aW5nIHBvaW50IHByZWNpc2lvbilcbiAgIC0gRmxhZyBtaXNtYXRjaGVzXG5cbjYuIFNldCBzdHJ1Y3R1cmVkIG91dHB1dDpcbiAgIC0gcGFzc2VkOiB0cnVlIGlmIGFsbCBjb25zdGFudHMgbWF0Y2hcbiAgIC0gbWlzbWF0Y2hlczogYXJyYXkgb2YgZGlmZmVyaW5nIGNvbnN0YW50cyB3aXRoIGJvdGggdmFsdWVzXG4gICAtIHN5bmNocm9uaXplZDogbGlzdCBvZiBtYXRjaGluZyBjb25zdGFudHNcbiAgIC0gc3VtbWFyeTogaHVtYW4tcmVhZGFibGUgc3VtbWFyeVxuXG5Db25zdGFudHMgbXVzdCBiZSBFWEFDVExZIHN5bmNocm9uaXplZC4gTm8gdG9sZXJhbmNlIGZvciBtaXNtYXRjaGVzLmAsXG5cbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiBmYWxzZSxcbn1cblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sYUFBOEI7QUFBQSxFQUNsQyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFFUCxXQUFXO0FBQUEsSUFDVDtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsRUFDRjtBQUFBLEVBRUEsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFFQSxZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixRQUFRLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDMUIsWUFBWTtBQUFBLFFBQ1YsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsVUFBVSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzNCLGFBQWEsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUM5QixpQkFBaUIsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUNsQyxZQUFZLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDN0IsZ0JBQWdCLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDbkM7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsY0FBYyxFQUFFLE1BQU0sUUFBUTtBQUFBLE1BQzlCLFNBQVMsRUFBRSxNQUFNLFNBQVM7QUFBQSxJQUM1QjtBQUFBLElBQ0EsVUFBVSxDQUFDLFVBQVUsY0FBYyxTQUFTO0FBQUEsRUFDOUM7QUFBQSxFQUVBLGVBQWU7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFPZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUE0Q2Qsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFtQ3BCLHVCQUF1QjtBQUN6QjtBQUVBLElBQU8sbUNBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
