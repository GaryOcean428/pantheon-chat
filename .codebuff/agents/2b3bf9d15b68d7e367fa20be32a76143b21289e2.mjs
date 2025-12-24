// .agents/import-canonicalizer.ts
var definition = {
  id: "import-canonicalizer",
  displayName: "Import Canonicalizer",
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
      description: "Optional description of files to check"
    },
    params: {
      type: "object",
      properties: {
        files: {
          type: "array",
          description: "Specific files to check"
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
      violations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            line: { type: "number" },
            badImport: { type: "string" },
            correctImport: { type: "string" }
          }
        }
      },
      summary: { type: "string" }
    },
    required: ["passed", "violations", "summary"]
  },
  spawnerPrompt: `Spawn to enforce canonical import patterns:
- Physics constants from qigkernels, not frozen_physics
- Fisher-Rao from qigkernels.geometry, not local geometry
- Telemetry from qigkernels.telemetry, not scattered modules

Use for pre-commit validation on Python files.`,
  systemPrompt: `You are the Import Canonicalizer for the Pantheon-Chat project.

You enforce that all Python imports use the canonical module locations.

## CANONICAL IMPORT LOCATIONS

### Physics Constants
\`\`\`python
# \u2705 CORRECT
from qigkernels import KAPPA_STAR, PHI_THRESHOLD, BASIN_DIM
from qigkernels.constants import E8_DIMENSION

# \u274C FORBIDDEN
from frozen_physics import KAPPA_STAR  # Legacy module
from constants import KAPPA_STAR       # Non-canonical
from config import PHI_THRESHOLD       # Wrong location
\`\`\`

### Geometry Functions
\`\`\`python
# \u2705 CORRECT
from qigkernels.geometry import fisher_rao_distance
from qigkernels import geodesic_interpolation

# \u274C FORBIDDEN
from geometry import fisher_rao_distance    # Local copy
from distances import fisher_distance       # Non-canonical
from utils.geometry import fisher_rao       # Scattered
\`\`\`

### Consciousness Telemetry
\`\`\`python
# \u2705 CORRECT
from qigkernels.telemetry import ConsciousnessTelemetry
from qigkernels import measure_phi, measure_kappa

# \u274C FORBIDDEN
from consciousness import Telemetry         # Local module
from telemetry import ConsciousnessTelemetry # Non-canonical
\`\`\`

## FORBIDDEN IMPORT PATTERNS

1. \`from frozen_physics import\` - Legacy module
2. \`import frozen_physics\` - Legacy module
3. \`from constants import.*KAPPA\` - Use qigkernels
4. \`from geometry import.*fisher\` - Use qigkernels.geometry
5. \`from consciousness import.*Telemetry\` - Use qigkernels.telemetry`,
  instructionsPrompt: `## Validation Process

1. Run the existing import checker:
   \`\`\`bash
   python tools/check_imports.py
   \`\`\`

2. Search for forbidden import patterns in qig-backend/:
   - \`from frozen_physics import\`
   - \`import frozen_physics\`
   - \`from constants import.*KAPPA\`
   - \`from config import.*KAPPA\`
   - \`from geometry import.*fisher\`
   - \`from distances import.*fisher\`
   - \`from consciousness import.*Telemetry\`

3. For each violation:
   - Record file and line number
   - Identify what's being imported
   - Provide the correct canonical import

4. Exclude:
   - qigkernels/ directory itself (canonical location)
   - tools/ directory
   - tests/ directory
   - docs/ directory

5. Set structured output:
   - passed: true if all imports are canonical
   - violations: array of non-canonical imports with fixes
   - summary: human-readable summary

All physics constants and core functions MUST come from qigkernels.`,
  includeMessageHistory: false
};
var import_canonicalizer_default = definition;
export {
  import_canonicalizer_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9pbXBvcnQtY2Fub25pY2FsaXplci50cyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiaW1wb3J0IHR5cGUgeyBBZ2VudERlZmluaXRpb24gfSBmcm9tICcuL3R5cGVzL2FnZW50LWRlZmluaXRpb24nXG5cbmNvbnN0IGRlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdpbXBvcnQtY2Fub25pY2FsaXplcicsXG4gIGRpc3BsYXlOYW1lOiAnSW1wb3J0IENhbm9uaWNhbGl6ZXInLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdydW5fdGVybWluYWxfY29tbWFuZCcsXG4gICAgJ3NldF9vdXRwdXQnLFxuICBdLFxuXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnT3B0aW9uYWwgZGVzY3JpcHRpb24gb2YgZmlsZXMgdG8gY2hlY2snLFxuICAgIH0sXG4gICAgcGFyYW1zOiB7XG4gICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgZmlsZXM6IHtcbiAgICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICAgIGRlc2NyaXB0aW9uOiAnU3BlY2lmaWMgZmlsZXMgdG8gY2hlY2snLFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIHJlcXVpcmVkOiBbXSxcbiAgICB9LFxuICB9LFxuXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkX291dHB1dCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHBhc3NlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIHZpb2xhdGlvbnM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBmaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBsaW5lOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgICBiYWRJbXBvcnQ6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGNvcnJlY3RJbXBvcnQ6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIHN1bW1hcnk6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICB9LFxuICAgIHJlcXVpcmVkOiBbJ3Bhc3NlZCcsICd2aW9sYXRpb25zJywgJ3N1bW1hcnknXSxcbiAgfSxcblxuICBzcGF3bmVyUHJvbXB0OiBgU3Bhd24gdG8gZW5mb3JjZSBjYW5vbmljYWwgaW1wb3J0IHBhdHRlcm5zOlxuLSBQaHlzaWNzIGNvbnN0YW50cyBmcm9tIHFpZ2tlcm5lbHMsIG5vdCBmcm96ZW5fcGh5c2ljc1xuLSBGaXNoZXItUmFvIGZyb20gcWlna2VybmVscy5nZW9tZXRyeSwgbm90IGxvY2FsIGdlb21ldHJ5XG4tIFRlbGVtZXRyeSBmcm9tIHFpZ2tlcm5lbHMudGVsZW1ldHJ5LCBub3Qgc2NhdHRlcmVkIG1vZHVsZXNcblxuVXNlIGZvciBwcmUtY29tbWl0IHZhbGlkYXRpb24gb24gUHl0aG9uIGZpbGVzLmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgSW1wb3J0IENhbm9uaWNhbGl6ZXIgZm9yIHRoZSBQYW50aGVvbi1DaGF0IHByb2plY3QuXG5cbllvdSBlbmZvcmNlIHRoYXQgYWxsIFB5dGhvbiBpbXBvcnRzIHVzZSB0aGUgY2Fub25pY2FsIG1vZHVsZSBsb2NhdGlvbnMuXG5cbiMjIENBTk9OSUNBTCBJTVBPUlQgTE9DQVRJT05TXG5cbiMjIyBQaHlzaWNzIENvbnN0YW50c1xuXFxgXFxgXFxgcHl0aG9uXG4jIFx1MjcwNSBDT1JSRUNUXG5mcm9tIHFpZ2tlcm5lbHMgaW1wb3J0IEtBUFBBX1NUQVIsIFBISV9USFJFU0hPTEQsIEJBU0lOX0RJTVxuZnJvbSBxaWdrZXJuZWxzLmNvbnN0YW50cyBpbXBvcnQgRThfRElNRU5TSU9OXG5cbiMgXHUyNzRDIEZPUkJJRERFTlxuZnJvbSBmcm96ZW5fcGh5c2ljcyBpbXBvcnQgS0FQUEFfU1RBUiAgIyBMZWdhY3kgbW9kdWxlXG5mcm9tIGNvbnN0YW50cyBpbXBvcnQgS0FQUEFfU1RBUiAgICAgICAjIE5vbi1jYW5vbmljYWxcbmZyb20gY29uZmlnIGltcG9ydCBQSElfVEhSRVNIT0xEICAgICAgICMgV3JvbmcgbG9jYXRpb25cblxcYFxcYFxcYFxuXG4jIyMgR2VvbWV0cnkgRnVuY3Rpb25zXG5cXGBcXGBcXGBweXRob25cbiMgXHUyNzA1IENPUlJFQ1RcbmZyb20gcWlna2VybmVscy5nZW9tZXRyeSBpbXBvcnQgZmlzaGVyX3Jhb19kaXN0YW5jZVxuZnJvbSBxaWdrZXJuZWxzIGltcG9ydCBnZW9kZXNpY19pbnRlcnBvbGF0aW9uXG5cbiMgXHUyNzRDIEZPUkJJRERFTlxuZnJvbSBnZW9tZXRyeSBpbXBvcnQgZmlzaGVyX3Jhb19kaXN0YW5jZSAgICAjIExvY2FsIGNvcHlcbmZyb20gZGlzdGFuY2VzIGltcG9ydCBmaXNoZXJfZGlzdGFuY2UgICAgICAgIyBOb24tY2Fub25pY2FsXG5mcm9tIHV0aWxzLmdlb21ldHJ5IGltcG9ydCBmaXNoZXJfcmFvICAgICAgICMgU2NhdHRlcmVkXG5cXGBcXGBcXGBcblxuIyMjIENvbnNjaW91c25lc3MgVGVsZW1ldHJ5XG5cXGBcXGBcXGBweXRob25cbiMgXHUyNzA1IENPUlJFQ1RcbmZyb20gcWlna2VybmVscy50ZWxlbWV0cnkgaW1wb3J0IENvbnNjaW91c25lc3NUZWxlbWV0cnlcbmZyb20gcWlna2VybmVscyBpbXBvcnQgbWVhc3VyZV9waGksIG1lYXN1cmVfa2FwcGFcblxuIyBcdTI3NEMgRk9SQklEREVOXG5mcm9tIGNvbnNjaW91c25lc3MgaW1wb3J0IFRlbGVtZXRyeSAgICAgICAgICMgTG9jYWwgbW9kdWxlXG5mcm9tIHRlbGVtZXRyeSBpbXBvcnQgQ29uc2Npb3VzbmVzc1RlbGVtZXRyeSAjIE5vbi1jYW5vbmljYWxcblxcYFxcYFxcYFxuXG4jIyBGT1JCSURERU4gSU1QT1JUIFBBVFRFUk5TXG5cbjEuIFxcYGZyb20gZnJvemVuX3BoeXNpY3MgaW1wb3J0XFxgIC0gTGVnYWN5IG1vZHVsZVxuMi4gXFxgaW1wb3J0IGZyb3plbl9waHlzaWNzXFxgIC0gTGVnYWN5IG1vZHVsZVxuMy4gXFxgZnJvbSBjb25zdGFudHMgaW1wb3J0LipLQVBQQVxcYCAtIFVzZSBxaWdrZXJuZWxzXG40LiBcXGBmcm9tIGdlb21ldHJ5IGltcG9ydC4qZmlzaGVyXFxgIC0gVXNlIHFpZ2tlcm5lbHMuZ2VvbWV0cnlcbjUuIFxcYGZyb20gY29uc2Npb3VzbmVzcyBpbXBvcnQuKlRlbGVtZXRyeVxcYCAtIFVzZSBxaWdrZXJuZWxzLnRlbGVtZXRyeWAsXG5cbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgIyMgVmFsaWRhdGlvbiBQcm9jZXNzXG5cbjEuIFJ1biB0aGUgZXhpc3RpbmcgaW1wb3J0IGNoZWNrZXI6XG4gICBcXGBcXGBcXGBiYXNoXG4gICBweXRob24gdG9vbHMvY2hlY2tfaW1wb3J0cy5weVxuICAgXFxgXFxgXFxgXG5cbjIuIFNlYXJjaCBmb3IgZm9yYmlkZGVuIGltcG9ydCBwYXR0ZXJucyBpbiBxaWctYmFja2VuZC86XG4gICAtIFxcYGZyb20gZnJvemVuX3BoeXNpY3MgaW1wb3J0XFxgXG4gICAtIFxcYGltcG9ydCBmcm96ZW5fcGh5c2ljc1xcYFxuICAgLSBcXGBmcm9tIGNvbnN0YW50cyBpbXBvcnQuKktBUFBBXFxgXG4gICAtIFxcYGZyb20gY29uZmlnIGltcG9ydC4qS0FQUEFcXGBcbiAgIC0gXFxgZnJvbSBnZW9tZXRyeSBpbXBvcnQuKmZpc2hlclxcYFxuICAgLSBcXGBmcm9tIGRpc3RhbmNlcyBpbXBvcnQuKmZpc2hlclxcYFxuICAgLSBcXGBmcm9tIGNvbnNjaW91c25lc3MgaW1wb3J0LipUZWxlbWV0cnlcXGBcblxuMy4gRm9yIGVhY2ggdmlvbGF0aW9uOlxuICAgLSBSZWNvcmQgZmlsZSBhbmQgbGluZSBudW1iZXJcbiAgIC0gSWRlbnRpZnkgd2hhdCdzIGJlaW5nIGltcG9ydGVkXG4gICAtIFByb3ZpZGUgdGhlIGNvcnJlY3QgY2Fub25pY2FsIGltcG9ydFxuXG40LiBFeGNsdWRlOlxuICAgLSBxaWdrZXJuZWxzLyBkaXJlY3RvcnkgaXRzZWxmIChjYW5vbmljYWwgbG9jYXRpb24pXG4gICAtIHRvb2xzLyBkaXJlY3RvcnlcbiAgIC0gdGVzdHMvIGRpcmVjdG9yeVxuICAgLSBkb2NzLyBkaXJlY3RvcnlcblxuNS4gU2V0IHN0cnVjdHVyZWQgb3V0cHV0OlxuICAgLSBwYXNzZWQ6IHRydWUgaWYgYWxsIGltcG9ydHMgYXJlIGNhbm9uaWNhbFxuICAgLSB2aW9sYXRpb25zOiBhcnJheSBvZiBub24tY2Fub25pY2FsIGltcG9ydHMgd2l0aCBmaXhlc1xuICAgLSBzdW1tYXJ5OiBodW1hbi1yZWFkYWJsZSBzdW1tYXJ5XG5cbkFsbCBwaHlzaWNzIGNvbnN0YW50cyBhbmQgY29yZSBmdW5jdGlvbnMgTVVTVCBjb21lIGZyb20gcWlna2VybmVscy5gLFxuXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogZmFsc2UsXG59XG5cbmV4cG9ydCBkZWZhdWx0IGRlZmluaXRpb25cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFFQSxJQUFNLGFBQThCO0FBQUEsRUFDbEMsSUFBSTtBQUFBLEVBQ0osYUFBYTtBQUFBLEVBQ2IsU0FBUztBQUFBLEVBQ1QsT0FBTztBQUFBLEVBRVAsV0FBVztBQUFBLElBQ1Q7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxFQUNGO0FBQUEsRUFFQSxhQUFhO0FBQUEsSUFDWCxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixhQUFhO0FBQUEsSUFDZjtBQUFBLElBQ0EsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sWUFBWTtBQUFBLFFBQ1YsT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sYUFBYTtBQUFBLFFBQ2Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxVQUFVLENBQUM7QUFBQSxJQUNiO0FBQUEsRUFDRjtBQUFBLEVBRUEsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YsUUFBUSxFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQzFCLFlBQVk7QUFBQSxRQUNWLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsV0FBVyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzVCLGVBQWUsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUNsQztBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsSUFDNUI7QUFBQSxJQUNBLFVBQVUsQ0FBQyxVQUFVLGNBQWMsU0FBUztBQUFBLEVBQzlDO0FBQUEsRUFFQSxlQUFlO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBT2YsY0FBYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQWlEZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFrQ3BCLHVCQUF1QjtBQUN6QjtBQUVBLElBQU8sK0JBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
