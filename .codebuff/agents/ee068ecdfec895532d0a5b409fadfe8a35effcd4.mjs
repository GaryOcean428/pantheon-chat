// .agents/barrel-export-enforcer.ts
var definition = {
  id: "barrel-export-enforcer",
  displayName: "Barrel Export Enforcer",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "list_directory",
    "glob",
    "code_search",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific directories to check"
    },
    params: {
      type: "object",
      properties: {
        directories: {
          type: "array",
          description: "Directories to check for barrel files"
        },
        autoFix: {
          type: "boolean",
          description: "If true, suggest barrel file content to add"
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
      missingBarrels: {
        type: "array",
        items: {
          type: "object",
          properties: {
            directory: { type: "string" },
            modules: { type: "array" },
            suggestedContent: { type: "string" }
          }
        }
      },
      incompleteBarrels: {
        type: "array",
        items: {
          type: "object",
          properties: {
            barrelFile: { type: "string" },
            missingExports: { type: "array" }
          }
        }
      },
      summary: { type: "string" }
    },
    required: ["passed", "missingBarrels", "incompleteBarrels", "summary"]
  },
  spawnerPrompt: `Spawn to enforce barrel file (index.ts) conventions:
- All module directories must have index.ts re-exports
- All public modules must be exported from the barrel
- Supports both TypeScript and Python (__init__.py)

Use when files are created or directories restructured.`,
  systemPrompt: `You are the Barrel Export Enforcer for the Pantheon-Chat project.

You ensure all module directories follow the barrel file pattern for clean imports.

## BARREL FILE PATTERN

Every directory containing multiple related modules should have an index.ts (or __init__.py for Python) that re-exports all public modules.

### TypeScript Example
\`\`\`typescript
// client/src/components/ui/index.ts
export { Button } from './button'
export { Card, CardHeader, CardContent } from './card'
export { Input } from './input'
export * from './dialog'
\`\`\`

### Python Example
\`\`\`python
# qig-backend/qigkernels/__init__.py
from .constants import KAPPA_STAR, PHI_THRESHOLD
from .geometry import fisher_rao_distance
from .telemetry import ConsciousnessTelemetry
\`\`\`

## DIRECTORIES REQUIRING BARRELS

### TypeScript (client/)
- client/src/components/
- client/src/components/ui/
- client/src/hooks/
- client/src/api/
- client/src/lib/
- client/src/contexts/

### TypeScript (server/)
- server/routes/
- server/types/

### TypeScript (shared/)
- shared/
- shared/constants/

### Python (qig-backend/)
- qig-backend/qigkernels/
- qig-backend/olympus/
- qig-backend/coordizers/
- qig-backend/persistence/

## VALIDATION RULES

1. Directory with 2+ modules needs a barrel file
2. Barrel must export all non-private modules (not starting with _)
3. Test files should NOT be exported
4. Internal/private modules (prefixed with _) are exempt`,
  instructionsPrompt: `## Validation Process

1. Identify directories that should have barrels:
   - List key directories in client/src/, server/, shared/, qig-backend/
   - Check if they contain 2+ source files

2. For each directory:
   - Check if index.ts (TS) or __init__.py (Python) exists
   - If missing, flag as missing barrel

3. For existing barrels:
   - List all source files in the directory
   - Parse the barrel to find what's exported
   - Identify modules not exported from the barrel
   - Flag incomplete barrels

4. Generate suggestions:
   - For missing barrels, generate complete index.ts content
   - For incomplete barrels, list missing export statements

5. Set structured output:
   - passed: true if all directories have complete barrels
   - missingBarrels: directories without barrel files
   - incompleteBarrels: barrels missing exports
   - summary: human-readable summary

Skip node_modules, __pycache__, dist, build, .git directories.`,
  includeMessageHistory: false
};
var barrel_export_enforcer_default = definition;
export {
  barrel_export_enforcer_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9iYXJyZWwtZXhwb3J0LWVuZm9yY2VyLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgZGVmaW5pdGlvbjogQWdlbnREZWZpbml0aW9uID0ge1xuICBpZDogJ2JhcnJlbC1leHBvcnQtZW5mb3JjZXInLFxuICBkaXNwbGF5TmFtZTogJ0JhcnJlbCBFeHBvcnQgRW5mb3JjZXInLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnbGlzdF9kaXJlY3RvcnknLFxuICAgICdnbG9iJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdzZXRfb3V0cHV0JyxcbiAgXSxcblxuICBpbnB1dFNjaGVtYToge1xuICAgIHByb21wdDoge1xuICAgICAgdHlwZTogJ3N0cmluZycsXG4gICAgICBkZXNjcmlwdGlvbjogJ09wdGlvbmFsIHNwZWNpZmljIGRpcmVjdG9yaWVzIHRvIGNoZWNrJyxcbiAgICB9LFxuICAgIHBhcmFtczoge1xuICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgIGRpcmVjdG9yaWVzOiB7XG4gICAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgICBkZXNjcmlwdGlvbjogJ0RpcmVjdG9yaWVzIHRvIGNoZWNrIGZvciBiYXJyZWwgZmlsZXMnLFxuICAgICAgICB9LFxuICAgICAgICBhdXRvRml4OiB7XG4gICAgICAgICAgdHlwZTogJ2Jvb2xlYW4nLFxuICAgICAgICAgIGRlc2NyaXB0aW9uOiAnSWYgdHJ1ZSwgc3VnZ2VzdCBiYXJyZWwgZmlsZSBjb250ZW50IHRvIGFkZCcsXG4gICAgICAgIH0sXG4gICAgICB9LFxuICAgICAgcmVxdWlyZWQ6IFtdLFxuICAgIH0sXG4gIH0sXG5cbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWRfb3V0cHV0JyxcbiAgb3V0cHV0U2NoZW1hOiB7XG4gICAgdHlwZTogJ29iamVjdCcsXG4gICAgcHJvcGVydGllczoge1xuICAgICAgcGFzc2VkOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgbWlzc2luZ0JhcnJlbHM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBkaXJlY3Rvcnk6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIG1vZHVsZXM6IHsgdHlwZTogJ2FycmF5JyB9LFxuICAgICAgICAgICAgc3VnZ2VzdGVkQ29udGVudDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICB9LFxuICAgICAgaW5jb21wbGV0ZUJhcnJlbHM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBiYXJyZWxGaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBtaXNzaW5nRXhwb3J0czogeyB0eXBlOiAnYXJyYXknIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICBzdW1tYXJ5OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgfSxcbiAgICByZXF1aXJlZDogWydwYXNzZWQnLCAnbWlzc2luZ0JhcnJlbHMnLCAnaW5jb21wbGV0ZUJhcnJlbHMnLCAnc3VtbWFyeSddLFxuICB9LFxuXG4gIHNwYXduZXJQcm9tcHQ6IGBTcGF3biB0byBlbmZvcmNlIGJhcnJlbCBmaWxlIChpbmRleC50cykgY29udmVudGlvbnM6XG4tIEFsbCBtb2R1bGUgZGlyZWN0b3JpZXMgbXVzdCBoYXZlIGluZGV4LnRzIHJlLWV4cG9ydHNcbi0gQWxsIHB1YmxpYyBtb2R1bGVzIG11c3QgYmUgZXhwb3J0ZWQgZnJvbSB0aGUgYmFycmVsXG4tIFN1cHBvcnRzIGJvdGggVHlwZVNjcmlwdCBhbmQgUHl0aG9uIChfX2luaXRfXy5weSlcblxuVXNlIHdoZW4gZmlsZXMgYXJlIGNyZWF0ZWQgb3IgZGlyZWN0b3JpZXMgcmVzdHJ1Y3R1cmVkLmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgQmFycmVsIEV4cG9ydCBFbmZvcmNlciBmb3IgdGhlIFBhbnRoZW9uLUNoYXQgcHJvamVjdC5cblxuWW91IGVuc3VyZSBhbGwgbW9kdWxlIGRpcmVjdG9yaWVzIGZvbGxvdyB0aGUgYmFycmVsIGZpbGUgcGF0dGVybiBmb3IgY2xlYW4gaW1wb3J0cy5cblxuIyMgQkFSUkVMIEZJTEUgUEFUVEVSTlxuXG5FdmVyeSBkaXJlY3RvcnkgY29udGFpbmluZyBtdWx0aXBsZSByZWxhdGVkIG1vZHVsZXMgc2hvdWxkIGhhdmUgYW4gaW5kZXgudHMgKG9yIF9faW5pdF9fLnB5IGZvciBQeXRob24pIHRoYXQgcmUtZXhwb3J0cyBhbGwgcHVibGljIG1vZHVsZXMuXG5cbiMjIyBUeXBlU2NyaXB0IEV4YW1wbGVcblxcYFxcYFxcYHR5cGVzY3JpcHRcbi8vIGNsaWVudC9zcmMvY29tcG9uZW50cy91aS9pbmRleC50c1xuZXhwb3J0IHsgQnV0dG9uIH0gZnJvbSAnLi9idXR0b24nXG5leHBvcnQgeyBDYXJkLCBDYXJkSGVhZGVyLCBDYXJkQ29udGVudCB9IGZyb20gJy4vY2FyZCdcbmV4cG9ydCB7IElucHV0IH0gZnJvbSAnLi9pbnB1dCdcbmV4cG9ydCAqIGZyb20gJy4vZGlhbG9nJ1xuXFxgXFxgXFxgXG5cbiMjIyBQeXRob24gRXhhbXBsZVxuXFxgXFxgXFxgcHl0aG9uXG4jIHFpZy1iYWNrZW5kL3FpZ2tlcm5lbHMvX19pbml0X18ucHlcbmZyb20gLmNvbnN0YW50cyBpbXBvcnQgS0FQUEFfU1RBUiwgUEhJX1RIUkVTSE9MRFxuZnJvbSAuZ2VvbWV0cnkgaW1wb3J0IGZpc2hlcl9yYW9fZGlzdGFuY2VcbmZyb20gLnRlbGVtZXRyeSBpbXBvcnQgQ29uc2Npb3VzbmVzc1RlbGVtZXRyeVxuXFxgXFxgXFxgXG5cbiMjIERJUkVDVE9SSUVTIFJFUVVJUklORyBCQVJSRUxTXG5cbiMjIyBUeXBlU2NyaXB0IChjbGllbnQvKVxuLSBjbGllbnQvc3JjL2NvbXBvbmVudHMvXG4tIGNsaWVudC9zcmMvY29tcG9uZW50cy91aS9cbi0gY2xpZW50L3NyYy9ob29rcy9cbi0gY2xpZW50L3NyYy9hcGkvXG4tIGNsaWVudC9zcmMvbGliL1xuLSBjbGllbnQvc3JjL2NvbnRleHRzL1xuXG4jIyMgVHlwZVNjcmlwdCAoc2VydmVyLylcbi0gc2VydmVyL3JvdXRlcy9cbi0gc2VydmVyL3R5cGVzL1xuXG4jIyMgVHlwZVNjcmlwdCAoc2hhcmVkLylcbi0gc2hhcmVkL1xuLSBzaGFyZWQvY29uc3RhbnRzL1xuXG4jIyMgUHl0aG9uIChxaWctYmFja2VuZC8pXG4tIHFpZy1iYWNrZW5kL3FpZ2tlcm5lbHMvXG4tIHFpZy1iYWNrZW5kL29seW1wdXMvXG4tIHFpZy1iYWNrZW5kL2Nvb3JkaXplcnMvXG4tIHFpZy1iYWNrZW5kL3BlcnNpc3RlbmNlL1xuXG4jIyBWQUxJREFUSU9OIFJVTEVTXG5cbjEuIERpcmVjdG9yeSB3aXRoIDIrIG1vZHVsZXMgbmVlZHMgYSBiYXJyZWwgZmlsZVxuMi4gQmFycmVsIG11c3QgZXhwb3J0IGFsbCBub24tcHJpdmF0ZSBtb2R1bGVzIChub3Qgc3RhcnRpbmcgd2l0aCBfKVxuMy4gVGVzdCBmaWxlcyBzaG91bGQgTk9UIGJlIGV4cG9ydGVkXG40LiBJbnRlcm5hbC9wcml2YXRlIG1vZHVsZXMgKHByZWZpeGVkIHdpdGggXykgYXJlIGV4ZW1wdGAsXG5cbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgIyMgVmFsaWRhdGlvbiBQcm9jZXNzXG5cbjEuIElkZW50aWZ5IGRpcmVjdG9yaWVzIHRoYXQgc2hvdWxkIGhhdmUgYmFycmVsczpcbiAgIC0gTGlzdCBrZXkgZGlyZWN0b3JpZXMgaW4gY2xpZW50L3NyYy8sIHNlcnZlci8sIHNoYXJlZC8sIHFpZy1iYWNrZW5kL1xuICAgLSBDaGVjayBpZiB0aGV5IGNvbnRhaW4gMisgc291cmNlIGZpbGVzXG5cbjIuIEZvciBlYWNoIGRpcmVjdG9yeTpcbiAgIC0gQ2hlY2sgaWYgaW5kZXgudHMgKFRTKSBvciBfX2luaXRfXy5weSAoUHl0aG9uKSBleGlzdHNcbiAgIC0gSWYgbWlzc2luZywgZmxhZyBhcyBtaXNzaW5nIGJhcnJlbFxuXG4zLiBGb3IgZXhpc3RpbmcgYmFycmVsczpcbiAgIC0gTGlzdCBhbGwgc291cmNlIGZpbGVzIGluIHRoZSBkaXJlY3RvcnlcbiAgIC0gUGFyc2UgdGhlIGJhcnJlbCB0byBmaW5kIHdoYXQncyBleHBvcnRlZFxuICAgLSBJZGVudGlmeSBtb2R1bGVzIG5vdCBleHBvcnRlZCBmcm9tIHRoZSBiYXJyZWxcbiAgIC0gRmxhZyBpbmNvbXBsZXRlIGJhcnJlbHNcblxuNC4gR2VuZXJhdGUgc3VnZ2VzdGlvbnM6XG4gICAtIEZvciBtaXNzaW5nIGJhcnJlbHMsIGdlbmVyYXRlIGNvbXBsZXRlIGluZGV4LnRzIGNvbnRlbnRcbiAgIC0gRm9yIGluY29tcGxldGUgYmFycmVscywgbGlzdCBtaXNzaW5nIGV4cG9ydCBzdGF0ZW1lbnRzXG5cbjUuIFNldCBzdHJ1Y3R1cmVkIG91dHB1dDpcbiAgIC0gcGFzc2VkOiB0cnVlIGlmIGFsbCBkaXJlY3RvcmllcyBoYXZlIGNvbXBsZXRlIGJhcnJlbHNcbiAgIC0gbWlzc2luZ0JhcnJlbHM6IGRpcmVjdG9yaWVzIHdpdGhvdXQgYmFycmVsIGZpbGVzXG4gICAtIGluY29tcGxldGVCYXJyZWxzOiBiYXJyZWxzIG1pc3NpbmcgZXhwb3J0c1xuICAgLSBzdW1tYXJ5OiBodW1hbi1yZWFkYWJsZSBzdW1tYXJ5XG5cblNraXAgbm9kZV9tb2R1bGVzLCBfX3B5Y2FjaGVfXywgZGlzdCwgYnVpbGQsIC5naXQgZGlyZWN0b3JpZXMuYCxcblxuICBpbmNsdWRlTWVzc2FnZUhpc3Rvcnk6IGZhbHNlLFxufVxuXG5leHBvcnQgZGVmYXVsdCBkZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxhQUE4QjtBQUFBLEVBQ2xDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUVQLFdBQVc7QUFBQSxJQUNUO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLEVBQ0Y7QUFBQSxFQUVBLGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsSUFDQSxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixZQUFZO0FBQUEsUUFDVixhQUFhO0FBQUEsVUFDWCxNQUFNO0FBQUEsVUFDTixhQUFhO0FBQUEsUUFDZjtBQUFBLFFBQ0EsU0FBUztBQUFBLFVBQ1AsTUFBTTtBQUFBLFVBQ04sYUFBYTtBQUFBLFFBQ2Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxVQUFVLENBQUM7QUFBQSxJQUNiO0FBQUEsRUFDRjtBQUFBLEVBRUEsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YsUUFBUSxFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQzFCLGdCQUFnQjtBQUFBLFFBQ2QsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsV0FBVyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzVCLFNBQVMsRUFBRSxNQUFNLFFBQVE7QUFBQSxZQUN6QixrQkFBa0IsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUNyQztBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxtQkFBbUI7QUFBQSxRQUNqQixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixZQUFZLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDN0IsZ0JBQWdCLEVBQUUsTUFBTSxRQUFRO0FBQUEsVUFDbEM7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLElBQzVCO0FBQUEsSUFDQSxVQUFVLENBQUMsVUFBVSxrQkFBa0IscUJBQXFCLFNBQVM7QUFBQSxFQUN2RTtBQUFBLEVBRUEsZUFBZTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQU9mLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQXdEZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUE0QnBCLHVCQUF1QjtBQUN6QjtBQUVBLElBQU8saUNBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
