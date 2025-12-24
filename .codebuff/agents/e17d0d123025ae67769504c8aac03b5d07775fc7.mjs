// .agents/codebase-cleanup-auditor.ts
var agentDefinition = {
  id: "codebase-cleanup-auditor",
  displayName: "Codebase Cleanup Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search", "run_terminal_command"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4", "codebuff/deep-thinker@0.0.3"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Audit codebase for cleanup and refactoring opportunities"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      cleanupNeeded: { type: "boolean" },
      deadCode: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            type: { type: "string", enum: ["unused-export", "unused-import", "unused-variable", "orphaned-file"] },
            symbol: { type: "string" }
          }
        }
      },
      refactoringOpportunities: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            opportunity: { type: "string" },
            effort: { type: "string", enum: ["small", "medium", "large"] },
            impact: { type: "string", enum: ["high", "medium", "low"] }
          }
        }
      },
      codeSmells: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            smell: { type: "string" },
            description: { type: "string" }
          }
        }
      },
      housekeeping: {
        type: "array",
        items: { type: "string" }
      }
    }
  },
  spawnerPrompt: "Spawn to audit codebase for cleanup and maintainability improvements",
  systemPrompt: `You are a code quality and maintainability expert.

Audit areas:
1. Dead code (unused exports, imports, variables)
2. Orphaned files (not imported anywhere)
3. Large files that need splitting
4. Complex functions that need refactoring
5. Code smells (long parameter lists, deep nesting)
6. Inconsistent patterns
7. TODO/FIXME comments
8. Console.log statements in production code
9. Commented-out code blocks

Housekeeping Checks:
- Remove unused dependencies
- Clean up .gitignore
- Update outdated comments
- Consolidate duplicate styles
- Remove temporary files
- Clean up build artifacts`,
  instructionsPrompt: `Audit codebase for cleanup:

1. Find unused exports with code search
2. Look for orphaned files (no importers)
3. Find large files (>500 lines)
4. Search for TODO/FIXME comments
5. Find console.log in production code
6. Look for commented-out code blocks
7. Check for duplicate code patterns
8. Find deeply nested code (>4 levels)
9. Report all cleanup opportunities`
};
var codebase_cleanup_auditor_default = agentDefinition;
export {
  codebase_cleanup_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9jb2RlYmFzZS1jbGVhbnVwLWF1ZGl0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBhZ2VudERlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdjb2RlYmFzZS1jbGVhbnVwLWF1ZGl0b3InLFxuICBkaXNwbGF5TmFtZTogJ0NvZGViYXNlIENsZWFudXAgQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnLCAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnLCAnY29kZWJ1ZmYvZGVlcC10aGlua2VyQDAuMC4zJ10sXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnQXVkaXQgY29kZWJhc2UgZm9yIGNsZWFudXAgYW5kIHJlZmFjdG9yaW5nIG9wcG9ydHVuaXRpZXMnXG4gICAgfVxuICB9LFxuICBpbmNsdWRlTWVzc2FnZUhpc3Rvcnk6IHRydWUsXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkJyxcbiAgb3V0cHV0U2NoZW1hOiB7XG4gICAgdHlwZTogJ29iamVjdCcsXG4gICAgcHJvcGVydGllczoge1xuICAgICAgY2xlYW51cE5lZWRlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIGRlYWRDb2RlOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgdHlwZTogeyB0eXBlOiAnc3RyaW5nJywgZW51bTogWyd1bnVzZWQtZXhwb3J0JywgJ3VudXNlZC1pbXBvcnQnLCAndW51c2VkLXZhcmlhYmxlJywgJ29ycGhhbmVkLWZpbGUnXSB9LFxuICAgICAgICAgICAgc3ltYm9sOiB7IHR5cGU6ICdzdHJpbmcnIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICByZWZhY3RvcmluZ09wcG9ydHVuaXRpZXM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBmaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBvcHBvcnR1bml0eTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgZWZmb3J0OiB7IHR5cGU6ICdzdHJpbmcnLCBlbnVtOiBbJ3NtYWxsJywgJ21lZGl1bScsICdsYXJnZSddIH0sXG4gICAgICAgICAgICBpbXBhY3Q6IHsgdHlwZTogJ3N0cmluZycsIGVudW06IFsnaGlnaCcsICdtZWRpdW0nLCAnbG93J10gfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIGNvZGVTbWVsbHM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBmaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBzbWVsbDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgZGVzY3JpcHRpb246IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIGhvdXNla2VlcGluZzoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczogeyB0eXBlOiAnc3RyaW5nJyB9XG4gICAgICB9XG4gICAgfVxuICB9LFxuICBzcGF3bmVyUHJvbXB0OiAnU3Bhd24gdG8gYXVkaXQgY29kZWJhc2UgZm9yIGNsZWFudXAgYW5kIG1haW50YWluYWJpbGl0eSBpbXByb3ZlbWVudHMnLFxuICBzeXN0ZW1Qcm9tcHQ6IGBZb3UgYXJlIGEgY29kZSBxdWFsaXR5IGFuZCBtYWludGFpbmFiaWxpdHkgZXhwZXJ0LlxuXG5BdWRpdCBhcmVhczpcbjEuIERlYWQgY29kZSAodW51c2VkIGV4cG9ydHMsIGltcG9ydHMsIHZhcmlhYmxlcylcbjIuIE9ycGhhbmVkIGZpbGVzIChub3QgaW1wb3J0ZWQgYW55d2hlcmUpXG4zLiBMYXJnZSBmaWxlcyB0aGF0IG5lZWQgc3BsaXR0aW5nXG40LiBDb21wbGV4IGZ1bmN0aW9ucyB0aGF0IG5lZWQgcmVmYWN0b3JpbmdcbjUuIENvZGUgc21lbGxzIChsb25nIHBhcmFtZXRlciBsaXN0cywgZGVlcCBuZXN0aW5nKVxuNi4gSW5jb25zaXN0ZW50IHBhdHRlcm5zXG43LiBUT0RPL0ZJWE1FIGNvbW1lbnRzXG44LiBDb25zb2xlLmxvZyBzdGF0ZW1lbnRzIGluIHByb2R1Y3Rpb24gY29kZVxuOS4gQ29tbWVudGVkLW91dCBjb2RlIGJsb2Nrc1xuXG5Ib3VzZWtlZXBpbmcgQ2hlY2tzOlxuLSBSZW1vdmUgdW51c2VkIGRlcGVuZGVuY2llc1xuLSBDbGVhbiB1cCAuZ2l0aWdub3JlXG4tIFVwZGF0ZSBvdXRkYXRlZCBjb21tZW50c1xuLSBDb25zb2xpZGF0ZSBkdXBsaWNhdGUgc3R5bGVzXG4tIFJlbW92ZSB0ZW1wb3JhcnkgZmlsZXNcbi0gQ2xlYW4gdXAgYnVpbGQgYXJ0aWZhY3RzYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgQXVkaXQgY29kZWJhc2UgZm9yIGNsZWFudXA6XG5cbjEuIEZpbmQgdW51c2VkIGV4cG9ydHMgd2l0aCBjb2RlIHNlYXJjaFxuMi4gTG9vayBmb3Igb3JwaGFuZWQgZmlsZXMgKG5vIGltcG9ydGVycylcbjMuIEZpbmQgbGFyZ2UgZmlsZXMgKD41MDAgbGluZXMpXG40LiBTZWFyY2ggZm9yIFRPRE8vRklYTUUgY29tbWVudHNcbjUuIEZpbmQgY29uc29sZS5sb2cgaW4gcHJvZHVjdGlvbiBjb2RlXG42LiBMb29rIGZvciBjb21tZW50ZWQtb3V0IGNvZGUgYmxvY2tzXG43LiBDaGVjayBmb3IgZHVwbGljYXRlIGNvZGUgcGF0dGVybnNcbjguIEZpbmQgZGVlcGx5IG5lc3RlZCBjb2RlICg+NCBsZXZlbHMpXG45LiBSZXBvcnQgYWxsIGNsZWFudXAgb3Bwb3J0dW5pdGllc2Bcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyxlQUFlLHNCQUFzQjtBQUFBLEVBQy9ELGlCQUFpQixDQUFDLGdDQUFnQyw2QkFBNkI7QUFBQSxFQUMvRSxhQUFhO0FBQUEsSUFDWCxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixhQUFhO0FBQUEsSUFDZjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLHVCQUF1QjtBQUFBLEVBQ3ZCLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLGVBQWUsRUFBRSxNQUFNLFVBQVU7QUFBQSxNQUNqQyxVQUFVO0FBQUEsUUFDUixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsTUFBTSxFQUFFLE1BQU0sVUFBVSxNQUFNLENBQUMsaUJBQWlCLGlCQUFpQixtQkFBbUIsZUFBZSxFQUFFO0FBQUEsWUFDckcsUUFBUSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQzNCO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLDBCQUEwQjtBQUFBLFFBQ3hCLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixhQUFhLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDOUIsUUFBUSxFQUFFLE1BQU0sVUFBVSxNQUFNLENBQUMsU0FBUyxVQUFVLE9BQU8sRUFBRTtBQUFBLFlBQzdELFFBQVEsRUFBRSxNQUFNLFVBQVUsTUFBTSxDQUFDLFFBQVEsVUFBVSxLQUFLLEVBQUU7QUFBQSxVQUM1RDtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxZQUFZO0FBQUEsUUFDVixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsT0FBTyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3hCLGFBQWEsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUNoQztBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxjQUFjO0FBQUEsUUFDWixNQUFNO0FBQUEsUUFDTixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsTUFDMUI7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUFBLEVBQ0EsZUFBZTtBQUFBLEVBQ2YsY0FBYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFvQmQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFXdEI7QUFFQSxJQUFPLG1DQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
