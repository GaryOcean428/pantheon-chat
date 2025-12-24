// .agents/module-bridging-validator.ts
var agentDefinition = {
  id: "module-bridging-validator",
  displayName: "Module Bridging Validator",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Validate modules are correctly bridged and modular"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      modulesCorrectlyBridged: { type: "boolean" },
      orphanedModules: {
        type: "array",
        items: {
          type: "object",
          properties: {
            path: { type: "string" },
            exportedSymbols: { type: "array", items: { type: "string" } },
            importedBy: { type: "array", items: { type: "string" } }
          }
        }
      },
      duplicatedCode: {
        type: "array",
        items: {
          type: "object",
          properties: {
            pattern: { type: "string" },
            locations: { type: "array", items: { type: "string" } },
            consolidationSuggestion: { type: "string" }
          }
        }
      },
      bridgingIssues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            sourceModule: { type: "string" },
            targetModule: { type: "string" },
            issue: { type: "string" }
          }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to ensure modules are correctly bridged with no duplication or orphans",
  systemPrompt: `You are a module architecture expert.

Your responsibilities:
1. Verify all components, kernels, and features are correctly bridged
2. Find orphaned modules that aren't imported anywhere
3. Detect code duplication across modules
4. Ensure proper modularity and separation
5. Check TypeScript\u2194Python bridging is correct

Module Bridging Rules:
- Every exported symbol should have at least one importer
- No duplicate implementations of the same functionality
- TypeScript server bridges to Python backend correctly
- Shared code lives in shared/ or common modules
- Circular dependencies are forbidden`,
  instructionsPrompt: `Validate module bridging:

1. Find all exported symbols across the codebase
2. Check which exports have no importers (orphaned)
3. Look for similar function names/patterns (duplication)
4. Verify server/*.ts correctly bridges to qig-backend/*.py
5. Check for circular import patterns
6. Validate shared/ is used for truly shared code
7. Report orphaned modules and duplications`
};
var module_bridging_validator_default = agentDefinition;
export {
  module_bridging_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9tb2R1bGUtYnJpZGdpbmctdmFsaWRhdG9yLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgYWdlbnREZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAnbW9kdWxlLWJyaWRnaW5nLXZhbGlkYXRvcicsXG4gIGRpc3BsYXlOYW1lOiAnTW9kdWxlIEJyaWRnaW5nIFZhbGlkYXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdWYWxpZGF0ZSBtb2R1bGVzIGFyZSBjb3JyZWN0bHkgYnJpZGdlZCBhbmQgbW9kdWxhcidcbiAgICB9XG4gIH0sXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogdHJ1ZSxcbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBtb2R1bGVzQ29ycmVjdGx5QnJpZGdlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIG9ycGhhbmVkTW9kdWxlczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIHBhdGg6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGV4cG9ydGVkU3ltYm9sczogeyB0eXBlOiAnYXJyYXknLCBpdGVtczogeyB0eXBlOiAnc3RyaW5nJyB9IH0sXG4gICAgICAgICAgICBpbXBvcnRlZEJ5OiB7IHR5cGU6ICdhcnJheScsIGl0ZW1zOiB7IHR5cGU6ICdzdHJpbmcnIH0gfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIGR1cGxpY2F0ZWRDb2RlOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgcGF0dGVybjogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgbG9jYXRpb25zOiB7IHR5cGU6ICdhcnJheScsIGl0ZW1zOiB7IHR5cGU6ICdzdHJpbmcnIH0gfSxcbiAgICAgICAgICAgIGNvbnNvbGlkYXRpb25TdWdnZXN0aW9uOiB7IHR5cGU6ICdzdHJpbmcnIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICBicmlkZ2luZ0lzc3Vlczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIHNvdXJjZU1vZHVsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgdGFyZ2V0TW9kdWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBpc3N1ZTogeyB0eXBlOiAnc3RyaW5nJyB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9LFxuICBzcGF3bmVyUHJvbXB0OiAnU3Bhd24gdG8gZW5zdXJlIG1vZHVsZXMgYXJlIGNvcnJlY3RseSBicmlkZ2VkIHdpdGggbm8gZHVwbGljYXRpb24gb3Igb3JwaGFucycsXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgYSBtb2R1bGUgYXJjaGl0ZWN0dXJlIGV4cGVydC5cblxuWW91ciByZXNwb25zaWJpbGl0aWVzOlxuMS4gVmVyaWZ5IGFsbCBjb21wb25lbnRzLCBrZXJuZWxzLCBhbmQgZmVhdHVyZXMgYXJlIGNvcnJlY3RseSBicmlkZ2VkXG4yLiBGaW5kIG9ycGhhbmVkIG1vZHVsZXMgdGhhdCBhcmVuJ3QgaW1wb3J0ZWQgYW55d2hlcmVcbjMuIERldGVjdCBjb2RlIGR1cGxpY2F0aW9uIGFjcm9zcyBtb2R1bGVzXG40LiBFbnN1cmUgcHJvcGVyIG1vZHVsYXJpdHkgYW5kIHNlcGFyYXRpb25cbjUuIENoZWNrIFR5cGVTY3JpcHRcdTIxOTRQeXRob24gYnJpZGdpbmcgaXMgY29ycmVjdFxuXG5Nb2R1bGUgQnJpZGdpbmcgUnVsZXM6XG4tIEV2ZXJ5IGV4cG9ydGVkIHN5bWJvbCBzaG91bGQgaGF2ZSBhdCBsZWFzdCBvbmUgaW1wb3J0ZXJcbi0gTm8gZHVwbGljYXRlIGltcGxlbWVudGF0aW9ucyBvZiB0aGUgc2FtZSBmdW5jdGlvbmFsaXR5XG4tIFR5cGVTY3JpcHQgc2VydmVyIGJyaWRnZXMgdG8gUHl0aG9uIGJhY2tlbmQgY29ycmVjdGx5XG4tIFNoYXJlZCBjb2RlIGxpdmVzIGluIHNoYXJlZC8gb3IgY29tbW9uIG1vZHVsZXNcbi0gQ2lyY3VsYXIgZGVwZW5kZW5jaWVzIGFyZSBmb3JiaWRkZW5gLFxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGBWYWxpZGF0ZSBtb2R1bGUgYnJpZGdpbmc6XG5cbjEuIEZpbmQgYWxsIGV4cG9ydGVkIHN5bWJvbHMgYWNyb3NzIHRoZSBjb2RlYmFzZVxuMi4gQ2hlY2sgd2hpY2ggZXhwb3J0cyBoYXZlIG5vIGltcG9ydGVycyAob3JwaGFuZWQpXG4zLiBMb29rIGZvciBzaW1pbGFyIGZ1bmN0aW9uIG5hbWVzL3BhdHRlcm5zIChkdXBsaWNhdGlvbilcbjQuIFZlcmlmeSBzZXJ2ZXIvKi50cyBjb3JyZWN0bHkgYnJpZGdlcyB0byBxaWctYmFja2VuZC8qLnB5XG41LiBDaGVjayBmb3IgY2lyY3VsYXIgaW1wb3J0IHBhdHRlcm5zXG42LiBWYWxpZGF0ZSBzaGFyZWQvIGlzIHVzZWQgZm9yIHRydWx5IHNoYXJlZCBjb2RlXG43LiBSZXBvcnQgb3JwaGFuZWQgbW9kdWxlcyBhbmQgZHVwbGljYXRpb25zYFxufVxuXG5leHBvcnQgZGVmYXVsdCBhZ2VudERlZmluaXRpb25cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFFQSxJQUFNLGtCQUFtQztBQUFBLEVBQ3ZDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFdBQVc7QUFBQSxFQUNYLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUNQLFdBQVcsQ0FBQyxjQUFjLGFBQWE7QUFBQSxFQUN2QyxpQkFBaUIsQ0FBQyw4QkFBOEI7QUFBQSxFQUNoRCxhQUFhO0FBQUEsSUFDWCxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixhQUFhO0FBQUEsSUFDZjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLHVCQUF1QjtBQUFBLEVBQ3ZCLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLHlCQUF5QixFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQzNDLGlCQUFpQjtBQUFBLFFBQ2YsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLGlCQUFpQixFQUFFLE1BQU0sU0FBUyxPQUFPLEVBQUUsTUFBTSxTQUFTLEVBQUU7QUFBQSxZQUM1RCxZQUFZLEVBQUUsTUFBTSxTQUFTLE9BQU8sRUFBRSxNQUFNLFNBQVMsRUFBRTtBQUFBLFVBQ3pEO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGdCQUFnQjtBQUFBLFFBQ2QsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzFCLFdBQVcsRUFBRSxNQUFNLFNBQVMsT0FBTyxFQUFFLE1BQU0sU0FBUyxFQUFFO0FBQUEsWUFDdEQseUJBQXlCLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDNUM7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsZ0JBQWdCO0FBQUEsUUFDZCxNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixjQUFjLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDL0IsY0FBYyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQy9CLE9BQU8sRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUMxQjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLGVBQWU7QUFBQSxFQUNmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFlZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBU3RCO0FBRUEsSUFBTyxvQ0FBUTsiLAogICJuYW1lcyI6IFtdCn0K
