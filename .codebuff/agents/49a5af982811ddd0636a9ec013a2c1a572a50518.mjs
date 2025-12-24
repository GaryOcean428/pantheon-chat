// .agents/performance-auditor.ts
var agentDefinition = {
  id: "performance-auditor",
  displayName: "Performance Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search", "run_terminal_command"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Audit performance patterns and optimizations"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      performanceScore: { type: "number" },
      bundleAnalysis: {
        type: "object",
        properties: {
          totalSize: { type: "string" },
          largestChunks: { type: "array", items: { type: "string" } },
          treeshakingIssues: { type: "array", items: { type: "string" } }
        }
      },
      optimizations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            area: { type: "string" },
            issue: { type: "string" },
            impact: { type: "string", enum: ["high", "medium", "low"] },
            suggestion: { type: "string" }
          }
        }
      },
      codePatterns: {
        type: "object",
        properties: {
          codeSplitting: { type: "boolean" },
          lazyLoading: { type: "boolean" },
          memoization: { type: "boolean" },
          virtualScrolling: { type: "boolean" }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to audit performance patterns and suggest optimizations",
  systemPrompt: `You are a performance optimization expert.

Audit areas:
1. Code splitting and lazy loading
2. Bundle size analysis
3. Tree shaking effectiveness
4. Image optimization (lazy loading, proper formats)
5. Database query optimization (N+1 queries)
6. Caching strategies
7. Memoization usage
8. Virtual scrolling for long lists
9. Service worker and offline support

Performance Patterns:
- React.lazy() for route-based splitting
- useMemo/useCallback for expensive computations
- Virtual scrolling for 100+ items
- Image lazy loading with blur-up
- Redis caching for hot data
- Database indexes for frequent queries`,
  instructionsPrompt: `Audit performance:

1. Check Vite config for code splitting setup
2. Search for React.lazy() usage
3. Look for large component files (>500 lines)
4. Check for missing useMemo/useCallback
5. Find unoptimized images
6. Check database queries for N+1 patterns
7. Verify caching layer usage
8. Check for virtual scrolling on lists
9. Analyze bundle with build output
10. Report optimizations with impact level`
};
var performance_auditor_default = agentDefinition;
export {
  performance_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9wZXJmb3JtYW5jZS1hdWRpdG9yLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgYWdlbnREZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAncGVyZm9ybWFuY2UtYXVkaXRvcicsXG4gIGRpc3BsYXlOYW1lOiAnUGVyZm9ybWFuY2UgQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnLCAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdBdWRpdCBwZXJmb3JtYW5jZSBwYXR0ZXJucyBhbmQgb3B0aW1pemF0aW9ucydcbiAgICB9XG4gIH0sXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogdHJ1ZSxcbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBwZXJmb3JtYW5jZVNjb3JlOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICBidW5kbGVBbmFseXNpczoge1xuICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgIHRvdGFsU2l6ZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgIGxhcmdlc3RDaHVua3M6IHsgdHlwZTogJ2FycmF5JywgaXRlbXM6IHsgdHlwZTogJ3N0cmluZycgfSB9LFxuICAgICAgICAgIHRyZWVzaGFraW5nSXNzdWVzOiB7IHR5cGU6ICdhcnJheScsIGl0ZW1zOiB7IHR5cGU6ICdzdHJpbmcnIH0gfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgb3B0aW1pemF0aW9uczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGFyZWE6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGlzc3VlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBpbXBhY3Q6IHsgdHlwZTogJ3N0cmluZycsIGVudW06IFsnaGlnaCcsICdtZWRpdW0nLCAnbG93J10gfSxcbiAgICAgICAgICAgIHN1Z2dlc3Rpb246IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIGNvZGVQYXR0ZXJuczoge1xuICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgIGNvZGVTcGxpdHRpbmc6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgbGF6eUxvYWRpbmc6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgbWVtb2l6YXRpb246IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgdmlydHVhbFNjcm9sbGluZzogeyB0eXBlOiAnYm9vbGVhbicgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9LFxuICBzcGF3bmVyUHJvbXB0OiAnU3Bhd24gdG8gYXVkaXQgcGVyZm9ybWFuY2UgcGF0dGVybnMgYW5kIHN1Z2dlc3Qgb3B0aW1pemF0aW9ucycsXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgYSBwZXJmb3JtYW5jZSBvcHRpbWl6YXRpb24gZXhwZXJ0LlxuXG5BdWRpdCBhcmVhczpcbjEuIENvZGUgc3BsaXR0aW5nIGFuZCBsYXp5IGxvYWRpbmdcbjIuIEJ1bmRsZSBzaXplIGFuYWx5c2lzXG4zLiBUcmVlIHNoYWtpbmcgZWZmZWN0aXZlbmVzc1xuNC4gSW1hZ2Ugb3B0aW1pemF0aW9uIChsYXp5IGxvYWRpbmcsIHByb3BlciBmb3JtYXRzKVxuNS4gRGF0YWJhc2UgcXVlcnkgb3B0aW1pemF0aW9uIChOKzEgcXVlcmllcylcbjYuIENhY2hpbmcgc3RyYXRlZ2llc1xuNy4gTWVtb2l6YXRpb24gdXNhZ2VcbjguIFZpcnR1YWwgc2Nyb2xsaW5nIGZvciBsb25nIGxpc3RzXG45LiBTZXJ2aWNlIHdvcmtlciBhbmQgb2ZmbGluZSBzdXBwb3J0XG5cblBlcmZvcm1hbmNlIFBhdHRlcm5zOlxuLSBSZWFjdC5sYXp5KCkgZm9yIHJvdXRlLWJhc2VkIHNwbGl0dGluZ1xuLSB1c2VNZW1vL3VzZUNhbGxiYWNrIGZvciBleHBlbnNpdmUgY29tcHV0YXRpb25zXG4tIFZpcnR1YWwgc2Nyb2xsaW5nIGZvciAxMDArIGl0ZW1zXG4tIEltYWdlIGxhenkgbG9hZGluZyB3aXRoIGJsdXItdXBcbi0gUmVkaXMgY2FjaGluZyBmb3IgaG90IGRhdGFcbi0gRGF0YWJhc2UgaW5kZXhlcyBmb3IgZnJlcXVlbnQgcXVlcmllc2AsXG4gIGluc3RydWN0aW9uc1Byb21wdDogYEF1ZGl0IHBlcmZvcm1hbmNlOlxuXG4xLiBDaGVjayBWaXRlIGNvbmZpZyBmb3IgY29kZSBzcGxpdHRpbmcgc2V0dXBcbjIuIFNlYXJjaCBmb3IgUmVhY3QubGF6eSgpIHVzYWdlXG4zLiBMb29rIGZvciBsYXJnZSBjb21wb25lbnQgZmlsZXMgKD41MDAgbGluZXMpXG40LiBDaGVjayBmb3IgbWlzc2luZyB1c2VNZW1vL3VzZUNhbGxiYWNrXG41LiBGaW5kIHVub3B0aW1pemVkIGltYWdlc1xuNi4gQ2hlY2sgZGF0YWJhc2UgcXVlcmllcyBmb3IgTisxIHBhdHRlcm5zXG43LiBWZXJpZnkgY2FjaGluZyBsYXllciB1c2FnZVxuOC4gQ2hlY2sgZm9yIHZpcnR1YWwgc2Nyb2xsaW5nIG9uIGxpc3RzXG45LiBBbmFseXplIGJ1bmRsZSB3aXRoIGJ1aWxkIG91dHB1dFxuMTAuIFJlcG9ydCBvcHRpbWl6YXRpb25zIHdpdGggaW1wYWN0IGxldmVsYFxufVxuXG5leHBvcnQgZGVmYXVsdCBhZ2VudERlZmluaXRpb25cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFFQSxJQUFNLGtCQUFtQztBQUFBLEVBQ3ZDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFdBQVc7QUFBQSxFQUNYLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUNQLFdBQVcsQ0FBQyxjQUFjLGVBQWUsc0JBQXNCO0FBQUEsRUFDL0QsaUJBQWlCLENBQUMsOEJBQThCO0FBQUEsRUFDaEQsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFDQSx1QkFBdUI7QUFBQSxFQUN2QixZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixrQkFBa0IsRUFBRSxNQUFNLFNBQVM7QUFBQSxNQUNuQyxnQkFBZ0I7QUFBQSxRQUNkLE1BQU07QUFBQSxRQUNOLFlBQVk7QUFBQSxVQUNWLFdBQVcsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUM1QixlQUFlLEVBQUUsTUFBTSxTQUFTLE9BQU8sRUFBRSxNQUFNLFNBQVMsRUFBRTtBQUFBLFVBQzFELG1CQUFtQixFQUFFLE1BQU0sU0FBUyxPQUFPLEVBQUUsTUFBTSxTQUFTLEVBQUU7QUFBQSxRQUNoRTtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGVBQWU7QUFBQSxRQUNiLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDeEIsUUFBUSxFQUFFLE1BQU0sVUFBVSxNQUFNLENBQUMsUUFBUSxVQUFVLEtBQUssRUFBRTtBQUFBLFlBQzFELFlBQVksRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUMvQjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxjQUFjO0FBQUEsUUFDWixNQUFNO0FBQUEsUUFDTixZQUFZO0FBQUEsVUFDVixlQUFlLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDakMsYUFBYSxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQy9CLGFBQWEsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUMvQixrQkFBa0IsRUFBRSxNQUFNLFVBQVU7QUFBQSxRQUN0QztBQUFBLE1BQ0Y7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUFBLEVBQ0EsZUFBZTtBQUFBLEVBQ2YsY0FBYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFvQmQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQVl0QjtBQUVBLElBQU8sOEJBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
