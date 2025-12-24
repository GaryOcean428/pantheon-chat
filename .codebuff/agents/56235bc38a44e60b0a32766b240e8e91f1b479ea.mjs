// .agents/state-management-auditor.ts
var agentDefinition = {
  id: "state-management-auditor",
  displayName: "State Management Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Audit state management patterns"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      stateManagementHealthy: { type: "boolean" },
      patterns: {
        type: "object",
        properties: {
          globalState: { type: "string" },
          serverState: { type: "string" },
          formState: { type: "string" },
          urlState: { type: "boolean" }
        }
      },
      issues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            component: { type: "string" },
            issue: { type: "string" },
            pattern: { type: "string" },
            suggestion: { type: "string" }
          }
        }
      },
      optimizations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            area: { type: "string" },
            current: { type: "string" },
            recommended: { type: "string" }
          }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to audit state management patterns and optimizations",
  systemPrompt: `You are a state management expert.

Audit areas:
1. Context usage and optimization
2. Global state management (Zustand, Redux)
3. Server state (React Query, SWR)
4. Form state management
5. URL state synchronization
6. State machine usage for complex flows

State Management Rules:
- Split contexts by update frequency
- Use server state libraries for API data
- Form state should use dedicated libraries
- URL should reflect important state
- Avoid prop drilling >3 levels
- State machines for complex workflows`,
  instructionsPrompt: `Audit state management:

1. Find all Context definitions
2. Check for state management libraries
3. Look for prop drilling patterns
4. Check URL state synchronization
5. Find complex state that needs machines
6. Check for unnecessary re-renders
7. Report issues and optimizations`
};
var state_management_auditor_default = agentDefinition;
export {
  state_management_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9zdGF0ZS1tYW5hZ2VtZW50LWF1ZGl0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBhZ2VudERlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdzdGF0ZS1tYW5hZ2VtZW50LWF1ZGl0b3InLFxuICBkaXNwbGF5TmFtZTogJ1N0YXRlIE1hbmFnZW1lbnQgQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdBdWRpdCBzdGF0ZSBtYW5hZ2VtZW50IHBhdHRlcm5zJ1xuICAgIH1cbiAgfSxcbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiB0cnVlLFxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHN0YXRlTWFuYWdlbWVudEhlYWx0aHk6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBwYXR0ZXJuczoge1xuICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgIGdsb2JhbFN0YXRlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgc2VydmVyU3RhdGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICBmb3JtU3RhdGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICB1cmxTdGF0ZTogeyB0eXBlOiAnYm9vbGVhbicgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgaXNzdWVzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgY29tcG9uZW50OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBpc3N1ZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgcGF0dGVybjogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgc3VnZ2VzdGlvbjogeyB0eXBlOiAnc3RyaW5nJyB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgb3B0aW1pemF0aW9uczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGFyZWE6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGN1cnJlbnQ6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHJlY29tbWVuZGVkOiB7IHR5cGU6ICdzdHJpbmcnIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH0sXG4gIHNwYXduZXJQcm9tcHQ6ICdTcGF3biB0byBhdWRpdCBzdGF0ZSBtYW5hZ2VtZW50IHBhdHRlcm5zIGFuZCBvcHRpbWl6YXRpb25zJyxcbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSBhIHN0YXRlIG1hbmFnZW1lbnQgZXhwZXJ0LlxuXG5BdWRpdCBhcmVhczpcbjEuIENvbnRleHQgdXNhZ2UgYW5kIG9wdGltaXphdGlvblxuMi4gR2xvYmFsIHN0YXRlIG1hbmFnZW1lbnQgKFp1c3RhbmQsIFJlZHV4KVxuMy4gU2VydmVyIHN0YXRlIChSZWFjdCBRdWVyeSwgU1dSKVxuNC4gRm9ybSBzdGF0ZSBtYW5hZ2VtZW50XG41LiBVUkwgc3RhdGUgc3luY2hyb25pemF0aW9uXG42LiBTdGF0ZSBtYWNoaW5lIHVzYWdlIGZvciBjb21wbGV4IGZsb3dzXG5cblN0YXRlIE1hbmFnZW1lbnQgUnVsZXM6XG4tIFNwbGl0IGNvbnRleHRzIGJ5IHVwZGF0ZSBmcmVxdWVuY3lcbi0gVXNlIHNlcnZlciBzdGF0ZSBsaWJyYXJpZXMgZm9yIEFQSSBkYXRhXG4tIEZvcm0gc3RhdGUgc2hvdWxkIHVzZSBkZWRpY2F0ZWQgbGlicmFyaWVzXG4tIFVSTCBzaG91bGQgcmVmbGVjdCBpbXBvcnRhbnQgc3RhdGVcbi0gQXZvaWQgcHJvcCBkcmlsbGluZyA+MyBsZXZlbHNcbi0gU3RhdGUgbWFjaGluZXMgZm9yIGNvbXBsZXggd29ya2Zsb3dzYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgQXVkaXQgc3RhdGUgbWFuYWdlbWVudDpcblxuMS4gRmluZCBhbGwgQ29udGV4dCBkZWZpbml0aW9uc1xuMi4gQ2hlY2sgZm9yIHN0YXRlIG1hbmFnZW1lbnQgbGlicmFyaWVzXG4zLiBMb29rIGZvciBwcm9wIGRyaWxsaW5nIHBhdHRlcm5zXG40LiBDaGVjayBVUkwgc3RhdGUgc3luY2hyb25pemF0aW9uXG41LiBGaW5kIGNvbXBsZXggc3RhdGUgdGhhdCBuZWVkcyBtYWNoaW5lc1xuNi4gQ2hlY2sgZm9yIHVubmVjZXNzYXJ5IHJlLXJlbmRlcnNcbjcuIFJlcG9ydCBpc3N1ZXMgYW5kIG9wdGltaXphdGlvbnNgXG59XG5cbmV4cG9ydCBkZWZhdWx0IGFnZW50RGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sa0JBQW1DO0FBQUEsRUFDdkMsSUFBSTtBQUFBLEVBQ0osYUFBYTtBQUFBLEVBQ2IsV0FBVztBQUFBLEVBQ1gsU0FBUztBQUFBLEVBQ1QsT0FBTztBQUFBLEVBQ1AsV0FBVyxDQUFDLGNBQWMsYUFBYTtBQUFBLEVBQ3ZDLGlCQUFpQixDQUFDLDhCQUE4QjtBQUFBLEVBQ2hELGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsRUFDRjtBQUFBLEVBQ0EsdUJBQXVCO0FBQUEsRUFDdkIsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1Ysd0JBQXdCLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDMUMsVUFBVTtBQUFBLFFBQ1IsTUFBTTtBQUFBLFFBQ04sWUFBWTtBQUFBLFVBQ1YsYUFBYSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQzlCLGFBQWEsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUM5QixXQUFXLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDNUIsVUFBVSxFQUFFLE1BQU0sVUFBVTtBQUFBLFFBQzlCO0FBQUEsTUFDRjtBQUFBLE1BQ0EsUUFBUTtBQUFBLFFBQ04sTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsV0FBVyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzVCLE9BQU8sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN4QixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDMUIsWUFBWSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQy9CO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGVBQWU7QUFBQSxRQUNiLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDMUIsYUFBYSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQ2hDO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUFBLEVBQ0EsZUFBZTtBQUFBLEVBQ2YsY0FBYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFpQmQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQVN0QjtBQUVBLElBQU8sbUNBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
