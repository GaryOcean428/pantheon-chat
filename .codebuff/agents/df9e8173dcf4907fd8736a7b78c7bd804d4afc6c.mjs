// .agents/component-architecture-auditor.ts
var agentDefinition = {
  id: "component-architecture-auditor",
  displayName: "Component Architecture Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Audit component architecture patterns"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      architectureHealthy: { type: "boolean" },
      patterns: {
        type: "object",
        properties: {
          compoundComponents: { type: "boolean" },
          renderProps: { type: "boolean" },
          hocs: { type: "boolean" },
          headlessUi: { type: "boolean" },
          polymorphic: { type: "boolean" }
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
      componentMetrics: {
        type: "object",
        properties: {
          totalComponents: { type: "number" },
          averageSize: { type: "number" },
          largestComponents: { type: "array", items: { type: "string" } }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to audit React component architecture patterns",
  systemPrompt: `You are a React component architecture expert.

Audit areas:
1. Compound component patterns
2. Render props usage
3. HOC patterns for cross-cutting concerns
4. Headless UI separation
5. Polymorphic components (as prop)
6. Slot patterns for flexible layouts
7. Component composition vs inheritance

Architecture Rules:
- Prefer composition over inheritance
- Separate logic from presentation (headless)
- Use compound components for related UI
- HOCs for authentication, analytics
- Polymorphic for flexible rendering
- Keep components focused and small`,
  instructionsPrompt: `Audit component architecture:

1. Find all component definitions
2. Check for large components (>200 lines)
3. Look for compound component patterns
4. Check for render props usage
5. Find HOC patterns
6. Look for tightly coupled components
7. Check component prop count
8. Report architecture issues`
};
var component_architecture_auditor_default = agentDefinition;
export {
  component_architecture_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9jb21wb25lbnQtYXJjaGl0ZWN0dXJlLWF1ZGl0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBhZ2VudERlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdjb21wb25lbnQtYXJjaGl0ZWN0dXJlLWF1ZGl0b3InLFxuICBkaXNwbGF5TmFtZTogJ0NvbXBvbmVudCBBcmNoaXRlY3R1cmUgQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdBdWRpdCBjb21wb25lbnQgYXJjaGl0ZWN0dXJlIHBhdHRlcm5zJ1xuICAgIH1cbiAgfSxcbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiB0cnVlLFxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIGFyY2hpdGVjdHVyZUhlYWx0aHk6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBwYXR0ZXJuczoge1xuICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgIGNvbXBvdW5kQ29tcG9uZW50czogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICByZW5kZXJQcm9wczogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBob2NzOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIGhlYWRsZXNzVWk6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgcG9seW1vcnBoaWM6IHsgdHlwZTogJ2Jvb2xlYW4nIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIGlzc3Vlczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGNvbXBvbmVudDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgaXNzdWU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHBhdHRlcm46IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHN1Z2dlc3Rpb246IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIGNvbXBvbmVudE1ldHJpY3M6IHtcbiAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICB0b3RhbENvbXBvbmVudHM6IHsgdHlwZTogJ251bWJlcicgfSxcbiAgICAgICAgICBhdmVyYWdlU2l6ZTogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgICAgIGxhcmdlc3RDb21wb25lbnRzOiB7IHR5cGU6ICdhcnJheScsIGl0ZW1zOiB7IHR5cGU6ICdzdHJpbmcnIH0gfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9LFxuICBzcGF3bmVyUHJvbXB0OiAnU3Bhd24gdG8gYXVkaXQgUmVhY3QgY29tcG9uZW50IGFyY2hpdGVjdHVyZSBwYXR0ZXJucycsXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgYSBSZWFjdCBjb21wb25lbnQgYXJjaGl0ZWN0dXJlIGV4cGVydC5cblxuQXVkaXQgYXJlYXM6XG4xLiBDb21wb3VuZCBjb21wb25lbnQgcGF0dGVybnNcbjIuIFJlbmRlciBwcm9wcyB1c2FnZVxuMy4gSE9DIHBhdHRlcm5zIGZvciBjcm9zcy1jdXR0aW5nIGNvbmNlcm5zXG40LiBIZWFkbGVzcyBVSSBzZXBhcmF0aW9uXG41LiBQb2x5bW9ycGhpYyBjb21wb25lbnRzIChhcyBwcm9wKVxuNi4gU2xvdCBwYXR0ZXJucyBmb3IgZmxleGlibGUgbGF5b3V0c1xuNy4gQ29tcG9uZW50IGNvbXBvc2l0aW9uIHZzIGluaGVyaXRhbmNlXG5cbkFyY2hpdGVjdHVyZSBSdWxlczpcbi0gUHJlZmVyIGNvbXBvc2l0aW9uIG92ZXIgaW5oZXJpdGFuY2Vcbi0gU2VwYXJhdGUgbG9naWMgZnJvbSBwcmVzZW50YXRpb24gKGhlYWRsZXNzKVxuLSBVc2UgY29tcG91bmQgY29tcG9uZW50cyBmb3IgcmVsYXRlZCBVSVxuLSBIT0NzIGZvciBhdXRoZW50aWNhdGlvbiwgYW5hbHl0aWNzXG4tIFBvbHltb3JwaGljIGZvciBmbGV4aWJsZSByZW5kZXJpbmdcbi0gS2VlcCBjb21wb25lbnRzIGZvY3VzZWQgYW5kIHNtYWxsYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgQXVkaXQgY29tcG9uZW50IGFyY2hpdGVjdHVyZTpcblxuMS4gRmluZCBhbGwgY29tcG9uZW50IGRlZmluaXRpb25zXG4yLiBDaGVjayBmb3IgbGFyZ2UgY29tcG9uZW50cyAoPjIwMCBsaW5lcylcbjMuIExvb2sgZm9yIGNvbXBvdW5kIGNvbXBvbmVudCBwYXR0ZXJuc1xuNC4gQ2hlY2sgZm9yIHJlbmRlciBwcm9wcyB1c2FnZVxuNS4gRmluZCBIT0MgcGF0dGVybnNcbjYuIExvb2sgZm9yIHRpZ2h0bHkgY291cGxlZCBjb21wb25lbnRzXG43LiBDaGVjayBjb21wb25lbnQgcHJvcCBjb3VudFxuOC4gUmVwb3J0IGFyY2hpdGVjdHVyZSBpc3N1ZXNgXG59XG5cbmV4cG9ydCBkZWZhdWx0IGFnZW50RGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sa0JBQW1DO0FBQUEsRUFDdkMsSUFBSTtBQUFBLEVBQ0osYUFBYTtBQUFBLEVBQ2IsV0FBVztBQUFBLEVBQ1gsU0FBUztBQUFBLEVBQ1QsT0FBTztBQUFBLEVBQ1AsV0FBVyxDQUFDLGNBQWMsYUFBYTtBQUFBLEVBQ3ZDLGlCQUFpQixDQUFDLDhCQUE4QjtBQUFBLEVBQ2hELGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsRUFDRjtBQUFBLEVBQ0EsdUJBQXVCO0FBQUEsRUFDdkIsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YscUJBQXFCLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDdkMsVUFBVTtBQUFBLFFBQ1IsTUFBTTtBQUFBLFFBQ04sWUFBWTtBQUFBLFVBQ1Ysb0JBQW9CLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDdEMsYUFBYSxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQy9CLE1BQU0sRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUN4QixZQUFZLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDOUIsYUFBYSxFQUFFLE1BQU0sVUFBVTtBQUFBLFFBQ2pDO0FBQUEsTUFDRjtBQUFBLE1BQ0EsUUFBUTtBQUFBLFFBQ04sTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsV0FBVyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzVCLE9BQU8sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN4QixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDMUIsWUFBWSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQy9CO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGtCQUFrQjtBQUFBLFFBQ2hCLE1BQU07QUFBQSxRQUNOLFlBQVk7QUFBQSxVQUNWLGlCQUFpQixFQUFFLE1BQU0sU0FBUztBQUFBLFVBQ2xDLGFBQWEsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUM5QixtQkFBbUIsRUFBRSxNQUFNLFNBQVMsT0FBTyxFQUFFLE1BQU0sU0FBUyxFQUFFO0FBQUEsUUFDaEU7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLGVBQWU7QUFBQSxFQUNmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFrQmQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBVXRCO0FBRUEsSUFBTyx5Q0FBUTsiLAogICJuYW1lcyI6IFtdCn0K
