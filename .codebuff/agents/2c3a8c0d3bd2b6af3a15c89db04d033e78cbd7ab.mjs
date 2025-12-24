// .agents/ui-ux-auditor.ts
var agentDefinition = {
  id: "ui-ux-auditor",
  displayName: "UI/UX Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Audit UI/UX patterns and improvements"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      designSystemConsistent: { type: "boolean" },
      missingPatterns: {
        type: "array",
        items: {
          type: "object",
          properties: {
            pattern: { type: "string" },
            description: { type: "string" },
            priority: { type: "string", enum: ["high", "medium", "low"] }
          }
        }
      },
      improvements: {
        type: "array",
        items: {
          type: "object",
          properties: {
            component: { type: "string" },
            suggestion: { type: "string" },
            category: { type: "string", enum: ["micro-interactions", "loading-states", "error-states", "empty-states", "responsive", "dark-mode", "accessibility"] }
          }
        }
      },
      mobileReadiness: {
        type: "object",
        properties: {
          responsive: { type: "boolean" },
          touchFriendly: { type: "boolean" },
          performanceOptimized: { type: "boolean" }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to audit UI/UX patterns and suggest improvements",
  systemPrompt: `You are a UI/UX expert auditor.

Audit areas:
1. Design system consistency (spacing, typography, colors)
2. Micro-interactions (hover states, transitions, animations)
3. Loading states (skeletons, spinners, optimistic updates)
4. Error states (user-friendly messages, recovery actions)
5. Empty states (illustrations, actionable CTAs)
6. Mobile responsiveness (320px to 4K)
7. Dark mode polish (contrast ratios)
8. Progressive disclosure (collapsible sections)
9. Navigation patterns (breadcrumbs, command palette)

Best Practices:
- Implement loading skeletons, not spinners
- Add hover states and transitions to all interactive elements
- Use optimistic UI updates
- Design engaging empty states with CTAs
- Ensure WCAG AA contrast ratios`,
  instructionsPrompt: `Audit UI/UX patterns:

1. Read client/src/components for existing patterns
2. Check for loading state implementations
3. Look for error boundary usage
4. Check Tailwind config for design tokens
5. Find components missing hover states
6. Check for responsive breakpoint usage
7. Audit dark mode implementation
8. Report all improvements with priority`
};
var ui_ux_auditor_default = agentDefinition;
export {
  ui_ux_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy91aS11eC1hdWRpdG9yLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgYWdlbnREZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAndWktdXgtYXVkaXRvcicsXG4gIGRpc3BsYXlOYW1lOiAnVUkvVVggQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdBdWRpdCBVSS9VWCBwYXR0ZXJucyBhbmQgaW1wcm92ZW1lbnRzJ1xuICAgIH1cbiAgfSxcbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiB0cnVlLFxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIGRlc2lnblN5c3RlbUNvbnNpc3RlbnQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBtaXNzaW5nUGF0dGVybnM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBwYXR0ZXJuOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBkZXNjcmlwdGlvbjogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgcHJpb3JpdHk6IHsgdHlwZTogJ3N0cmluZycsIGVudW06IFsnaGlnaCcsICdtZWRpdW0nLCAnbG93J10gfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIGltcHJvdmVtZW50czoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGNvbXBvbmVudDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgc3VnZ2VzdGlvbjogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgY2F0ZWdvcnk6IHsgdHlwZTogJ3N0cmluZycsIGVudW06IFsnbWljcm8taW50ZXJhY3Rpb25zJywgJ2xvYWRpbmctc3RhdGVzJywgJ2Vycm9yLXN0YXRlcycsICdlbXB0eS1zdGF0ZXMnLCAncmVzcG9uc2l2ZScsICdkYXJrLW1vZGUnLCAnYWNjZXNzaWJpbGl0eSddIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICBtb2JpbGVSZWFkaW5lc3M6IHtcbiAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICByZXNwb25zaXZlOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIHRvdWNoRnJpZW5kbHk6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgcGVyZm9ybWFuY2VPcHRpbWl6ZWQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfSxcbiAgc3Bhd25lclByb21wdDogJ1NwYXduIHRvIGF1ZGl0IFVJL1VYIHBhdHRlcm5zIGFuZCBzdWdnZXN0IGltcHJvdmVtZW50cycsXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgYSBVSS9VWCBleHBlcnQgYXVkaXRvci5cblxuQXVkaXQgYXJlYXM6XG4xLiBEZXNpZ24gc3lzdGVtIGNvbnNpc3RlbmN5IChzcGFjaW5nLCB0eXBvZ3JhcGh5LCBjb2xvcnMpXG4yLiBNaWNyby1pbnRlcmFjdGlvbnMgKGhvdmVyIHN0YXRlcywgdHJhbnNpdGlvbnMsIGFuaW1hdGlvbnMpXG4zLiBMb2FkaW5nIHN0YXRlcyAoc2tlbGV0b25zLCBzcGlubmVycywgb3B0aW1pc3RpYyB1cGRhdGVzKVxuNC4gRXJyb3Igc3RhdGVzICh1c2VyLWZyaWVuZGx5IG1lc3NhZ2VzLCByZWNvdmVyeSBhY3Rpb25zKVxuNS4gRW1wdHkgc3RhdGVzIChpbGx1c3RyYXRpb25zLCBhY3Rpb25hYmxlIENUQXMpXG42LiBNb2JpbGUgcmVzcG9uc2l2ZW5lc3MgKDMyMHB4IHRvIDRLKVxuNy4gRGFyayBtb2RlIHBvbGlzaCAoY29udHJhc3QgcmF0aW9zKVxuOC4gUHJvZ3Jlc3NpdmUgZGlzY2xvc3VyZSAoY29sbGFwc2libGUgc2VjdGlvbnMpXG45LiBOYXZpZ2F0aW9uIHBhdHRlcm5zIChicmVhZGNydW1icywgY29tbWFuZCBwYWxldHRlKVxuXG5CZXN0IFByYWN0aWNlczpcbi0gSW1wbGVtZW50IGxvYWRpbmcgc2tlbGV0b25zLCBub3Qgc3Bpbm5lcnNcbi0gQWRkIGhvdmVyIHN0YXRlcyBhbmQgdHJhbnNpdGlvbnMgdG8gYWxsIGludGVyYWN0aXZlIGVsZW1lbnRzXG4tIFVzZSBvcHRpbWlzdGljIFVJIHVwZGF0ZXNcbi0gRGVzaWduIGVuZ2FnaW5nIGVtcHR5IHN0YXRlcyB3aXRoIENUQXNcbi0gRW5zdXJlIFdDQUcgQUEgY29udHJhc3QgcmF0aW9zYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgQXVkaXQgVUkvVVggcGF0dGVybnM6XG5cbjEuIFJlYWQgY2xpZW50L3NyYy9jb21wb25lbnRzIGZvciBleGlzdGluZyBwYXR0ZXJuc1xuMi4gQ2hlY2sgZm9yIGxvYWRpbmcgc3RhdGUgaW1wbGVtZW50YXRpb25zXG4zLiBMb29rIGZvciBlcnJvciBib3VuZGFyeSB1c2FnZVxuNC4gQ2hlY2sgVGFpbHdpbmQgY29uZmlnIGZvciBkZXNpZ24gdG9rZW5zXG41LiBGaW5kIGNvbXBvbmVudHMgbWlzc2luZyBob3ZlciBzdGF0ZXNcbjYuIENoZWNrIGZvciByZXNwb25zaXZlIGJyZWFrcG9pbnQgdXNhZ2VcbjcuIEF1ZGl0IGRhcmsgbW9kZSBpbXBsZW1lbnRhdGlvblxuOC4gUmVwb3J0IGFsbCBpbXByb3ZlbWVudHMgd2l0aCBwcmlvcml0eWBcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyxhQUFhO0FBQUEsRUFDdkMsaUJBQWlCLENBQUMsOEJBQThCO0FBQUEsRUFDaEQsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFDQSx1QkFBdUI7QUFBQSxFQUN2QixZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVix3QkFBd0IsRUFBRSxNQUFNLFVBQVU7QUFBQSxNQUMxQyxpQkFBaUI7QUFBQSxRQUNmLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLFNBQVMsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUMxQixhQUFhLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDOUIsVUFBVSxFQUFFLE1BQU0sVUFBVSxNQUFNLENBQUMsUUFBUSxVQUFVLEtBQUssRUFBRTtBQUFBLFVBQzlEO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGNBQWM7QUFBQSxRQUNaLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLFdBQVcsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUM1QixZQUFZLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDN0IsVUFBVSxFQUFFLE1BQU0sVUFBVSxNQUFNLENBQUMsc0JBQXNCLGtCQUFrQixnQkFBZ0IsZ0JBQWdCLGNBQWMsYUFBYSxlQUFlLEVBQUU7QUFBQSxVQUN6SjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxpQkFBaUI7QUFBQSxRQUNmLE1BQU07QUFBQSxRQUNOLFlBQVk7QUFBQSxVQUNWLFlBQVksRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUM5QixlQUFlLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDakMsc0JBQXNCLEVBQUUsTUFBTSxVQUFVO0FBQUEsUUFDMUM7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLGVBQWU7QUFBQSxFQUNmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQW1CZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFVdEI7QUFFQSxJQUFPLHdCQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
