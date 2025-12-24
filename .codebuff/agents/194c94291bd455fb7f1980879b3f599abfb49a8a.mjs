// .agents/accessibility-auditor.ts
var agentDefinition = {
  id: "accessibility-auditor",
  displayName: "Accessibility Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Audit accessibility (a11y) compliance"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      wcagLevel: { type: "string", enum: ["none", "A", "AA", "AAA"] },
      issues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            component: { type: "string" },
            issue: { type: "string" },
            wcagCriteria: { type: "string" },
            severity: { type: "string", enum: ["critical", "serious", "moderate", "minor"] },
            fix: { type: "string" }
          }
        }
      },
      checklist: {
        type: "object",
        properties: {
          ariaLabels: { type: "boolean" },
          keyboardNav: { type: "boolean" },
          focusManagement: { type: "boolean" },
          colorContrast: { type: "boolean" },
          altText: { type: "boolean" },
          skipLinks: { type: "boolean" },
          motionPreferences: { type: "boolean" },
          textScaling: { type: "boolean" }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to audit accessibility compliance and WCAG conformance",
  systemPrompt: `You are an accessibility (a11y) expert.

Audit for WCAG 2.1 AA compliance:
1. ARIA labels and roles
2. Keyboard navigation (Tab, Enter, Escape)
3. Focus management and visible focus states
4. Color contrast ratios (4.5:1 normal, 3:1 large text)
5. Alternative text for images
6. Skip navigation links
7. Motion preferences (prefers-reduced-motion)
8. Text scaling support (up to 200%)
9. Form labels and error messages
10. Screen reader compatibility

Common Issues:
- Missing aria-label on icon buttons
- No visible focus indicator
- Non-semantic HTML (div instead of button)
- Missing form labels
- Color-only information
- Auto-playing media
- Keyboard traps in modals`,
  instructionsPrompt: `Audit accessibility:

1. Search for buttons without aria-label
2. Check for onClick on non-button elements
3. Look for images missing alt text
4. Check form inputs for labels
5. Verify focus trap in modals
6. Check for prefers-reduced-motion usage
7. Look for color-only information conveyance
8. Check heading hierarchy (h1, h2, h3)
9. Verify skip navigation link exists
10. Report all issues with WCAG criteria and fixes`
};
var accessibility_auditor_default = agentDefinition;
export {
  accessibility_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9hY2Nlc3NpYmlsaXR5LWF1ZGl0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBhZ2VudERlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdhY2Nlc3NpYmlsaXR5LWF1ZGl0b3InLFxuICBkaXNwbGF5TmFtZTogJ0FjY2Vzc2liaWxpdHkgQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdBdWRpdCBhY2Nlc3NpYmlsaXR5IChhMTF5KSBjb21wbGlhbmNlJ1xuICAgIH1cbiAgfSxcbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiB0cnVlLFxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHdjYWdMZXZlbDogeyB0eXBlOiAnc3RyaW5nJywgZW51bTogWydub25lJywgJ0EnLCAnQUEnLCAnQUFBJ10gfSxcbiAgICAgIGlzc3Vlczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGNvbXBvbmVudDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgaXNzdWU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHdjYWdDcml0ZXJpYTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgc2V2ZXJpdHk6IHsgdHlwZTogJ3N0cmluZycsIGVudW06IFsnY3JpdGljYWwnLCAnc2VyaW91cycsICdtb2RlcmF0ZScsICdtaW5vciddIH0sXG4gICAgICAgICAgICBmaXg6IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIGNoZWNrbGlzdDoge1xuICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgIGFyaWFMYWJlbHM6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAga2V5Ym9hcmROYXY6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgZm9jdXNNYW5hZ2VtZW50OiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIGNvbG9yQ29udHJhc3Q6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgYWx0VGV4dDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBza2lwTGlua3M6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgbW90aW9uUHJlZmVyZW5jZXM6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgdGV4dFNjYWxpbmc6IHsgdHlwZTogJ2Jvb2xlYW4nIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfSxcbiAgc3Bhd25lclByb21wdDogJ1NwYXduIHRvIGF1ZGl0IGFjY2Vzc2liaWxpdHkgY29tcGxpYW5jZSBhbmQgV0NBRyBjb25mb3JtYW5jZScsXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgYW4gYWNjZXNzaWJpbGl0eSAoYTExeSkgZXhwZXJ0LlxuXG5BdWRpdCBmb3IgV0NBRyAyLjEgQUEgY29tcGxpYW5jZTpcbjEuIEFSSUEgbGFiZWxzIGFuZCByb2xlc1xuMi4gS2V5Ym9hcmQgbmF2aWdhdGlvbiAoVGFiLCBFbnRlciwgRXNjYXBlKVxuMy4gRm9jdXMgbWFuYWdlbWVudCBhbmQgdmlzaWJsZSBmb2N1cyBzdGF0ZXNcbjQuIENvbG9yIGNvbnRyYXN0IHJhdGlvcyAoNC41OjEgbm9ybWFsLCAzOjEgbGFyZ2UgdGV4dClcbjUuIEFsdGVybmF0aXZlIHRleHQgZm9yIGltYWdlc1xuNi4gU2tpcCBuYXZpZ2F0aW9uIGxpbmtzXG43LiBNb3Rpb24gcHJlZmVyZW5jZXMgKHByZWZlcnMtcmVkdWNlZC1tb3Rpb24pXG44LiBUZXh0IHNjYWxpbmcgc3VwcG9ydCAodXAgdG8gMjAwJSlcbjkuIEZvcm0gbGFiZWxzIGFuZCBlcnJvciBtZXNzYWdlc1xuMTAuIFNjcmVlbiByZWFkZXIgY29tcGF0aWJpbGl0eVxuXG5Db21tb24gSXNzdWVzOlxuLSBNaXNzaW5nIGFyaWEtbGFiZWwgb24gaWNvbiBidXR0b25zXG4tIE5vIHZpc2libGUgZm9jdXMgaW5kaWNhdG9yXG4tIE5vbi1zZW1hbnRpYyBIVE1MIChkaXYgaW5zdGVhZCBvZiBidXR0b24pXG4tIE1pc3NpbmcgZm9ybSBsYWJlbHNcbi0gQ29sb3Itb25seSBpbmZvcm1hdGlvblxuLSBBdXRvLXBsYXlpbmcgbWVkaWFcbi0gS2V5Ym9hcmQgdHJhcHMgaW4gbW9kYWxzYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgQXVkaXQgYWNjZXNzaWJpbGl0eTpcblxuMS4gU2VhcmNoIGZvciBidXR0b25zIHdpdGhvdXQgYXJpYS1sYWJlbFxuMi4gQ2hlY2sgZm9yIG9uQ2xpY2sgb24gbm9uLWJ1dHRvbiBlbGVtZW50c1xuMy4gTG9vayBmb3IgaW1hZ2VzIG1pc3NpbmcgYWx0IHRleHRcbjQuIENoZWNrIGZvcm0gaW5wdXRzIGZvciBsYWJlbHNcbjUuIFZlcmlmeSBmb2N1cyB0cmFwIGluIG1vZGFsc1xuNi4gQ2hlY2sgZm9yIHByZWZlcnMtcmVkdWNlZC1tb3Rpb24gdXNhZ2VcbjcuIExvb2sgZm9yIGNvbG9yLW9ubHkgaW5mb3JtYXRpb24gY29udmV5YW5jZVxuOC4gQ2hlY2sgaGVhZGluZyBoaWVyYXJjaHkgKGgxLCBoMiwgaDMpXG45LiBWZXJpZnkgc2tpcCBuYXZpZ2F0aW9uIGxpbmsgZXhpc3RzXG4xMC4gUmVwb3J0IGFsbCBpc3N1ZXMgd2l0aCBXQ0FHIGNyaXRlcmlhIGFuZCBmaXhlc2Bcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyxhQUFhO0FBQUEsRUFDdkMsaUJBQWlCLENBQUMsOEJBQThCO0FBQUEsRUFDaEQsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFDQSx1QkFBdUI7QUFBQSxFQUN2QixZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixXQUFXLEVBQUUsTUFBTSxVQUFVLE1BQU0sQ0FBQyxRQUFRLEtBQUssTUFBTSxLQUFLLEVBQUU7QUFBQSxNQUM5RCxRQUFRO0FBQUEsUUFDTixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixXQUFXLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDNUIsT0FBTyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3hCLGNBQWMsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUMvQixVQUFVLEVBQUUsTUFBTSxVQUFVLE1BQU0sQ0FBQyxZQUFZLFdBQVcsWUFBWSxPQUFPLEVBQUU7QUFBQSxZQUMvRSxLQUFLLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDeEI7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsV0FBVztBQUFBLFFBQ1QsTUFBTTtBQUFBLFFBQ04sWUFBWTtBQUFBLFVBQ1YsWUFBWSxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQzlCLGFBQWEsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUMvQixpQkFBaUIsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUNuQyxlQUFlLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDakMsU0FBUyxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQzNCLFdBQVcsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUM3QixtQkFBbUIsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUNyQyxhQUFhLEVBQUUsTUFBTSxVQUFVO0FBQUEsUUFDakM7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLGVBQWU7QUFBQSxFQUNmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQXNCZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBWXRCO0FBRUEsSUFBTyxnQ0FBUTsiLAogICJuYW1lcyI6IFtdCn0K
