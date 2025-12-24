// .agents/error-handling-auditor.ts
var agentDefinition = {
  id: "error-handling-auditor",
  displayName: "Error Handling Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Audit error handling patterns"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      errorHandlingComplete: { type: "boolean" },
      patterns: {
        type: "object",
        properties: {
          errorBoundaries: { type: "boolean" },
          apiErrorHandling: { type: "boolean" },
          formValidation: { type: "boolean" },
          globalErrorHandler: { type: "boolean" },
          errorTracking: { type: "boolean" }
        }
      },
      issues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            issue: { type: "string" },
            severity: { type: "string", enum: ["critical", "high", "medium", "low"] },
            suggestion: { type: "string" }
          }
        }
      },
      swallowedErrors: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            line: { type: "number" },
            pattern: { type: "string" }
          }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to audit error handling completeness",
  systemPrompt: `You are an error handling expert.

Audit areas:
1. React Error Boundaries
2. API error handling and retries
3. Form validation errors
4. Global error handler
5. Error tracking integration (Sentry, etc.)
6. User-friendly error messages
7. Error recovery options

Error Handling Rules:
- Never swallow errors silently
- Log all errors with context
- Show user-friendly messages
- Provide recovery actions
- Use error boundaries for component trees
- Retry transient failures
- Report errors to tracking service`,
  instructionsPrompt: `Audit error handling:

1. Search for empty catch blocks
2. Find Error Boundary implementations
3. Check API call error handling
4. Look for form validation patterns
5. Check for error tracking setup
6. Find unhandled promise rejections
7. Check error message quality
8. Report gaps and improvements`
};
var error_handling_auditor_default = agentDefinition;
export {
  error_handling_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9lcnJvci1oYW5kbGluZy1hdWRpdG9yLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgYWdlbnREZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAnZXJyb3ItaGFuZGxpbmctYXVkaXRvcicsXG4gIGRpc3BsYXlOYW1lOiAnRXJyb3IgSGFuZGxpbmcgQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdBdWRpdCBlcnJvciBoYW5kbGluZyBwYXR0ZXJucydcbiAgICB9XG4gIH0sXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogdHJ1ZSxcbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBlcnJvckhhbmRsaW5nQ29tcGxldGU6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBwYXR0ZXJuczoge1xuICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgIGVycm9yQm91bmRhcmllczogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBhcGlFcnJvckhhbmRsaW5nOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIGZvcm1WYWxpZGF0aW9uOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIGdsb2JhbEVycm9ySGFuZGxlcjogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBlcnJvclRyYWNraW5nOiB7IHR5cGU6ICdib29sZWFuJyB9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICBpc3N1ZXM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBmaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBpc3N1ZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgc2V2ZXJpdHk6IHsgdHlwZTogJ3N0cmluZycsIGVudW06IFsnY3JpdGljYWwnLCAnaGlnaCcsICdtZWRpdW0nLCAnbG93J10gfSxcbiAgICAgICAgICAgIHN1Z2dlc3Rpb246IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIHN3YWxsb3dlZEVycm9yczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGZpbGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGxpbmU6IHsgdHlwZTogJ251bWJlcicgfSxcbiAgICAgICAgICAgIHBhdHRlcm46IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfSxcbiAgc3Bhd25lclByb21wdDogJ1NwYXduIHRvIGF1ZGl0IGVycm9yIGhhbmRsaW5nIGNvbXBsZXRlbmVzcycsXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgYW4gZXJyb3IgaGFuZGxpbmcgZXhwZXJ0LlxuXG5BdWRpdCBhcmVhczpcbjEuIFJlYWN0IEVycm9yIEJvdW5kYXJpZXNcbjIuIEFQSSBlcnJvciBoYW5kbGluZyBhbmQgcmV0cmllc1xuMy4gRm9ybSB2YWxpZGF0aW9uIGVycm9yc1xuNC4gR2xvYmFsIGVycm9yIGhhbmRsZXJcbjUuIEVycm9yIHRyYWNraW5nIGludGVncmF0aW9uIChTZW50cnksIGV0Yy4pXG42LiBVc2VyLWZyaWVuZGx5IGVycm9yIG1lc3NhZ2VzXG43LiBFcnJvciByZWNvdmVyeSBvcHRpb25zXG5cbkVycm9yIEhhbmRsaW5nIFJ1bGVzOlxuLSBOZXZlciBzd2FsbG93IGVycm9ycyBzaWxlbnRseVxuLSBMb2cgYWxsIGVycm9ycyB3aXRoIGNvbnRleHRcbi0gU2hvdyB1c2VyLWZyaWVuZGx5IG1lc3NhZ2VzXG4tIFByb3ZpZGUgcmVjb3ZlcnkgYWN0aW9uc1xuLSBVc2UgZXJyb3IgYm91bmRhcmllcyBmb3IgY29tcG9uZW50IHRyZWVzXG4tIFJldHJ5IHRyYW5zaWVudCBmYWlsdXJlc1xuLSBSZXBvcnQgZXJyb3JzIHRvIHRyYWNraW5nIHNlcnZpY2VgLFxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGBBdWRpdCBlcnJvciBoYW5kbGluZzpcblxuMS4gU2VhcmNoIGZvciBlbXB0eSBjYXRjaCBibG9ja3NcbjIuIEZpbmQgRXJyb3IgQm91bmRhcnkgaW1wbGVtZW50YXRpb25zXG4zLiBDaGVjayBBUEkgY2FsbCBlcnJvciBoYW5kbGluZ1xuNC4gTG9vayBmb3IgZm9ybSB2YWxpZGF0aW9uIHBhdHRlcm5zXG41LiBDaGVjayBmb3IgZXJyb3IgdHJhY2tpbmcgc2V0dXBcbjYuIEZpbmQgdW5oYW5kbGVkIHByb21pc2UgcmVqZWN0aW9uc1xuNy4gQ2hlY2sgZXJyb3IgbWVzc2FnZSBxdWFsaXR5XG44LiBSZXBvcnQgZ2FwcyBhbmQgaW1wcm92ZW1lbnRzYFxufVxuXG5leHBvcnQgZGVmYXVsdCBhZ2VudERlZmluaXRpb25cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFFQSxJQUFNLGtCQUFtQztBQUFBLEVBQ3ZDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFdBQVc7QUFBQSxFQUNYLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUNQLFdBQVcsQ0FBQyxjQUFjLGFBQWE7QUFBQSxFQUN2QyxpQkFBaUIsQ0FBQyw4QkFBOEI7QUFBQSxFQUNoRCxhQUFhO0FBQUEsSUFDWCxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixhQUFhO0FBQUEsSUFDZjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLHVCQUF1QjtBQUFBLEVBQ3ZCLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLHVCQUF1QixFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQ3pDLFVBQVU7QUFBQSxRQUNSLE1BQU07QUFBQSxRQUNOLFlBQVk7QUFBQSxVQUNWLGlCQUFpQixFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQ25DLGtCQUFrQixFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQ3BDLGdCQUFnQixFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQ2xDLG9CQUFvQixFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQ3RDLGVBQWUsRUFBRSxNQUFNLFVBQVU7QUFBQSxRQUNuQztBQUFBLE1BQ0Y7QUFBQSxNQUNBLFFBQVE7QUFBQSxRQUNOLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDeEIsVUFBVSxFQUFFLE1BQU0sVUFBVSxNQUFNLENBQUMsWUFBWSxRQUFRLFVBQVUsS0FBSyxFQUFFO0FBQUEsWUFDeEUsWUFBWSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQy9CO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGlCQUFpQjtBQUFBLFFBQ2YsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDNUI7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQUEsRUFDQSxlQUFlO0FBQUEsRUFDZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFtQmQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBVXRCO0FBRUEsSUFBTyxpQ0FBUTsiLAogICJuYW1lcyI6IFtdCn0K
