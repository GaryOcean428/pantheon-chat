// .agents/api-versioning-validator.ts
var agentDefinition = {
  id: "api-versioning-validator",
  displayName: "API Versioning Validator",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Validate API versioning and route organization"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      versioningCorrect: { type: "boolean" },
      apiRoutes: {
        type: "array",
        items: {
          type: "object",
          properties: {
            path: { type: "string" },
            version: { type: "string" },
            methods: { type: "array", items: { type: "string" } },
            documented: { type: "boolean" }
          }
        }
      },
      issues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            route: { type: "string" },
            issue: { type: "string" },
            suggestion: { type: "string" }
          }
        }
      },
      openApiSync: { type: "boolean" }
    }
  },
  spawnerPrompt: "Spawn to validate API versioning and route organization",
  systemPrompt: `You are an API design expert.

Validation areas:
1. API versioning consistency (/api/v1/, /api/v2/)
2. RESTful route naming conventions
3. HTTP method usage (GET, POST, PUT, DELETE)
4. Response format consistency
5. Error code standardization
6. OpenAPI spec synchronization
7. Rate limiting configuration
8. Authentication middleware

API Rules:
- All routes should be versioned (/api/v1/...)
- Use plural nouns for resources
- Consistent response envelope
- Standardized error codes
- OpenAPI spec must match implementation
- Internal routes use /internal/ prefix`,
  instructionsPrompt: `Validate API versioning:

1. Read server/routes.ts for route definitions
2. Check for /api/v1/ versioning pattern
3. Verify OpenAPI spec in docs/api/
4. Compare spec to actual routes
5. Check for unversioned routes
6. Verify response format consistency
7. Check rate limiting middleware
8. Report issues and suggestions`
};
var api_versioning_validator_default = agentDefinition;
export {
  api_versioning_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9hcGktdmVyc2lvbmluZy12YWxpZGF0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBhZ2VudERlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdhcGktdmVyc2lvbmluZy12YWxpZGF0b3InLFxuICBkaXNwbGF5TmFtZTogJ0FQSSBWZXJzaW9uaW5nIFZhbGlkYXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdWYWxpZGF0ZSBBUEkgdmVyc2lvbmluZyBhbmQgcm91dGUgb3JnYW5pemF0aW9uJ1xuICAgIH1cbiAgfSxcbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiB0cnVlLFxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHZlcnNpb25pbmdDb3JyZWN0OiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgYXBpUm91dGVzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgcGF0aDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgdmVyc2lvbjogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgbWV0aG9kczogeyB0eXBlOiAnYXJyYXknLCBpdGVtczogeyB0eXBlOiAnc3RyaW5nJyB9IH0sXG4gICAgICAgICAgICBkb2N1bWVudGVkOiB7IHR5cGU6ICdib29sZWFuJyB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgaXNzdWVzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgcm91dGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGlzc3VlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBzdWdnZXN0aW9uOiB7IHR5cGU6ICdzdHJpbmcnIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICBvcGVuQXBpU3luYzogeyB0eXBlOiAnYm9vbGVhbicgfVxuICAgIH1cbiAgfSxcbiAgc3Bhd25lclByb21wdDogJ1NwYXduIHRvIHZhbGlkYXRlIEFQSSB2ZXJzaW9uaW5nIGFuZCByb3V0ZSBvcmdhbml6YXRpb24nLFxuICBzeXN0ZW1Qcm9tcHQ6IGBZb3UgYXJlIGFuIEFQSSBkZXNpZ24gZXhwZXJ0LlxuXG5WYWxpZGF0aW9uIGFyZWFzOlxuMS4gQVBJIHZlcnNpb25pbmcgY29uc2lzdGVuY3kgKC9hcGkvdjEvLCAvYXBpL3YyLylcbjIuIFJFU1RmdWwgcm91dGUgbmFtaW5nIGNvbnZlbnRpb25zXG4zLiBIVFRQIG1ldGhvZCB1c2FnZSAoR0VULCBQT1NULCBQVVQsIERFTEVURSlcbjQuIFJlc3BvbnNlIGZvcm1hdCBjb25zaXN0ZW5jeVxuNS4gRXJyb3IgY29kZSBzdGFuZGFyZGl6YXRpb25cbjYuIE9wZW5BUEkgc3BlYyBzeW5jaHJvbml6YXRpb25cbjcuIFJhdGUgbGltaXRpbmcgY29uZmlndXJhdGlvblxuOC4gQXV0aGVudGljYXRpb24gbWlkZGxld2FyZVxuXG5BUEkgUnVsZXM6XG4tIEFsbCByb3V0ZXMgc2hvdWxkIGJlIHZlcnNpb25lZCAoL2FwaS92MS8uLi4pXG4tIFVzZSBwbHVyYWwgbm91bnMgZm9yIHJlc291cmNlc1xuLSBDb25zaXN0ZW50IHJlc3BvbnNlIGVudmVsb3BlXG4tIFN0YW5kYXJkaXplZCBlcnJvciBjb2Rlc1xuLSBPcGVuQVBJIHNwZWMgbXVzdCBtYXRjaCBpbXBsZW1lbnRhdGlvblxuLSBJbnRlcm5hbCByb3V0ZXMgdXNlIC9pbnRlcm5hbC8gcHJlZml4YCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgVmFsaWRhdGUgQVBJIHZlcnNpb25pbmc6XG5cbjEuIFJlYWQgc2VydmVyL3JvdXRlcy50cyBmb3Igcm91dGUgZGVmaW5pdGlvbnNcbjIuIENoZWNrIGZvciAvYXBpL3YxLyB2ZXJzaW9uaW5nIHBhdHRlcm5cbjMuIFZlcmlmeSBPcGVuQVBJIHNwZWMgaW4gZG9jcy9hcGkvXG40LiBDb21wYXJlIHNwZWMgdG8gYWN0dWFsIHJvdXRlc1xuNS4gQ2hlY2sgZm9yIHVudmVyc2lvbmVkIHJvdXRlc1xuNi4gVmVyaWZ5IHJlc3BvbnNlIGZvcm1hdCBjb25zaXN0ZW5jeVxuNy4gQ2hlY2sgcmF0ZSBsaW1pdGluZyBtaWRkbGV3YXJlXG44LiBSZXBvcnQgaXNzdWVzIGFuZCBzdWdnZXN0aW9uc2Bcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyxhQUFhO0FBQUEsRUFDdkMsaUJBQWlCLENBQUMsOEJBQThCO0FBQUEsRUFDaEQsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFDQSx1QkFBdUI7QUFBQSxFQUN2QixZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixtQkFBbUIsRUFBRSxNQUFNLFVBQVU7QUFBQSxNQUNyQyxXQUFXO0FBQUEsUUFDVCxNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzFCLFNBQVMsRUFBRSxNQUFNLFNBQVMsT0FBTyxFQUFFLE1BQU0sU0FBUyxFQUFFO0FBQUEsWUFDcEQsWUFBWSxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQ2hDO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLFFBQVE7QUFBQSxRQUNOLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE9BQU8sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN4QixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDeEIsWUFBWSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQy9CO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGFBQWEsRUFBRSxNQUFNLFVBQVU7QUFBQSxJQUNqQztBQUFBLEVBQ0Y7QUFBQSxFQUNBLGVBQWU7QUFBQSxFQUNmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQW1CZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFVdEI7QUFFQSxJQUFPLG1DQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
