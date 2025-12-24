// .agents/testing-coverage-auditor.ts
var agentDefinition = {
  id: "testing-coverage-auditor",
  displayName: "Testing Coverage Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search", "run_terminal_command"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Audit test coverage and testing patterns"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      coveragePercentage: { type: "number" },
      testingGaps: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            untestedFunctions: { type: "array", items: { type: "string" } },
            priority: { type: "string", enum: ["critical", "high", "medium", "low"] }
          }
        }
      },
      testTypes: {
        type: "object",
        properties: {
          unit: { type: "number" },
          integration: { type: "number" },
          e2e: { type: "number" },
          visual: { type: "number" }
        }
      },
      recommendations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            area: { type: "string" },
            testType: { type: "string" },
            description: { type: "string" }
          }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to audit test coverage and identify testing gaps",
  systemPrompt: `You are a testing and quality assurance expert.

Audit areas:
1. Unit test coverage for utilities and hooks
2. Component test coverage
3. Integration test coverage for APIs
4. E2E test coverage for critical paths
5. Visual regression testing
6. Accessibility testing
7. Performance testing

Testing Priorities:
- Critical paths must have E2E tests
- All utilities should have unit tests
- API endpoints need integration tests
- Complex components need component tests
- QIG core functions need extensive testing
- Consciousness metrics need validation tests`,
  instructionsPrompt: `Audit test coverage:

1. Run npm test -- --coverage to get coverage report
2. Find files without corresponding test files
3. Check test file patterns (.test.ts, .spec.ts)
4. Identify critical paths without E2E tests
5. Check qig-backend/ for Python test coverage
6. Look for mock patterns and test utilities
7. Check for Playwright E2E tests
8. Report testing gaps with priority`
};
var testing_coverage_auditor_default = agentDefinition;
export {
  testing_coverage_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy90ZXN0aW5nLWNvdmVyYWdlLWF1ZGl0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBhZ2VudERlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICd0ZXN0aW5nLWNvdmVyYWdlLWF1ZGl0b3InLFxuICBkaXNwbGF5TmFtZTogJ1Rlc3RpbmcgQ292ZXJhZ2UgQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnLCAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdBdWRpdCB0ZXN0IGNvdmVyYWdlIGFuZCB0ZXN0aW5nIHBhdHRlcm5zJ1xuICAgIH1cbiAgfSxcbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiB0cnVlLFxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIGNvdmVyYWdlUGVyY2VudGFnZTogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgdGVzdGluZ0dhcHM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBmaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICB1bnRlc3RlZEZ1bmN0aW9uczogeyB0eXBlOiAnYXJyYXknLCBpdGVtczogeyB0eXBlOiAnc3RyaW5nJyB9IH0sXG4gICAgICAgICAgICBwcmlvcml0eTogeyB0eXBlOiAnc3RyaW5nJywgZW51bTogWydjcml0aWNhbCcsICdoaWdoJywgJ21lZGl1bScsICdsb3cnXSB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgdGVzdFR5cGVzOiB7XG4gICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgdW5pdDogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgICAgIGludGVncmF0aW9uOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgZTJlOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgdmlzdWFsOiB7IHR5cGU6ICdudW1iZXInIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIHJlY29tbWVuZGF0aW9uczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGFyZWE6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHRlc3RUeXBlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBkZXNjcmlwdGlvbjogeyB0eXBlOiAnc3RyaW5nJyB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9LFxuICBzcGF3bmVyUHJvbXB0OiAnU3Bhd24gdG8gYXVkaXQgdGVzdCBjb3ZlcmFnZSBhbmQgaWRlbnRpZnkgdGVzdGluZyBnYXBzJyxcbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSBhIHRlc3RpbmcgYW5kIHF1YWxpdHkgYXNzdXJhbmNlIGV4cGVydC5cblxuQXVkaXQgYXJlYXM6XG4xLiBVbml0IHRlc3QgY292ZXJhZ2UgZm9yIHV0aWxpdGllcyBhbmQgaG9va3NcbjIuIENvbXBvbmVudCB0ZXN0IGNvdmVyYWdlXG4zLiBJbnRlZ3JhdGlvbiB0ZXN0IGNvdmVyYWdlIGZvciBBUElzXG40LiBFMkUgdGVzdCBjb3ZlcmFnZSBmb3IgY3JpdGljYWwgcGF0aHNcbjUuIFZpc3VhbCByZWdyZXNzaW9uIHRlc3RpbmdcbjYuIEFjY2Vzc2liaWxpdHkgdGVzdGluZ1xuNy4gUGVyZm9ybWFuY2UgdGVzdGluZ1xuXG5UZXN0aW5nIFByaW9yaXRpZXM6XG4tIENyaXRpY2FsIHBhdGhzIG11c3QgaGF2ZSBFMkUgdGVzdHNcbi0gQWxsIHV0aWxpdGllcyBzaG91bGQgaGF2ZSB1bml0IHRlc3RzXG4tIEFQSSBlbmRwb2ludHMgbmVlZCBpbnRlZ3JhdGlvbiB0ZXN0c1xuLSBDb21wbGV4IGNvbXBvbmVudHMgbmVlZCBjb21wb25lbnQgdGVzdHNcbi0gUUlHIGNvcmUgZnVuY3Rpb25zIG5lZWQgZXh0ZW5zaXZlIHRlc3Rpbmdcbi0gQ29uc2Npb3VzbmVzcyBtZXRyaWNzIG5lZWQgdmFsaWRhdGlvbiB0ZXN0c2AsXG4gIGluc3RydWN0aW9uc1Byb21wdDogYEF1ZGl0IHRlc3QgY292ZXJhZ2U6XG5cbjEuIFJ1biBucG0gdGVzdCAtLSAtLWNvdmVyYWdlIHRvIGdldCBjb3ZlcmFnZSByZXBvcnRcbjIuIEZpbmQgZmlsZXMgd2l0aG91dCBjb3JyZXNwb25kaW5nIHRlc3QgZmlsZXNcbjMuIENoZWNrIHRlc3QgZmlsZSBwYXR0ZXJucyAoLnRlc3QudHMsIC5zcGVjLnRzKVxuNC4gSWRlbnRpZnkgY3JpdGljYWwgcGF0aHMgd2l0aG91dCBFMkUgdGVzdHNcbjUuIENoZWNrIHFpZy1iYWNrZW5kLyBmb3IgUHl0aG9uIHRlc3QgY292ZXJhZ2VcbjYuIExvb2sgZm9yIG1vY2sgcGF0dGVybnMgYW5kIHRlc3QgdXRpbGl0aWVzXG43LiBDaGVjayBmb3IgUGxheXdyaWdodCBFMkUgdGVzdHNcbjguIFJlcG9ydCB0ZXN0aW5nIGdhcHMgd2l0aCBwcmlvcml0eWBcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyxlQUFlLHNCQUFzQjtBQUFBLEVBQy9ELGlCQUFpQixDQUFDLDhCQUE4QjtBQUFBLEVBQ2hELGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsRUFDRjtBQUFBLEVBQ0EsdUJBQXVCO0FBQUEsRUFDdkIsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1Ysb0JBQW9CLEVBQUUsTUFBTSxTQUFTO0FBQUEsTUFDckMsYUFBYTtBQUFBLFFBQ1gsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLG1CQUFtQixFQUFFLE1BQU0sU0FBUyxPQUFPLEVBQUUsTUFBTSxTQUFTLEVBQUU7QUFBQSxZQUM5RCxVQUFVLEVBQUUsTUFBTSxVQUFVLE1BQU0sQ0FBQyxZQUFZLFFBQVEsVUFBVSxLQUFLLEVBQUU7QUFBQSxVQUMxRTtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxXQUFXO0FBQUEsUUFDVCxNQUFNO0FBQUEsUUFDTixZQUFZO0FBQUEsVUFDVixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDdkIsYUFBYSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQzlCLEtBQUssRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUN0QixRQUFRLEVBQUUsTUFBTSxTQUFTO0FBQUEsUUFDM0I7QUFBQSxNQUNGO0FBQUEsTUFDQSxpQkFBaUI7QUFBQSxRQUNmLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixVQUFVLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDM0IsYUFBYSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQ2hDO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUFBLEVBQ0EsZUFBZTtBQUFBLEVBQ2YsY0FBYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQWtCZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFVdEI7QUFFQSxJQUFPLG1DQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
