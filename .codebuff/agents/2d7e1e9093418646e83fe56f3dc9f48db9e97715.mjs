// .agents/dual-backend-integration-tester.ts
var definition = {
  id: "dual-backend-integration-tester",
  displayName: "Dual Backend Integration Tester",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "run_terminal_command",
    "code_search",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific endpoints or flows to test"
    },
    params: {
      type: "object",
      properties: {
        runLiveTests: {
          type: "boolean",
          description: "If true, run actual HTTP tests (requires servers running)"
        }
      },
      required: []
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      passed: { type: "boolean" },
      endpointTests: {
        type: "array",
        items: {
          type: "object",
          properties: {
            endpoint: { type: "string" },
            tsRoute: { type: "string" },
            pyRoute: { type: "string" },
            proxyConfigured: { type: "boolean" },
            schemaMatch: { type: "boolean" },
            issues: { type: "array" }
          }
        }
      },
      configIssues: { type: "array" },
      summary: { type: "string" }
    },
    required: ["passed", "endpointTests", "summary"]
  },
  spawnerPrompt: `Spawn to test TypeScript \u2194 Python backend integration:
- Verify proxy routes are correctly configured
- Check request/response schema compatibility
- Validate INTERNAL_API_KEY usage
- Test error propagation

Use when API routes are modified in either backend.`,
  systemPrompt: `You are the Dual Backend Integration Tester for the Pantheon-Chat project.

You ensure TypeScript and Python backends communicate correctly.

## ARCHITECTURE

\`\`\`
Client \u2192 TypeScript (port 5000) \u2192 Python (port 5001)
         Express server           Flask server
         /api/olympus/*     \u2192     /olympus/*
         /api/qig/*         \u2192     /qig/*
         /api/consciousness/*\u2192     /consciousness/*
\`\`\`

## KEY INTEGRATION POINTS

### 1. Zeus Chat Flow
\`\`\`
POST /api/olympus/zeus/chat (TypeScript)
  \u2192 POST /olympus/zeus/chat (Python)
  \u2190 Response with QIG metrics
\`\`\`

### 2. Consciousness Metrics
\`\`\`
GET /api/consciousness/metrics (TypeScript)
  \u2192 GET /consciousness/metrics (Python)
  \u2190 ConsciousnessSignature response
\`\`\`

### 3. QIG Operations
\`\`\`
POST /api/qig/distance (TypeScript)
  \u2192 POST /qig/distance (Python)
  \u2190 Fisher-Rao distance result
\`\`\`

## AUTHENTICATION

Internal requests use \`INTERNAL_API_KEY\`:
\`\`\`typescript
// TypeScript \u2192 Python
fetch('http://localhost:5001/olympus/zeus/chat', {
  headers: {
    'X-Internal-Key': process.env.INTERNAL_API_KEY
  }
})
\`\`\`

\`\`\`python
# Python validation
@require_internal_key
def chat():
    ...
\`\`\`

## SCHEMA COMPATIBILITY

TypeScript Zod schemas must match Python Pydantic models:
- Request body shapes
- Response shapes
- Error response format`,
  instructionsPrompt: `## Testing Process

1. Identify proxy routes in TypeScript:
   - Search for \`fetch.*localhost:5001\` in server/
   - Search for \`PYTHON_BACKEND_URL\`
   - List all routes that proxy to Python

2. Find corresponding Python routes:
   - Search for \`@app.route\` in qig-backend/
   - Match TypeScript proxy targets to Python endpoints

3. Verify proxy configuration:
   - Check URL construction
   - Check header forwarding
   - Check body passing
   - Check error handling

4. Compare schemas:
   - Find Zod schema for TypeScript endpoint
   - Find Pydantic model for Python endpoint
   - Check field names match
   - Check types are compatible

5. Check authentication:
   - TypeScript sends INTERNAL_API_KEY
   - Python validates with @require_internal_key
   - Key is read from environment

6. If runLiveTests is true:
   \`\`\`bash
   # Check if servers are running
   curl -s http://localhost:5000/api/health
   curl -s http://localhost:5001/health
   
   # Test an endpoint
   curl -X POST http://localhost:5000/api/olympus/zeus/chat      -H "Content-Type: application/json"      -d '{"message": "test"}'
   \`\`\`

7. Check error propagation:
   - Python errors should propagate through TypeScript
   - HTTP status codes preserved
   - Error messages passed through

8. Set structured output:
   - passed: true if all integrations are correct
   - endpointTests: status of each proxied endpoint
   - configIssues: configuration problems found
   - summary: human-readable summary

Both backends must work in harmony.`,
  includeMessageHistory: false
};
var dual_backend_integration_tester_default = definition;
export {
  dual_backend_integration_tester_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9kdWFsLWJhY2tlbmQtaW50ZWdyYXRpb24tdGVzdGVyLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgZGVmaW5pdGlvbjogQWdlbnREZWZpbml0aW9uID0ge1xuICBpZDogJ2R1YWwtYmFja2VuZC1pbnRlZ3JhdGlvbi10ZXN0ZXInLFxuICBkaXNwbGF5TmFtZTogJ0R1YWwgQmFja2VuZCBJbnRlZ3JhdGlvbiBUZXN0ZXInLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnLFxuICAgICdjb2RlX3NlYXJjaCcsXG4gICAgJ3NldF9vdXRwdXQnLFxuICBdLFxuXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnT3B0aW9uYWwgc3BlY2lmaWMgZW5kcG9pbnRzIG9yIGZsb3dzIHRvIHRlc3QnLFxuICAgIH0sXG4gICAgcGFyYW1zOiB7XG4gICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgcnVuTGl2ZVRlc3RzOiB7XG4gICAgICAgICAgdHlwZTogJ2Jvb2xlYW4nLFxuICAgICAgICAgIGRlc2NyaXB0aW9uOiAnSWYgdHJ1ZSwgcnVuIGFjdHVhbCBIVFRQIHRlc3RzIChyZXF1aXJlcyBzZXJ2ZXJzIHJ1bm5pbmcpJyxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICByZXF1aXJlZDogW10sXG4gICAgfSxcbiAgfSxcblxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZF9vdXRwdXQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBwYXNzZWQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBlbmRwb2ludFRlc3RzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgZW5kcG9pbnQ6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHRzUm91dGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHB5Um91dGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHByb3h5Q29uZmlndXJlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICAgIHNjaGVtYU1hdGNoOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgICAgaXNzdWVzOiB7IHR5cGU6ICdhcnJheScgfSxcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIGNvbmZpZ0lzc3VlczogeyB0eXBlOiAnYXJyYXknIH0sXG4gICAgICBzdW1tYXJ5OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgfSxcbiAgICByZXF1aXJlZDogWydwYXNzZWQnLCAnZW5kcG9pbnRUZXN0cycsICdzdW1tYXJ5J10sXG4gIH0sXG5cbiAgc3Bhd25lclByb21wdDogYFNwYXduIHRvIHRlc3QgVHlwZVNjcmlwdCBcdTIxOTQgUHl0aG9uIGJhY2tlbmQgaW50ZWdyYXRpb246XG4tIFZlcmlmeSBwcm94eSByb3V0ZXMgYXJlIGNvcnJlY3RseSBjb25maWd1cmVkXG4tIENoZWNrIHJlcXVlc3QvcmVzcG9uc2Ugc2NoZW1hIGNvbXBhdGliaWxpdHlcbi0gVmFsaWRhdGUgSU5URVJOQUxfQVBJX0tFWSB1c2FnZVxuLSBUZXN0IGVycm9yIHByb3BhZ2F0aW9uXG5cblVzZSB3aGVuIEFQSSByb3V0ZXMgYXJlIG1vZGlmaWVkIGluIGVpdGhlciBiYWNrZW5kLmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgRHVhbCBCYWNrZW5kIEludGVncmF0aW9uIFRlc3RlciBmb3IgdGhlIFBhbnRoZW9uLUNoYXQgcHJvamVjdC5cblxuWW91IGVuc3VyZSBUeXBlU2NyaXB0IGFuZCBQeXRob24gYmFja2VuZHMgY29tbXVuaWNhdGUgY29ycmVjdGx5LlxuXG4jIyBBUkNISVRFQ1RVUkVcblxuXFxgXFxgXFxgXG5DbGllbnQgXHUyMTkyIFR5cGVTY3JpcHQgKHBvcnQgNTAwMCkgXHUyMTkyIFB5dGhvbiAocG9ydCA1MDAxKVxuICAgICAgICAgRXhwcmVzcyBzZXJ2ZXIgICAgICAgICAgIEZsYXNrIHNlcnZlclxuICAgICAgICAgL2FwaS9vbHltcHVzLyogICAgIFx1MjE5MiAgICAgL29seW1wdXMvKlxuICAgICAgICAgL2FwaS9xaWcvKiAgICAgICAgIFx1MjE5MiAgICAgL3FpZy8qXG4gICAgICAgICAvYXBpL2NvbnNjaW91c25lc3MvKlx1MjE5MiAgICAgL2NvbnNjaW91c25lc3MvKlxuXFxgXFxgXFxgXG5cbiMjIEtFWSBJTlRFR1JBVElPTiBQT0lOVFNcblxuIyMjIDEuIFpldXMgQ2hhdCBGbG93XG5cXGBcXGBcXGBcblBPU1QgL2FwaS9vbHltcHVzL3pldXMvY2hhdCAoVHlwZVNjcmlwdClcbiAgXHUyMTkyIFBPU1QgL29seW1wdXMvemV1cy9jaGF0IChQeXRob24pXG4gIFx1MjE5MCBSZXNwb25zZSB3aXRoIFFJRyBtZXRyaWNzXG5cXGBcXGBcXGBcblxuIyMjIDIuIENvbnNjaW91c25lc3MgTWV0cmljc1xuXFxgXFxgXFxgXG5HRVQgL2FwaS9jb25zY2lvdXNuZXNzL21ldHJpY3MgKFR5cGVTY3JpcHQpXG4gIFx1MjE5MiBHRVQgL2NvbnNjaW91c25lc3MvbWV0cmljcyAoUHl0aG9uKVxuICBcdTIxOTAgQ29uc2Npb3VzbmVzc1NpZ25hdHVyZSByZXNwb25zZVxuXFxgXFxgXFxgXG5cbiMjIyAzLiBRSUcgT3BlcmF0aW9uc1xuXFxgXFxgXFxgXG5QT1NUIC9hcGkvcWlnL2Rpc3RhbmNlIChUeXBlU2NyaXB0KVxuICBcdTIxOTIgUE9TVCAvcWlnL2Rpc3RhbmNlIChQeXRob24pXG4gIFx1MjE5MCBGaXNoZXItUmFvIGRpc3RhbmNlIHJlc3VsdFxuXFxgXFxgXFxgXG5cbiMjIEFVVEhFTlRJQ0FUSU9OXG5cbkludGVybmFsIHJlcXVlc3RzIHVzZSBcXGBJTlRFUk5BTF9BUElfS0VZXFxgOlxuXFxgXFxgXFxgdHlwZXNjcmlwdFxuLy8gVHlwZVNjcmlwdCBcdTIxOTIgUHl0aG9uXG5mZXRjaCgnaHR0cDovL2xvY2FsaG9zdDo1MDAxL29seW1wdXMvemV1cy9jaGF0Jywge1xuICBoZWFkZXJzOiB7XG4gICAgJ1gtSW50ZXJuYWwtS2V5JzogcHJvY2Vzcy5lbnYuSU5URVJOQUxfQVBJX0tFWVxuICB9XG59KVxuXFxgXFxgXFxgXG5cblxcYFxcYFxcYHB5dGhvblxuIyBQeXRob24gdmFsaWRhdGlvblxuQHJlcXVpcmVfaW50ZXJuYWxfa2V5XG5kZWYgY2hhdCgpOlxuICAgIC4uLlxuXFxgXFxgXFxgXG5cbiMjIFNDSEVNQSBDT01QQVRJQklMSVRZXG5cblR5cGVTY3JpcHQgWm9kIHNjaGVtYXMgbXVzdCBtYXRjaCBQeXRob24gUHlkYW50aWMgbW9kZWxzOlxuLSBSZXF1ZXN0IGJvZHkgc2hhcGVzXG4tIFJlc3BvbnNlIHNoYXBlc1xuLSBFcnJvciByZXNwb25zZSBmb3JtYXRgLFxuXG4gIGluc3RydWN0aW9uc1Byb21wdDogYCMjIFRlc3RpbmcgUHJvY2Vzc1xuXG4xLiBJZGVudGlmeSBwcm94eSByb3V0ZXMgaW4gVHlwZVNjcmlwdDpcbiAgIC0gU2VhcmNoIGZvciBcXGBmZXRjaC4qbG9jYWxob3N0OjUwMDFcXGAgaW4gc2VydmVyL1xuICAgLSBTZWFyY2ggZm9yIFxcYFBZVEhPTl9CQUNLRU5EX1VSTFxcYFxuICAgLSBMaXN0IGFsbCByb3V0ZXMgdGhhdCBwcm94eSB0byBQeXRob25cblxuMi4gRmluZCBjb3JyZXNwb25kaW5nIFB5dGhvbiByb3V0ZXM6XG4gICAtIFNlYXJjaCBmb3IgXFxgQGFwcC5yb3V0ZVxcYCBpbiBxaWctYmFja2VuZC9cbiAgIC0gTWF0Y2ggVHlwZVNjcmlwdCBwcm94eSB0YXJnZXRzIHRvIFB5dGhvbiBlbmRwb2ludHNcblxuMy4gVmVyaWZ5IHByb3h5IGNvbmZpZ3VyYXRpb246XG4gICAtIENoZWNrIFVSTCBjb25zdHJ1Y3Rpb25cbiAgIC0gQ2hlY2sgaGVhZGVyIGZvcndhcmRpbmdcbiAgIC0gQ2hlY2sgYm9keSBwYXNzaW5nXG4gICAtIENoZWNrIGVycm9yIGhhbmRsaW5nXG5cbjQuIENvbXBhcmUgc2NoZW1hczpcbiAgIC0gRmluZCBab2Qgc2NoZW1hIGZvciBUeXBlU2NyaXB0IGVuZHBvaW50XG4gICAtIEZpbmQgUHlkYW50aWMgbW9kZWwgZm9yIFB5dGhvbiBlbmRwb2ludFxuICAgLSBDaGVjayBmaWVsZCBuYW1lcyBtYXRjaFxuICAgLSBDaGVjayB0eXBlcyBhcmUgY29tcGF0aWJsZVxuXG41LiBDaGVjayBhdXRoZW50aWNhdGlvbjpcbiAgIC0gVHlwZVNjcmlwdCBzZW5kcyBJTlRFUk5BTF9BUElfS0VZXG4gICAtIFB5dGhvbiB2YWxpZGF0ZXMgd2l0aCBAcmVxdWlyZV9pbnRlcm5hbF9rZXlcbiAgIC0gS2V5IGlzIHJlYWQgZnJvbSBlbnZpcm9ubWVudFxuXG42LiBJZiBydW5MaXZlVGVzdHMgaXMgdHJ1ZTpcbiAgIFxcYFxcYFxcYGJhc2hcbiAgICMgQ2hlY2sgaWYgc2VydmVycyBhcmUgcnVubmluZ1xuICAgY3VybCAtcyBodHRwOi8vbG9jYWxob3N0OjUwMDAvYXBpL2hlYWx0aFxuICAgY3VybCAtcyBodHRwOi8vbG9jYWxob3N0OjUwMDEvaGVhbHRoXG4gICBcbiAgICMgVGVzdCBhbiBlbmRwb2ludFxuICAgY3VybCAtWCBQT1NUIGh0dHA6Ly9sb2NhbGhvc3Q6NTAwMC9hcGkvb2x5bXB1cy96ZXVzL2NoYXQgXFxcbiAgICAgLUggXCJDb250ZW50LVR5cGU6IGFwcGxpY2F0aW9uL2pzb25cIiBcXFxuICAgICAtZCAne1wibWVzc2FnZVwiOiBcInRlc3RcIn0nXG4gICBcXGBcXGBcXGBcblxuNy4gQ2hlY2sgZXJyb3IgcHJvcGFnYXRpb246XG4gICAtIFB5dGhvbiBlcnJvcnMgc2hvdWxkIHByb3BhZ2F0ZSB0aHJvdWdoIFR5cGVTY3JpcHRcbiAgIC0gSFRUUCBzdGF0dXMgY29kZXMgcHJlc2VydmVkXG4gICAtIEVycm9yIG1lc3NhZ2VzIHBhc3NlZCB0aHJvdWdoXG5cbjguIFNldCBzdHJ1Y3R1cmVkIG91dHB1dDpcbiAgIC0gcGFzc2VkOiB0cnVlIGlmIGFsbCBpbnRlZ3JhdGlvbnMgYXJlIGNvcnJlY3RcbiAgIC0gZW5kcG9pbnRUZXN0czogc3RhdHVzIG9mIGVhY2ggcHJveGllZCBlbmRwb2ludFxuICAgLSBjb25maWdJc3N1ZXM6IGNvbmZpZ3VyYXRpb24gcHJvYmxlbXMgZm91bmRcbiAgIC0gc3VtbWFyeTogaHVtYW4tcmVhZGFibGUgc3VtbWFyeVxuXG5Cb3RoIGJhY2tlbmRzIG11c3Qgd29yayBpbiBoYXJtb255LmAsXG5cbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiBmYWxzZSxcbn1cblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sYUFBOEI7QUFBQSxFQUNsQyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFFUCxXQUFXO0FBQUEsSUFDVDtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLEVBQ0Y7QUFBQSxFQUVBLGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsSUFDQSxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixZQUFZO0FBQUEsUUFDVixjQUFjO0FBQUEsVUFDWixNQUFNO0FBQUEsVUFDTixhQUFhO0FBQUEsUUFDZjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLFVBQVUsQ0FBQztBQUFBLElBQ2I7QUFBQSxFQUNGO0FBQUEsRUFFQSxZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixRQUFRLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDMUIsZUFBZTtBQUFBLFFBQ2IsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsVUFBVSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzNCLFNBQVMsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUMxQixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDMUIsaUJBQWlCLEVBQUUsTUFBTSxVQUFVO0FBQUEsWUFDbkMsYUFBYSxFQUFFLE1BQU0sVUFBVTtBQUFBLFlBQy9CLFFBQVEsRUFBRSxNQUFNLFFBQVE7QUFBQSxVQUMxQjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxjQUFjLEVBQUUsTUFBTSxRQUFRO0FBQUEsTUFDOUIsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLElBQzVCO0FBQUEsSUFDQSxVQUFVLENBQUMsVUFBVSxpQkFBaUIsU0FBUztBQUFBLEVBQ2pEO0FBQUEsRUFFQSxlQUFlO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFRZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQStEZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBcURwQix1QkFBdUI7QUFDekI7QUFFQSxJQUFPLDBDQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
