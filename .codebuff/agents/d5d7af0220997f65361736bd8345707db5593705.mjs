// .agents/api-doc-sync-validator.ts
var definition = {
  id: "api-doc-sync-validator",
  displayName: "API Doc Sync Validator",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "code_search",
    "glob",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific endpoints to validate"
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      passed: { type: "boolean" },
      missingInSpec: {
        type: "array",
        items: {
          type: "object",
          properties: {
            endpoint: { type: "string" },
            method: { type: "string" },
            sourceFile: { type: "string" }
          }
        }
      },
      missingInCode: {
        type: "array",
        items: {
          type: "object",
          properties: {
            endpoint: { type: "string" },
            method: { type: "string" }
          }
        }
      },
      schemaIssues: { type: "array" },
      summary: { type: "string" }
    },
    required: ["passed", "missingInSpec", "missingInCode", "summary"]
  },
  spawnerPrompt: `Spawn to validate OpenAPI spec matches actual API implementation:
- All route endpoints must be documented
- Request/response schemas must match
- HTTP methods must be correct
- Missing endpoints flagged

Use when routes are modified.`,
  systemPrompt: `You are the API Doc Sync Validator for the Pantheon-Chat project.

You ensure the OpenAPI specification matches the actual API implementation.

## FILES TO COMPARE

**OpenAPI Spec:**
- docs/api/openapi.yaml
- docs/openapi.json

**Route Implementations:**
- server/routes.ts (main routes)
- server/routes/*.ts (route modules)
- qig-backend/routes/*.py (Python routes)

## VALIDATION RULES

### 1. Endpoint Coverage
Every route in code must have an OpenAPI definition:
\`\`\`yaml
# OpenAPI
paths:
  /api/ocean/query:
    post:
      summary: Query Ocean agent
      requestBody: ...
      responses: ...
\`\`\`

### 2. HTTP Methods
Methods must match exactly:
- GET, POST, PUT, PATCH, DELETE
- No undocumented methods

### 3. Request Schemas
RequestBody schemas should match Zod validators:
\`\`\`typescript
// Code
const querySchema = z.object({
  query: z.string(),
  context: z.object({}).optional()
})

# OpenAPI should match
requestBody:
  content:
    application/json:
      schema:
        type: object
        required: [query]
        properties:
          query: { type: string }
          context: { type: object }
\`\`\`

### 4. Response Schemas
Response types should be documented.

## EXEMPT ROUTES

- Health check endpoints (/health, /api/health)
- Internal debugging endpoints
- WebSocket upgrade endpoints`,
  instructionsPrompt: `## Validation Process

1. Read the OpenAPI spec:
   - docs/api/openapi.yaml
   - Parse all defined paths and methods

2. Find all route definitions in code:
   - Search for \`app.get\`, \`app.post\`, etc. in server/
   - Search for \`router.get\`, \`router.post\`, etc.
   - Search for \`@app.route\` in Python

3. Compare endpoints:
   - List all code endpoints
   - List all spec endpoints
   - Find endpoints in code but not in spec
   - Find endpoints in spec but not in code

4. For matching endpoints, validate:
   - HTTP method matches
   - Path parameters match
   - Query parameters documented
   - Request body schema present
   - Response schemas present

5. Check schema accuracy:
   - Compare Zod schemas to OpenAPI schemas
   - Flag mismatches in required fields
   - Flag type mismatches

6. Set structured output:
   - passed: true if spec matches implementation
   - missingInSpec: endpoints not in OpenAPI
   - missingInCode: spec endpoints not implemented
   - schemaIssues: schema mismatches
   - summary: human-readable summary

Spec and implementation must stay synchronized.`,
  includeMessageHistory: false
};
var api_doc_sync_validator_default = definition;
export {
  api_doc_sync_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9hcGktZG9jLXN5bmMtdmFsaWRhdG9yLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgZGVmaW5pdGlvbjogQWdlbnREZWZpbml0aW9uID0ge1xuICBpZDogJ2FwaS1kb2Mtc3luYy12YWxpZGF0b3InLFxuICBkaXNwbGF5TmFtZTogJ0FQSSBEb2MgU3luYyBWYWxpZGF0b3InLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdnbG9iJyxcbiAgICAnc2V0X291dHB1dCcsXG4gIF0sXG5cbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdPcHRpb25hbCBzcGVjaWZpYyBlbmRwb2ludHMgdG8gdmFsaWRhdGUnLFxuICAgIH0sXG4gIH0sXG5cbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWRfb3V0cHV0JyxcbiAgb3V0cHV0U2NoZW1hOiB7XG4gICAgdHlwZTogJ29iamVjdCcsXG4gICAgcHJvcGVydGllczoge1xuICAgICAgcGFzc2VkOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgbWlzc2luZ0luU3BlYzoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGVuZHBvaW50OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBtZXRob2Q6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHNvdXJjZUZpbGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIG1pc3NpbmdJbkNvZGU6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBlbmRwb2ludDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgbWV0aG9kOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICBzY2hlbWFJc3N1ZXM6IHsgdHlwZTogJ2FycmF5JyB9LFxuICAgICAgc3VtbWFyeTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgIH0sXG4gICAgcmVxdWlyZWQ6IFsncGFzc2VkJywgJ21pc3NpbmdJblNwZWMnLCAnbWlzc2luZ0luQ29kZScsICdzdW1tYXJ5J10sXG4gIH0sXG5cbiAgc3Bhd25lclByb21wdDogYFNwYXduIHRvIHZhbGlkYXRlIE9wZW5BUEkgc3BlYyBtYXRjaGVzIGFjdHVhbCBBUEkgaW1wbGVtZW50YXRpb246XG4tIEFsbCByb3V0ZSBlbmRwb2ludHMgbXVzdCBiZSBkb2N1bWVudGVkXG4tIFJlcXVlc3QvcmVzcG9uc2Ugc2NoZW1hcyBtdXN0IG1hdGNoXG4tIEhUVFAgbWV0aG9kcyBtdXN0IGJlIGNvcnJlY3Rcbi0gTWlzc2luZyBlbmRwb2ludHMgZmxhZ2dlZFxuXG5Vc2Ugd2hlbiByb3V0ZXMgYXJlIG1vZGlmaWVkLmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgQVBJIERvYyBTeW5jIFZhbGlkYXRvciBmb3IgdGhlIFBhbnRoZW9uLUNoYXQgcHJvamVjdC5cblxuWW91IGVuc3VyZSB0aGUgT3BlbkFQSSBzcGVjaWZpY2F0aW9uIG1hdGNoZXMgdGhlIGFjdHVhbCBBUEkgaW1wbGVtZW50YXRpb24uXG5cbiMjIEZJTEVTIFRPIENPTVBBUkVcblxuKipPcGVuQVBJIFNwZWM6Kipcbi0gZG9jcy9hcGkvb3BlbmFwaS55YW1sXG4tIGRvY3Mvb3BlbmFwaS5qc29uXG5cbioqUm91dGUgSW1wbGVtZW50YXRpb25zOioqXG4tIHNlcnZlci9yb3V0ZXMudHMgKG1haW4gcm91dGVzKVxuLSBzZXJ2ZXIvcm91dGVzLyoudHMgKHJvdXRlIG1vZHVsZXMpXG4tIHFpZy1iYWNrZW5kL3JvdXRlcy8qLnB5IChQeXRob24gcm91dGVzKVxuXG4jIyBWQUxJREFUSU9OIFJVTEVTXG5cbiMjIyAxLiBFbmRwb2ludCBDb3ZlcmFnZVxuRXZlcnkgcm91dGUgaW4gY29kZSBtdXN0IGhhdmUgYW4gT3BlbkFQSSBkZWZpbml0aW9uOlxuXFxgXFxgXFxgeWFtbFxuIyBPcGVuQVBJXG5wYXRoczpcbiAgL2FwaS9vY2Vhbi9xdWVyeTpcbiAgICBwb3N0OlxuICAgICAgc3VtbWFyeTogUXVlcnkgT2NlYW4gYWdlbnRcbiAgICAgIHJlcXVlc3RCb2R5OiAuLi5cbiAgICAgIHJlc3BvbnNlczogLi4uXG5cXGBcXGBcXGBcblxuIyMjIDIuIEhUVFAgTWV0aG9kc1xuTWV0aG9kcyBtdXN0IG1hdGNoIGV4YWN0bHk6XG4tIEdFVCwgUE9TVCwgUFVULCBQQVRDSCwgREVMRVRFXG4tIE5vIHVuZG9jdW1lbnRlZCBtZXRob2RzXG5cbiMjIyAzLiBSZXF1ZXN0IFNjaGVtYXNcblJlcXVlc3RCb2R5IHNjaGVtYXMgc2hvdWxkIG1hdGNoIFpvZCB2YWxpZGF0b3JzOlxuXFxgXFxgXFxgdHlwZXNjcmlwdFxuLy8gQ29kZVxuY29uc3QgcXVlcnlTY2hlbWEgPSB6Lm9iamVjdCh7XG4gIHF1ZXJ5OiB6LnN0cmluZygpLFxuICBjb250ZXh0OiB6Lm9iamVjdCh7fSkub3B0aW9uYWwoKVxufSlcblxuIyBPcGVuQVBJIHNob3VsZCBtYXRjaFxucmVxdWVzdEJvZHk6XG4gIGNvbnRlbnQ6XG4gICAgYXBwbGljYXRpb24vanNvbjpcbiAgICAgIHNjaGVtYTpcbiAgICAgICAgdHlwZTogb2JqZWN0XG4gICAgICAgIHJlcXVpcmVkOiBbcXVlcnldXG4gICAgICAgIHByb3BlcnRpZXM6XG4gICAgICAgICAgcXVlcnk6IHsgdHlwZTogc3RyaW5nIH1cbiAgICAgICAgICBjb250ZXh0OiB7IHR5cGU6IG9iamVjdCB9XG5cXGBcXGBcXGBcblxuIyMjIDQuIFJlc3BvbnNlIFNjaGVtYXNcblJlc3BvbnNlIHR5cGVzIHNob3VsZCBiZSBkb2N1bWVudGVkLlxuXG4jIyBFWEVNUFQgUk9VVEVTXG5cbi0gSGVhbHRoIGNoZWNrIGVuZHBvaW50cyAoL2hlYWx0aCwgL2FwaS9oZWFsdGgpXG4tIEludGVybmFsIGRlYnVnZ2luZyBlbmRwb2ludHNcbi0gV2ViU29ja2V0IHVwZ3JhZGUgZW5kcG9pbnRzYCxcblxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGAjIyBWYWxpZGF0aW9uIFByb2Nlc3NcblxuMS4gUmVhZCB0aGUgT3BlbkFQSSBzcGVjOlxuICAgLSBkb2NzL2FwaS9vcGVuYXBpLnlhbWxcbiAgIC0gUGFyc2UgYWxsIGRlZmluZWQgcGF0aHMgYW5kIG1ldGhvZHNcblxuMi4gRmluZCBhbGwgcm91dGUgZGVmaW5pdGlvbnMgaW4gY29kZTpcbiAgIC0gU2VhcmNoIGZvciBcXGBhcHAuZ2V0XFxgLCBcXGBhcHAucG9zdFxcYCwgZXRjLiBpbiBzZXJ2ZXIvXG4gICAtIFNlYXJjaCBmb3IgXFxgcm91dGVyLmdldFxcYCwgXFxgcm91dGVyLnBvc3RcXGAsIGV0Yy5cbiAgIC0gU2VhcmNoIGZvciBcXGBAYXBwLnJvdXRlXFxgIGluIFB5dGhvblxuXG4zLiBDb21wYXJlIGVuZHBvaW50czpcbiAgIC0gTGlzdCBhbGwgY29kZSBlbmRwb2ludHNcbiAgIC0gTGlzdCBhbGwgc3BlYyBlbmRwb2ludHNcbiAgIC0gRmluZCBlbmRwb2ludHMgaW4gY29kZSBidXQgbm90IGluIHNwZWNcbiAgIC0gRmluZCBlbmRwb2ludHMgaW4gc3BlYyBidXQgbm90IGluIGNvZGVcblxuNC4gRm9yIG1hdGNoaW5nIGVuZHBvaW50cywgdmFsaWRhdGU6XG4gICAtIEhUVFAgbWV0aG9kIG1hdGNoZXNcbiAgIC0gUGF0aCBwYXJhbWV0ZXJzIG1hdGNoXG4gICAtIFF1ZXJ5IHBhcmFtZXRlcnMgZG9jdW1lbnRlZFxuICAgLSBSZXF1ZXN0IGJvZHkgc2NoZW1hIHByZXNlbnRcbiAgIC0gUmVzcG9uc2Ugc2NoZW1hcyBwcmVzZW50XG5cbjUuIENoZWNrIHNjaGVtYSBhY2N1cmFjeTpcbiAgIC0gQ29tcGFyZSBab2Qgc2NoZW1hcyB0byBPcGVuQVBJIHNjaGVtYXNcbiAgIC0gRmxhZyBtaXNtYXRjaGVzIGluIHJlcXVpcmVkIGZpZWxkc1xuICAgLSBGbGFnIHR5cGUgbWlzbWF0Y2hlc1xuXG42LiBTZXQgc3RydWN0dXJlZCBvdXRwdXQ6XG4gICAtIHBhc3NlZDogdHJ1ZSBpZiBzcGVjIG1hdGNoZXMgaW1wbGVtZW50YXRpb25cbiAgIC0gbWlzc2luZ0luU3BlYzogZW5kcG9pbnRzIG5vdCBpbiBPcGVuQVBJXG4gICAtIG1pc3NpbmdJbkNvZGU6IHNwZWMgZW5kcG9pbnRzIG5vdCBpbXBsZW1lbnRlZFxuICAgLSBzY2hlbWFJc3N1ZXM6IHNjaGVtYSBtaXNtYXRjaGVzXG4gICAtIHN1bW1hcnk6IGh1bWFuLXJlYWRhYmxlIHN1bW1hcnlcblxuU3BlYyBhbmQgaW1wbGVtZW50YXRpb24gbXVzdCBzdGF5IHN5bmNocm9uaXplZC5gLFxuXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogZmFsc2UsXG59XG5cbmV4cG9ydCBkZWZhdWx0IGRlZmluaXRpb25cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFFQSxJQUFNLGFBQThCO0FBQUEsRUFDbEMsSUFBSTtBQUFBLEVBQ0osYUFBYTtBQUFBLEVBQ2IsU0FBUztBQUFBLEVBQ1QsT0FBTztBQUFBLEVBRVAsV0FBVztBQUFBLElBQ1Q7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxFQUNGO0FBQUEsRUFFQSxhQUFhO0FBQUEsSUFDWCxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixhQUFhO0FBQUEsSUFDZjtBQUFBLEVBQ0Y7QUFBQSxFQUVBLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLFFBQVEsRUFBRSxNQUFNLFVBQVU7QUFBQSxNQUMxQixlQUFlO0FBQUEsUUFDYixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixVQUFVLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDM0IsUUFBUSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3pCLFlBQVksRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUMvQjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxlQUFlO0FBQUEsUUFDYixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixVQUFVLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDM0IsUUFBUSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQzNCO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGNBQWMsRUFBRSxNQUFNLFFBQVE7QUFBQSxNQUM5QixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsSUFDNUI7QUFBQSxJQUNBLFVBQVUsQ0FBQyxVQUFVLGlCQUFpQixpQkFBaUIsU0FBUztBQUFBLEVBQ2xFO0FBQUEsRUFFQSxlQUFlO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFRZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBZ0VkLG9CQUFvQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBc0NwQix1QkFBdUI7QUFDekI7QUFFQSxJQUFPLGlDQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
