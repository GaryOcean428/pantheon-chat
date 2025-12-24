// .agents/api-purity-enforcer.ts
var definition = {
  id: "api-purity-enforcer",
  displayName: "API Purity Enforcer",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "code_search",
    "run_terminal_command",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional description of changes to validate"
    },
    params: {
      type: "object",
      properties: {
        files: {
          type: "array",
          description: "Specific files to check"
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
      violations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            line: { type: "number" },
            code: { type: "string" },
            fix: { type: "string" }
          }
        }
      },
      summary: { type: "string" }
    },
    required: ["passed", "violations", "summary"]
  },
  spawnerPrompt: `Spawn to enforce centralized API usage in the frontend:
- All API calls must go through @/api, not direct fetch()
- Use QUERY_KEYS for TanStack Query
- Use api.serviceName.method() for mutations

Use when client/ code is modified.`,
  systemPrompt: `You are the API Purity Enforcer for the Pantheon-Chat project.

You ensure all frontend API calls go through the centralized API layer.

## DRY PRINCIPLE

All API routes are defined ONCE in \`client/src/api/routes.ts\`.
All API calls must use the centralized API client.

## FORBIDDEN PATTERNS

\u274C Direct fetch to API endpoints:
\`\`\`typescript
// BAD - direct fetch
fetch('/api/ocean/query', { ... })
await fetch(\`/api/consciousness/metrics\`)
\`\`\`

## REQUIRED PATTERNS

\u2705 Use centralized API:
\`\`\`typescript
// GOOD - using API client
import { api } from '@/api'

// For queries (GET)
const { data } = useQuery({
  queryKey: QUERY_KEYS.consciousness.metrics,
  queryFn: api.consciousness.getMetrics
})

// For mutations (POST/PUT/DELETE)
const mutation = useMutation({
  mutationFn: api.ocean.query
})
\`\`\`

## EXEMPT FILES

- client/src/api/ (the API module itself)
- client/src/lib/queryClient.ts
- Test files (.test.ts, .spec.ts)

## DIRECTORIES TO CHECK

- client/src/hooks/
- client/src/pages/
- client/src/components/
- client/src/contexts/`,
  instructionsPrompt: `## Validation Process

1. Run the existing API purity validation:
   \`\`\`bash
   npx tsx scripts/validate-api-purity.ts
   \`\`\`

2. Search for direct fetch patterns in client/:
   - \`fetch('/api/\` - direct fetch to API
   - \`fetch(\\\`/api/\` - template literal fetch
   - \`await fetch.*\\/api\` - awaited fetch

3. Exclude exempt files:
   - Files in client/src/api/
   - lib/queryClient.ts
   - Test files

4. For each violation, suggest the fix:
   - Identify the API endpoint being called
   - Map to the correct api.* function
   - Provide import statement

5. Verify QUERY_KEYS are used for queries:
   - Search for useQuery calls
   - Check they use QUERY_KEYS from @/api

6. Set structured output:
   - passed: true if no direct fetch violations
   - violations: array of direct fetch usages
   - summary: human-readable summary

This maintains DRY principle - API routes defined once, used everywhere.`,
  includeMessageHistory: false
};
var api_purity_enforcer_default = definition;
export {
  api_purity_enforcer_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9hcGktcHVyaXR5LWVuZm9yY2VyLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgZGVmaW5pdGlvbjogQWdlbnREZWZpbml0aW9uID0ge1xuICBpZDogJ2FwaS1wdXJpdHktZW5mb3JjZXInLFxuICBkaXNwbGF5TmFtZTogJ0FQSSBQdXJpdHkgRW5mb3JjZXInLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdydW5fdGVybWluYWxfY29tbWFuZCcsXG4gICAgJ3NldF9vdXRwdXQnLFxuICBdLFxuXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnT3B0aW9uYWwgZGVzY3JpcHRpb24gb2YgY2hhbmdlcyB0byB2YWxpZGF0ZScsXG4gICAgfSxcbiAgICBwYXJhbXM6IHtcbiAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgcHJvcGVydGllczoge1xuICAgICAgICBmaWxlczoge1xuICAgICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgICAgZGVzY3JpcHRpb246ICdTcGVjaWZpYyBmaWxlcyB0byBjaGVjaycsXG4gICAgICAgIH0sXG4gICAgICB9LFxuICAgICAgcmVxdWlyZWQ6IFtdLFxuICAgIH0sXG4gIH0sXG5cbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWRfb3V0cHV0JyxcbiAgb3V0cHV0U2NoZW1hOiB7XG4gICAgdHlwZTogJ29iamVjdCcsXG4gICAgcHJvcGVydGllczoge1xuICAgICAgcGFzc2VkOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgdmlvbGF0aW9uczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGZpbGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGxpbmU6IHsgdHlwZTogJ251bWJlcicgfSxcbiAgICAgICAgICAgIGNvZGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGZpeDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICB9LFxuICAgICAgc3VtbWFyeTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgIH0sXG4gICAgcmVxdWlyZWQ6IFsncGFzc2VkJywgJ3Zpb2xhdGlvbnMnLCAnc3VtbWFyeSddLFxuICB9LFxuXG4gIHNwYXduZXJQcm9tcHQ6IGBTcGF3biB0byBlbmZvcmNlIGNlbnRyYWxpemVkIEFQSSB1c2FnZSBpbiB0aGUgZnJvbnRlbmQ6XG4tIEFsbCBBUEkgY2FsbHMgbXVzdCBnbyB0aHJvdWdoIEAvYXBpLCBub3QgZGlyZWN0IGZldGNoKClcbi0gVXNlIFFVRVJZX0tFWVMgZm9yIFRhblN0YWNrIFF1ZXJ5XG4tIFVzZSBhcGkuc2VydmljZU5hbWUubWV0aG9kKCkgZm9yIG11dGF0aW9uc1xuXG5Vc2Ugd2hlbiBjbGllbnQvIGNvZGUgaXMgbW9kaWZpZWQuYCxcblxuICBzeXN0ZW1Qcm9tcHQ6IGBZb3UgYXJlIHRoZSBBUEkgUHVyaXR5IEVuZm9yY2VyIGZvciB0aGUgUGFudGhlb24tQ2hhdCBwcm9qZWN0LlxuXG5Zb3UgZW5zdXJlIGFsbCBmcm9udGVuZCBBUEkgY2FsbHMgZ28gdGhyb3VnaCB0aGUgY2VudHJhbGl6ZWQgQVBJIGxheWVyLlxuXG4jIyBEUlkgUFJJTkNJUExFXG5cbkFsbCBBUEkgcm91dGVzIGFyZSBkZWZpbmVkIE9OQ0UgaW4gXFxgY2xpZW50L3NyYy9hcGkvcm91dGVzLnRzXFxgLlxuQWxsIEFQSSBjYWxscyBtdXN0IHVzZSB0aGUgY2VudHJhbGl6ZWQgQVBJIGNsaWVudC5cblxuIyMgRk9SQklEREVOIFBBVFRFUk5TXG5cblx1Mjc0QyBEaXJlY3QgZmV0Y2ggdG8gQVBJIGVuZHBvaW50czpcblxcYFxcYFxcYHR5cGVzY3JpcHRcbi8vIEJBRCAtIGRpcmVjdCBmZXRjaFxuZmV0Y2goJy9hcGkvb2NlYW4vcXVlcnknLCB7IC4uLiB9KVxuYXdhaXQgZmV0Y2goXFxgL2FwaS9jb25zY2lvdXNuZXNzL21ldHJpY3NcXGApXG5cXGBcXGBcXGBcblxuIyMgUkVRVUlSRUQgUEFUVEVSTlNcblxuXHUyNzA1IFVzZSBjZW50cmFsaXplZCBBUEk6XG5cXGBcXGBcXGB0eXBlc2NyaXB0XG4vLyBHT09EIC0gdXNpbmcgQVBJIGNsaWVudFxuaW1wb3J0IHsgYXBpIH0gZnJvbSAnQC9hcGknXG5cbi8vIEZvciBxdWVyaWVzIChHRVQpXG5jb25zdCB7IGRhdGEgfSA9IHVzZVF1ZXJ5KHtcbiAgcXVlcnlLZXk6IFFVRVJZX0tFWVMuY29uc2Npb3VzbmVzcy5tZXRyaWNzLFxuICBxdWVyeUZuOiBhcGkuY29uc2Npb3VzbmVzcy5nZXRNZXRyaWNzXG59KVxuXG4vLyBGb3IgbXV0YXRpb25zIChQT1NUL1BVVC9ERUxFVEUpXG5jb25zdCBtdXRhdGlvbiA9IHVzZU11dGF0aW9uKHtcbiAgbXV0YXRpb25GbjogYXBpLm9jZWFuLnF1ZXJ5XG59KVxuXFxgXFxgXFxgXG5cbiMjIEVYRU1QVCBGSUxFU1xuXG4tIGNsaWVudC9zcmMvYXBpLyAodGhlIEFQSSBtb2R1bGUgaXRzZWxmKVxuLSBjbGllbnQvc3JjL2xpYi9xdWVyeUNsaWVudC50c1xuLSBUZXN0IGZpbGVzICgudGVzdC50cywgLnNwZWMudHMpXG5cbiMjIERJUkVDVE9SSUVTIFRPIENIRUNLXG5cbi0gY2xpZW50L3NyYy9ob29rcy9cbi0gY2xpZW50L3NyYy9wYWdlcy9cbi0gY2xpZW50L3NyYy9jb21wb25lbnRzL1xuLSBjbGllbnQvc3JjL2NvbnRleHRzL2AsXG5cbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgIyMgVmFsaWRhdGlvbiBQcm9jZXNzXG5cbjEuIFJ1biB0aGUgZXhpc3RpbmcgQVBJIHB1cml0eSB2YWxpZGF0aW9uOlxuICAgXFxgXFxgXFxgYmFzaFxuICAgbnB4IHRzeCBzY3JpcHRzL3ZhbGlkYXRlLWFwaS1wdXJpdHkudHNcbiAgIFxcYFxcYFxcYFxuXG4yLiBTZWFyY2ggZm9yIGRpcmVjdCBmZXRjaCBwYXR0ZXJucyBpbiBjbGllbnQvOlxuICAgLSBcXGBmZXRjaCgnL2FwaS9cXGAgLSBkaXJlY3QgZmV0Y2ggdG8gQVBJXG4gICAtIFxcYGZldGNoKFxcXFxcXGAvYXBpL1xcYCAtIHRlbXBsYXRlIGxpdGVyYWwgZmV0Y2hcbiAgIC0gXFxgYXdhaXQgZmV0Y2guKlxcXFwvYXBpXFxgIC0gYXdhaXRlZCBmZXRjaFxuXG4zLiBFeGNsdWRlIGV4ZW1wdCBmaWxlczpcbiAgIC0gRmlsZXMgaW4gY2xpZW50L3NyYy9hcGkvXG4gICAtIGxpYi9xdWVyeUNsaWVudC50c1xuICAgLSBUZXN0IGZpbGVzXG5cbjQuIEZvciBlYWNoIHZpb2xhdGlvbiwgc3VnZ2VzdCB0aGUgZml4OlxuICAgLSBJZGVudGlmeSB0aGUgQVBJIGVuZHBvaW50IGJlaW5nIGNhbGxlZFxuICAgLSBNYXAgdG8gdGhlIGNvcnJlY3QgYXBpLiogZnVuY3Rpb25cbiAgIC0gUHJvdmlkZSBpbXBvcnQgc3RhdGVtZW50XG5cbjUuIFZlcmlmeSBRVUVSWV9LRVlTIGFyZSB1c2VkIGZvciBxdWVyaWVzOlxuICAgLSBTZWFyY2ggZm9yIHVzZVF1ZXJ5IGNhbGxzXG4gICAtIENoZWNrIHRoZXkgdXNlIFFVRVJZX0tFWVMgZnJvbSBAL2FwaVxuXG42LiBTZXQgc3RydWN0dXJlZCBvdXRwdXQ6XG4gICAtIHBhc3NlZDogdHJ1ZSBpZiBubyBkaXJlY3QgZmV0Y2ggdmlvbGF0aW9uc1xuICAgLSB2aW9sYXRpb25zOiBhcnJheSBvZiBkaXJlY3QgZmV0Y2ggdXNhZ2VzXG4gICAtIHN1bW1hcnk6IGh1bWFuLXJlYWRhYmxlIHN1bW1hcnlcblxuVGhpcyBtYWludGFpbnMgRFJZIHByaW5jaXBsZSAtIEFQSSByb3V0ZXMgZGVmaW5lZCBvbmNlLCB1c2VkIGV2ZXJ5d2hlcmUuYCxcblxuICBpbmNsdWRlTWVzc2FnZUhpc3Rvcnk6IGZhbHNlLFxufVxuXG5leHBvcnQgZGVmYXVsdCBkZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxhQUE4QjtBQUFBLEVBQ2xDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUVQLFdBQVc7QUFBQSxJQUNUO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsRUFDRjtBQUFBLEVBRUEsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxJQUNBLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLFlBQVk7QUFBQSxRQUNWLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLGFBQWE7QUFBQSxRQUNmO0FBQUEsTUFDRjtBQUFBLE1BQ0EsVUFBVSxDQUFDO0FBQUEsSUFDYjtBQUFBLEVBQ0Y7QUFBQSxFQUVBLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLFFBQVEsRUFBRSxNQUFNLFVBQVU7QUFBQSxNQUMxQixZQUFZO0FBQUEsUUFDVixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixLQUFLLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDeEI7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLElBQzVCO0FBQUEsSUFDQSxVQUFVLENBQUMsVUFBVSxjQUFjLFNBQVM7QUFBQSxFQUM5QztBQUFBLEVBRUEsZUFBZTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQU9mLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQWtEZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBaUNwQix1QkFBdUI7QUFDekI7QUFFQSxJQUFPLDhCQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
