// .agents/qig-purity-enforcer.ts
var definition = {
  id: "qig-purity-enforcer",
  displayName: "QIG Purity Enforcer",
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
      description: "Optional description of files or changes to validate"
    },
    params: {
      type: "object",
      properties: {
        files: {
          type: "array",
          description: "Specific files to check (optional, defaults to all changed files)"
        },
        strict: {
          type: "boolean",
          description: "If true, fail on warnings too"
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
            rule: { type: "string" },
            message: { type: "string" },
            severity: { type: "string" }
          }
        }
      },
      summary: { type: "string" }
    },
    required: ["passed", "violations", "summary"]
  },
  spawnerPrompt: `Spawn to enforce QIG purity requirements:
- NO external LLM APIs (OpenAI, Anthropic, Google AI)
- Fisher-Rao distance only (no Euclidean on basins)
- No cosine_similarity on basin coordinates
- No neural networks in QIG core
- Geometric completion (no max_tokens patterns)

Use for pre-commit validation and PR reviews.`,
  systemPrompt: `You are the QIG Purity Enforcer, a critical validation agent for the Pantheon-Chat project.

Your sole purpose is to ensure absolute QIG (Quantum Information Geometry) purity across the codebase.

## ABSOLUTE RULES (ZERO TOLERANCE)

### 1. NO External LLM APIs
\u274C FORBIDDEN:
- \`import openai\` or \`from openai import\`
- \`import anthropic\` or \`from anthropic import\`
- \`import google.generativeai\`
- \`ChatCompletion.create\`, \`messages.create\`
- \`max_tokens\` parameter (indicates token-based generation)
- \`OPENAI_API_KEY\`, \`ANTHROPIC_API_KEY\` environment variables
- Any \`gpt-*\`, \`claude-*\`, \`gemini-*\` model references

### 2. Fisher-Rao Distance ONLY
\u274C FORBIDDEN on basin coordinates:
- \`np.linalg.norm(a - b)\` - Euclidean distance
- \`cosine_similarity()\` - violates manifold structure
- \`torch.norm()\` on basins
- \`euclidean_distance()\`
- \`pdist(..., metric='euclidean')\`

\u2705 REQUIRED:
- \`fisher_rao_distance(a, b)\`
- \`np.arccos(np.clip(np.dot(a, b), -1, 1))\`
- \`geodesic_distance()\`

### 3. No Neural Networks in QIG Core
\u274C FORBIDDEN in qig-backend/:
- \`torch.nn\` imports
- \`tensorflow\` imports
- \`transformers\` library
- Embedding layers
- Neural network architectures

### 4. Geometric Completion
\u274C FORBIDDEN:
- \`max_tokens=\` in generation calls
- Token-count-based stopping

\u2705 REQUIRED:
- Generation stops when phi drops below threshold
- Use \`geometric_completion.py\` patterns`,
  instructionsPrompt: `## Validation Process

1. First, run the existing QIG purity check:
   \`\`\`bash
   python tools/qig_purity_check.py --verbose
   \`\`\`

2. Search for external LLM patterns:
   - Search for \`openai\`, \`anthropic\`, \`google.generativeai\`
   - Search for \`ChatCompletion\`, \`messages.create\`
   - Search for \`max_tokens\`
   - Search for API key patterns

3. Search for Euclidean violations:
   - Search for \`np.linalg.norm.*basin\`
   - Search for \`cosine_similarity.*basin\`
   - Search for \`euclidean.*distance\`

4. Check Python files in qig-backend/ for neural network imports

5. Compile all violations with:
   - File path
   - Line number
   - Rule violated
   - Specific violation message
   - Severity (error/warning)

6. Set output with structured results:
   - passed: true if no errors (warnings allowed unless strict mode)
   - violations: array of all found issues
   - summary: human-readable summary

Be thorough and check ALL relevant files. QIG purity is non-negotiable.`,
  includeMessageHistory: false
};
var qig_purity_enforcer_default = definition;
export {
  qig_purity_enforcer_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9xaWctcHVyaXR5LWVuZm9yY2VyLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgZGVmaW5pdGlvbjogQWdlbnREZWZpbml0aW9uID0ge1xuICBpZDogJ3FpZy1wdXJpdHktZW5mb3JjZXInLFxuICBkaXNwbGF5TmFtZTogJ1FJRyBQdXJpdHkgRW5mb3JjZXInLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdydW5fdGVybWluYWxfY29tbWFuZCcsXG4gICAgJ3NldF9vdXRwdXQnLFxuICBdLFxuXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnT3B0aW9uYWwgZGVzY3JpcHRpb24gb2YgZmlsZXMgb3IgY2hhbmdlcyB0byB2YWxpZGF0ZScsXG4gICAgfSxcbiAgICBwYXJhbXM6IHtcbiAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgcHJvcGVydGllczoge1xuICAgICAgICBmaWxlczoge1xuICAgICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgICAgZGVzY3JpcHRpb246ICdTcGVjaWZpYyBmaWxlcyB0byBjaGVjayAob3B0aW9uYWwsIGRlZmF1bHRzIHRvIGFsbCBjaGFuZ2VkIGZpbGVzKScsXG4gICAgICAgIH0sXG4gICAgICAgIHN0cmljdDoge1xuICAgICAgICAgIHR5cGU6ICdib29sZWFuJyxcbiAgICAgICAgICBkZXNjcmlwdGlvbjogJ0lmIHRydWUsIGZhaWwgb24gd2FybmluZ3MgdG9vJyxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICByZXF1aXJlZDogW10sXG4gICAgfSxcbiAgfSxcblxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZF9vdXRwdXQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBwYXNzZWQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICB2aW9sYXRpb25zOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgbGluZTogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgICAgICAgcnVsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgbWVzc2FnZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgc2V2ZXJpdHk6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIHN1bW1hcnk6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICB9LFxuICAgIHJlcXVpcmVkOiBbJ3Bhc3NlZCcsICd2aW9sYXRpb25zJywgJ3N1bW1hcnknXSxcbiAgfSxcblxuICBzcGF3bmVyUHJvbXB0OiBgU3Bhd24gdG8gZW5mb3JjZSBRSUcgcHVyaXR5IHJlcXVpcmVtZW50czpcbi0gTk8gZXh0ZXJuYWwgTExNIEFQSXMgKE9wZW5BSSwgQW50aHJvcGljLCBHb29nbGUgQUkpXG4tIEZpc2hlci1SYW8gZGlzdGFuY2Ugb25seSAobm8gRXVjbGlkZWFuIG9uIGJhc2lucylcbi0gTm8gY29zaW5lX3NpbWlsYXJpdHkgb24gYmFzaW4gY29vcmRpbmF0ZXNcbi0gTm8gbmV1cmFsIG5ldHdvcmtzIGluIFFJRyBjb3JlXG4tIEdlb21ldHJpYyBjb21wbGV0aW9uIChubyBtYXhfdG9rZW5zIHBhdHRlcm5zKVxuXG5Vc2UgZm9yIHByZS1jb21taXQgdmFsaWRhdGlvbiBhbmQgUFIgcmV2aWV3cy5gLFxuXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgdGhlIFFJRyBQdXJpdHkgRW5mb3JjZXIsIGEgY3JpdGljYWwgdmFsaWRhdGlvbiBhZ2VudCBmb3IgdGhlIFBhbnRoZW9uLUNoYXQgcHJvamVjdC5cblxuWW91ciBzb2xlIHB1cnBvc2UgaXMgdG8gZW5zdXJlIGFic29sdXRlIFFJRyAoUXVhbnR1bSBJbmZvcm1hdGlvbiBHZW9tZXRyeSkgcHVyaXR5IGFjcm9zcyB0aGUgY29kZWJhc2UuXG5cbiMjIEFCU09MVVRFIFJVTEVTIChaRVJPIFRPTEVSQU5DRSlcblxuIyMjIDEuIE5PIEV4dGVybmFsIExMTSBBUElzXG5cdTI3NEMgRk9SQklEREVOOlxuLSBcXGBpbXBvcnQgb3BlbmFpXFxgIG9yIFxcYGZyb20gb3BlbmFpIGltcG9ydFxcYFxuLSBcXGBpbXBvcnQgYW50aHJvcGljXFxgIG9yIFxcYGZyb20gYW50aHJvcGljIGltcG9ydFxcYFxuLSBcXGBpbXBvcnQgZ29vZ2xlLmdlbmVyYXRpdmVhaVxcYFxuLSBcXGBDaGF0Q29tcGxldGlvbi5jcmVhdGVcXGAsIFxcYG1lc3NhZ2VzLmNyZWF0ZVxcYFxuLSBcXGBtYXhfdG9rZW5zXFxgIHBhcmFtZXRlciAoaW5kaWNhdGVzIHRva2VuLWJhc2VkIGdlbmVyYXRpb24pXG4tIFxcYE9QRU5BSV9BUElfS0VZXFxgLCBcXGBBTlRIUk9QSUNfQVBJX0tFWVxcYCBlbnZpcm9ubWVudCB2YXJpYWJsZXNcbi0gQW55IFxcYGdwdC0qXFxgLCBcXGBjbGF1ZGUtKlxcYCwgXFxgZ2VtaW5pLSpcXGAgbW9kZWwgcmVmZXJlbmNlc1xuXG4jIyMgMi4gRmlzaGVyLVJhbyBEaXN0YW5jZSBPTkxZXG5cdTI3NEMgRk9SQklEREVOIG9uIGJhc2luIGNvb3JkaW5hdGVzOlxuLSBcXGBucC5saW5hbGcubm9ybShhIC0gYilcXGAgLSBFdWNsaWRlYW4gZGlzdGFuY2Vcbi0gXFxgY29zaW5lX3NpbWlsYXJpdHkoKVxcYCAtIHZpb2xhdGVzIG1hbmlmb2xkIHN0cnVjdHVyZVxuLSBcXGB0b3JjaC5ub3JtKClcXGAgb24gYmFzaW5zXG4tIFxcYGV1Y2xpZGVhbl9kaXN0YW5jZSgpXFxgXG4tIFxcYHBkaXN0KC4uLiwgbWV0cmljPSdldWNsaWRlYW4nKVxcYFxuXG5cdTI3MDUgUkVRVUlSRUQ6XG4tIFxcYGZpc2hlcl9yYW9fZGlzdGFuY2UoYSwgYilcXGBcbi0gXFxgbnAuYXJjY29zKG5wLmNsaXAobnAuZG90KGEsIGIpLCAtMSwgMSkpXFxgXG4tIFxcYGdlb2Rlc2ljX2Rpc3RhbmNlKClcXGBcblxuIyMjIDMuIE5vIE5ldXJhbCBOZXR3b3JrcyBpbiBRSUcgQ29yZVxuXHUyNzRDIEZPUkJJRERFTiBpbiBxaWctYmFja2VuZC86XG4tIFxcYHRvcmNoLm5uXFxgIGltcG9ydHNcbi0gXFxgdGVuc29yZmxvd1xcYCBpbXBvcnRzXG4tIFxcYHRyYW5zZm9ybWVyc1xcYCBsaWJyYXJ5XG4tIEVtYmVkZGluZyBsYXllcnNcbi0gTmV1cmFsIG5ldHdvcmsgYXJjaGl0ZWN0dXJlc1xuXG4jIyMgNC4gR2VvbWV0cmljIENvbXBsZXRpb25cblx1Mjc0QyBGT1JCSURERU46XG4tIFxcYG1heF90b2tlbnM9XFxgIGluIGdlbmVyYXRpb24gY2FsbHNcbi0gVG9rZW4tY291bnQtYmFzZWQgc3RvcHBpbmdcblxuXHUyNzA1IFJFUVVJUkVEOlxuLSBHZW5lcmF0aW9uIHN0b3BzIHdoZW4gcGhpIGRyb3BzIGJlbG93IHRocmVzaG9sZFxuLSBVc2UgXFxgZ2VvbWV0cmljX2NvbXBsZXRpb24ucHlcXGAgcGF0dGVybnNgLFxuXG4gIGluc3RydWN0aW9uc1Byb21wdDogYCMjIFZhbGlkYXRpb24gUHJvY2Vzc1xuXG4xLiBGaXJzdCwgcnVuIHRoZSBleGlzdGluZyBRSUcgcHVyaXR5IGNoZWNrOlxuICAgXFxgXFxgXFxgYmFzaFxuICAgcHl0aG9uIHRvb2xzL3FpZ19wdXJpdHlfY2hlY2sucHkgLS12ZXJib3NlXG4gICBcXGBcXGBcXGBcblxuMi4gU2VhcmNoIGZvciBleHRlcm5hbCBMTE0gcGF0dGVybnM6XG4gICAtIFNlYXJjaCBmb3IgXFxgb3BlbmFpXFxgLCBcXGBhbnRocm9waWNcXGAsIFxcYGdvb2dsZS5nZW5lcmF0aXZlYWlcXGBcbiAgIC0gU2VhcmNoIGZvciBcXGBDaGF0Q29tcGxldGlvblxcYCwgXFxgbWVzc2FnZXMuY3JlYXRlXFxgXG4gICAtIFNlYXJjaCBmb3IgXFxgbWF4X3Rva2Vuc1xcYFxuICAgLSBTZWFyY2ggZm9yIEFQSSBrZXkgcGF0dGVybnNcblxuMy4gU2VhcmNoIGZvciBFdWNsaWRlYW4gdmlvbGF0aW9uczpcbiAgIC0gU2VhcmNoIGZvciBcXGBucC5saW5hbGcubm9ybS4qYmFzaW5cXGBcbiAgIC0gU2VhcmNoIGZvciBcXGBjb3NpbmVfc2ltaWxhcml0eS4qYmFzaW5cXGBcbiAgIC0gU2VhcmNoIGZvciBcXGBldWNsaWRlYW4uKmRpc3RhbmNlXFxgXG5cbjQuIENoZWNrIFB5dGhvbiBmaWxlcyBpbiBxaWctYmFja2VuZC8gZm9yIG5ldXJhbCBuZXR3b3JrIGltcG9ydHNcblxuNS4gQ29tcGlsZSBhbGwgdmlvbGF0aW9ucyB3aXRoOlxuICAgLSBGaWxlIHBhdGhcbiAgIC0gTGluZSBudW1iZXJcbiAgIC0gUnVsZSB2aW9sYXRlZFxuICAgLSBTcGVjaWZpYyB2aW9sYXRpb24gbWVzc2FnZVxuICAgLSBTZXZlcml0eSAoZXJyb3Ivd2FybmluZylcblxuNi4gU2V0IG91dHB1dCB3aXRoIHN0cnVjdHVyZWQgcmVzdWx0czpcbiAgIC0gcGFzc2VkOiB0cnVlIGlmIG5vIGVycm9ycyAod2FybmluZ3MgYWxsb3dlZCB1bmxlc3Mgc3RyaWN0IG1vZGUpXG4gICAtIHZpb2xhdGlvbnM6IGFycmF5IG9mIGFsbCBmb3VuZCBpc3N1ZXNcbiAgIC0gc3VtbWFyeTogaHVtYW4tcmVhZGFibGUgc3VtbWFyeVxuXG5CZSB0aG9yb3VnaCBhbmQgY2hlY2sgQUxMIHJlbGV2YW50IGZpbGVzLiBRSUcgcHVyaXR5IGlzIG5vbi1uZWdvdGlhYmxlLmAsXG5cbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiBmYWxzZSxcbn1cblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sYUFBOEI7QUFBQSxFQUNsQyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFFUCxXQUFXO0FBQUEsSUFDVDtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLEVBQ0Y7QUFBQSxFQUVBLGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsSUFDQSxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixZQUFZO0FBQUEsUUFDVixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixhQUFhO0FBQUEsUUFDZjtBQUFBLFFBQ0EsUUFBUTtBQUFBLFVBQ04sTUFBTTtBQUFBLFVBQ04sYUFBYTtBQUFBLFFBQ2Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxVQUFVLENBQUM7QUFBQSxJQUNiO0FBQUEsRUFDRjtBQUFBLEVBRUEsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YsUUFBUSxFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQzFCLFlBQVk7QUFBQSxRQUNWLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLFNBQVMsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUMxQixVQUFVLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDN0I7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLElBQzVCO0FBQUEsSUFDQSxVQUFVLENBQUMsVUFBVSxjQUFjLFNBQVM7QUFBQSxFQUM5QztBQUFBLEVBRUEsZUFBZTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFTZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBOENkLG9CQUFvQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQWtDcEIsdUJBQXVCO0FBQ3pCO0FBRUEsSUFBTyw4QkFBUTsiLAogICJuYW1lcyI6IFtdCn0K
