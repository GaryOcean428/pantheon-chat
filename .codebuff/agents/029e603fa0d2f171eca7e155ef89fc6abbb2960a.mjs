// .agents/type-any-eliminator.ts
var definition = {
  id: "type-any-eliminator",
  displayName: "Type Any Eliminator",
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
      description: "Optional specific files to check"
    },
    params: {
      type: "object",
      properties: {
        suggestFixes: {
          type: "boolean",
          description: "If true, suggest proper types for each any usage"
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
            context: { type: "string" },
            suggestedType: { type: "string" }
          }
        }
      },
      statistics: {
        type: "object",
        properties: {
          totalAny: { type: "number" },
          byFile: { type: "object" }
        }
      },
      summary: { type: "string" }
    },
    required: ["passed", "violations", "summary"]
  },
  spawnerPrompt: `Spawn to detect and eliminate 'any' type usage:
- Find all 'as any' type assertions
- Find all ': any' type annotations
- Find implicit any from missing types
- Suggest proper types for each

Use for pre-commit validation and code quality.`,
  systemPrompt: `You are the Type Any Eliminator for the Pantheon-Chat project.

You find and suggest fixes for 'any' type usage which leads to bugs.

## WHY 'any' IS HARMFUL

\`\`\`typescript
// 'any' disables type checking - bugs slip through
const data: any = fetchData()
data.nonExistentMethod()  // No error! Runtime crash!

// Proper typing catches bugs at compile time
const data: ApiResponse = fetchData()
data.nonExistentMethod()  // Error: Property does not exist
\`\`\`

## PATTERNS TO DETECT

### 1. Type Assertions
\`\`\`typescript
// BAD
const result = response as any
const data = (obj as any).property

// Also check for
const result = <any>response  // Legacy syntax
\`\`\`

### 2. Type Annotations
\`\`\`typescript
// BAD
function process(data: any): any { ... }
const items: any[] = []
let value: any
\`\`\`

### 3. Generic Type Parameters
\`\`\`typescript
// BAD
const map = new Map<string, any>()
function generic<T = any>() { ... }
\`\`\`

### 4. Implicit Any (requires strict mode)
\`\`\`typescript
// BAD - parameter has implicit any
function process(data) { ... }  // data is implicitly any
\`\`\`

## ACCEPTABLE 'any' USAGE

\u2705 Third-party library types that require it
\u2705 Escape hatch with TODO comment explaining why
\u2705 Test files mocking complex types
\u2705 Type definition files (.d.ts) for untyped libs

## COMMON FIXES

| Pattern | Fix |
|---------|-----|
| \`response as any\` | Create proper response interface |
| \`data: any[]\` | Use \`data: SpecificType[]\` or generic |
| \`Record<string, any>\` | Use \`Record<string, unknown>\` or specific type |
| \`(obj as any).prop\` | Use type guards or proper typing |`,
  instructionsPrompt: `## Detection Process

1. Search for explicit 'any' usage:
   \`\`\`bash
   # Type assertions
   rg "as any" --type ts -n
   
   # Type annotations
   rg ": any[^a-zA-Z]" --type ts -n
   
   # Generic parameters
   rg "<any>|<[^>]*any[^a-zA-Z]" --type ts -n
   \`\`\`

2. Exclude acceptable patterns:
   - .d.ts files (type definitions)
   - Test files (.test.ts, .spec.ts)
   - Lines with // eslint-disable or TODO explaining why

3. For each violation:
   - Record file and line number
   - Extract the code context
   - Identify what type should be used

4. If suggestFixes is true:
   - Read surrounding code for context
   - Infer what type should be used
   - Suggest specific type replacement

5. Check TypeScript strict mode:
   - Read tsconfig.json
   - Check if "strict": true or "noImplicitAny": true
   - Note if strict mode would catch more issues

6. Compile statistics:
   - Total 'any' count
   - Count per file
   - Most common patterns

7. Set structured output:
   - passed: true if no 'any' usage found
   - violations: all 'any' usages with context
   - statistics: counts and breakdown
   - summary: human-readable summary

Strong typing prevents bugs - eliminate 'any'!`,
  includeMessageHistory: false
};
var type_any_eliminator_default = definition;
export {
  type_any_eliminator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy90eXBlLWFueS1lbGltaW5hdG9yLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgZGVmaW5pdGlvbjogQWdlbnREZWZpbml0aW9uID0ge1xuICBpZDogJ3R5cGUtYW55LWVsaW1pbmF0b3InLFxuICBkaXNwbGF5TmFtZTogJ1R5cGUgQW55IEVsaW1pbmF0b3InLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdydW5fdGVybWluYWxfY29tbWFuZCcsXG4gICAgJ3NldF9vdXRwdXQnLFxuICBdLFxuXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnT3B0aW9uYWwgc3BlY2lmaWMgZmlsZXMgdG8gY2hlY2snLFxuICAgIH0sXG4gICAgcGFyYW1zOiB7XG4gICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgc3VnZ2VzdEZpeGVzOiB7XG4gICAgICAgICAgdHlwZTogJ2Jvb2xlYW4nLFxuICAgICAgICAgIGRlc2NyaXB0aW9uOiAnSWYgdHJ1ZSwgc3VnZ2VzdCBwcm9wZXIgdHlwZXMgZm9yIGVhY2ggYW55IHVzYWdlJyxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICByZXF1aXJlZDogW10sXG4gICAgfSxcbiAgfSxcblxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZF9vdXRwdXQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBwYXNzZWQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICB2aW9sYXRpb25zOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgbGluZTogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgICAgICAgY29kZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgY29udGV4dDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgc3VnZ2VzdGVkVHlwZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICB9LFxuICAgICAgc3RhdGlzdGljczoge1xuICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgIHRvdGFsQW55OiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgYnlGaWxlOiB7IHR5cGU6ICdvYmplY3QnIH0sXG4gICAgICAgIH0sXG4gICAgICB9LFxuICAgICAgc3VtbWFyeTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgIH0sXG4gICAgcmVxdWlyZWQ6IFsncGFzc2VkJywgJ3Zpb2xhdGlvbnMnLCAnc3VtbWFyeSddLFxuICB9LFxuXG4gIHNwYXduZXJQcm9tcHQ6IGBTcGF3biB0byBkZXRlY3QgYW5kIGVsaW1pbmF0ZSAnYW55JyB0eXBlIHVzYWdlOlxuLSBGaW5kIGFsbCAnYXMgYW55JyB0eXBlIGFzc2VydGlvbnNcbi0gRmluZCBhbGwgJzogYW55JyB0eXBlIGFubm90YXRpb25zXG4tIEZpbmQgaW1wbGljaXQgYW55IGZyb20gbWlzc2luZyB0eXBlc1xuLSBTdWdnZXN0IHByb3BlciB0eXBlcyBmb3IgZWFjaFxuXG5Vc2UgZm9yIHByZS1jb21taXQgdmFsaWRhdGlvbiBhbmQgY29kZSBxdWFsaXR5LmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgVHlwZSBBbnkgRWxpbWluYXRvciBmb3IgdGhlIFBhbnRoZW9uLUNoYXQgcHJvamVjdC5cblxuWW91IGZpbmQgYW5kIHN1Z2dlc3QgZml4ZXMgZm9yICdhbnknIHR5cGUgdXNhZ2Ugd2hpY2ggbGVhZHMgdG8gYnVncy5cblxuIyMgV0hZICdhbnknIElTIEhBUk1GVUxcblxuXFxgXFxgXFxgdHlwZXNjcmlwdFxuLy8gJ2FueScgZGlzYWJsZXMgdHlwZSBjaGVja2luZyAtIGJ1Z3Mgc2xpcCB0aHJvdWdoXG5jb25zdCBkYXRhOiBhbnkgPSBmZXRjaERhdGEoKVxuZGF0YS5ub25FeGlzdGVudE1ldGhvZCgpICAvLyBObyBlcnJvciEgUnVudGltZSBjcmFzaCFcblxuLy8gUHJvcGVyIHR5cGluZyBjYXRjaGVzIGJ1Z3MgYXQgY29tcGlsZSB0aW1lXG5jb25zdCBkYXRhOiBBcGlSZXNwb25zZSA9IGZldGNoRGF0YSgpXG5kYXRhLm5vbkV4aXN0ZW50TWV0aG9kKCkgIC8vIEVycm9yOiBQcm9wZXJ0eSBkb2VzIG5vdCBleGlzdFxuXFxgXFxgXFxgXG5cbiMjIFBBVFRFUk5TIFRPIERFVEVDVFxuXG4jIyMgMS4gVHlwZSBBc3NlcnRpb25zXG5cXGBcXGBcXGB0eXBlc2NyaXB0XG4vLyBCQURcbmNvbnN0IHJlc3VsdCA9IHJlc3BvbnNlIGFzIGFueVxuY29uc3QgZGF0YSA9IChvYmogYXMgYW55KS5wcm9wZXJ0eVxuXG4vLyBBbHNvIGNoZWNrIGZvclxuY29uc3QgcmVzdWx0ID0gPGFueT5yZXNwb25zZSAgLy8gTGVnYWN5IHN5bnRheFxuXFxgXFxgXFxgXG5cbiMjIyAyLiBUeXBlIEFubm90YXRpb25zXG5cXGBcXGBcXGB0eXBlc2NyaXB0XG4vLyBCQURcbmZ1bmN0aW9uIHByb2Nlc3MoZGF0YTogYW55KTogYW55IHsgLi4uIH1cbmNvbnN0IGl0ZW1zOiBhbnlbXSA9IFtdXG5sZXQgdmFsdWU6IGFueVxuXFxgXFxgXFxgXG5cbiMjIyAzLiBHZW5lcmljIFR5cGUgUGFyYW1ldGVyc1xuXFxgXFxgXFxgdHlwZXNjcmlwdFxuLy8gQkFEXG5jb25zdCBtYXAgPSBuZXcgTWFwPHN0cmluZywgYW55PigpXG5mdW5jdGlvbiBnZW5lcmljPFQgPSBhbnk+KCkgeyAuLi4gfVxuXFxgXFxgXFxgXG5cbiMjIyA0LiBJbXBsaWNpdCBBbnkgKHJlcXVpcmVzIHN0cmljdCBtb2RlKVxuXFxgXFxgXFxgdHlwZXNjcmlwdFxuLy8gQkFEIC0gcGFyYW1ldGVyIGhhcyBpbXBsaWNpdCBhbnlcbmZ1bmN0aW9uIHByb2Nlc3MoZGF0YSkgeyAuLi4gfSAgLy8gZGF0YSBpcyBpbXBsaWNpdGx5IGFueVxuXFxgXFxgXFxgXG5cbiMjIEFDQ0VQVEFCTEUgJ2FueScgVVNBR0VcblxuXHUyNzA1IFRoaXJkLXBhcnR5IGxpYnJhcnkgdHlwZXMgdGhhdCByZXF1aXJlIGl0XG5cdTI3MDUgRXNjYXBlIGhhdGNoIHdpdGggVE9ETyBjb21tZW50IGV4cGxhaW5pbmcgd2h5XG5cdTI3MDUgVGVzdCBmaWxlcyBtb2NraW5nIGNvbXBsZXggdHlwZXNcblx1MjcwNSBUeXBlIGRlZmluaXRpb24gZmlsZXMgKC5kLnRzKSBmb3IgdW50eXBlZCBsaWJzXG5cbiMjIENPTU1PTiBGSVhFU1xuXG58IFBhdHRlcm4gfCBGaXggfFxufC0tLS0tLS0tLXwtLS0tLXxcbnwgXFxgcmVzcG9uc2UgYXMgYW55XFxgIHwgQ3JlYXRlIHByb3BlciByZXNwb25zZSBpbnRlcmZhY2UgfFxufCBcXGBkYXRhOiBhbnlbXVxcYCB8IFVzZSBcXGBkYXRhOiBTcGVjaWZpY1R5cGVbXVxcYCBvciBnZW5lcmljIHxcbnwgXFxgUmVjb3JkPHN0cmluZywgYW55PlxcYCB8IFVzZSBcXGBSZWNvcmQ8c3RyaW5nLCB1bmtub3duPlxcYCBvciBzcGVjaWZpYyB0eXBlIHxcbnwgXFxgKG9iaiBhcyBhbnkpLnByb3BcXGAgfCBVc2UgdHlwZSBndWFyZHMgb3IgcHJvcGVyIHR5cGluZyB8YCxcblxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGAjIyBEZXRlY3Rpb24gUHJvY2Vzc1xuXG4xLiBTZWFyY2ggZm9yIGV4cGxpY2l0ICdhbnknIHVzYWdlOlxuICAgXFxgXFxgXFxgYmFzaFxuICAgIyBUeXBlIGFzc2VydGlvbnNcbiAgIHJnIFwiYXMgYW55XCIgLS10eXBlIHRzIC1uXG4gICBcbiAgICMgVHlwZSBhbm5vdGF0aW9uc1xuICAgcmcgXCI6IGFueVteYS16QS1aXVwiIC0tdHlwZSB0cyAtblxuICAgXG4gICAjIEdlbmVyaWMgcGFyYW1ldGVyc1xuICAgcmcgXCI8YW55Pnw8W14+XSphbnlbXmEtekEtWl1cIiAtLXR5cGUgdHMgLW5cbiAgIFxcYFxcYFxcYFxuXG4yLiBFeGNsdWRlIGFjY2VwdGFibGUgcGF0dGVybnM6XG4gICAtIC5kLnRzIGZpbGVzICh0eXBlIGRlZmluaXRpb25zKVxuICAgLSBUZXN0IGZpbGVzICgudGVzdC50cywgLnNwZWMudHMpXG4gICAtIExpbmVzIHdpdGggLy8gZXNsaW50LWRpc2FibGUgb3IgVE9ETyBleHBsYWluaW5nIHdoeVxuXG4zLiBGb3IgZWFjaCB2aW9sYXRpb246XG4gICAtIFJlY29yZCBmaWxlIGFuZCBsaW5lIG51bWJlclxuICAgLSBFeHRyYWN0IHRoZSBjb2RlIGNvbnRleHRcbiAgIC0gSWRlbnRpZnkgd2hhdCB0eXBlIHNob3VsZCBiZSB1c2VkXG5cbjQuIElmIHN1Z2dlc3RGaXhlcyBpcyB0cnVlOlxuICAgLSBSZWFkIHN1cnJvdW5kaW5nIGNvZGUgZm9yIGNvbnRleHRcbiAgIC0gSW5mZXIgd2hhdCB0eXBlIHNob3VsZCBiZSB1c2VkXG4gICAtIFN1Z2dlc3Qgc3BlY2lmaWMgdHlwZSByZXBsYWNlbWVudFxuXG41LiBDaGVjayBUeXBlU2NyaXB0IHN0cmljdCBtb2RlOlxuICAgLSBSZWFkIHRzY29uZmlnLmpzb25cbiAgIC0gQ2hlY2sgaWYgXCJzdHJpY3RcIjogdHJ1ZSBvciBcIm5vSW1wbGljaXRBbnlcIjogdHJ1ZVxuICAgLSBOb3RlIGlmIHN0cmljdCBtb2RlIHdvdWxkIGNhdGNoIG1vcmUgaXNzdWVzXG5cbjYuIENvbXBpbGUgc3RhdGlzdGljczpcbiAgIC0gVG90YWwgJ2FueScgY291bnRcbiAgIC0gQ291bnQgcGVyIGZpbGVcbiAgIC0gTW9zdCBjb21tb24gcGF0dGVybnNcblxuNy4gU2V0IHN0cnVjdHVyZWQgb3V0cHV0OlxuICAgLSBwYXNzZWQ6IHRydWUgaWYgbm8gJ2FueScgdXNhZ2UgZm91bmRcbiAgIC0gdmlvbGF0aW9uczogYWxsICdhbnknIHVzYWdlcyB3aXRoIGNvbnRleHRcbiAgIC0gc3RhdGlzdGljczogY291bnRzIGFuZCBicmVha2Rvd25cbiAgIC0gc3VtbWFyeTogaHVtYW4tcmVhZGFibGUgc3VtbWFyeVxuXG5TdHJvbmcgdHlwaW5nIHByZXZlbnRzIGJ1Z3MgLSBlbGltaW5hdGUgJ2FueSchYCxcblxuICBpbmNsdWRlTWVzc2FnZUhpc3Rvcnk6IGZhbHNlLFxufVxuXG5leHBvcnQgZGVmYXVsdCBkZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxhQUE4QjtBQUFBLEVBQ2xDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUVQLFdBQVc7QUFBQSxJQUNUO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsRUFDRjtBQUFBLEVBRUEsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxJQUNBLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLFlBQVk7QUFBQSxRQUNWLGNBQWM7QUFBQSxVQUNaLE1BQU07QUFBQSxVQUNOLGFBQWE7QUFBQSxRQUNmO0FBQUEsTUFDRjtBQUFBLE1BQ0EsVUFBVSxDQUFDO0FBQUEsSUFDYjtBQUFBLEVBQ0Y7QUFBQSxFQUVBLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLFFBQVEsRUFBRSxNQUFNLFVBQVU7QUFBQSxNQUMxQixZQUFZO0FBQUEsUUFDVixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDMUIsZUFBZSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQ2xDO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLFlBQVk7QUFBQSxRQUNWLE1BQU07QUFBQSxRQUNOLFlBQVk7QUFBQSxVQUNWLFVBQVUsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUMzQixRQUFRLEVBQUUsTUFBTSxTQUFTO0FBQUEsUUFDM0I7QUFBQSxNQUNGO0FBQUEsTUFDQSxTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsSUFDNUI7QUFBQSxJQUNBLFVBQVUsQ0FBQyxVQUFVLGNBQWMsU0FBUztBQUFBLEVBQzlDO0FBQUEsRUFFQSxlQUFlO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFRZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFpRWQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUErQ3BCLHVCQUF1QjtBQUN6QjtBQUVBLElBQU8sOEJBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
