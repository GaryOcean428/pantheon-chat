// .agents/dead-code-detector.ts
var definition = {
  id: "dead-code-detector",
  displayName: "Dead Code Detector",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "code_search",
    "glob",
    "run_terminal_command",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific directories or files to check"
    },
    params: {
      type: "object",
      properties: {
        directories: {
          type: "array",
          description: "Directories to scan (defaults to all source)"
        },
        includeTests: {
          type: "boolean",
          description: "Include test files in analysis"
        }
      },
      required: []
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      unusedExports: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            export: { type: "string" },
            type: { type: "string" }
          }
        }
      },
      orphanedFiles: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            reason: { type: "string" }
          }
        }
      },
      unusedDependencies: { type: "array" },
      summary: { type: "string" }
    },
    required: ["unusedExports", "orphanedFiles", "summary"]
  },
  spawnerPrompt: `Spawn to detect dead code in the codebase:
- Unused exported functions/classes/variables
- Orphaned files (not imported anywhere)
- Unused npm/pip dependencies
- Commented-out code blocks

Use for periodic codebase cleanup.`,
  systemPrompt: `You are the Dead Code Detector for the Pantheon-Chat project.

You find unused code that can be safely removed.

## WHAT TO DETECT

### 1. Unused Exports
\`\`\`typescript
// Exported but never imported elsewhere
export function unusedHelper() { ... }  // Dead code!
export const UNUSED_CONSTANT = 42       // Dead code!
export class UnusedClass { }            // Dead code!
\`\`\`

### 2. Orphaned Files
Files that exist but are never imported:
- Components not used in any page
- Utilities not imported anywhere
- Old implementations replaced but not deleted

### 3. Unused Dependencies
\`\`\`json
// package.json
"dependencies": {
  "never-used-package": "^1.0.0"  // Dead dependency!
}
\`\`\`

### 4. Commented-Out Code
\`\`\`typescript
// function oldImplementation() {
//   // This was replaced
//   return legacy();
// }
\`\`\`

## SAFE TO REMOVE

\u2705 Functions/classes with zero imports
\u2705 Files with zero imports (check barrel exports first)
\u2705 Dependencies not in any import statement
\u2705 Large commented code blocks (>10 lines)

## NOT SAFE TO REMOVE

\u274C Dynamic imports (\`import()\`)
\u274C Entry points (main.ts, index.ts of root)
\u274C CLI scripts referenced in package.json
\u274C Test files (may have isolated tests)
\u274C Type definitions used in .d.ts
\u274C Exports used via barrel files`,
  instructionsPrompt: `## Detection Process

1. Find all exports in the codebase:
   - TypeScript: \`export function\`, \`export const\`, \`export class\`
   - Python: Functions/classes in __all__ or not prefixed with _

2. For each export, search for imports:
   \`\`\`bash
   # Search for import of specific symbol
   rg "import.*{.*symbolName.*}" --type ts
   rg "from.*import.*symbolName" --type py
   \`\`\`

3. Check barrel file re-exports:
   - Symbol may be re-exported from index.ts
   - Track transitive exports

4. Find orphaned files:
   - List all source files
   - For each, search for imports of that file
   - Flag files with zero imports

5. Check npm dependencies:
   - Read package.json dependencies
   - Search for import of each package
   - Flag packages never imported

6. Check pip dependencies:
   - Read requirements.txt
   - Search for imports of each package
   - Flag packages never imported

7. Find commented code blocks:
   - Search for multi-line comments containing code patterns
   - Flag blocks > 10 lines of commented code

8. Exclude false positives:
   - Entry points
   - CLI scripts
   - Dynamic imports
   - Type-only imports

9. Set structured output:
   - unusedExports: exports with no importers
   - orphanedFiles: files never imported
   - unusedDependencies: packages never used
   - summary: human-readable summary with safe removal recommendations

Remove dead code to reduce maintenance burden.`,
  includeMessageHistory: false
};
var dead_code_detector_default = definition;
export {
  dead_code_detector_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9kZWFkLWNvZGUtZGV0ZWN0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBkZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAnZGVhZC1jb2RlLWRldGVjdG9yJyxcbiAgZGlzcGxheU5hbWU6ICdEZWFkIENvZGUgRGV0ZWN0b3InLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdnbG9iJyxcbiAgICAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnLFxuICAgICdzZXRfb3V0cHV0JyxcbiAgXSxcblxuICBpbnB1dFNjaGVtYToge1xuICAgIHByb21wdDoge1xuICAgICAgdHlwZTogJ3N0cmluZycsXG4gICAgICBkZXNjcmlwdGlvbjogJ09wdGlvbmFsIHNwZWNpZmljIGRpcmVjdG9yaWVzIG9yIGZpbGVzIHRvIGNoZWNrJyxcbiAgICB9LFxuICAgIHBhcmFtczoge1xuICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgIGRpcmVjdG9yaWVzOiB7XG4gICAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgICBkZXNjcmlwdGlvbjogJ0RpcmVjdG9yaWVzIHRvIHNjYW4gKGRlZmF1bHRzIHRvIGFsbCBzb3VyY2UpJyxcbiAgICAgICAgfSxcbiAgICAgICAgaW5jbHVkZVRlc3RzOiB7XG4gICAgICAgICAgdHlwZTogJ2Jvb2xlYW4nLFxuICAgICAgICAgIGRlc2NyaXB0aW9uOiAnSW5jbHVkZSB0ZXN0IGZpbGVzIGluIGFuYWx5c2lzJyxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICByZXF1aXJlZDogW10sXG4gICAgfSxcbiAgfSxcblxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZF9vdXRwdXQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICB1bnVzZWRFeHBvcnRzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgZXhwb3J0OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICB0eXBlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICBvcnBoYW5lZEZpbGVzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgcmVhc29uOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICB1bnVzZWREZXBlbmRlbmNpZXM6IHsgdHlwZTogJ2FycmF5JyB9LFxuICAgICAgc3VtbWFyeTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgIH0sXG4gICAgcmVxdWlyZWQ6IFsndW51c2VkRXhwb3J0cycsICdvcnBoYW5lZEZpbGVzJywgJ3N1bW1hcnknXSxcbiAgfSxcblxuICBzcGF3bmVyUHJvbXB0OiBgU3Bhd24gdG8gZGV0ZWN0IGRlYWQgY29kZSBpbiB0aGUgY29kZWJhc2U6XG4tIFVudXNlZCBleHBvcnRlZCBmdW5jdGlvbnMvY2xhc3Nlcy92YXJpYWJsZXNcbi0gT3JwaGFuZWQgZmlsZXMgKG5vdCBpbXBvcnRlZCBhbnl3aGVyZSlcbi0gVW51c2VkIG5wbS9waXAgZGVwZW5kZW5jaWVzXG4tIENvbW1lbnRlZC1vdXQgY29kZSBibG9ja3NcblxuVXNlIGZvciBwZXJpb2RpYyBjb2RlYmFzZSBjbGVhbnVwLmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgRGVhZCBDb2RlIERldGVjdG9yIGZvciB0aGUgUGFudGhlb24tQ2hhdCBwcm9qZWN0LlxuXG5Zb3UgZmluZCB1bnVzZWQgY29kZSB0aGF0IGNhbiBiZSBzYWZlbHkgcmVtb3ZlZC5cblxuIyMgV0hBVCBUTyBERVRFQ1RcblxuIyMjIDEuIFVudXNlZCBFeHBvcnRzXG5cXGBcXGBcXGB0eXBlc2NyaXB0XG4vLyBFeHBvcnRlZCBidXQgbmV2ZXIgaW1wb3J0ZWQgZWxzZXdoZXJlXG5leHBvcnQgZnVuY3Rpb24gdW51c2VkSGVscGVyKCkgeyAuLi4gfSAgLy8gRGVhZCBjb2RlIVxuZXhwb3J0IGNvbnN0IFVOVVNFRF9DT05TVEFOVCA9IDQyICAgICAgIC8vIERlYWQgY29kZSFcbmV4cG9ydCBjbGFzcyBVbnVzZWRDbGFzcyB7IH0gICAgICAgICAgICAvLyBEZWFkIGNvZGUhXG5cXGBcXGBcXGBcblxuIyMjIDIuIE9ycGhhbmVkIEZpbGVzXG5GaWxlcyB0aGF0IGV4aXN0IGJ1dCBhcmUgbmV2ZXIgaW1wb3J0ZWQ6XG4tIENvbXBvbmVudHMgbm90IHVzZWQgaW4gYW55IHBhZ2Vcbi0gVXRpbGl0aWVzIG5vdCBpbXBvcnRlZCBhbnl3aGVyZVxuLSBPbGQgaW1wbGVtZW50YXRpb25zIHJlcGxhY2VkIGJ1dCBub3QgZGVsZXRlZFxuXG4jIyMgMy4gVW51c2VkIERlcGVuZGVuY2llc1xuXFxgXFxgXFxganNvblxuLy8gcGFja2FnZS5qc29uXG5cImRlcGVuZGVuY2llc1wiOiB7XG4gIFwibmV2ZXItdXNlZC1wYWNrYWdlXCI6IFwiXjEuMC4wXCIgIC8vIERlYWQgZGVwZW5kZW5jeSFcbn1cblxcYFxcYFxcYFxuXG4jIyMgNC4gQ29tbWVudGVkLU91dCBDb2RlXG5cXGBcXGBcXGB0eXBlc2NyaXB0XG4vLyBmdW5jdGlvbiBvbGRJbXBsZW1lbnRhdGlvbigpIHtcbi8vICAgLy8gVGhpcyB3YXMgcmVwbGFjZWRcbi8vICAgcmV0dXJuIGxlZ2FjeSgpO1xuLy8gfVxuXFxgXFxgXFxgXG5cbiMjIFNBRkUgVE8gUkVNT1ZFXG5cblx1MjcwNSBGdW5jdGlvbnMvY2xhc3NlcyB3aXRoIHplcm8gaW1wb3J0c1xuXHUyNzA1IEZpbGVzIHdpdGggemVybyBpbXBvcnRzIChjaGVjayBiYXJyZWwgZXhwb3J0cyBmaXJzdClcblx1MjcwNSBEZXBlbmRlbmNpZXMgbm90IGluIGFueSBpbXBvcnQgc3RhdGVtZW50XG5cdTI3MDUgTGFyZ2UgY29tbWVudGVkIGNvZGUgYmxvY2tzICg+MTAgbGluZXMpXG5cbiMjIE5PVCBTQUZFIFRPIFJFTU9WRVxuXG5cdTI3NEMgRHluYW1pYyBpbXBvcnRzIChcXGBpbXBvcnQoKVxcYClcblx1Mjc0QyBFbnRyeSBwb2ludHMgKG1haW4udHMsIGluZGV4LnRzIG9mIHJvb3QpXG5cdTI3NEMgQ0xJIHNjcmlwdHMgcmVmZXJlbmNlZCBpbiBwYWNrYWdlLmpzb25cblx1Mjc0QyBUZXN0IGZpbGVzIChtYXkgaGF2ZSBpc29sYXRlZCB0ZXN0cylcblx1Mjc0QyBUeXBlIGRlZmluaXRpb25zIHVzZWQgaW4gLmQudHNcblx1Mjc0QyBFeHBvcnRzIHVzZWQgdmlhIGJhcnJlbCBmaWxlc2AsXG5cbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgIyMgRGV0ZWN0aW9uIFByb2Nlc3NcblxuMS4gRmluZCBhbGwgZXhwb3J0cyBpbiB0aGUgY29kZWJhc2U6XG4gICAtIFR5cGVTY3JpcHQ6IFxcYGV4cG9ydCBmdW5jdGlvblxcYCwgXFxgZXhwb3J0IGNvbnN0XFxgLCBcXGBleHBvcnQgY2xhc3NcXGBcbiAgIC0gUHl0aG9uOiBGdW5jdGlvbnMvY2xhc3NlcyBpbiBfX2FsbF9fIG9yIG5vdCBwcmVmaXhlZCB3aXRoIF9cblxuMi4gRm9yIGVhY2ggZXhwb3J0LCBzZWFyY2ggZm9yIGltcG9ydHM6XG4gICBcXGBcXGBcXGBiYXNoXG4gICAjIFNlYXJjaCBmb3IgaW1wb3J0IG9mIHNwZWNpZmljIHN5bWJvbFxuICAgcmcgXCJpbXBvcnQuKnsuKnN5bWJvbE5hbWUuKn1cIiAtLXR5cGUgdHNcbiAgIHJnIFwiZnJvbS4qaW1wb3J0LipzeW1ib2xOYW1lXCIgLS10eXBlIHB5XG4gICBcXGBcXGBcXGBcblxuMy4gQ2hlY2sgYmFycmVsIGZpbGUgcmUtZXhwb3J0czpcbiAgIC0gU3ltYm9sIG1heSBiZSByZS1leHBvcnRlZCBmcm9tIGluZGV4LnRzXG4gICAtIFRyYWNrIHRyYW5zaXRpdmUgZXhwb3J0c1xuXG40LiBGaW5kIG9ycGhhbmVkIGZpbGVzOlxuICAgLSBMaXN0IGFsbCBzb3VyY2UgZmlsZXNcbiAgIC0gRm9yIGVhY2gsIHNlYXJjaCBmb3IgaW1wb3J0cyBvZiB0aGF0IGZpbGVcbiAgIC0gRmxhZyBmaWxlcyB3aXRoIHplcm8gaW1wb3J0c1xuXG41LiBDaGVjayBucG0gZGVwZW5kZW5jaWVzOlxuICAgLSBSZWFkIHBhY2thZ2UuanNvbiBkZXBlbmRlbmNpZXNcbiAgIC0gU2VhcmNoIGZvciBpbXBvcnQgb2YgZWFjaCBwYWNrYWdlXG4gICAtIEZsYWcgcGFja2FnZXMgbmV2ZXIgaW1wb3J0ZWRcblxuNi4gQ2hlY2sgcGlwIGRlcGVuZGVuY2llczpcbiAgIC0gUmVhZCByZXF1aXJlbWVudHMudHh0XG4gICAtIFNlYXJjaCBmb3IgaW1wb3J0cyBvZiBlYWNoIHBhY2thZ2VcbiAgIC0gRmxhZyBwYWNrYWdlcyBuZXZlciBpbXBvcnRlZFxuXG43LiBGaW5kIGNvbW1lbnRlZCBjb2RlIGJsb2NrczpcbiAgIC0gU2VhcmNoIGZvciBtdWx0aS1saW5lIGNvbW1lbnRzIGNvbnRhaW5pbmcgY29kZSBwYXR0ZXJuc1xuICAgLSBGbGFnIGJsb2NrcyA+IDEwIGxpbmVzIG9mIGNvbW1lbnRlZCBjb2RlXG5cbjguIEV4Y2x1ZGUgZmFsc2UgcG9zaXRpdmVzOlxuICAgLSBFbnRyeSBwb2ludHNcbiAgIC0gQ0xJIHNjcmlwdHNcbiAgIC0gRHluYW1pYyBpbXBvcnRzXG4gICAtIFR5cGUtb25seSBpbXBvcnRzXG5cbjkuIFNldCBzdHJ1Y3R1cmVkIG91dHB1dDpcbiAgIC0gdW51c2VkRXhwb3J0czogZXhwb3J0cyB3aXRoIG5vIGltcG9ydGVyc1xuICAgLSBvcnBoYW5lZEZpbGVzOiBmaWxlcyBuZXZlciBpbXBvcnRlZFxuICAgLSB1bnVzZWREZXBlbmRlbmNpZXM6IHBhY2thZ2VzIG5ldmVyIHVzZWRcbiAgIC0gc3VtbWFyeTogaHVtYW4tcmVhZGFibGUgc3VtbWFyeSB3aXRoIHNhZmUgcmVtb3ZhbCByZWNvbW1lbmRhdGlvbnNcblxuUmVtb3ZlIGRlYWQgY29kZSB0byByZWR1Y2UgbWFpbnRlbmFuY2UgYnVyZGVuLmAsXG5cbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiBmYWxzZSxcbn1cblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sYUFBOEI7QUFBQSxFQUNsQyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFFUCxXQUFXO0FBQUEsSUFDVDtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxFQUNGO0FBQUEsRUFFQSxhQUFhO0FBQUEsSUFDWCxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixhQUFhO0FBQUEsSUFDZjtBQUFBLElBQ0EsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sWUFBWTtBQUFBLFFBQ1YsYUFBYTtBQUFBLFVBQ1gsTUFBTTtBQUFBLFVBQ04sYUFBYTtBQUFBLFFBQ2Y7QUFBQSxRQUNBLGNBQWM7QUFBQSxVQUNaLE1BQU07QUFBQSxVQUNOLGFBQWE7QUFBQSxRQUNmO0FBQUEsTUFDRjtBQUFBLE1BQ0EsVUFBVSxDQUFDO0FBQUEsSUFDYjtBQUFBLEVBQ0Y7QUFBQSxFQUVBLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLGVBQWU7QUFBQSxRQUNiLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixRQUFRLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDekIsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQ3pCO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGVBQWU7QUFBQSxRQUNiLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixRQUFRLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDM0I7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0Esb0JBQW9CLEVBQUUsTUFBTSxRQUFRO0FBQUEsTUFDcEMsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLElBQzVCO0FBQUEsSUFDQSxVQUFVLENBQUMsaUJBQWlCLGlCQUFpQixTQUFTO0FBQUEsRUFDeEQ7QUFBQSxFQUVBLGVBQWU7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQVFmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFvRGQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFrRHBCLHVCQUF1QjtBQUN6QjtBQUVBLElBQU8sNkJBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
