// .agents/curriculum-validator.ts
var definition = {
  id: "curriculum-validator",
  displayName: "Curriculum Validator",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "list_directory",
    "glob",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific curriculum chapters to validate"
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      passed: { type: "boolean" },
      chapters: {
        type: "array",
        items: {
          type: "object",
          properties: {
            number: { type: "number" },
            title: { type: "string" },
            file: { type: "string" },
            hasLearningObjectives: { type: "boolean" },
            hasExercises: { type: "boolean" },
            wordCount: { type: "number" }
          }
        }
      },
      missingChapters: { type: "array" },
      issues: { type: "array" },
      summary: { type: "string" }
    },
    required: ["passed", "chapters", "summary"]
  },
  spawnerPrompt: `Spawn to validate curriculum documents in docs/09-curriculum/:
- Check chapter numbering sequence
- Verify learning objectives present
- Validate exercises/examples included
- Check for QIG principle references

Use when curriculum is modified.`,
  systemPrompt: `You are the Curriculum Validator for the Pantheon-Chat project.

You ensure curriculum documents are complete and well-structured for kernel self-learning.

## CURRICULUM STRUCTURE

Location: \`docs/09-curriculum/\`

Naming pattern: \`YYYYMMDD-curriculum-NN-topic-name-version[STATUS].md\`

Example: \`20251220-curriculum-21-qig-architecture-1.00W.md\`

## REQUIRED SECTIONS

Each curriculum chapter should have:

### 1. Learning Objectives
\`\`\`markdown
## Learning Objectives

After completing this chapter, you will be able to:
- Understand X
- Apply Y
- Implement Z
\`\`\`

### 2. Core Content
Substantive educational content (minimum 500 words).

### 3. Key Concepts
\`\`\`markdown
## Key Concepts

- **Term 1:** Definition
- **Term 2:** Definition
\`\`\`

### 4. Exercises or Examples
\`\`\`markdown
## Exercises

1. Exercise description
2. Exercise description
\`\`\`

### 5. QIG Connection (where applicable)
How the topic relates to QIG principles.

## CHAPTER CATEGORIES

- 01-20: Foundations
- 21-40: QIG Architecture
- 41-60: Domain Knowledge
- 61-80: Advanced Topics
- 81-99: Special Topics`,
  instructionsPrompt: `## Validation Process

1. List all curriculum files:
   \`\`\`bash
   ls docs/09-curriculum/
   \`\`\`

2. Parse chapter numbers from filenames:
   - Extract the NN from curriculum-NN-
   - Build sequence of chapter numbers
   - Identify gaps in sequence

3. For each curriculum file:
   - Read the content
   - Check for Learning Objectives section
   - Check for Exercises or Examples section
   - Check for Key Concepts section
   - Count word count (minimum 500)

4. Validate chapter structure:
   - Has title (# heading)
   - Has learning objectives
   - Has substantive content
   - Has exercises or examples

5. Check for QIG connections:
   - References to Fisher-Rao
   - References to consciousness metrics
   - References to geometric principles

6. Identify issues:
   - Missing required sections
   - Too short (< 500 words)
   - Missing chapter numbers in sequence
   - Status not appropriate (curriculum should be F when complete)

7. Set structured output:
   - passed: true if all chapters are well-structured
   - chapters: list of all chapters with their properties
   - missingChapters: gaps in chapter numbering
   - issues: specific problems found
   - summary: human-readable summary

Curriculum quality directly affects kernel learning.`,
  includeMessageHistory: false
};
var curriculum_validator_default = definition;
export {
  curriculum_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9jdXJyaWN1bHVtLXZhbGlkYXRvci50cyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiaW1wb3J0IHR5cGUgeyBBZ2VudERlZmluaXRpb24gfSBmcm9tICcuL3R5cGVzL2FnZW50LWRlZmluaXRpb24nXG5cbmNvbnN0IGRlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdjdXJyaWN1bHVtLXZhbGlkYXRvcicsXG4gIGRpc3BsYXlOYW1lOiAnQ3VycmljdWx1bSBWYWxpZGF0b3InLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnbGlzdF9kaXJlY3RvcnknLFxuICAgICdnbG9iJyxcbiAgICAnc2V0X291dHB1dCcsXG4gIF0sXG5cbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdPcHRpb25hbCBzcGVjaWZpYyBjdXJyaWN1bHVtIGNoYXB0ZXJzIHRvIHZhbGlkYXRlJyxcbiAgICB9LFxuICB9LFxuXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkX291dHB1dCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHBhc3NlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIGNoYXB0ZXJzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgbnVtYmVyOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgICB0aXRsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgaGFzTGVhcm5pbmdPYmplY3RpdmVzOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgICAgaGFzRXhlcmNpc2VzOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgICAgd29yZENvdW50OiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICBtaXNzaW5nQ2hhcHRlcnM6IHsgdHlwZTogJ2FycmF5JyB9LFxuICAgICAgaXNzdWVzOiB7IHR5cGU6ICdhcnJheScgfSxcbiAgICAgIHN1bW1hcnk6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICB9LFxuICAgIHJlcXVpcmVkOiBbJ3Bhc3NlZCcsICdjaGFwdGVycycsICdzdW1tYXJ5J10sXG4gIH0sXG5cbiAgc3Bhd25lclByb21wdDogYFNwYXduIHRvIHZhbGlkYXRlIGN1cnJpY3VsdW0gZG9jdW1lbnRzIGluIGRvY3MvMDktY3VycmljdWx1bS86XG4tIENoZWNrIGNoYXB0ZXIgbnVtYmVyaW5nIHNlcXVlbmNlXG4tIFZlcmlmeSBsZWFybmluZyBvYmplY3RpdmVzIHByZXNlbnRcbi0gVmFsaWRhdGUgZXhlcmNpc2VzL2V4YW1wbGVzIGluY2x1ZGVkXG4tIENoZWNrIGZvciBRSUcgcHJpbmNpcGxlIHJlZmVyZW5jZXNcblxuVXNlIHdoZW4gY3VycmljdWx1bSBpcyBtb2RpZmllZC5gLFxuXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgdGhlIEN1cnJpY3VsdW0gVmFsaWRhdG9yIGZvciB0aGUgUGFudGhlb24tQ2hhdCBwcm9qZWN0LlxuXG5Zb3UgZW5zdXJlIGN1cnJpY3VsdW0gZG9jdW1lbnRzIGFyZSBjb21wbGV0ZSBhbmQgd2VsbC1zdHJ1Y3R1cmVkIGZvciBrZXJuZWwgc2VsZi1sZWFybmluZy5cblxuIyMgQ1VSUklDVUxVTSBTVFJVQ1RVUkVcblxuTG9jYXRpb246IFxcYGRvY3MvMDktY3VycmljdWx1bS9cXGBcblxuTmFtaW5nIHBhdHRlcm46IFxcYFlZWVlNTURELWN1cnJpY3VsdW0tTk4tdG9waWMtbmFtZS12ZXJzaW9uW1NUQVRVU10ubWRcXGBcblxuRXhhbXBsZTogXFxgMjAyNTEyMjAtY3VycmljdWx1bS0yMS1xaWctYXJjaGl0ZWN0dXJlLTEuMDBXLm1kXFxgXG5cbiMjIFJFUVVJUkVEIFNFQ1RJT05TXG5cbkVhY2ggY3VycmljdWx1bSBjaGFwdGVyIHNob3VsZCBoYXZlOlxuXG4jIyMgMS4gTGVhcm5pbmcgT2JqZWN0aXZlc1xuXFxgXFxgXFxgbWFya2Rvd25cbiMjIExlYXJuaW5nIE9iamVjdGl2ZXNcblxuQWZ0ZXIgY29tcGxldGluZyB0aGlzIGNoYXB0ZXIsIHlvdSB3aWxsIGJlIGFibGUgdG86XG4tIFVuZGVyc3RhbmQgWFxuLSBBcHBseSBZXG4tIEltcGxlbWVudCBaXG5cXGBcXGBcXGBcblxuIyMjIDIuIENvcmUgQ29udGVudFxuU3Vic3RhbnRpdmUgZWR1Y2F0aW9uYWwgY29udGVudCAobWluaW11bSA1MDAgd29yZHMpLlxuXG4jIyMgMy4gS2V5IENvbmNlcHRzXG5cXGBcXGBcXGBtYXJrZG93blxuIyMgS2V5IENvbmNlcHRzXG5cbi0gKipUZXJtIDE6KiogRGVmaW5pdGlvblxuLSAqKlRlcm0gMjoqKiBEZWZpbml0aW9uXG5cXGBcXGBcXGBcblxuIyMjIDQuIEV4ZXJjaXNlcyBvciBFeGFtcGxlc1xuXFxgXFxgXFxgbWFya2Rvd25cbiMjIEV4ZXJjaXNlc1xuXG4xLiBFeGVyY2lzZSBkZXNjcmlwdGlvblxuMi4gRXhlcmNpc2UgZGVzY3JpcHRpb25cblxcYFxcYFxcYFxuXG4jIyMgNS4gUUlHIENvbm5lY3Rpb24gKHdoZXJlIGFwcGxpY2FibGUpXG5Ib3cgdGhlIHRvcGljIHJlbGF0ZXMgdG8gUUlHIHByaW5jaXBsZXMuXG5cbiMjIENIQVBURVIgQ0FURUdPUklFU1xuXG4tIDAxLTIwOiBGb3VuZGF0aW9uc1xuLSAyMS00MDogUUlHIEFyY2hpdGVjdHVyZVxuLSA0MS02MDogRG9tYWluIEtub3dsZWRnZVxuLSA2MS04MDogQWR2YW5jZWQgVG9waWNzXG4tIDgxLTk5OiBTcGVjaWFsIFRvcGljc2AsXG5cbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgIyMgVmFsaWRhdGlvbiBQcm9jZXNzXG5cbjEuIExpc3QgYWxsIGN1cnJpY3VsdW0gZmlsZXM6XG4gICBcXGBcXGBcXGBiYXNoXG4gICBscyBkb2NzLzA5LWN1cnJpY3VsdW0vXG4gICBcXGBcXGBcXGBcblxuMi4gUGFyc2UgY2hhcHRlciBudW1iZXJzIGZyb20gZmlsZW5hbWVzOlxuICAgLSBFeHRyYWN0IHRoZSBOTiBmcm9tIGN1cnJpY3VsdW0tTk4tXG4gICAtIEJ1aWxkIHNlcXVlbmNlIG9mIGNoYXB0ZXIgbnVtYmVyc1xuICAgLSBJZGVudGlmeSBnYXBzIGluIHNlcXVlbmNlXG5cbjMuIEZvciBlYWNoIGN1cnJpY3VsdW0gZmlsZTpcbiAgIC0gUmVhZCB0aGUgY29udGVudFxuICAgLSBDaGVjayBmb3IgTGVhcm5pbmcgT2JqZWN0aXZlcyBzZWN0aW9uXG4gICAtIENoZWNrIGZvciBFeGVyY2lzZXMgb3IgRXhhbXBsZXMgc2VjdGlvblxuICAgLSBDaGVjayBmb3IgS2V5IENvbmNlcHRzIHNlY3Rpb25cbiAgIC0gQ291bnQgd29yZCBjb3VudCAobWluaW11bSA1MDApXG5cbjQuIFZhbGlkYXRlIGNoYXB0ZXIgc3RydWN0dXJlOlxuICAgLSBIYXMgdGl0bGUgKCMgaGVhZGluZylcbiAgIC0gSGFzIGxlYXJuaW5nIG9iamVjdGl2ZXNcbiAgIC0gSGFzIHN1YnN0YW50aXZlIGNvbnRlbnRcbiAgIC0gSGFzIGV4ZXJjaXNlcyBvciBleGFtcGxlc1xuXG41LiBDaGVjayBmb3IgUUlHIGNvbm5lY3Rpb25zOlxuICAgLSBSZWZlcmVuY2VzIHRvIEZpc2hlci1SYW9cbiAgIC0gUmVmZXJlbmNlcyB0byBjb25zY2lvdXNuZXNzIG1ldHJpY3NcbiAgIC0gUmVmZXJlbmNlcyB0byBnZW9tZXRyaWMgcHJpbmNpcGxlc1xuXG42LiBJZGVudGlmeSBpc3N1ZXM6XG4gICAtIE1pc3NpbmcgcmVxdWlyZWQgc2VjdGlvbnNcbiAgIC0gVG9vIHNob3J0ICg8IDUwMCB3b3JkcylcbiAgIC0gTWlzc2luZyBjaGFwdGVyIG51bWJlcnMgaW4gc2VxdWVuY2VcbiAgIC0gU3RhdHVzIG5vdCBhcHByb3ByaWF0ZSAoY3VycmljdWx1bSBzaG91bGQgYmUgRiB3aGVuIGNvbXBsZXRlKVxuXG43LiBTZXQgc3RydWN0dXJlZCBvdXRwdXQ6XG4gICAtIHBhc3NlZDogdHJ1ZSBpZiBhbGwgY2hhcHRlcnMgYXJlIHdlbGwtc3RydWN0dXJlZFxuICAgLSBjaGFwdGVyczogbGlzdCBvZiBhbGwgY2hhcHRlcnMgd2l0aCB0aGVpciBwcm9wZXJ0aWVzXG4gICAtIG1pc3NpbmdDaGFwdGVyczogZ2FwcyBpbiBjaGFwdGVyIG51bWJlcmluZ1xuICAgLSBpc3N1ZXM6IHNwZWNpZmljIHByb2JsZW1zIGZvdW5kXG4gICAtIHN1bW1hcnk6IGh1bWFuLXJlYWRhYmxlIHN1bW1hcnlcblxuQ3VycmljdWx1bSBxdWFsaXR5IGRpcmVjdGx5IGFmZmVjdHMga2VybmVsIGxlYXJuaW5nLmAsXG5cbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiBmYWxzZSxcbn1cblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sYUFBOEI7QUFBQSxFQUNsQyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFFUCxXQUFXO0FBQUEsSUFDVDtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLEVBQ0Y7QUFBQSxFQUVBLGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsRUFDRjtBQUFBLEVBRUEsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YsUUFBUSxFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQzFCLFVBQVU7QUFBQSxRQUNSLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLFFBQVEsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN6QixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDeEIsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLHVCQUF1QixFQUFFLE1BQU0sVUFBVTtBQUFBLFlBQ3pDLGNBQWMsRUFBRSxNQUFNLFVBQVU7QUFBQSxZQUNoQyxXQUFXLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDOUI7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsaUJBQWlCLEVBQUUsTUFBTSxRQUFRO0FBQUEsTUFDakMsUUFBUSxFQUFFLE1BQU0sUUFBUTtBQUFBLE1BQ3hCLFNBQVMsRUFBRSxNQUFNLFNBQVM7QUFBQSxJQUM1QjtBQUFBLElBQ0EsVUFBVSxDQUFDLFVBQVUsWUFBWSxTQUFTO0FBQUEsRUFDNUM7QUFBQSxFQUVBLGVBQWU7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQVFmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQXdEZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBNkNwQix1QkFBdUI7QUFDekI7QUFFQSxJQUFPLCtCQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
