// .agents/doc-status-tracker.ts
var definition = {
  id: "doc-status-tracker",
  displayName: "Doc Status Tracker",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "glob",
    "list_directory",
    "run_terminal_command",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional focus area or specific docs to check"
    },
    params: {
      type: "object",
      properties: {
        staleDays: {
          type: "number",
          description: "Days after which Working docs are considered stale (default: 30)"
        }
      },
      required: []
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      statusCounts: {
        type: "object",
        properties: {
          frozen: { type: "number" },
          working: { type: "number" },
          draft: { type: "number" },
          hypothesis: { type: "number" },
          approved: { type: "number" }
        }
      },
      staleDocs: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            status: { type: "string" },
            date: { type: "string" },
            daysSinceUpdate: { type: "number" }
          }
        }
      },
      recommendations: { type: "array" },
      summary: { type: "string" }
    },
    required: ["statusCounts", "staleDocs", "summary"]
  },
  spawnerPrompt: `Spawn to track documentation status across the project:
- Count documents by status (F/W/D/H/A)
- Identify stale Working docs (>30 days)
- Recommend status transitions
- Generate documentation health report

Use for weekly documentation audits.`,
  systemPrompt: `You are the Doc Status Tracker for the Pantheon-Chat project.

You monitor documentation health and status transitions.

## STATUS CODES

| Code | Status | Description | Lifespan |
|------|--------|-------------|----------|
| F | Frozen | Immutable facts, validated | Permanent |
| W | Working | Active development | Should transition within 30 days |
| D | Draft | Early stage | Should transition within 14 days |
| H | Hypothesis | Needs validation | Until validated/rejected |
| A | Approved | Reviewed and approved | Until superseded |

## HEALTH INDICATORS

### Healthy Documentation
- Most docs are Frozen (validated, stable)
- Working docs are actively being updated
- Clear path from Draft \u2192 Working \u2192 Frozen

### Warning Signs
- Too many Working docs (>30% of total)
- Stale Working docs (>30 days since date)
- Draft docs older than 14 days
- No Frozen docs in a category

## STATUS TRANSITIONS

\`\`\`
Draft (D) \u2192 Working (W) \u2192 Frozen (F)
                \u2193
          Approved (A)

Hypothesis (H) \u2192 Frozen (F) [if validated]
              \u2192 Deprecated [if rejected]
\`\`\`

## DIRECTORY STRUCTURE

- 01-policies/ - Should be mostly F (Frozen)
- 02-procedures/ - Mix of F and W
- 03-technical/ - Can have W, H documents
- 04-records/ - Should be F after completion
- 05-decisions/ - ADRs should be F
- 06-implementation/ - Often W during development
- 07-user-guides/ - Should be F for published
- 08-experiments/ - Can have H documents
- 09-curriculum/ - Should be F when complete`,
  instructionsPrompt: `## Tracking Process

1. Find all documentation files:
   \`\`\`bash
   find docs -name "*.md" -type f | grep -E "[0-9]{8}.*[FWDHA].md$"
   \`\`\`

2. Parse each filename:
   - Extract date (YYYYMMDD)
   - Extract status code (last char before .md)
   - Calculate days since document date

3. Compile statistics:
   - Count by status (F, W, D, H, A)
   - Count by directory
   - Identify percentages

4. Find stale documents:
   - Working (W) docs older than 30 days
   - Draft (D) docs older than 14 days
   - Hypothesis (H) docs without recent updates

5. Generate recommendations:
   - Stale Working docs should be Frozen or updated
   - Old Drafts should progress or be archived
   - Hypothesis docs should be validated

6. Check directory health:
   - policies/ should be >80% Frozen
   - procedures/ should be >60% Frozen
   - decisions/ should be 100% Frozen

7. Set structured output:
   - statusCounts: breakdown by status code
   - staleDocs: documents needing attention
   - recommendations: specific actions to take
   - summary: overall documentation health

Provide actionable recommendations for improving doc health.`,
  includeMessageHistory: false
};
var doc_status_tracker_default = definition;
export {
  doc_status_tracker_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9kb2Mtc3RhdHVzLXRyYWNrZXIudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBkZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAnZG9jLXN0YXR1cy10cmFja2VyJyxcbiAgZGlzcGxheU5hbWU6ICdEb2MgU3RhdHVzIFRyYWNrZXInLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnZ2xvYicsXG4gICAgJ2xpc3RfZGlyZWN0b3J5JyxcbiAgICAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnLFxuICAgICdzZXRfb3V0cHV0JyxcbiAgXSxcblxuICBpbnB1dFNjaGVtYToge1xuICAgIHByb21wdDoge1xuICAgICAgdHlwZTogJ3N0cmluZycsXG4gICAgICBkZXNjcmlwdGlvbjogJ09wdGlvbmFsIGZvY3VzIGFyZWEgb3Igc3BlY2lmaWMgZG9jcyB0byBjaGVjaycsXG4gICAgfSxcbiAgICBwYXJhbXM6IHtcbiAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgcHJvcGVydGllczoge1xuICAgICAgICBzdGFsZURheXM6IHtcbiAgICAgICAgICB0eXBlOiAnbnVtYmVyJyxcbiAgICAgICAgICBkZXNjcmlwdGlvbjogJ0RheXMgYWZ0ZXIgd2hpY2ggV29ya2luZyBkb2NzIGFyZSBjb25zaWRlcmVkIHN0YWxlIChkZWZhdWx0OiAzMCknLFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIHJlcXVpcmVkOiBbXSxcbiAgICB9LFxuICB9LFxuXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkX291dHB1dCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHN0YXR1c0NvdW50czoge1xuICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgIGZyb3plbjogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgICAgIHdvcmtpbmc6IHsgdHlwZTogJ251bWJlcicgfSxcbiAgICAgICAgICBkcmFmdDogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgICAgIGh5cG90aGVzaXM6IHsgdHlwZTogJ251bWJlcicgfSxcbiAgICAgICAgICBhcHByb3ZlZDogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIHN0YWxlRG9jczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGZpbGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHN0YXR1czogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgZGF0ZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgZGF5c1NpbmNlVXBkYXRlOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICByZWNvbW1lbmRhdGlvbnM6IHsgdHlwZTogJ2FycmF5JyB9LFxuICAgICAgc3VtbWFyeTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgIH0sXG4gICAgcmVxdWlyZWQ6IFsnc3RhdHVzQ291bnRzJywgJ3N0YWxlRG9jcycsICdzdW1tYXJ5J10sXG4gIH0sXG5cbiAgc3Bhd25lclByb21wdDogYFNwYXduIHRvIHRyYWNrIGRvY3VtZW50YXRpb24gc3RhdHVzIGFjcm9zcyB0aGUgcHJvamVjdDpcbi0gQ291bnQgZG9jdW1lbnRzIGJ5IHN0YXR1cyAoRi9XL0QvSC9BKVxuLSBJZGVudGlmeSBzdGFsZSBXb3JraW5nIGRvY3MgKD4zMCBkYXlzKVxuLSBSZWNvbW1lbmQgc3RhdHVzIHRyYW5zaXRpb25zXG4tIEdlbmVyYXRlIGRvY3VtZW50YXRpb24gaGVhbHRoIHJlcG9ydFxuXG5Vc2UgZm9yIHdlZWtseSBkb2N1bWVudGF0aW9uIGF1ZGl0cy5gLFxuXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgdGhlIERvYyBTdGF0dXMgVHJhY2tlciBmb3IgdGhlIFBhbnRoZW9uLUNoYXQgcHJvamVjdC5cblxuWW91IG1vbml0b3IgZG9jdW1lbnRhdGlvbiBoZWFsdGggYW5kIHN0YXR1cyB0cmFuc2l0aW9ucy5cblxuIyMgU1RBVFVTIENPREVTXG5cbnwgQ29kZSB8IFN0YXR1cyB8IERlc2NyaXB0aW9uIHwgTGlmZXNwYW4gfFxufC0tLS0tLXwtLS0tLS0tLXwtLS0tLS0tLS0tLS0tfC0tLS0tLS0tLS18XG58IEYgfCBGcm96ZW4gfCBJbW11dGFibGUgZmFjdHMsIHZhbGlkYXRlZCB8IFBlcm1hbmVudCB8XG58IFcgfCBXb3JraW5nIHwgQWN0aXZlIGRldmVsb3BtZW50IHwgU2hvdWxkIHRyYW5zaXRpb24gd2l0aGluIDMwIGRheXMgfFxufCBEIHwgRHJhZnQgfCBFYXJseSBzdGFnZSB8IFNob3VsZCB0cmFuc2l0aW9uIHdpdGhpbiAxNCBkYXlzIHxcbnwgSCB8IEh5cG90aGVzaXMgfCBOZWVkcyB2YWxpZGF0aW9uIHwgVW50aWwgdmFsaWRhdGVkL3JlamVjdGVkIHxcbnwgQSB8IEFwcHJvdmVkIHwgUmV2aWV3ZWQgYW5kIGFwcHJvdmVkIHwgVW50aWwgc3VwZXJzZWRlZCB8XG5cbiMjIEhFQUxUSCBJTkRJQ0FUT1JTXG5cbiMjIyBIZWFsdGh5IERvY3VtZW50YXRpb25cbi0gTW9zdCBkb2NzIGFyZSBGcm96ZW4gKHZhbGlkYXRlZCwgc3RhYmxlKVxuLSBXb3JraW5nIGRvY3MgYXJlIGFjdGl2ZWx5IGJlaW5nIHVwZGF0ZWRcbi0gQ2xlYXIgcGF0aCBmcm9tIERyYWZ0IFx1MjE5MiBXb3JraW5nIFx1MjE5MiBGcm96ZW5cblxuIyMjIFdhcm5pbmcgU2lnbnNcbi0gVG9vIG1hbnkgV29ya2luZyBkb2NzICg+MzAlIG9mIHRvdGFsKVxuLSBTdGFsZSBXb3JraW5nIGRvY3MgKD4zMCBkYXlzIHNpbmNlIGRhdGUpXG4tIERyYWZ0IGRvY3Mgb2xkZXIgdGhhbiAxNCBkYXlzXG4tIE5vIEZyb3plbiBkb2NzIGluIGEgY2F0ZWdvcnlcblxuIyMgU1RBVFVTIFRSQU5TSVRJT05TXG5cblxcYFxcYFxcYFxuRHJhZnQgKEQpIFx1MjE5MiBXb3JraW5nIChXKSBcdTIxOTIgRnJvemVuIChGKVxuICAgICAgICAgICAgICAgIFx1MjE5M1xuICAgICAgICAgIEFwcHJvdmVkIChBKVxuXG5IeXBvdGhlc2lzIChIKSBcdTIxOTIgRnJvemVuIChGKSBbaWYgdmFsaWRhdGVkXVxuICAgICAgICAgICAgICBcdTIxOTIgRGVwcmVjYXRlZCBbaWYgcmVqZWN0ZWRdXG5cXGBcXGBcXGBcblxuIyMgRElSRUNUT1JZIFNUUlVDVFVSRVxuXG4tIDAxLXBvbGljaWVzLyAtIFNob3VsZCBiZSBtb3N0bHkgRiAoRnJvemVuKVxuLSAwMi1wcm9jZWR1cmVzLyAtIE1peCBvZiBGIGFuZCBXXG4tIDAzLXRlY2huaWNhbC8gLSBDYW4gaGF2ZSBXLCBIIGRvY3VtZW50c1xuLSAwNC1yZWNvcmRzLyAtIFNob3VsZCBiZSBGIGFmdGVyIGNvbXBsZXRpb25cbi0gMDUtZGVjaXNpb25zLyAtIEFEUnMgc2hvdWxkIGJlIEZcbi0gMDYtaW1wbGVtZW50YXRpb24vIC0gT2Z0ZW4gVyBkdXJpbmcgZGV2ZWxvcG1lbnRcbi0gMDctdXNlci1ndWlkZXMvIC0gU2hvdWxkIGJlIEYgZm9yIHB1Ymxpc2hlZFxuLSAwOC1leHBlcmltZW50cy8gLSBDYW4gaGF2ZSBIIGRvY3VtZW50c1xuLSAwOS1jdXJyaWN1bHVtLyAtIFNob3VsZCBiZSBGIHdoZW4gY29tcGxldGVgLFxuXG4gIGluc3RydWN0aW9uc1Byb21wdDogYCMjIFRyYWNraW5nIFByb2Nlc3NcblxuMS4gRmluZCBhbGwgZG9jdW1lbnRhdGlvbiBmaWxlczpcbiAgIFxcYFxcYFxcYGJhc2hcbiAgIGZpbmQgZG9jcyAtbmFtZSBcIioubWRcIiAtdHlwZSBmIHwgZ3JlcCAtRSBcIlswLTldezh9LipbRldESEFdXFwubWQkXCJcbiAgIFxcYFxcYFxcYFxuXG4yLiBQYXJzZSBlYWNoIGZpbGVuYW1lOlxuICAgLSBFeHRyYWN0IGRhdGUgKFlZWVlNTUREKVxuICAgLSBFeHRyYWN0IHN0YXR1cyBjb2RlIChsYXN0IGNoYXIgYmVmb3JlIC5tZClcbiAgIC0gQ2FsY3VsYXRlIGRheXMgc2luY2UgZG9jdW1lbnQgZGF0ZVxuXG4zLiBDb21waWxlIHN0YXRpc3RpY3M6XG4gICAtIENvdW50IGJ5IHN0YXR1cyAoRiwgVywgRCwgSCwgQSlcbiAgIC0gQ291bnQgYnkgZGlyZWN0b3J5XG4gICAtIElkZW50aWZ5IHBlcmNlbnRhZ2VzXG5cbjQuIEZpbmQgc3RhbGUgZG9jdW1lbnRzOlxuICAgLSBXb3JraW5nIChXKSBkb2NzIG9sZGVyIHRoYW4gMzAgZGF5c1xuICAgLSBEcmFmdCAoRCkgZG9jcyBvbGRlciB0aGFuIDE0IGRheXNcbiAgIC0gSHlwb3RoZXNpcyAoSCkgZG9jcyB3aXRob3V0IHJlY2VudCB1cGRhdGVzXG5cbjUuIEdlbmVyYXRlIHJlY29tbWVuZGF0aW9uczpcbiAgIC0gU3RhbGUgV29ya2luZyBkb2NzIHNob3VsZCBiZSBGcm96ZW4gb3IgdXBkYXRlZFxuICAgLSBPbGQgRHJhZnRzIHNob3VsZCBwcm9ncmVzcyBvciBiZSBhcmNoaXZlZFxuICAgLSBIeXBvdGhlc2lzIGRvY3Mgc2hvdWxkIGJlIHZhbGlkYXRlZFxuXG42LiBDaGVjayBkaXJlY3RvcnkgaGVhbHRoOlxuICAgLSBwb2xpY2llcy8gc2hvdWxkIGJlID44MCUgRnJvemVuXG4gICAtIHByb2NlZHVyZXMvIHNob3VsZCBiZSA+NjAlIEZyb3plblxuICAgLSBkZWNpc2lvbnMvIHNob3VsZCBiZSAxMDAlIEZyb3plblxuXG43LiBTZXQgc3RydWN0dXJlZCBvdXRwdXQ6XG4gICAtIHN0YXR1c0NvdW50czogYnJlYWtkb3duIGJ5IHN0YXR1cyBjb2RlXG4gICAtIHN0YWxlRG9jczogZG9jdW1lbnRzIG5lZWRpbmcgYXR0ZW50aW9uXG4gICAtIHJlY29tbWVuZGF0aW9uczogc3BlY2lmaWMgYWN0aW9ucyB0byB0YWtlXG4gICAtIHN1bW1hcnk6IG92ZXJhbGwgZG9jdW1lbnRhdGlvbiBoZWFsdGhcblxuUHJvdmlkZSBhY3Rpb25hYmxlIHJlY29tbWVuZGF0aW9ucyBmb3IgaW1wcm92aW5nIGRvYyBoZWFsdGguYCxcblxuICBpbmNsdWRlTWVzc2FnZUhpc3Rvcnk6IGZhbHNlLFxufVxuXG5leHBvcnQgZGVmYXVsdCBkZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxhQUE4QjtBQUFBLEVBQ2xDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUVQLFdBQVc7QUFBQSxJQUNUO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLEVBQ0Y7QUFBQSxFQUVBLGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsSUFDQSxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixZQUFZO0FBQUEsUUFDVixXQUFXO0FBQUEsVUFDVCxNQUFNO0FBQUEsVUFDTixhQUFhO0FBQUEsUUFDZjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLFVBQVUsQ0FBQztBQUFBLElBQ2I7QUFBQSxFQUNGO0FBQUEsRUFFQSxZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixjQUFjO0FBQUEsUUFDWixNQUFNO0FBQUEsUUFDTixZQUFZO0FBQUEsVUFDVixRQUFRLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDekIsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQzFCLE9BQU8sRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUN4QixZQUFZLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDN0IsVUFBVSxFQUFFLE1BQU0sU0FBUztBQUFBLFFBQzdCO0FBQUEsTUFDRjtBQUFBLE1BQ0EsV0FBVztBQUFBLFFBQ1QsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLFFBQVEsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN6QixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsaUJBQWlCLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDcEM7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsaUJBQWlCLEVBQUUsTUFBTSxRQUFRO0FBQUEsTUFDakMsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLElBQzVCO0FBQUEsSUFDQSxVQUFVLENBQUMsZ0JBQWdCLGFBQWEsU0FBUztBQUFBLEVBQ25EO0FBQUEsRUFFQSxlQUFlO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFRZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFrRGQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBd0NwQix1QkFBdUI7QUFDekI7QUFFQSxJQUFPLDZCQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
