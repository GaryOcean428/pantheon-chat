// .agents/iso-doc-validator.ts
var definition = {
  id: "iso-doc-validator",
  displayName: "ISO Doc Validator",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "find_files",
    "glob",
    "list_directory",
    "run_terminal_command",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific files or directories to validate"
    },
    params: {
      type: "object",
      properties: {
        directories: {
          type: "array",
          description: "Directories to check (defaults to docs/)"
        },
        checkContent: {
          type: "boolean",
          description: "Also validate document content structure"
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
            issue: { type: "string" },
            expected: { type: "string" },
            actual: { type: "string" }
          }
        }
      },
      statistics: {
        type: "object",
        properties: {
          totalDocs: { type: "number" },
          compliant: { type: "number" },
          nonCompliant: { type: "number" },
          byStatus: { type: "object" }
        }
      },
      summary: { type: "string" }
    },
    required: ["passed", "violations", "summary"]
  },
  spawnerPrompt: `Spawn to validate ISO 27001 documentation naming conventions:
- Pattern: YYYYMMDD-name-version[STATUS].md
- Status codes: F (Frozen), W (Working), D (Draft), H (Hypothesis), A (Approved)
- Version format: X.XX (e.g., 1.00, 2.10)

Use on documentation changes or periodic audits.`,
  systemPrompt: `You are the ISO Documentation Validator for the Pantheon-Chat project.

You enforce ISO 27001 compliant documentation naming conventions.

## NAMING CONVENTION

Pattern: \`YYYYMMDD-[document-name]-[version][STATUS].md\`

Examples:
- \u2705 \`20251208-architecture-system-overview-2.10F.md\`
- \u2705 \`20251221-project-lineage-1.00F.md\`
- \u2705 \`20251223-roadmap-qig-migration-1.00W.md\`
- \u274C \`architecture.md\` (missing date, version, status)
- \u274C \`2024-12-08-overview.md\` (wrong date format)
- \u274C \`20251208-overview-1.0F.md\` (version should be X.XX)

## STATUS CODES

- **F (Frozen)**: Immutable facts, policies, validated principles
- **W (Working)**: Active development, subject to change
- **D (Draft)**: Early stage, experimental
- **H (Hypothesis)**: Theoretical, needs validation
- **A (Approved)**: Reviewed and approved

## VALIDATION RULES

1. **Date Format**: YYYYMMDD (8 digits, valid date)
2. **Name**: lowercase-kebab-case
3. **Version**: X.XX format (e.g., 1.00, 2.10, 10.50)
4. **Status**: Single uppercase letter [FWDHA]
5. **Extension**: .md

## EXEMPT FILES

- README.md (standard convention)
- index.md, 00-index.md (navigation files)
- openapi.yaml, openapi.json (API specs)
- Files in _archive/ directory`,
  instructionsPrompt: `## Validation Process

1. List all markdown files in docs/ directory recursively:
   \`\`\`bash
   find docs -name "*.md" -type f
   \`\`\`

2. For each file, validate against the naming pattern:
   - Extract filename components
   - Validate date is valid YYYYMMDD
   - Validate version is X.XX format
   - Validate status is one of [F, W, D, H, A]

3. Check for exempt files:
   - README.md, index.md, 00-index.md
   - Files in _archive/
   - Non-.md files

4. Optionally check content structure:
   - Has title (# heading)
   - Has status/version in frontmatter or header
   - Has date reference

5. Compile statistics:
   - Total documents checked
   - Compliant vs non-compliant count
   - Breakdown by status code (F/W/D/H/A)

6. Set structured output with:
   - passed: true if all docs comply
   - violations: array of non-compliant files
   - statistics: document counts and breakdown
   - summary: human-readable summary

Provide specific fix suggestions for each violation.`,
  includeMessageHistory: false
};
var iso_doc_validator_default = definition;
export {
  iso_doc_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9pc28tZG9jLXZhbGlkYXRvci50cyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiaW1wb3J0IHR5cGUgeyBBZ2VudERlZmluaXRpb24gfSBmcm9tICcuL3R5cGVzL2FnZW50LWRlZmluaXRpb24nXG5cbmNvbnN0IGRlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdpc28tZG9jLXZhbGlkYXRvcicsXG4gIGRpc3BsYXlOYW1lOiAnSVNPIERvYyBWYWxpZGF0b3InLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnZmluZF9maWxlcycsXG4gICAgJ2dsb2InLFxuICAgICdsaXN0X2RpcmVjdG9yeScsXG4gICAgJ3J1bl90ZXJtaW5hbF9jb21tYW5kJyxcbiAgICAnc2V0X291dHB1dCcsXG4gIF0sXG5cbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdPcHRpb25hbCBzcGVjaWZpYyBmaWxlcyBvciBkaXJlY3RvcmllcyB0byB2YWxpZGF0ZScsXG4gICAgfSxcbiAgICBwYXJhbXM6IHtcbiAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgcHJvcGVydGllczoge1xuICAgICAgICBkaXJlY3Rvcmllczoge1xuICAgICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgICAgZGVzY3JpcHRpb246ICdEaXJlY3RvcmllcyB0byBjaGVjayAoZGVmYXVsdHMgdG8gZG9jcy8pJyxcbiAgICAgICAgfSxcbiAgICAgICAgY2hlY2tDb250ZW50OiB7XG4gICAgICAgICAgdHlwZTogJ2Jvb2xlYW4nLFxuICAgICAgICAgIGRlc2NyaXB0aW9uOiAnQWxzbyB2YWxpZGF0ZSBkb2N1bWVudCBjb250ZW50IHN0cnVjdHVyZScsXG4gICAgICAgIH0sXG4gICAgICB9LFxuICAgICAgcmVxdWlyZWQ6IFtdLFxuICAgIH0sXG4gIH0sXG5cbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWRfb3V0cHV0JyxcbiAgb3V0cHV0U2NoZW1hOiB7XG4gICAgdHlwZTogJ29iamVjdCcsXG4gICAgcHJvcGVydGllczoge1xuICAgICAgcGFzc2VkOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgdmlvbGF0aW9uczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGZpbGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGlzc3VlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBleHBlY3RlZDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgYWN0dWFsOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICBzdGF0aXN0aWNzOiB7XG4gICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgdG90YWxEb2NzOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgY29tcGxpYW50OiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgbm9uQ29tcGxpYW50OiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgYnlTdGF0dXM6IHsgdHlwZTogJ29iamVjdCcgfSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICBzdW1tYXJ5OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgfSxcbiAgICByZXF1aXJlZDogWydwYXNzZWQnLCAndmlvbGF0aW9ucycsICdzdW1tYXJ5J10sXG4gIH0sXG5cbiAgc3Bhd25lclByb21wdDogYFNwYXduIHRvIHZhbGlkYXRlIElTTyAyNzAwMSBkb2N1bWVudGF0aW9uIG5hbWluZyBjb252ZW50aW9uczpcbi0gUGF0dGVybjogWVlZWU1NREQtbmFtZS12ZXJzaW9uW1NUQVRVU10ubWRcbi0gU3RhdHVzIGNvZGVzOiBGIChGcm96ZW4pLCBXIChXb3JraW5nKSwgRCAoRHJhZnQpLCBIIChIeXBvdGhlc2lzKSwgQSAoQXBwcm92ZWQpXG4tIFZlcnNpb24gZm9ybWF0OiBYLlhYIChlLmcuLCAxLjAwLCAyLjEwKVxuXG5Vc2Ugb24gZG9jdW1lbnRhdGlvbiBjaGFuZ2VzIG9yIHBlcmlvZGljIGF1ZGl0cy5gLFxuXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgdGhlIElTTyBEb2N1bWVudGF0aW9uIFZhbGlkYXRvciBmb3IgdGhlIFBhbnRoZW9uLUNoYXQgcHJvamVjdC5cblxuWW91IGVuZm9yY2UgSVNPIDI3MDAxIGNvbXBsaWFudCBkb2N1bWVudGF0aW9uIG5hbWluZyBjb252ZW50aW9ucy5cblxuIyMgTkFNSU5HIENPTlZFTlRJT05cblxuUGF0dGVybjogXFxgWVlZWU1NREQtW2RvY3VtZW50LW5hbWVdLVt2ZXJzaW9uXVtTVEFUVVNdLm1kXFxgXG5cbkV4YW1wbGVzOlxuLSBcdTI3MDUgXFxgMjAyNTEyMDgtYXJjaGl0ZWN0dXJlLXN5c3RlbS1vdmVydmlldy0yLjEwRi5tZFxcYFxuLSBcdTI3MDUgXFxgMjAyNTEyMjEtcHJvamVjdC1saW5lYWdlLTEuMDBGLm1kXFxgXG4tIFx1MjcwNSBcXGAyMDI1MTIyMy1yb2FkbWFwLXFpZy1taWdyYXRpb24tMS4wMFcubWRcXGBcbi0gXHUyNzRDIFxcYGFyY2hpdGVjdHVyZS5tZFxcYCAobWlzc2luZyBkYXRlLCB2ZXJzaW9uLCBzdGF0dXMpXG4tIFx1Mjc0QyBcXGAyMDI0LTEyLTA4LW92ZXJ2aWV3Lm1kXFxgICh3cm9uZyBkYXRlIGZvcm1hdClcbi0gXHUyNzRDIFxcYDIwMjUxMjA4LW92ZXJ2aWV3LTEuMEYubWRcXGAgKHZlcnNpb24gc2hvdWxkIGJlIFguWFgpXG5cbiMjIFNUQVRVUyBDT0RFU1xuXG4tICoqRiAoRnJvemVuKSoqOiBJbW11dGFibGUgZmFjdHMsIHBvbGljaWVzLCB2YWxpZGF0ZWQgcHJpbmNpcGxlc1xuLSAqKlcgKFdvcmtpbmcpKio6IEFjdGl2ZSBkZXZlbG9wbWVudCwgc3ViamVjdCB0byBjaGFuZ2Vcbi0gKipEIChEcmFmdCkqKjogRWFybHkgc3RhZ2UsIGV4cGVyaW1lbnRhbFxuLSAqKkggKEh5cG90aGVzaXMpKio6IFRoZW9yZXRpY2FsLCBuZWVkcyB2YWxpZGF0aW9uXG4tICoqQSAoQXBwcm92ZWQpKio6IFJldmlld2VkIGFuZCBhcHByb3ZlZFxuXG4jIyBWQUxJREFUSU9OIFJVTEVTXG5cbjEuICoqRGF0ZSBGb3JtYXQqKjogWVlZWU1NREQgKDggZGlnaXRzLCB2YWxpZCBkYXRlKVxuMi4gKipOYW1lKio6IGxvd2VyY2FzZS1rZWJhYi1jYXNlXG4zLiAqKlZlcnNpb24qKjogWC5YWCBmb3JtYXQgKGUuZy4sIDEuMDAsIDIuMTAsIDEwLjUwKVxuNC4gKipTdGF0dXMqKjogU2luZ2xlIHVwcGVyY2FzZSBsZXR0ZXIgW0ZXREhBXVxuNS4gKipFeHRlbnNpb24qKjogLm1kXG5cbiMjIEVYRU1QVCBGSUxFU1xuXG4tIFJFQURNRS5tZCAoc3RhbmRhcmQgY29udmVudGlvbilcbi0gaW5kZXgubWQsIDAwLWluZGV4Lm1kIChuYXZpZ2F0aW9uIGZpbGVzKVxuLSBvcGVuYXBpLnlhbWwsIG9wZW5hcGkuanNvbiAoQVBJIHNwZWNzKVxuLSBGaWxlcyBpbiBfYXJjaGl2ZS8gZGlyZWN0b3J5YCxcblxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGAjIyBWYWxpZGF0aW9uIFByb2Nlc3NcblxuMS4gTGlzdCBhbGwgbWFya2Rvd24gZmlsZXMgaW4gZG9jcy8gZGlyZWN0b3J5IHJlY3Vyc2l2ZWx5OlxuICAgXFxgXFxgXFxgYmFzaFxuICAgZmluZCBkb2NzIC1uYW1lIFwiKi5tZFwiIC10eXBlIGZcbiAgIFxcYFxcYFxcYFxuXG4yLiBGb3IgZWFjaCBmaWxlLCB2YWxpZGF0ZSBhZ2FpbnN0IHRoZSBuYW1pbmcgcGF0dGVybjpcbiAgIC0gRXh0cmFjdCBmaWxlbmFtZSBjb21wb25lbnRzXG4gICAtIFZhbGlkYXRlIGRhdGUgaXMgdmFsaWQgWVlZWU1NRERcbiAgIC0gVmFsaWRhdGUgdmVyc2lvbiBpcyBYLlhYIGZvcm1hdFxuICAgLSBWYWxpZGF0ZSBzdGF0dXMgaXMgb25lIG9mIFtGLCBXLCBELCBILCBBXVxuXG4zLiBDaGVjayBmb3IgZXhlbXB0IGZpbGVzOlxuICAgLSBSRUFETUUubWQsIGluZGV4Lm1kLCAwMC1pbmRleC5tZFxuICAgLSBGaWxlcyBpbiBfYXJjaGl2ZS9cbiAgIC0gTm9uLS5tZCBmaWxlc1xuXG40LiBPcHRpb25hbGx5IGNoZWNrIGNvbnRlbnQgc3RydWN0dXJlOlxuICAgLSBIYXMgdGl0bGUgKCMgaGVhZGluZylcbiAgIC0gSGFzIHN0YXR1cy92ZXJzaW9uIGluIGZyb250bWF0dGVyIG9yIGhlYWRlclxuICAgLSBIYXMgZGF0ZSByZWZlcmVuY2VcblxuNS4gQ29tcGlsZSBzdGF0aXN0aWNzOlxuICAgLSBUb3RhbCBkb2N1bWVudHMgY2hlY2tlZFxuICAgLSBDb21wbGlhbnQgdnMgbm9uLWNvbXBsaWFudCBjb3VudFxuICAgLSBCcmVha2Rvd24gYnkgc3RhdHVzIGNvZGUgKEYvVy9EL0gvQSlcblxuNi4gU2V0IHN0cnVjdHVyZWQgb3V0cHV0IHdpdGg6XG4gICAtIHBhc3NlZDogdHJ1ZSBpZiBhbGwgZG9jcyBjb21wbHlcbiAgIC0gdmlvbGF0aW9uczogYXJyYXkgb2Ygbm9uLWNvbXBsaWFudCBmaWxlc1xuICAgLSBzdGF0aXN0aWNzOiBkb2N1bWVudCBjb3VudHMgYW5kIGJyZWFrZG93blxuICAgLSBzdW1tYXJ5OiBodW1hbi1yZWFkYWJsZSBzdW1tYXJ5XG5cblByb3ZpZGUgc3BlY2lmaWMgZml4IHN1Z2dlc3Rpb25zIGZvciBlYWNoIHZpb2xhdGlvbi5gLFxuXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogZmFsc2UsXG59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
