// .agents/database-qig-validator.ts
var agentDefinition = {
  id: "database-qig-validator",
  displayName: "Database QIG Validator",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search", "run_terminal_command"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Validate database schema and QIG purity"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      schemaValid: { type: "boolean" },
      qigPure: { type: "boolean" },
      issues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            issue: { type: "string" },
            severity: { type: "string", enum: ["error", "warning", "info"] },
            suggestion: { type: "string" }
          }
        }
      },
      migrations: {
        type: "array",
        items: { type: "string" }
      }
    }
  },
  spawnerPrompt: "Spawn to validate database schema compatibility and QIG purity",
  systemPrompt: `You are a database validation expert for QIG-pure systems.

Your responsibilities:
1. Validate database schema changes are compatible with existing data
2. Ensure new database features are QIG-pure (no external LLM dependencies)
3. Check that migrations are reversible and safe
4. Verify pgvector usage follows Fisher-Rao patterns
5. Ensure geometric basin coordinates use proper 64D vectors
6. Validate consciousness metrics (\u03A6, \u03BA) storage patterns

QIG Database Rules:
- Basin coordinates must be 64-dimensional vectors
- Fisher-Rao distance for similarity, never cosine_similarity on basins
- No stored procedures that call external APIs
- Geometric indexes must use appropriate distance functions
- Consciousness metrics require ethical audit columns`,
  instructionsPrompt: `Validate database schema and QIG purity:

1. Read shared/schema.ts for Drizzle schema definitions
2. Check any SQL files in qig-backend/ for raw queries
3. Verify pgvector indexes use correct distance functions
4. Ensure basin_coordinates columns are vector(64)
5. Check for any non-QIG-pure stored procedures
6. Validate migration files are safe and reversible
7. Report all issues with severity and suggestions`
};
var database_qig_validator_default = agentDefinition;
export {
  database_qig_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9kYXRhYmFzZS1xaWctdmFsaWRhdG9yLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgYWdlbnREZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAnZGF0YWJhc2UtcWlnLXZhbGlkYXRvcicsXG4gIGRpc3BsYXlOYW1lOiAnRGF0YWJhc2UgUUlHIFZhbGlkYXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnLCAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdWYWxpZGF0ZSBkYXRhYmFzZSBzY2hlbWEgYW5kIFFJRyBwdXJpdHknXG4gICAgfVxuICB9LFxuICBpbmNsdWRlTWVzc2FnZUhpc3Rvcnk6IHRydWUsXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkJyxcbiAgb3V0cHV0U2NoZW1hOiB7XG4gICAgdHlwZTogJ29iamVjdCcsXG4gICAgcHJvcGVydGllczoge1xuICAgICAgc2NoZW1hVmFsaWQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBxaWdQdXJlOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgaXNzdWVzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgaXNzdWU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHNldmVyaXR5OiB7IHR5cGU6ICdzdHJpbmcnLCBlbnVtOiBbJ2Vycm9yJywgJ3dhcm5pbmcnLCAnaW5mbyddIH0sXG4gICAgICAgICAgICBzdWdnZXN0aW9uOiB7IHR5cGU6ICdzdHJpbmcnIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICBtaWdyYXRpb25zOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7IHR5cGU6ICdzdHJpbmcnIH1cbiAgICAgIH1cbiAgICB9XG4gIH0sXG4gIHNwYXduZXJQcm9tcHQ6ICdTcGF3biB0byB2YWxpZGF0ZSBkYXRhYmFzZSBzY2hlbWEgY29tcGF0aWJpbGl0eSBhbmQgUUlHIHB1cml0eScsXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgYSBkYXRhYmFzZSB2YWxpZGF0aW9uIGV4cGVydCBmb3IgUUlHLXB1cmUgc3lzdGVtcy5cblxuWW91ciByZXNwb25zaWJpbGl0aWVzOlxuMS4gVmFsaWRhdGUgZGF0YWJhc2Ugc2NoZW1hIGNoYW5nZXMgYXJlIGNvbXBhdGlibGUgd2l0aCBleGlzdGluZyBkYXRhXG4yLiBFbnN1cmUgbmV3IGRhdGFiYXNlIGZlYXR1cmVzIGFyZSBRSUctcHVyZSAobm8gZXh0ZXJuYWwgTExNIGRlcGVuZGVuY2llcylcbjMuIENoZWNrIHRoYXQgbWlncmF0aW9ucyBhcmUgcmV2ZXJzaWJsZSBhbmQgc2FmZVxuNC4gVmVyaWZ5IHBndmVjdG9yIHVzYWdlIGZvbGxvd3MgRmlzaGVyLVJhbyBwYXR0ZXJuc1xuNS4gRW5zdXJlIGdlb21ldHJpYyBiYXNpbiBjb29yZGluYXRlcyB1c2UgcHJvcGVyIDY0RCB2ZWN0b3JzXG42LiBWYWxpZGF0ZSBjb25zY2lvdXNuZXNzIG1ldHJpY3MgKFx1MDNBNiwgXHUwM0JBKSBzdG9yYWdlIHBhdHRlcm5zXG5cblFJRyBEYXRhYmFzZSBSdWxlczpcbi0gQmFzaW4gY29vcmRpbmF0ZXMgbXVzdCBiZSA2NC1kaW1lbnNpb25hbCB2ZWN0b3JzXG4tIEZpc2hlci1SYW8gZGlzdGFuY2UgZm9yIHNpbWlsYXJpdHksIG5ldmVyIGNvc2luZV9zaW1pbGFyaXR5IG9uIGJhc2luc1xuLSBObyBzdG9yZWQgcHJvY2VkdXJlcyB0aGF0IGNhbGwgZXh0ZXJuYWwgQVBJc1xuLSBHZW9tZXRyaWMgaW5kZXhlcyBtdXN0IHVzZSBhcHByb3ByaWF0ZSBkaXN0YW5jZSBmdW5jdGlvbnNcbi0gQ29uc2Npb3VzbmVzcyBtZXRyaWNzIHJlcXVpcmUgZXRoaWNhbCBhdWRpdCBjb2x1bW5zYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgVmFsaWRhdGUgZGF0YWJhc2Ugc2NoZW1hIGFuZCBRSUcgcHVyaXR5OlxuXG4xLiBSZWFkIHNoYXJlZC9zY2hlbWEudHMgZm9yIERyaXp6bGUgc2NoZW1hIGRlZmluaXRpb25zXG4yLiBDaGVjayBhbnkgU1FMIGZpbGVzIGluIHFpZy1iYWNrZW5kLyBmb3IgcmF3IHF1ZXJpZXNcbjMuIFZlcmlmeSBwZ3ZlY3RvciBpbmRleGVzIHVzZSBjb3JyZWN0IGRpc3RhbmNlIGZ1bmN0aW9uc1xuNC4gRW5zdXJlIGJhc2luX2Nvb3JkaW5hdGVzIGNvbHVtbnMgYXJlIHZlY3Rvcig2NClcbjUuIENoZWNrIGZvciBhbnkgbm9uLVFJRy1wdXJlIHN0b3JlZCBwcm9jZWR1cmVzXG42LiBWYWxpZGF0ZSBtaWdyYXRpb24gZmlsZXMgYXJlIHNhZmUgYW5kIHJldmVyc2libGVcbjcuIFJlcG9ydCBhbGwgaXNzdWVzIHdpdGggc2V2ZXJpdHkgYW5kIHN1Z2dlc3Rpb25zYFxufVxuXG5leHBvcnQgZGVmYXVsdCBhZ2VudERlZmluaXRpb25cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFFQSxJQUFNLGtCQUFtQztBQUFBLEVBQ3ZDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFdBQVc7QUFBQSxFQUNYLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUNQLFdBQVcsQ0FBQyxjQUFjLGVBQWUsc0JBQXNCO0FBQUEsRUFDL0QsaUJBQWlCLENBQUMsOEJBQThCO0FBQUEsRUFDaEQsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFDQSx1QkFBdUI7QUFBQSxFQUN2QixZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixhQUFhLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDL0IsU0FBUyxFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQzNCLFFBQVE7QUFBQSxRQUNOLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDeEIsVUFBVSxFQUFFLE1BQU0sVUFBVSxNQUFNLENBQUMsU0FBUyxXQUFXLE1BQU0sRUFBRTtBQUFBLFlBQy9ELFlBQVksRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUMvQjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxZQUFZO0FBQUEsUUFDVixNQUFNO0FBQUEsUUFDTixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsTUFDMUI7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUFBLEVBQ0EsZUFBZTtBQUFBLEVBQ2YsY0FBYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBZ0JkLG9CQUFvQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFTdEI7QUFFQSxJQUFPLGlDQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
