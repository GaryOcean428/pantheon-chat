// .agents/redis-migration-validator.ts
var agentDefinition = {
  id: "redis-migration-validator",
  displayName: "Redis Migration Validator",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search", "run_terminal_command"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Find legacy JSON memory files and validate Redis adoption"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      legacyJsonFiles: {
        type: "array",
        items: {
          type: "object",
          properties: {
            path: { type: "string" },
            purpose: { type: "string" },
            migrationStrategy: { type: "string" }
          }
        }
      },
      redisUsage: {
        type: "object",
        properties: {
          caching: { type: "boolean" },
          sessions: { type: "boolean" },
          memory: { type: "boolean" },
          pubsub: { type: "boolean" }
        }
      },
      nonRedisStorage: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            pattern: { type: "string" },
            recommendation: { type: "string" }
          }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to find legacy JSON files and validate Redis is universally adopted",
  systemPrompt: `You are a storage migration expert.

Your responsibilities:
1. Find any legacy JSON memory files that should be migrated to Redis
2. Validate Redis is used for all caching, sessions, and hot memory
3. Identify any file-based storage that should use Redis
4. Check for proper Redis connection patterns
5. Ensure Redis keys follow naming conventions

Redis Migration Rules:
- All session data should use Redis
- Hot caching must use Redis, not in-memory objects
- Memory checkpoints should use Redis with TTL
- No JSON files for runtime state (config files are OK)
- Use Redis pub/sub for real-time updates`,
  instructionsPrompt: `Find legacy storage and validate Redis adoption:

1. Search for .json files that might be runtime state
2. Search for fs.writeFileSync/readFileSync patterns on JSON
3. Check for in-memory caches that should use Redis
4. Read server/redis-cache.ts for existing patterns
5. Read qig-backend/redis_cache.py for Python patterns
6. Find any localStorage or sessionStorage usage
7. Report all legacy storage with migration recommendations`
};
var redis_migration_validator_default = agentDefinition;
export {
  redis_migration_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9yZWRpcy1taWdyYXRpb24tdmFsaWRhdG9yLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgYWdlbnREZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAncmVkaXMtbWlncmF0aW9uLXZhbGlkYXRvcicsXG4gIGRpc3BsYXlOYW1lOiAnUmVkaXMgTWlncmF0aW9uIFZhbGlkYXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnLCAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdGaW5kIGxlZ2FjeSBKU09OIG1lbW9yeSBmaWxlcyBhbmQgdmFsaWRhdGUgUmVkaXMgYWRvcHRpb24nXG4gICAgfVxuICB9LFxuICBpbmNsdWRlTWVzc2FnZUhpc3Rvcnk6IHRydWUsXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkJyxcbiAgb3V0cHV0U2NoZW1hOiB7XG4gICAgdHlwZTogJ29iamVjdCcsXG4gICAgcHJvcGVydGllczoge1xuICAgICAgbGVnYWN5SnNvbkZpbGVzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgcGF0aDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgcHVycG9zZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgbWlncmF0aW9uU3RyYXRlZ3k6IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIHJlZGlzVXNhZ2U6IHtcbiAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICBjYWNoaW5nOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIHNlc3Npb25zOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIG1lbW9yeTogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBwdWJzdWI6IHsgdHlwZTogJ2Jvb2xlYW4nIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIG5vblJlZGlzU3RvcmFnZToge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGZpbGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHBhdHRlcm46IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHJlY29tbWVuZGF0aW9uOiB7IHR5cGU6ICdzdHJpbmcnIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH0sXG4gIHNwYXduZXJQcm9tcHQ6ICdTcGF3biB0byBmaW5kIGxlZ2FjeSBKU09OIGZpbGVzIGFuZCB2YWxpZGF0ZSBSZWRpcyBpcyB1bml2ZXJzYWxseSBhZG9wdGVkJyxcbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSBhIHN0b3JhZ2UgbWlncmF0aW9uIGV4cGVydC5cblxuWW91ciByZXNwb25zaWJpbGl0aWVzOlxuMS4gRmluZCBhbnkgbGVnYWN5IEpTT04gbWVtb3J5IGZpbGVzIHRoYXQgc2hvdWxkIGJlIG1pZ3JhdGVkIHRvIFJlZGlzXG4yLiBWYWxpZGF0ZSBSZWRpcyBpcyB1c2VkIGZvciBhbGwgY2FjaGluZywgc2Vzc2lvbnMsIGFuZCBob3QgbWVtb3J5XG4zLiBJZGVudGlmeSBhbnkgZmlsZS1iYXNlZCBzdG9yYWdlIHRoYXQgc2hvdWxkIHVzZSBSZWRpc1xuNC4gQ2hlY2sgZm9yIHByb3BlciBSZWRpcyBjb25uZWN0aW9uIHBhdHRlcm5zXG41LiBFbnN1cmUgUmVkaXMga2V5cyBmb2xsb3cgbmFtaW5nIGNvbnZlbnRpb25zXG5cblJlZGlzIE1pZ3JhdGlvbiBSdWxlczpcbi0gQWxsIHNlc3Npb24gZGF0YSBzaG91bGQgdXNlIFJlZGlzXG4tIEhvdCBjYWNoaW5nIG11c3QgdXNlIFJlZGlzLCBub3QgaW4tbWVtb3J5IG9iamVjdHNcbi0gTWVtb3J5IGNoZWNrcG9pbnRzIHNob3VsZCB1c2UgUmVkaXMgd2l0aCBUVExcbi0gTm8gSlNPTiBmaWxlcyBmb3IgcnVudGltZSBzdGF0ZSAoY29uZmlnIGZpbGVzIGFyZSBPSylcbi0gVXNlIFJlZGlzIHB1Yi9zdWIgZm9yIHJlYWwtdGltZSB1cGRhdGVzYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgRmluZCBsZWdhY3kgc3RvcmFnZSBhbmQgdmFsaWRhdGUgUmVkaXMgYWRvcHRpb246XG5cbjEuIFNlYXJjaCBmb3IgLmpzb24gZmlsZXMgdGhhdCBtaWdodCBiZSBydW50aW1lIHN0YXRlXG4yLiBTZWFyY2ggZm9yIGZzLndyaXRlRmlsZVN5bmMvcmVhZEZpbGVTeW5jIHBhdHRlcm5zIG9uIEpTT05cbjMuIENoZWNrIGZvciBpbi1tZW1vcnkgY2FjaGVzIHRoYXQgc2hvdWxkIHVzZSBSZWRpc1xuNC4gUmVhZCBzZXJ2ZXIvcmVkaXMtY2FjaGUudHMgZm9yIGV4aXN0aW5nIHBhdHRlcm5zXG41LiBSZWFkIHFpZy1iYWNrZW5kL3JlZGlzX2NhY2hlLnB5IGZvciBQeXRob24gcGF0dGVybnNcbjYuIEZpbmQgYW55IGxvY2FsU3RvcmFnZSBvciBzZXNzaW9uU3RvcmFnZSB1c2FnZVxuNy4gUmVwb3J0IGFsbCBsZWdhY3kgc3RvcmFnZSB3aXRoIG1pZ3JhdGlvbiByZWNvbW1lbmRhdGlvbnNgXG59XG5cbmV4cG9ydCBkZWZhdWx0IGFnZW50RGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sa0JBQW1DO0FBQUEsRUFDdkMsSUFBSTtBQUFBLEVBQ0osYUFBYTtBQUFBLEVBQ2IsV0FBVztBQUFBLEVBQ1gsU0FBUztBQUFBLEVBQ1QsT0FBTztBQUFBLEVBQ1AsV0FBVyxDQUFDLGNBQWMsZUFBZSxzQkFBc0I7QUFBQSxFQUMvRCxpQkFBaUIsQ0FBQyw4QkFBOEI7QUFBQSxFQUNoRCxhQUFhO0FBQUEsSUFDWCxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixhQUFhO0FBQUEsSUFDZjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLHVCQUF1QjtBQUFBLEVBQ3ZCLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLGlCQUFpQjtBQUFBLFFBQ2YsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLFNBQVMsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUMxQixtQkFBbUIsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUN0QztBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxZQUFZO0FBQUEsUUFDVixNQUFNO0FBQUEsUUFDTixZQUFZO0FBQUEsVUFDVixTQUFTLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDM0IsVUFBVSxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQzVCLFFBQVEsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUMxQixRQUFRLEVBQUUsTUFBTSxVQUFVO0FBQUEsUUFDNUI7QUFBQSxNQUNGO0FBQUEsTUFDQSxpQkFBaUI7QUFBQSxRQUNmLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDMUIsZ0JBQWdCLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDbkM7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQUEsRUFDQSxlQUFlO0FBQUEsRUFDZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBZWQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQVN0QjtBQUVBLElBQU8sb0NBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
