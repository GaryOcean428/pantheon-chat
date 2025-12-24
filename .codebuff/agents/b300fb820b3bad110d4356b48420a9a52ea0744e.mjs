// .agents/devops-auditor.ts
var agentDefinition = {
  id: "devops-auditor",
  displayName: "DevOps Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search", "run_terminal_command"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Audit DevOps and deployment configuration"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      deploymentReady: { type: "boolean" },
      cicdStatus: {
        type: "object",
        properties: {
          pipelineExists: { type: "boolean" },
          testsInPipeline: { type: "boolean" },
          previewDeployments: { type: "boolean" },
          automatedReleases: { type: "boolean" }
        }
      },
      infrastructure: {
        type: "object",
        properties: {
          dockerized: { type: "boolean" },
          envParity: { type: "boolean" },
          secretsManaged: { type: "boolean" },
          backupsConfigured: { type: "boolean" }
        }
      },
      issues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            area: { type: "string" },
            issue: { type: "string" },
            severity: { type: "string" },
            recommendation: { type: "string" }
          }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to audit DevOps configuration and deployment readiness",
  systemPrompt: `You are a DevOps and infrastructure expert.

Audit areas:
1. CI/CD pipeline configuration
2. Docker configuration
3. Environment parity (dev/staging/prod)
4. Secrets management
5. Database backups
6. Deployment strategies (blue-green, canary)
7. Monitoring and logging
8. Auto-scaling configuration

Best Practices:
- Tests must run in CI pipeline
- Preview deployments for PRs
- Semantic versioning with automated releases
- Secrets in environment, not code
- Database backup strategy documented
- Zero-downtime deployments`,
  instructionsPrompt: `Audit DevOps configuration:

1. Check for .github/workflows/ or CI config
2. Read Dockerfile configurations
3. Check docker-compose files
4. Verify .env.example exists
5. Check for secrets in codebase
6. Read deployment configs (railway.json, etc.)
7. Check for backup scripts
8. Verify monitoring setup
9. Report issues with severity`
};
var devops_auditor_default = agentDefinition;
export {
  devops_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9kZXZvcHMtYXVkaXRvci50cyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiaW1wb3J0IHR5cGUgeyBBZ2VudERlZmluaXRpb24gfSBmcm9tICcuL3R5cGVzL2FnZW50LWRlZmluaXRpb24nXG5cbmNvbnN0IGFnZW50RGVmaW5pdGlvbjogQWdlbnREZWZpbml0aW9uID0ge1xuICBpZDogJ2Rldm9wcy1hdWRpdG9yJyxcbiAgZGlzcGxheU5hbWU6ICdEZXZPcHMgQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnLCAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdBdWRpdCBEZXZPcHMgYW5kIGRlcGxveW1lbnQgY29uZmlndXJhdGlvbidcbiAgICB9XG4gIH0sXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogdHJ1ZSxcbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBkZXBsb3ltZW50UmVhZHk6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBjaWNkU3RhdHVzOiB7XG4gICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgcGlwZWxpbmVFeGlzdHM6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgdGVzdHNJblBpcGVsaW5lOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIHByZXZpZXdEZXBsb3ltZW50czogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBhdXRvbWF0ZWRSZWxlYXNlczogeyB0eXBlOiAnYm9vbGVhbicgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgaW5mcmFzdHJ1Y3R1cmU6IHtcbiAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICBkb2NrZXJpemVkOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIGVudlBhcml0eTogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBzZWNyZXRzTWFuYWdlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBiYWNrdXBzQ29uZmlndXJlZDogeyB0eXBlOiAnYm9vbGVhbicgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgaXNzdWVzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgYXJlYTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgaXNzdWU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHNldmVyaXR5OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICByZWNvbW1lbmRhdGlvbjogeyB0eXBlOiAnc3RyaW5nJyB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9LFxuICBzcGF3bmVyUHJvbXB0OiAnU3Bhd24gdG8gYXVkaXQgRGV2T3BzIGNvbmZpZ3VyYXRpb24gYW5kIGRlcGxveW1lbnQgcmVhZGluZXNzJyxcbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSBhIERldk9wcyBhbmQgaW5mcmFzdHJ1Y3R1cmUgZXhwZXJ0LlxuXG5BdWRpdCBhcmVhczpcbjEuIENJL0NEIHBpcGVsaW5lIGNvbmZpZ3VyYXRpb25cbjIuIERvY2tlciBjb25maWd1cmF0aW9uXG4zLiBFbnZpcm9ubWVudCBwYXJpdHkgKGRldi9zdGFnaW5nL3Byb2QpXG40LiBTZWNyZXRzIG1hbmFnZW1lbnRcbjUuIERhdGFiYXNlIGJhY2t1cHNcbjYuIERlcGxveW1lbnQgc3RyYXRlZ2llcyAoYmx1ZS1ncmVlbiwgY2FuYXJ5KVxuNy4gTW9uaXRvcmluZyBhbmQgbG9nZ2luZ1xuOC4gQXV0by1zY2FsaW5nIGNvbmZpZ3VyYXRpb25cblxuQmVzdCBQcmFjdGljZXM6XG4tIFRlc3RzIG11c3QgcnVuIGluIENJIHBpcGVsaW5lXG4tIFByZXZpZXcgZGVwbG95bWVudHMgZm9yIFBSc1xuLSBTZW1hbnRpYyB2ZXJzaW9uaW5nIHdpdGggYXV0b21hdGVkIHJlbGVhc2VzXG4tIFNlY3JldHMgaW4gZW52aXJvbm1lbnQsIG5vdCBjb2RlXG4tIERhdGFiYXNlIGJhY2t1cCBzdHJhdGVneSBkb2N1bWVudGVkXG4tIFplcm8tZG93bnRpbWUgZGVwbG95bWVudHNgLFxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGBBdWRpdCBEZXZPcHMgY29uZmlndXJhdGlvbjpcblxuMS4gQ2hlY2sgZm9yIC5naXRodWIvd29ya2Zsb3dzLyBvciBDSSBjb25maWdcbjIuIFJlYWQgRG9ja2VyZmlsZSBjb25maWd1cmF0aW9uc1xuMy4gQ2hlY2sgZG9ja2VyLWNvbXBvc2UgZmlsZXNcbjQuIFZlcmlmeSAuZW52LmV4YW1wbGUgZXhpc3RzXG41LiBDaGVjayBmb3Igc2VjcmV0cyBpbiBjb2RlYmFzZVxuNi4gUmVhZCBkZXBsb3ltZW50IGNvbmZpZ3MgKHJhaWx3YXkuanNvbiwgZXRjLilcbjcuIENoZWNrIGZvciBiYWNrdXAgc2NyaXB0c1xuOC4gVmVyaWZ5IG1vbml0b3Jpbmcgc2V0dXBcbjkuIFJlcG9ydCBpc3N1ZXMgd2l0aCBzZXZlcml0eWBcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyxlQUFlLHNCQUFzQjtBQUFBLEVBQy9ELGlCQUFpQixDQUFDLDhCQUE4QjtBQUFBLEVBQ2hELGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsRUFDRjtBQUFBLEVBQ0EsdUJBQXVCO0FBQUEsRUFDdkIsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YsaUJBQWlCLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDbkMsWUFBWTtBQUFBLFFBQ1YsTUFBTTtBQUFBLFFBQ04sWUFBWTtBQUFBLFVBQ1YsZ0JBQWdCLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDbEMsaUJBQWlCLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDbkMsb0JBQW9CLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDdEMsbUJBQW1CLEVBQUUsTUFBTSxVQUFVO0FBQUEsUUFDdkM7QUFBQSxNQUNGO0FBQUEsTUFDQSxnQkFBZ0I7QUFBQSxRQUNkLE1BQU07QUFBQSxRQUNOLFlBQVk7QUFBQSxVQUNWLFlBQVksRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUM5QixXQUFXLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDN0IsZ0JBQWdCLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDbEMsbUJBQW1CLEVBQUUsTUFBTSxVQUFVO0FBQUEsUUFDdkM7QUFBQSxNQUNGO0FBQUEsTUFDQSxRQUFRO0FBQUEsUUFDTixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsT0FBTyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3hCLFVBQVUsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUMzQixnQkFBZ0IsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUNuQztBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLGVBQWU7QUFBQSxFQUNmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQW1CZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQVd0QjtBQUVBLElBQU8seUJBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
