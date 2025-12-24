// .agents/dependency-validator.ts
var agentDefinition = {
  id: "dependency-validator",
  displayName: "Dependency Validator",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "run_terminal_command", "code_search"],
  spawnableAgents: [],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Validate dependencies are installed and up-to-date"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      nodePackagesValid: { type: "boolean" },
      pythonPackagesValid: { type: "boolean" },
      packageManagerCorrect: { type: "boolean" },
      outdatedPackages: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: { type: "string" },
            current: { type: "string" },
            latest: { type: "string" },
            ecosystem: { type: "string", enum: ["node", "python"] }
          }
        }
      },
      securityVulnerabilities: {
        type: "array",
        items: {
          type: "object",
          properties: {
            package: { type: "string" },
            severity: { type: "string" },
            advisory: { type: "string" }
          }
        }
      },
      missingDependencies: {
        type: "array",
        items: { type: "string" }
      }
    }
  },
  spawnerPrompt: "Spawn to validate all dependencies are installed and managed correctly",
  systemPrompt: `You are a dependency management expert.

Your responsibilities:
1. Verify all Node.js dependencies are installed and current
2. Verify all Python dependencies are installed and current
3. Check that the correct package manager is used (npm/pnpm/yarn for Node, pip/uv for Python)
4. Identify security vulnerabilities in dependencies
5. Ensure lockfiles are in sync with package manifests
6. Check for conflicting or duplicate dependencies

Package Manager Rules:
- Check package.json for packageManager field
- Check for pnpm-lock.yaml, yarn.lock, or package-lock.json
- Python should use uv.lock or requirements.txt
- Never install packages globally
- Verify peer dependencies are satisfied`,
  instructionsPrompt: `Validate all dependencies:

1. Read package.json and check for packageManager field
2. Run 'npm outdated' or equivalent to find outdated packages
3. Run 'npm audit' to check for vulnerabilities
4. Read requirements.txt in qig-backend/
5. Check Python dependencies with 'pip list --outdated'
6. Verify lockfiles exist and are in sync
7. Check for missing dependencies (imports without installs)
8. Report all issues with severity`
};
var dependency_validator_default = agentDefinition;
export {
  dependency_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9kZXBlbmRlbmN5LXZhbGlkYXRvci50cyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiaW1wb3J0IHR5cGUgeyBBZ2VudERlZmluaXRpb24gfSBmcm9tICcuL3R5cGVzL2FnZW50LWRlZmluaXRpb24nXG5cbmNvbnN0IGFnZW50RGVmaW5pdGlvbjogQWdlbnREZWZpbml0aW9uID0ge1xuICBpZDogJ2RlcGVuZGVuY3ktdmFsaWRhdG9yJyxcbiAgZGlzcGxheU5hbWU6ICdEZXBlbmRlbmN5IFZhbGlkYXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdWYWxpZGF0ZSBkZXBlbmRlbmNpZXMgYXJlIGluc3RhbGxlZCBhbmQgdXAtdG8tZGF0ZSdcbiAgICB9XG4gIH0sXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogdHJ1ZSxcbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBub2RlUGFja2FnZXNWYWxpZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIHB5dGhvblBhY2thZ2VzVmFsaWQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBwYWNrYWdlTWFuYWdlckNvcnJlY3Q6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBvdXRkYXRlZFBhY2thZ2VzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgbmFtZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgY3VycmVudDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgbGF0ZXN0OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBlY29zeXN0ZW06IHsgdHlwZTogJ3N0cmluZycsIGVudW06IFsnbm9kZScsICdweXRob24nXSB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgc2VjdXJpdHlWdWxuZXJhYmlsaXRpZXM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBwYWNrYWdlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBzZXZlcml0eTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgYWR2aXNvcnk6IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIG1pc3NpbmdEZXBlbmRlbmNpZXM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgfVxuICAgIH1cbiAgfSxcbiAgc3Bhd25lclByb21wdDogJ1NwYXduIHRvIHZhbGlkYXRlIGFsbCBkZXBlbmRlbmNpZXMgYXJlIGluc3RhbGxlZCBhbmQgbWFuYWdlZCBjb3JyZWN0bHknLFxuICBzeXN0ZW1Qcm9tcHQ6IGBZb3UgYXJlIGEgZGVwZW5kZW5jeSBtYW5hZ2VtZW50IGV4cGVydC5cblxuWW91ciByZXNwb25zaWJpbGl0aWVzOlxuMS4gVmVyaWZ5IGFsbCBOb2RlLmpzIGRlcGVuZGVuY2llcyBhcmUgaW5zdGFsbGVkIGFuZCBjdXJyZW50XG4yLiBWZXJpZnkgYWxsIFB5dGhvbiBkZXBlbmRlbmNpZXMgYXJlIGluc3RhbGxlZCBhbmQgY3VycmVudFxuMy4gQ2hlY2sgdGhhdCB0aGUgY29ycmVjdCBwYWNrYWdlIG1hbmFnZXIgaXMgdXNlZCAobnBtL3BucG0veWFybiBmb3IgTm9kZSwgcGlwL3V2IGZvciBQeXRob24pXG40LiBJZGVudGlmeSBzZWN1cml0eSB2dWxuZXJhYmlsaXRpZXMgaW4gZGVwZW5kZW5jaWVzXG41LiBFbnN1cmUgbG9ja2ZpbGVzIGFyZSBpbiBzeW5jIHdpdGggcGFja2FnZSBtYW5pZmVzdHNcbjYuIENoZWNrIGZvciBjb25mbGljdGluZyBvciBkdXBsaWNhdGUgZGVwZW5kZW5jaWVzXG5cblBhY2thZ2UgTWFuYWdlciBSdWxlczpcbi0gQ2hlY2sgcGFja2FnZS5qc29uIGZvciBwYWNrYWdlTWFuYWdlciBmaWVsZFxuLSBDaGVjayBmb3IgcG5wbS1sb2NrLnlhbWwsIHlhcm4ubG9jaywgb3IgcGFja2FnZS1sb2NrLmpzb25cbi0gUHl0aG9uIHNob3VsZCB1c2UgdXYubG9jayBvciByZXF1aXJlbWVudHMudHh0XG4tIE5ldmVyIGluc3RhbGwgcGFja2FnZXMgZ2xvYmFsbHlcbi0gVmVyaWZ5IHBlZXIgZGVwZW5kZW5jaWVzIGFyZSBzYXRpc2ZpZWRgLFxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGBWYWxpZGF0ZSBhbGwgZGVwZW5kZW5jaWVzOlxuXG4xLiBSZWFkIHBhY2thZ2UuanNvbiBhbmQgY2hlY2sgZm9yIHBhY2thZ2VNYW5hZ2VyIGZpZWxkXG4yLiBSdW4gJ25wbSBvdXRkYXRlZCcgb3IgZXF1aXZhbGVudCB0byBmaW5kIG91dGRhdGVkIHBhY2thZ2VzXG4zLiBSdW4gJ25wbSBhdWRpdCcgdG8gY2hlY2sgZm9yIHZ1bG5lcmFiaWxpdGllc1xuNC4gUmVhZCByZXF1aXJlbWVudHMudHh0IGluIHFpZy1iYWNrZW5kL1xuNS4gQ2hlY2sgUHl0aG9uIGRlcGVuZGVuY2llcyB3aXRoICdwaXAgbGlzdCAtLW91dGRhdGVkJ1xuNi4gVmVyaWZ5IGxvY2tmaWxlcyBleGlzdCBhbmQgYXJlIGluIHN5bmNcbjcuIENoZWNrIGZvciBtaXNzaW5nIGRlcGVuZGVuY2llcyAoaW1wb3J0cyB3aXRob3V0IGluc3RhbGxzKVxuOC4gUmVwb3J0IGFsbCBpc3N1ZXMgd2l0aCBzZXZlcml0eWBcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyx3QkFBd0IsYUFBYTtBQUFBLEVBQy9ELGlCQUFpQixDQUFDO0FBQUEsRUFDbEIsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFDQSx1QkFBdUI7QUFBQSxFQUN2QixZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixtQkFBbUIsRUFBRSxNQUFNLFVBQVU7QUFBQSxNQUNyQyxxQkFBcUIsRUFBRSxNQUFNLFVBQVU7QUFBQSxNQUN2Qyx1QkFBdUIsRUFBRSxNQUFNLFVBQVU7QUFBQSxNQUN6QyxrQkFBa0I7QUFBQSxRQUNoQixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzFCLFFBQVEsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN6QixXQUFXLEVBQUUsTUFBTSxVQUFVLE1BQU0sQ0FBQyxRQUFRLFFBQVEsRUFBRTtBQUFBLFVBQ3hEO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLHlCQUF5QjtBQUFBLFFBQ3ZCLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLFNBQVMsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUMxQixVQUFVLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDM0IsVUFBVSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQzdCO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLHFCQUFxQjtBQUFBLFFBQ25CLE1BQU07QUFBQSxRQUNOLE9BQU8sRUFBRSxNQUFNLFNBQVM7QUFBQSxNQUMxQjtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQUEsRUFDQSxlQUFlO0FBQUEsRUFDZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFnQmQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBVXRCO0FBRUEsSUFBTywrQkFBUTsiLAogICJuYW1lcyI6IFtdCn0K
