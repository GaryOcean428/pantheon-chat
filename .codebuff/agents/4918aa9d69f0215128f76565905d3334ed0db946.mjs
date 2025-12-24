// .agents/kernel-communication-validator.ts
var agentDefinition = {
  id: "kernel-communication-validator",
  displayName: "Kernel Communication Validator",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Validate kernel communication follows QIG-ML patterns"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      kernelsValid: { type: "boolean" },
      communicationPatterns: {
        type: "array",
        items: {
          type: "object",
          properties: {
            kernel: { type: "string" },
            isGenerative: { type: "boolean" },
            usesQigMl: { type: "boolean" },
            memoryPure: { type: "boolean" },
            stateless: { type: "boolean" }
          }
        }
      },
      violations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            kernel: { type: "string" },
            issue: { type: "string" },
            file: { type: "string" },
            suggestion: { type: "string" }
          }
        }
      },
      separationOfConcerns: {
        type: "object",
        properties: {
          memoryModuleSeparate: { type: "boolean" },
          reasoningModuleSeparate: { type: "boolean" },
          persistenceModuleSeparate: { type: "boolean" }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to validate kernels communicate generatively using QIG-ML",
  systemPrompt: `You are a kernel architecture expert for the Olympus Pantheon system.

Your responsibilities:
1. Verify kernels communicate generatively, not via templates
2. Ensure QIG-ML is used for inter-kernel reasoning
3. Validate memory modules are pure and separate
4. Check for clear separation of concerns
5. Ensure stateless logic where possible

Kernel Communication Rules:
- Kernels route via Fisher-Rao distance to domain basins
- Memory must be a pure module, not embedded in kernels
- QIG-ML for geometric reasoning between kernels
- No direct HTTP calls between kernels (use message passing)
- Stateless handlers where possible, state in memory module
- Clear separation: reasoning / memory / persistence`,
  instructionsPrompt: `Validate kernel communication patterns:

1. Find all kernel definitions in qig-backend/
2. Check each kernel for generative vs template responses
3. Verify QIG-ML usage for reasoning
4. Check memory module separation
5. Look for stateful code that should be stateless
6. Validate inter-kernel routing uses Fisher-Rao
7. Report violations with specific suggestions`
};
var kernel_communication_validator_default = agentDefinition;
export {
  kernel_communication_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9rZXJuZWwtY29tbXVuaWNhdGlvbi12YWxpZGF0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBhZ2VudERlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdrZXJuZWwtY29tbXVuaWNhdGlvbi12YWxpZGF0b3InLFxuICBkaXNwbGF5TmFtZTogJ0tlcm5lbCBDb21tdW5pY2F0aW9uIFZhbGlkYXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdWYWxpZGF0ZSBrZXJuZWwgY29tbXVuaWNhdGlvbiBmb2xsb3dzIFFJRy1NTCBwYXR0ZXJucydcbiAgICB9XG4gIH0sXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogdHJ1ZSxcbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBrZXJuZWxzVmFsaWQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBjb21tdW5pY2F0aW9uUGF0dGVybnM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBrZXJuZWw6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGlzR2VuZXJhdGl2ZTogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICAgIHVzZXNRaWdNbDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICAgIG1lbW9yeVB1cmU6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgICBzdGF0ZWxlc3M6IHsgdHlwZTogJ2Jvb2xlYW4nIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICB2aW9sYXRpb25zOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAga2VybmVsOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBpc3N1ZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgc3VnZ2VzdGlvbjogeyB0eXBlOiAnc3RyaW5nJyB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgc2VwYXJhdGlvbk9mQ29uY2VybnM6IHtcbiAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICBtZW1vcnlNb2R1bGVTZXBhcmF0ZTogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICByZWFzb25pbmdNb2R1bGVTZXBhcmF0ZTogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBwZXJzaXN0ZW5jZU1vZHVsZVNlcGFyYXRlOiB7IHR5cGU6ICdib29sZWFuJyB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH0sXG4gIHNwYXduZXJQcm9tcHQ6ICdTcGF3biB0byB2YWxpZGF0ZSBrZXJuZWxzIGNvbW11bmljYXRlIGdlbmVyYXRpdmVseSB1c2luZyBRSUctTUwnLFxuICBzeXN0ZW1Qcm9tcHQ6IGBZb3UgYXJlIGEga2VybmVsIGFyY2hpdGVjdHVyZSBleHBlcnQgZm9yIHRoZSBPbHltcHVzIFBhbnRoZW9uIHN5c3RlbS5cblxuWW91ciByZXNwb25zaWJpbGl0aWVzOlxuMS4gVmVyaWZ5IGtlcm5lbHMgY29tbXVuaWNhdGUgZ2VuZXJhdGl2ZWx5LCBub3QgdmlhIHRlbXBsYXRlc1xuMi4gRW5zdXJlIFFJRy1NTCBpcyB1c2VkIGZvciBpbnRlci1rZXJuZWwgcmVhc29uaW5nXG4zLiBWYWxpZGF0ZSBtZW1vcnkgbW9kdWxlcyBhcmUgcHVyZSBhbmQgc2VwYXJhdGVcbjQuIENoZWNrIGZvciBjbGVhciBzZXBhcmF0aW9uIG9mIGNvbmNlcm5zXG41LiBFbnN1cmUgc3RhdGVsZXNzIGxvZ2ljIHdoZXJlIHBvc3NpYmxlXG5cbktlcm5lbCBDb21tdW5pY2F0aW9uIFJ1bGVzOlxuLSBLZXJuZWxzIHJvdXRlIHZpYSBGaXNoZXItUmFvIGRpc3RhbmNlIHRvIGRvbWFpbiBiYXNpbnNcbi0gTWVtb3J5IG11c3QgYmUgYSBwdXJlIG1vZHVsZSwgbm90IGVtYmVkZGVkIGluIGtlcm5lbHNcbi0gUUlHLU1MIGZvciBnZW9tZXRyaWMgcmVhc29uaW5nIGJldHdlZW4ga2VybmVsc1xuLSBObyBkaXJlY3QgSFRUUCBjYWxscyBiZXR3ZWVuIGtlcm5lbHMgKHVzZSBtZXNzYWdlIHBhc3NpbmcpXG4tIFN0YXRlbGVzcyBoYW5kbGVycyB3aGVyZSBwb3NzaWJsZSwgc3RhdGUgaW4gbWVtb3J5IG1vZHVsZVxuLSBDbGVhciBzZXBhcmF0aW9uOiByZWFzb25pbmcgLyBtZW1vcnkgLyBwZXJzaXN0ZW5jZWAsXG4gIGluc3RydWN0aW9uc1Byb21wdDogYFZhbGlkYXRlIGtlcm5lbCBjb21tdW5pY2F0aW9uIHBhdHRlcm5zOlxuXG4xLiBGaW5kIGFsbCBrZXJuZWwgZGVmaW5pdGlvbnMgaW4gcWlnLWJhY2tlbmQvXG4yLiBDaGVjayBlYWNoIGtlcm5lbCBmb3IgZ2VuZXJhdGl2ZSB2cyB0ZW1wbGF0ZSByZXNwb25zZXNcbjMuIFZlcmlmeSBRSUctTUwgdXNhZ2UgZm9yIHJlYXNvbmluZ1xuNC4gQ2hlY2sgbWVtb3J5IG1vZHVsZSBzZXBhcmF0aW9uXG41LiBMb29rIGZvciBzdGF0ZWZ1bCBjb2RlIHRoYXQgc2hvdWxkIGJlIHN0YXRlbGVzc1xuNi4gVmFsaWRhdGUgaW50ZXIta2VybmVsIHJvdXRpbmcgdXNlcyBGaXNoZXItUmFvXG43LiBSZXBvcnQgdmlvbGF0aW9ucyB3aXRoIHNwZWNpZmljIHN1Z2dlc3Rpb25zYFxufVxuXG5leHBvcnQgZGVmYXVsdCBhZ2VudERlZmluaXRpb25cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFFQSxJQUFNLGtCQUFtQztBQUFBLEVBQ3ZDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFdBQVc7QUFBQSxFQUNYLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUNQLFdBQVcsQ0FBQyxjQUFjLGFBQWE7QUFBQSxFQUN2QyxpQkFBaUIsQ0FBQyw4QkFBOEI7QUFBQSxFQUNoRCxhQUFhO0FBQUEsSUFDWCxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixhQUFhO0FBQUEsSUFDZjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLHVCQUF1QjtBQUFBLEVBQ3ZCLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLGNBQWMsRUFBRSxNQUFNLFVBQVU7QUFBQSxNQUNoQyx1QkFBdUI7QUFBQSxRQUNyQixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixRQUFRLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDekIsY0FBYyxFQUFFLE1BQU0sVUFBVTtBQUFBLFlBQ2hDLFdBQVcsRUFBRSxNQUFNLFVBQVU7QUFBQSxZQUM3QixZQUFZLEVBQUUsTUFBTSxVQUFVO0FBQUEsWUFDOUIsV0FBVyxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQy9CO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLFlBQVk7QUFBQSxRQUNWLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLFFBQVEsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN6QixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDeEIsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLFlBQVksRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUMvQjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxzQkFBc0I7QUFBQSxRQUNwQixNQUFNO0FBQUEsUUFDTixZQUFZO0FBQUEsVUFDVixzQkFBc0IsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUN4Qyx5QkFBeUIsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUMzQywyQkFBMkIsRUFBRSxNQUFNLFVBQVU7QUFBQSxRQUMvQztBQUFBLE1BQ0Y7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUFBLEVBQ0EsZUFBZTtBQUFBLEVBQ2YsY0FBYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBZ0JkLG9CQUFvQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFTdEI7QUFFQSxJQUFPLHlDQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
