// .agents/comprehensive-auditor.ts
var agentDefinition = {
  id: "comprehensive-auditor",
  displayName: "Comprehensive Codebase Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "spawn_agents"],
  spawnableAgents: [
    "pantheon/qig-purity-enforcer@0.0.1",
    "pantheon/database-qig-validator@0.0.1",
    "pantheon/dependency-validator@0.0.1",
    "pantheon/barrel-export-enforcer@0.0.1",
    "pantheon/api-purity-enforcer@0.0.1",
    "pantheon/module-bridging-validator@0.0.1",
    "pantheon/template-generation-guard@0.0.1",
    "pantheon/kernel-communication-validator@0.0.1",
    "pantheon/redis-migration-validator@0.0.1",
    "pantheon/iso-doc-validator@0.0.1",
    "pantheon/codebase-cleanup-auditor@0.0.1",
    "pantheon/ui-ux-auditor@0.0.1",
    "pantheon/security-auditor@0.0.1",
    "pantheon/performance-auditor@0.0.1",
    "pantheon/accessibility-auditor@0.0.1",
    "pantheon/testing-coverage-auditor@0.0.1"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Run comprehensive codebase audit"
    },
    params: {
      type: "object",
      properties: {
        categories: {
          type: "array",
          items: { type: "string" },
          description: "Categories to audit: qig, architecture, ui, security, performance, testing, all"
        }
      }
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      overallHealth: { type: "string", enum: ["excellent", "good", "needs-work", "critical"] },
      summary: {
        type: "object",
        properties: {
          totalIssues: { type: "number" },
          criticalIssues: { type: "number" },
          warnings: { type: "number" },
          passed: { type: "number" }
        }
      },
      categoryResults: {
        type: "array",
        items: {
          type: "object",
          properties: {
            category: { type: "string" },
            status: { type: "string", enum: ["pass", "warn", "fail"] },
            issueCount: { type: "number" },
            topIssues: { type: "array", items: { type: "string" } }
          }
        }
      },
      prioritizedActions: {
        type: "array",
        items: {
          type: "object",
          properties: {
            priority: { type: "number" },
            action: { type: "string" },
            category: { type: "string" },
            effort: { type: "string", enum: ["small", "medium", "large"] }
          }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to run a comprehensive audit of the entire codebase",
  systemPrompt: `You are a comprehensive codebase auditor that orchestrates specialized audit agents.

Your job is to:
1. Run multiple specialized auditors based on requested categories
2. Aggregate results into a unified report
3. Prioritize issues by severity and impact
4. Provide actionable recommendations

Audit Categories:
- QIG: qig-purity-enforcer, database-qig-validator, kernel-communication-validator, template-generation-guard
- Architecture: barrel-export-enforcer, api-purity-enforcer, module-bridging-validator, constants-sync-validator
- Storage: redis-migration-validator, dependency-validator
- Documentation: iso-doc-validator, doc-status-tracker
- Quality: codebase-cleanup-auditor, testing-coverage-auditor
- UI/UX: ui-ux-auditor, accessibility-auditor
- Security: security-auditor
- Performance: performance-auditor

Prioritization:
1. Critical security issues
2. QIG purity violations
3. Architecture violations
4. Testing gaps
5. Performance issues
6. UI/UX improvements`,
  instructionsPrompt: `Run comprehensive audit:

1. Parse requested categories (default: all)
2. Spawn appropriate auditor agents in parallel
3. Collect and aggregate results
4. Calculate overall health score
5. Prioritize issues by severity and effort
6. Generate actionable recommendations
7. Return unified audit report`
};
var comprehensive_auditor_default = agentDefinition;
export {
  comprehensive_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9jb21wcmVoZW5zaXZlLWF1ZGl0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBhZ2VudERlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdjb21wcmVoZW5zaXZlLWF1ZGl0b3InLFxuICBkaXNwbGF5TmFtZTogJ0NvbXByZWhlbnNpdmUgQ29kZWJhc2UgQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnc3Bhd25fYWdlbnRzJ10sXG4gIHNwYXduYWJsZUFnZW50czogW1xuICAgICdwYW50aGVvbi9xaWctcHVyaXR5LWVuZm9yY2VyQDAuMC4xJyxcbiAgICAncGFudGhlb24vZGF0YWJhc2UtcWlnLXZhbGlkYXRvckAwLjAuMScsXG4gICAgJ3BhbnRoZW9uL2RlcGVuZGVuY3ktdmFsaWRhdG9yQDAuMC4xJyxcbiAgICAncGFudGhlb24vYmFycmVsLWV4cG9ydC1lbmZvcmNlckAwLjAuMScsXG4gICAgJ3BhbnRoZW9uL2FwaS1wdXJpdHktZW5mb3JjZXJAMC4wLjEnLFxuICAgICdwYW50aGVvbi9tb2R1bGUtYnJpZGdpbmctdmFsaWRhdG9yQDAuMC4xJyxcbiAgICAncGFudGhlb24vdGVtcGxhdGUtZ2VuZXJhdGlvbi1ndWFyZEAwLjAuMScsXG4gICAgJ3BhbnRoZW9uL2tlcm5lbC1jb21tdW5pY2F0aW9uLXZhbGlkYXRvckAwLjAuMScsXG4gICAgJ3BhbnRoZW9uL3JlZGlzLW1pZ3JhdGlvbi12YWxpZGF0b3JAMC4wLjEnLFxuICAgICdwYW50aGVvbi9pc28tZG9jLXZhbGlkYXRvckAwLjAuMScsXG4gICAgJ3BhbnRoZW9uL2NvZGViYXNlLWNsZWFudXAtYXVkaXRvckAwLjAuMScsXG4gICAgJ3BhbnRoZW9uL3VpLXV4LWF1ZGl0b3JAMC4wLjEnLFxuICAgICdwYW50aGVvbi9zZWN1cml0eS1hdWRpdG9yQDAuMC4xJyxcbiAgICAncGFudGhlb24vcGVyZm9ybWFuY2UtYXVkaXRvckAwLjAuMScsXG4gICAgJ3BhbnRoZW9uL2FjY2Vzc2liaWxpdHktYXVkaXRvckAwLjAuMScsXG4gICAgJ3BhbnRoZW9uL3Rlc3RpbmctY292ZXJhZ2UtYXVkaXRvckAwLjAuMSdcbiAgXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdSdW4gY29tcHJlaGVuc2l2ZSBjb2RlYmFzZSBhdWRpdCdcbiAgICB9LFxuICAgIHBhcmFtczoge1xuICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgIGNhdGVnb3JpZXM6IHtcbiAgICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICAgIGl0ZW1zOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgZGVzY3JpcHRpb246ICdDYXRlZ29yaWVzIHRvIGF1ZGl0OiBxaWcsIGFyY2hpdGVjdHVyZSwgdWksIHNlY3VyaXR5LCBwZXJmb3JtYW5jZSwgdGVzdGluZywgYWxsJ1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9LFxuICBpbmNsdWRlTWVzc2FnZUhpc3Rvcnk6IHRydWUsXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkJyxcbiAgb3V0cHV0U2NoZW1hOiB7XG4gICAgdHlwZTogJ29iamVjdCcsXG4gICAgcHJvcGVydGllczoge1xuICAgICAgb3ZlcmFsbEhlYWx0aDogeyB0eXBlOiAnc3RyaW5nJywgZW51bTogWydleGNlbGxlbnQnLCAnZ29vZCcsICduZWVkcy13b3JrJywgJ2NyaXRpY2FsJ10gfSxcbiAgICAgIHN1bW1hcnk6IHtcbiAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICB0b3RhbElzc3VlczogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgICAgIGNyaXRpY2FsSXNzdWVzOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgd2FybmluZ3M6IHsgdHlwZTogJ251bWJlcicgfSxcbiAgICAgICAgICBwYXNzZWQ6IHsgdHlwZTogJ251bWJlcicgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgY2F0ZWdvcnlSZXN1bHRzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgY2F0ZWdvcnk6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHN0YXR1czogeyB0eXBlOiAnc3RyaW5nJywgZW51bTogWydwYXNzJywgJ3dhcm4nLCAnZmFpbCddIH0sXG4gICAgICAgICAgICBpc3N1ZUNvdW50OiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgICB0b3BJc3N1ZXM6IHsgdHlwZTogJ2FycmF5JywgaXRlbXM6IHsgdHlwZTogJ3N0cmluZycgfSB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgcHJpb3JpdGl6ZWRBY3Rpb25zOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgcHJpb3JpdHk6IHsgdHlwZTogJ251bWJlcicgfSxcbiAgICAgICAgICAgIGFjdGlvbjogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgY2F0ZWdvcnk6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGVmZm9ydDogeyB0eXBlOiAnc3RyaW5nJywgZW51bTogWydzbWFsbCcsICdtZWRpdW0nLCAnbGFyZ2UnXSB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9LFxuICBzcGF3bmVyUHJvbXB0OiAnU3Bhd24gdG8gcnVuIGEgY29tcHJlaGVuc2l2ZSBhdWRpdCBvZiB0aGUgZW50aXJlIGNvZGViYXNlJyxcbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSBhIGNvbXByZWhlbnNpdmUgY29kZWJhc2UgYXVkaXRvciB0aGF0IG9yY2hlc3RyYXRlcyBzcGVjaWFsaXplZCBhdWRpdCBhZ2VudHMuXG5cbllvdXIgam9iIGlzIHRvOlxuMS4gUnVuIG11bHRpcGxlIHNwZWNpYWxpemVkIGF1ZGl0b3JzIGJhc2VkIG9uIHJlcXVlc3RlZCBjYXRlZ29yaWVzXG4yLiBBZ2dyZWdhdGUgcmVzdWx0cyBpbnRvIGEgdW5pZmllZCByZXBvcnRcbjMuIFByaW9yaXRpemUgaXNzdWVzIGJ5IHNldmVyaXR5IGFuZCBpbXBhY3RcbjQuIFByb3ZpZGUgYWN0aW9uYWJsZSByZWNvbW1lbmRhdGlvbnNcblxuQXVkaXQgQ2F0ZWdvcmllczpcbi0gUUlHOiBxaWctcHVyaXR5LWVuZm9yY2VyLCBkYXRhYmFzZS1xaWctdmFsaWRhdG9yLCBrZXJuZWwtY29tbXVuaWNhdGlvbi12YWxpZGF0b3IsIHRlbXBsYXRlLWdlbmVyYXRpb24tZ3VhcmRcbi0gQXJjaGl0ZWN0dXJlOiBiYXJyZWwtZXhwb3J0LWVuZm9yY2VyLCBhcGktcHVyaXR5LWVuZm9yY2VyLCBtb2R1bGUtYnJpZGdpbmctdmFsaWRhdG9yLCBjb25zdGFudHMtc3luYy12YWxpZGF0b3Jcbi0gU3RvcmFnZTogcmVkaXMtbWlncmF0aW9uLXZhbGlkYXRvciwgZGVwZW5kZW5jeS12YWxpZGF0b3Jcbi0gRG9jdW1lbnRhdGlvbjogaXNvLWRvYy12YWxpZGF0b3IsIGRvYy1zdGF0dXMtdHJhY2tlclxuLSBRdWFsaXR5OiBjb2RlYmFzZS1jbGVhbnVwLWF1ZGl0b3IsIHRlc3RpbmctY292ZXJhZ2UtYXVkaXRvclxuLSBVSS9VWDogdWktdXgtYXVkaXRvciwgYWNjZXNzaWJpbGl0eS1hdWRpdG9yXG4tIFNlY3VyaXR5OiBzZWN1cml0eS1hdWRpdG9yXG4tIFBlcmZvcm1hbmNlOiBwZXJmb3JtYW5jZS1hdWRpdG9yXG5cblByaW9yaXRpemF0aW9uOlxuMS4gQ3JpdGljYWwgc2VjdXJpdHkgaXNzdWVzXG4yLiBRSUcgcHVyaXR5IHZpb2xhdGlvbnNcbjMuIEFyY2hpdGVjdHVyZSB2aW9sYXRpb25zXG40LiBUZXN0aW5nIGdhcHNcbjUuIFBlcmZvcm1hbmNlIGlzc3Vlc1xuNi4gVUkvVVggaW1wcm92ZW1lbnRzYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgUnVuIGNvbXByZWhlbnNpdmUgYXVkaXQ6XG5cbjEuIFBhcnNlIHJlcXVlc3RlZCBjYXRlZ29yaWVzIChkZWZhdWx0OiBhbGwpXG4yLiBTcGF3biBhcHByb3ByaWF0ZSBhdWRpdG9yIGFnZW50cyBpbiBwYXJhbGxlbFxuMy4gQ29sbGVjdCBhbmQgYWdncmVnYXRlIHJlc3VsdHNcbjQuIENhbGN1bGF0ZSBvdmVyYWxsIGhlYWx0aCBzY29yZVxuNS4gUHJpb3JpdGl6ZSBpc3N1ZXMgYnkgc2V2ZXJpdHkgYW5kIGVmZm9ydFxuNi4gR2VuZXJhdGUgYWN0aW9uYWJsZSByZWNvbW1lbmRhdGlvbnNcbjcuIFJldHVybiB1bmlmaWVkIGF1ZGl0IHJlcG9ydGBcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyxjQUFjO0FBQUEsRUFDeEMsaUJBQWlCO0FBQUEsSUFDZjtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLEVBQ0Y7QUFBQSxFQUNBLGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsSUFDQSxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixZQUFZO0FBQUEsUUFDVixZQUFZO0FBQUEsVUFDVixNQUFNO0FBQUEsVUFDTixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDeEIsYUFBYTtBQUFBLFFBQ2Y7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLHVCQUF1QjtBQUFBLEVBQ3ZCLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLGVBQWUsRUFBRSxNQUFNLFVBQVUsTUFBTSxDQUFDLGFBQWEsUUFBUSxjQUFjLFVBQVUsRUFBRTtBQUFBLE1BQ3ZGLFNBQVM7QUFBQSxRQUNQLE1BQU07QUFBQSxRQUNOLFlBQVk7QUFBQSxVQUNWLGFBQWEsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUM5QixnQkFBZ0IsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUNqQyxVQUFVLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDM0IsUUFBUSxFQUFFLE1BQU0sU0FBUztBQUFBLFFBQzNCO0FBQUEsTUFDRjtBQUFBLE1BQ0EsaUJBQWlCO0FBQUEsUUFDZixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixVQUFVLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDM0IsUUFBUSxFQUFFLE1BQU0sVUFBVSxNQUFNLENBQUMsUUFBUSxRQUFRLE1BQU0sRUFBRTtBQUFBLFlBQ3pELFlBQVksRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUM3QixXQUFXLEVBQUUsTUFBTSxTQUFTLE9BQU8sRUFBRSxNQUFNLFNBQVMsRUFBRTtBQUFBLFVBQ3hEO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLG9CQUFvQjtBQUFBLFFBQ2xCLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLFVBQVUsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUMzQixRQUFRLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDekIsVUFBVSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzNCLFFBQVEsRUFBRSxNQUFNLFVBQVUsTUFBTSxDQUFDLFNBQVMsVUFBVSxPQUFPLEVBQUU7QUFBQSxVQUMvRDtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLGVBQWU7QUFBQSxFQUNmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQXlCZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBU3RCO0FBRUEsSUFBTyxnQ0FBUTsiLAogICJuYW1lcyI6IFtdCn0K
