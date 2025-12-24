// .agents/security-auditor.ts
var agentDefinition = {
  id: "security-auditor",
  displayName: "Security Auditor",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search", "run_terminal_command"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Audit security vulnerabilities and best practices"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      overallSecure: { type: "boolean" },
      criticalIssues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            issue: { type: "string" },
            file: { type: "string" },
            line: { type: "number" },
            severity: { type: "string", enum: ["critical", "high", "medium", "low"] },
            remediation: { type: "string" }
          }
        }
      },
      securityChecks: {
        type: "object",
        properties: {
          cspHeaders: { type: "boolean" },
          inputSanitization: { type: "boolean" },
          rateLimiting: { type: "boolean" },
          csrfProtection: { type: "boolean" },
          secretsExposed: { type: "boolean" },
          sqlInjection: { type: "boolean" },
          xssVulnerabilities: { type: "boolean" }
        }
      },
      recommendations: {
        type: "array",
        items: { type: "string" }
      }
    }
  },
  spawnerPrompt: "Spawn to audit security vulnerabilities and compliance",
  systemPrompt: `You are a security auditor expert.

Audit areas:
1. Content Security Policy headers
2. Input sanitization for user content
3. Rate limiting on API endpoints
4. CSRF protection tokens
5. Exposed secrets in code or environment
6. SQL injection vulnerabilities
7. XSS vulnerabilities
8. Authentication/authorization flaws
9. Dependency vulnerabilities

Critical Checks:
- No hardcoded API keys or secrets
- No eval() or dangerous dynamic code
- Parameterized queries for all DB access
- HTML sanitization for user content
- Proper CORS configuration
- Secure cookie settings (httpOnly, secure, sameSite)`,
  instructionsPrompt: `Perform security audit:

1. Search for hardcoded secrets (API keys, passwords)
2. Check for eval(), new Function(), innerHTML usage
3. Verify SQL queries are parameterized
4. Check rate limiting middleware
5. Verify CSP headers in server config
6. Check authentication middleware
7. Run npm audit for dependency vulnerabilities
8. Check for exposed .env files in public
9. Report all issues with severity and remediation`
};
var security_auditor_default = agentDefinition;
export {
  security_auditor_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9zZWN1cml0eS1hdWRpdG9yLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgYWdlbnREZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAnc2VjdXJpdHktYXVkaXRvcicsXG4gIGRpc3BsYXlOYW1lOiAnU2VjdXJpdHkgQXVkaXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnLCAncnVuX3Rlcm1pbmFsX2NvbW1hbmQnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdBdWRpdCBzZWN1cml0eSB2dWxuZXJhYmlsaXRpZXMgYW5kIGJlc3QgcHJhY3RpY2VzJ1xuICAgIH1cbiAgfSxcbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiB0cnVlLFxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIG92ZXJhbGxTZWN1cmU6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBjcml0aWNhbElzc3Vlczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGlzc3VlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBmaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBsaW5lOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgICBzZXZlcml0eTogeyB0eXBlOiAnc3RyaW5nJywgZW51bTogWydjcml0aWNhbCcsICdoaWdoJywgJ21lZGl1bScsICdsb3cnXSB9LFxuICAgICAgICAgICAgcmVtZWRpYXRpb246IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIHNlY3VyaXR5Q2hlY2tzOiB7XG4gICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgY3NwSGVhZGVyczogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBpbnB1dFNhbml0aXphdGlvbjogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICByYXRlTGltaXRpbmc6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgY3NyZlByb3RlY3Rpb246IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgc2VjcmV0c0V4cG9zZWQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgc3FsSW5qZWN0aW9uOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIHhzc1Z1bG5lcmFiaWxpdGllczogeyB0eXBlOiAnYm9vbGVhbicgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgcmVjb21tZW5kYXRpb25zOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7IHR5cGU6ICdzdHJpbmcnIH1cbiAgICAgIH1cbiAgICB9XG4gIH0sXG4gIHNwYXduZXJQcm9tcHQ6ICdTcGF3biB0byBhdWRpdCBzZWN1cml0eSB2dWxuZXJhYmlsaXRpZXMgYW5kIGNvbXBsaWFuY2UnLFxuICBzeXN0ZW1Qcm9tcHQ6IGBZb3UgYXJlIGEgc2VjdXJpdHkgYXVkaXRvciBleHBlcnQuXG5cbkF1ZGl0IGFyZWFzOlxuMS4gQ29udGVudCBTZWN1cml0eSBQb2xpY3kgaGVhZGVyc1xuMi4gSW5wdXQgc2FuaXRpemF0aW9uIGZvciB1c2VyIGNvbnRlbnRcbjMuIFJhdGUgbGltaXRpbmcgb24gQVBJIGVuZHBvaW50c1xuNC4gQ1NSRiBwcm90ZWN0aW9uIHRva2Vuc1xuNS4gRXhwb3NlZCBzZWNyZXRzIGluIGNvZGUgb3IgZW52aXJvbm1lbnRcbjYuIFNRTCBpbmplY3Rpb24gdnVsbmVyYWJpbGl0aWVzXG43LiBYU1MgdnVsbmVyYWJpbGl0aWVzXG44LiBBdXRoZW50aWNhdGlvbi9hdXRob3JpemF0aW9uIGZsYXdzXG45LiBEZXBlbmRlbmN5IHZ1bG5lcmFiaWxpdGllc1xuXG5Dcml0aWNhbCBDaGVja3M6XG4tIE5vIGhhcmRjb2RlZCBBUEkga2V5cyBvciBzZWNyZXRzXG4tIE5vIGV2YWwoKSBvciBkYW5nZXJvdXMgZHluYW1pYyBjb2RlXG4tIFBhcmFtZXRlcml6ZWQgcXVlcmllcyBmb3IgYWxsIERCIGFjY2Vzc1xuLSBIVE1MIHNhbml0aXphdGlvbiBmb3IgdXNlciBjb250ZW50XG4tIFByb3BlciBDT1JTIGNvbmZpZ3VyYXRpb25cbi0gU2VjdXJlIGNvb2tpZSBzZXR0aW5ncyAoaHR0cE9ubHksIHNlY3VyZSwgc2FtZVNpdGUpYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgUGVyZm9ybSBzZWN1cml0eSBhdWRpdDpcblxuMS4gU2VhcmNoIGZvciBoYXJkY29kZWQgc2VjcmV0cyAoQVBJIGtleXMsIHBhc3N3b3JkcylcbjIuIENoZWNrIGZvciBldmFsKCksIG5ldyBGdW5jdGlvbigpLCBpbm5lckhUTUwgdXNhZ2VcbjMuIFZlcmlmeSBTUUwgcXVlcmllcyBhcmUgcGFyYW1ldGVyaXplZFxuNC4gQ2hlY2sgcmF0ZSBsaW1pdGluZyBtaWRkbGV3YXJlXG41LiBWZXJpZnkgQ1NQIGhlYWRlcnMgaW4gc2VydmVyIGNvbmZpZ1xuNi4gQ2hlY2sgYXV0aGVudGljYXRpb24gbWlkZGxld2FyZVxuNy4gUnVuIG5wbSBhdWRpdCBmb3IgZGVwZW5kZW5jeSB2dWxuZXJhYmlsaXRpZXNcbjguIENoZWNrIGZvciBleHBvc2VkIC5lbnYgZmlsZXMgaW4gcHVibGljXG45LiBSZXBvcnQgYWxsIGlzc3VlcyB3aXRoIHNldmVyaXR5IGFuZCByZW1lZGlhdGlvbmBcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyxlQUFlLHNCQUFzQjtBQUFBLEVBQy9ELGlCQUFpQixDQUFDLDhCQUE4QjtBQUFBLEVBQ2hELGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsRUFDRjtBQUFBLEVBQ0EsdUJBQXVCO0FBQUEsRUFDdkIsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YsZUFBZSxFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQ2pDLGdCQUFnQjtBQUFBLFFBQ2QsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsT0FBTyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3hCLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsVUFBVSxFQUFFLE1BQU0sVUFBVSxNQUFNLENBQUMsWUFBWSxRQUFRLFVBQVUsS0FBSyxFQUFFO0FBQUEsWUFDeEUsYUFBYSxFQUFFLE1BQU0sU0FBUztBQUFBLFVBQ2hDO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGdCQUFnQjtBQUFBLFFBQ2QsTUFBTTtBQUFBLFFBQ04sWUFBWTtBQUFBLFVBQ1YsWUFBWSxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQzlCLG1CQUFtQixFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQ3JDLGNBQWMsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUNoQyxnQkFBZ0IsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUNsQyxnQkFBZ0IsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUNsQyxjQUFjLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDaEMsb0JBQW9CLEVBQUUsTUFBTSxVQUFVO0FBQUEsUUFDeEM7QUFBQSxNQUNGO0FBQUEsTUFDQSxpQkFBaUI7QUFBQSxRQUNmLE1BQU07QUFBQSxRQUNOLE9BQU8sRUFBRSxNQUFNLFNBQVM7QUFBQSxNQUMxQjtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQUEsRUFDQSxlQUFlO0FBQUEsRUFDZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQW9CZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQVd0QjtBQUVBLElBQU8sMkJBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
