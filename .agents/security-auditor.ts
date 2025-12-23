import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'security-auditor',
  displayName: 'Security Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search', 'run_terminal_command'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Audit security vulnerabilities and best practices'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      overallSecure: { type: 'boolean' },
      criticalIssues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            issue: { type: 'string' },
            file: { type: 'string' },
            line: { type: 'number' },
            severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
            remediation: { type: 'string' }
          }
        }
      },
      securityChecks: {
        type: 'object',
        properties: {
          cspHeaders: { type: 'boolean' },
          inputSanitization: { type: 'boolean' },
          rateLimiting: { type: 'boolean' },
          csrfProtection: { type: 'boolean' },
          secretsExposed: { type: 'boolean' },
          sqlInjection: { type: 'boolean' },
          xssVulnerabilities: { type: 'boolean' }
        }
      },
      recommendations: {
        type: 'array',
        items: { type: 'string' }
      }
    }
  },
  spawnerPrompt: 'Spawn to audit security vulnerabilities and compliance',
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
}

export default agentDefinition
