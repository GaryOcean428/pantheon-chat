import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'devops-auditor',
  displayName: 'DevOps Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search', 'run_terminal_command'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Audit DevOps and deployment configuration'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      deploymentReady: { type: 'boolean' },
      cicdStatus: {
        type: 'object',
        properties: {
          pipelineExists: { type: 'boolean' },
          testsInPipeline: { type: 'boolean' },
          previewDeployments: { type: 'boolean' },
          automatedReleases: { type: 'boolean' }
        }
      },
      infrastructure: {
        type: 'object',
        properties: {
          dockerized: { type: 'boolean' },
          envParity: { type: 'boolean' },
          secretsManaged: { type: 'boolean' },
          backupsConfigured: { type: 'boolean' }
        }
      },
      issues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            area: { type: 'string' },
            issue: { type: 'string' },
            severity: { type: 'string' },
            recommendation: { type: 'string' }
          }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to audit DevOps configuration and deployment readiness',
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
}

export default agentDefinition
