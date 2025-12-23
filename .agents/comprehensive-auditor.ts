import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'comprehensive-auditor',
  displayName: 'Comprehensive Codebase Auditor',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'spawn_agents'],
  spawnableAgents: [
    'pantheon/qig-purity-enforcer@0.0.1',
    'pantheon/database-qig-validator@0.0.1',
    'pantheon/dependency-validator@0.0.1',
    'pantheon/barrel-export-enforcer@0.0.1',
    'pantheon/api-purity-enforcer@0.0.1',
    'pantheon/module-bridging-validator@0.0.1',
    'pantheon/template-generation-guard@0.0.1',
    'pantheon/kernel-communication-validator@0.0.1',
    'pantheon/redis-migration-validator@0.0.1',
    'pantheon/iso-doc-validator@0.0.1',
    'pantheon/codebase-cleanup-auditor@0.0.1',
    'pantheon/ui-ux-auditor@0.0.1',
    'pantheon/security-auditor@0.0.1',
    'pantheon/performance-auditor@0.0.1',
    'pantheon/accessibility-auditor@0.0.1',
    'pantheon/testing-coverage-auditor@0.0.1'
  ],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Run comprehensive codebase audit'
    },
    params: {
      type: 'object',
      properties: {
        categories: {
          type: 'array',
          items: { type: 'string' },
          description: 'Categories to audit: qig, architecture, ui, security, performance, testing, all'
        }
      }
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      overallHealth: { type: 'string', enum: ['excellent', 'good', 'needs-work', 'critical'] },
      summary: {
        type: 'object',
        properties: {
          totalIssues: { type: 'number' },
          criticalIssues: { type: 'number' },
          warnings: { type: 'number' },
          passed: { type: 'number' }
        }
      },
      categoryResults: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            category: { type: 'string' },
            status: { type: 'string', enum: ['pass', 'warn', 'fail'] },
            issueCount: { type: 'number' },
            topIssues: { type: 'array', items: { type: 'string' } }
          }
        }
      },
      prioritizedActions: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            priority: { type: 'number' },
            action: { type: 'string' },
            category: { type: 'string' },
            effort: { type: 'string', enum: ['small', 'medium', 'large'] }
          }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to run a comprehensive audit of the entire codebase',
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
}

export default agentDefinition
