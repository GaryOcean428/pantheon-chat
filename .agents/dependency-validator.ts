import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'dependency-validator',
  displayName: 'Dependency Validator',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'run_terminal_command', 'code_search'],
  spawnableAgents: [],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Validate dependencies are installed and up-to-date'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      nodePackagesValid: { type: 'boolean' },
      pythonPackagesValid: { type: 'boolean' },
      packageManagerCorrect: { type: 'boolean' },
      outdatedPackages: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            current: { type: 'string' },
            latest: { type: 'string' },
            ecosystem: { type: 'string', enum: ['node', 'python'] }
          }
        }
      },
      securityVulnerabilities: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            package: { type: 'string' },
            severity: { type: 'string' },
            advisory: { type: 'string' }
          }
        }
      },
      missingDependencies: {
        type: 'array',
        items: { type: 'string' }
      }
    }
  },
  spawnerPrompt: 'Spawn to validate all dependencies are installed and managed correctly',
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
}

export default agentDefinition
