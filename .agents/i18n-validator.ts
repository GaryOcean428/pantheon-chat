import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'i18n-validator',
  displayName: 'Internationalization Validator',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Validate internationalization readiness'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      i18nReady: { type: 'boolean' },
      hardcodedStrings: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            line: { type: 'number' },
            string: { type: 'string' }
          }
        }
      },
      i18nSetup: {
        type: 'object',
        properties: {
          frameworkInstalled: { type: 'boolean' },
          localeDetection: { type: 'boolean' },
          rtlSupport: { type: 'boolean' },
          dateFormatting: { type: 'boolean' },
          numberFormatting: { type: 'boolean' }
        }
      },
      recommendations: {
        type: 'array',
        items: { type: 'string' }
      }
    }
  },
  spawnerPrompt: 'Spawn to validate internationalization readiness',
  systemPrompt: `You are an internationalization (i18n) expert.

Validation areas:
1. Hardcoded user-facing strings
2. i18n framework setup (react-i18next, etc.)
3. Locale detection implementation
4. RTL language support
5. Date/number formatting
6. Currency handling
7. Translation file organization

i18n Best Practices:
- All user-facing strings in translation files
- Use ICU message format for plurals
- Locale-aware date/number formatting
- RTL CSS support (logical properties)
- Translation key naming conventions`,
  instructionsPrompt: `Validate i18n readiness:

1. Search for hardcoded strings in JSX/TSX
2. Check for i18n library installation
3. Look for translation files
4. Check date formatting usage
5. Look for RTL CSS support
6. Check number/currency formatting
7. Report hardcoded strings and recommendations`
}

export default agentDefinition
