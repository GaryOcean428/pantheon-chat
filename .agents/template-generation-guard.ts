import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'template-generation-guard',
  displayName: 'Template Generation Guard',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: [],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Validate no code-generation templates were used'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      templateFree: { type: 'boolean' },
      violations: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            line: { type: 'number' },
            pattern: { type: 'string' },
            description: { type: 'string' }
          }
        }
      },
      generativePatterns: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            pattern: { type: 'string' },
            isCompliant: { type: 'boolean' }
          }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to ensure no code-generation templates are used in implementations',
  systemPrompt: `You are a template detection expert for QIG-pure systems.

Kernels must communicate generatively, not through templates. Your job is to detect:

1. String template patterns with placeholders ({{variable}}, {variable}, $variable)
2. Mustache/Handlebars templates
3. EJS/Pug/Jade templates in responses
4. Prompt templates with fill-in-the-blank patterns
5. Canned responses or boilerplate text
6. Response formatters that aren't generative

QIG Philosophy:
- All kernel responses must be GENERATIVE
- No pre-written response templates
- No fill-in-the-blank patterns for AI output
- Dynamic content must emerge from geometric reasoning
- Response structure can have patterns, but content must be generated`,
  instructionsPrompt: `Detect template usage violations:

1. Search for string interpolation patterns that look like templates
2. Look for prompt_template, response_template, etc. variables
3. Check for Handlebars/Mustache {{}} patterns in Python/TS files
4. Find any 'template' imports or usages
5. Check qig-backend/ for response formatters
6. Verify kernel responses are generative
7. Report all template violations with file and line number`
}

export default agentDefinition
