import type { AgentDefinition } from './types/agent-definition'

const agentDefinition: AgentDefinition = {
  id: 'seo-validator',
  displayName: 'SEO Validator',
  publisher: 'pantheon',
  version: '0.0.1',
  model: 'anthropic/claude-sonnet-4',
  toolNames: ['read_files', 'code_search'],
  spawnableAgents: ['codebuff/file-explorer@0.0.4'],
  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Validate SEO and meta tag implementation'
    }
  },
  includeMessageHistory: true,
  outputMode: 'structured',
  outputSchema: {
    type: 'object',
    properties: {
      seoReady: { type: 'boolean' },
      metaTags: {
        type: 'object',
        properties: {
          title: { type: 'boolean' },
          description: { type: 'boolean' },
          ogTags: { type: 'boolean' },
          twitterCards: { type: 'boolean' },
          canonical: { type: 'boolean' }
        }
      },
      technicalSeo: {
        type: 'object',
        properties: {
          sitemap: { type: 'boolean' },
          robotsTxt: { type: 'boolean' },
          structuredData: { type: 'boolean' },
          semanticHtml: { type: 'boolean' }
        }
      },
      issues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            page: { type: 'string' },
            issue: { type: 'string' },
            impact: { type: 'string' }
          }
        }
      }
    }
  },
  spawnerPrompt: 'Spawn to validate SEO implementation and meta tags',
  systemPrompt: `You are an SEO expert.

Validation areas:
1. Meta tags (title, description)
2. Open Graph tags for social sharing
3. Twitter Card meta tags
4. Canonical URLs
5. Sitemap.xml generation
6. robots.txt configuration
7. Structured data (Schema.org)
8. Semantic HTML usage
9. Heading hierarchy

SEO Best Practices:
- Unique title and description per page
- OG image for all shareable pages
- Proper heading hierarchy (h1 > h2 > h3)
- Semantic HTML elements (nav, main, article)
- Internal linking structure`,
  instructionsPrompt: `Validate SEO implementation:

1. Check index.html for meta tags
2. Look for react-helmet or similar
3. Check for sitemap.xml
4. Check for robots.txt
5. Search for structured data (JSON-LD)
6. Verify heading hierarchy in pages
7. Check semantic HTML usage
8. Report SEO issues and impact`
}

export default agentDefinition
