// .agents/seo-validator.ts
var agentDefinition = {
  id: "seo-validator",
  displayName: "SEO Validator",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Validate SEO and meta tag implementation"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      seoReady: { type: "boolean" },
      metaTags: {
        type: "object",
        properties: {
          title: { type: "boolean" },
          description: { type: "boolean" },
          ogTags: { type: "boolean" },
          twitterCards: { type: "boolean" },
          canonical: { type: "boolean" }
        }
      },
      technicalSeo: {
        type: "object",
        properties: {
          sitemap: { type: "boolean" },
          robotsTxt: { type: "boolean" },
          structuredData: { type: "boolean" },
          semanticHtml: { type: "boolean" }
        }
      },
      issues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            page: { type: "string" },
            issue: { type: "string" },
            impact: { type: "string" }
          }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to validate SEO implementation and meta tags",
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
};
var seo_validator_default = agentDefinition;
export {
  seo_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9zZW8tdmFsaWRhdG9yLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgYWdlbnREZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAnc2VvLXZhbGlkYXRvcicsXG4gIGRpc3BsYXlOYW1lOiAnU0VPIFZhbGlkYXRvcicsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbJ2NvZGVidWZmL2ZpbGUtZXhwbG9yZXJAMC4wLjQnXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdWYWxpZGF0ZSBTRU8gYW5kIG1ldGEgdGFnIGltcGxlbWVudGF0aW9uJ1xuICAgIH1cbiAgfSxcbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiB0cnVlLFxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHNlb1JlYWR5OiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgbWV0YVRhZ3M6IHtcbiAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICB0aXRsZTogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBkZXNjcmlwdGlvbjogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBvZ1RhZ3M6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgdHdpdHRlckNhcmRzOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgIGNhbm9uaWNhbDogeyB0eXBlOiAnYm9vbGVhbicgfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgdGVjaG5pY2FsU2VvOiB7XG4gICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgc2l0ZW1hcDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICByb2JvdHNUeHQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgc3RydWN0dXJlZERhdGE6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgc2VtYW50aWNIdG1sOiB7IHR5cGU6ICdib29sZWFuJyB9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICBpc3N1ZXM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBwYWdlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBpc3N1ZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgaW1wYWN0OiB7IHR5cGU6ICdzdHJpbmcnIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH0sXG4gIHNwYXduZXJQcm9tcHQ6ICdTcGF3biB0byB2YWxpZGF0ZSBTRU8gaW1wbGVtZW50YXRpb24gYW5kIG1ldGEgdGFncycsXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgYW4gU0VPIGV4cGVydC5cblxuVmFsaWRhdGlvbiBhcmVhczpcbjEuIE1ldGEgdGFncyAodGl0bGUsIGRlc2NyaXB0aW9uKVxuMi4gT3BlbiBHcmFwaCB0YWdzIGZvciBzb2NpYWwgc2hhcmluZ1xuMy4gVHdpdHRlciBDYXJkIG1ldGEgdGFnc1xuNC4gQ2Fub25pY2FsIFVSTHNcbjUuIFNpdGVtYXAueG1sIGdlbmVyYXRpb25cbjYuIHJvYm90cy50eHQgY29uZmlndXJhdGlvblxuNy4gU3RydWN0dXJlZCBkYXRhIChTY2hlbWEub3JnKVxuOC4gU2VtYW50aWMgSFRNTCB1c2FnZVxuOS4gSGVhZGluZyBoaWVyYXJjaHlcblxuU0VPIEJlc3QgUHJhY3RpY2VzOlxuLSBVbmlxdWUgdGl0bGUgYW5kIGRlc2NyaXB0aW9uIHBlciBwYWdlXG4tIE9HIGltYWdlIGZvciBhbGwgc2hhcmVhYmxlIHBhZ2VzXG4tIFByb3BlciBoZWFkaW5nIGhpZXJhcmNoeSAoaDEgPiBoMiA+IGgzKVxuLSBTZW1hbnRpYyBIVE1MIGVsZW1lbnRzIChuYXYsIG1haW4sIGFydGljbGUpXG4tIEludGVybmFsIGxpbmtpbmcgc3RydWN0dXJlYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgVmFsaWRhdGUgU0VPIGltcGxlbWVudGF0aW9uOlxuXG4xLiBDaGVjayBpbmRleC5odG1sIGZvciBtZXRhIHRhZ3NcbjIuIExvb2sgZm9yIHJlYWN0LWhlbG1ldCBvciBzaW1pbGFyXG4zLiBDaGVjayBmb3Igc2l0ZW1hcC54bWxcbjQuIENoZWNrIGZvciByb2JvdHMudHh0XG41LiBTZWFyY2ggZm9yIHN0cnVjdHVyZWQgZGF0YSAoSlNPTi1MRClcbjYuIFZlcmlmeSBoZWFkaW5nIGhpZXJhcmNoeSBpbiBwYWdlc1xuNy4gQ2hlY2sgc2VtYW50aWMgSFRNTCB1c2FnZVxuOC4gUmVwb3J0IFNFTyBpc3N1ZXMgYW5kIGltcGFjdGBcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyxhQUFhO0FBQUEsRUFDdkMsaUJBQWlCLENBQUMsOEJBQThCO0FBQUEsRUFDaEQsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFDQSx1QkFBdUI7QUFBQSxFQUN2QixZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixVQUFVLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDNUIsVUFBVTtBQUFBLFFBQ1IsTUFBTTtBQUFBLFFBQ04sWUFBWTtBQUFBLFVBQ1YsT0FBTyxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQ3pCLGFBQWEsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUMvQixRQUFRLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDMUIsY0FBYyxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQ2hDLFdBQVcsRUFBRSxNQUFNLFVBQVU7QUFBQSxRQUMvQjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGNBQWM7QUFBQSxRQUNaLE1BQU07QUFBQSxRQUNOLFlBQVk7QUFBQSxVQUNWLFNBQVMsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUMzQixXQUFXLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDN0IsZ0JBQWdCLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDbEMsY0FBYyxFQUFFLE1BQU0sVUFBVTtBQUFBLFFBQ2xDO0FBQUEsTUFDRjtBQUFBLE1BQ0EsUUFBUTtBQUFBLFFBQ04sTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLE9BQU8sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN4QixRQUFRLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDM0I7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQUEsRUFDQSxlQUFlO0FBQUEsRUFDZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFtQmQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBVXRCO0FBRUEsSUFBTyx3QkFBUTsiLAogICJuYW1lcyI6IFtdCn0K
