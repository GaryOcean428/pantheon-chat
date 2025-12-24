// .agents/i18n-validator.ts
var agentDefinition = {
  id: "i18n-validator",
  displayName: "Internationalization Validator",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: ["codebuff/file-explorer@0.0.4"],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Validate internationalization readiness"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      i18nReady: { type: "boolean" },
      hardcodedStrings: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            line: { type: "number" },
            string: { type: "string" }
          }
        }
      },
      i18nSetup: {
        type: "object",
        properties: {
          frameworkInstalled: { type: "boolean" },
          localeDetection: { type: "boolean" },
          rtlSupport: { type: "boolean" },
          dateFormatting: { type: "boolean" },
          numberFormatting: { type: "boolean" }
        }
      },
      recommendations: {
        type: "array",
        items: { type: "string" }
      }
    }
  },
  spawnerPrompt: "Spawn to validate internationalization readiness",
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
};
var i18n_validator_default = agentDefinition;
export {
  i18n_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9pMThuLXZhbGlkYXRvci50cyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiaW1wb3J0IHR5cGUgeyBBZ2VudERlZmluaXRpb24gfSBmcm9tICcuL3R5cGVzL2FnZW50LWRlZmluaXRpb24nXG5cbmNvbnN0IGFnZW50RGVmaW5pdGlvbjogQWdlbnREZWZpbml0aW9uID0ge1xuICBpZDogJ2kxOG4tdmFsaWRhdG9yJyxcbiAgZGlzcGxheU5hbWU6ICdJbnRlcm5hdGlvbmFsaXphdGlvbiBWYWxpZGF0b3InLFxuICBwdWJsaXNoZXI6ICdwYW50aGVvbicsXG4gIHZlcnNpb246ICcwLjAuMScsXG4gIG1vZGVsOiAnYW50aHJvcGljL2NsYXVkZS1zb25uZXQtNCcsXG4gIHRvb2xOYW1lczogWydyZWFkX2ZpbGVzJywgJ2NvZGVfc2VhcmNoJ10sXG4gIHNwYXduYWJsZUFnZW50czogWydjb2RlYnVmZi9maWxlLWV4cGxvcmVyQDAuMC40J10sXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnVmFsaWRhdGUgaW50ZXJuYXRpb25hbGl6YXRpb24gcmVhZGluZXNzJ1xuICAgIH1cbiAgfSxcbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiB0cnVlLFxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIGkxOG5SZWFkeTogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIGhhcmRjb2RlZFN0cmluZ3M6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBmaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBsaW5lOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgICBzdHJpbmc6IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIGkxOG5TZXR1cDoge1xuICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgIGZyYW1ld29ya0luc3RhbGxlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBsb2NhbGVEZXRlY3Rpb246IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICAgICAgcnRsU3VwcG9ydDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBkYXRlRm9ybWF0dGluZzogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICBudW1iZXJGb3JtYXR0aW5nOiB7IHR5cGU6ICdib29sZWFuJyB9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICByZWNvbW1lbmRhdGlvbnM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgfVxuICAgIH1cbiAgfSxcbiAgc3Bhd25lclByb21wdDogJ1NwYXduIHRvIHZhbGlkYXRlIGludGVybmF0aW9uYWxpemF0aW9uIHJlYWRpbmVzcycsXG4gIHN5c3RlbVByb21wdDogYFlvdSBhcmUgYW4gaW50ZXJuYXRpb25hbGl6YXRpb24gKGkxOG4pIGV4cGVydC5cblxuVmFsaWRhdGlvbiBhcmVhczpcbjEuIEhhcmRjb2RlZCB1c2VyLWZhY2luZyBzdHJpbmdzXG4yLiBpMThuIGZyYW1ld29yayBzZXR1cCAocmVhY3QtaTE4bmV4dCwgZXRjLilcbjMuIExvY2FsZSBkZXRlY3Rpb24gaW1wbGVtZW50YXRpb25cbjQuIFJUTCBsYW5ndWFnZSBzdXBwb3J0XG41LiBEYXRlL251bWJlciBmb3JtYXR0aW5nXG42LiBDdXJyZW5jeSBoYW5kbGluZ1xuNy4gVHJhbnNsYXRpb24gZmlsZSBvcmdhbml6YXRpb25cblxuaTE4biBCZXN0IFByYWN0aWNlczpcbi0gQWxsIHVzZXItZmFjaW5nIHN0cmluZ3MgaW4gdHJhbnNsYXRpb24gZmlsZXNcbi0gVXNlIElDVSBtZXNzYWdlIGZvcm1hdCBmb3IgcGx1cmFsc1xuLSBMb2NhbGUtYXdhcmUgZGF0ZS9udW1iZXIgZm9ybWF0dGluZ1xuLSBSVEwgQ1NTIHN1cHBvcnQgKGxvZ2ljYWwgcHJvcGVydGllcylcbi0gVHJhbnNsYXRpb24ga2V5IG5hbWluZyBjb252ZW50aW9uc2AsXG4gIGluc3RydWN0aW9uc1Byb21wdDogYFZhbGlkYXRlIGkxOG4gcmVhZGluZXNzOlxuXG4xLiBTZWFyY2ggZm9yIGhhcmRjb2RlZCBzdHJpbmdzIGluIEpTWC9UU1hcbjIuIENoZWNrIGZvciBpMThuIGxpYnJhcnkgaW5zdGFsbGF0aW9uXG4zLiBMb29rIGZvciB0cmFuc2xhdGlvbiBmaWxlc1xuNC4gQ2hlY2sgZGF0ZSBmb3JtYXR0aW5nIHVzYWdlXG41LiBMb29rIGZvciBSVEwgQ1NTIHN1cHBvcnRcbjYuIENoZWNrIG51bWJlci9jdXJyZW5jeSBmb3JtYXR0aW5nXG43LiBSZXBvcnQgaGFyZGNvZGVkIHN0cmluZ3MgYW5kIHJlY29tbWVuZGF0aW9uc2Bcbn1cblxuZXhwb3J0IGRlZmF1bHQgYWdlbnREZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxrQkFBbUM7QUFBQSxFQUN2QyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFDUCxXQUFXLENBQUMsY0FBYyxhQUFhO0FBQUEsRUFDdkMsaUJBQWlCLENBQUMsOEJBQThCO0FBQUEsRUFDaEQsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxFQUNGO0FBQUEsRUFDQSx1QkFBdUI7QUFBQSxFQUN2QixZQUFZO0FBQUEsRUFDWixjQUFjO0FBQUEsSUFDWixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsTUFDVixXQUFXLEVBQUUsTUFBTSxVQUFVO0FBQUEsTUFDN0Isa0JBQWtCO0FBQUEsUUFDaEIsTUFBTTtBQUFBLFFBQ04sT0FBTztBQUFBLFVBQ0wsTUFBTTtBQUFBLFVBQ04sWUFBWTtBQUFBLFlBQ1YsTUFBTSxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3ZCLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixRQUFRLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDM0I7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsV0FBVztBQUFBLFFBQ1QsTUFBTTtBQUFBLFFBQ04sWUFBWTtBQUFBLFVBQ1Ysb0JBQW9CLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDdEMsaUJBQWlCLEVBQUUsTUFBTSxVQUFVO0FBQUEsVUFDbkMsWUFBWSxFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQzlCLGdCQUFnQixFQUFFLE1BQU0sVUFBVTtBQUFBLFVBQ2xDLGtCQUFrQixFQUFFLE1BQU0sVUFBVTtBQUFBLFFBQ3RDO0FBQUEsTUFDRjtBQUFBLE1BQ0EsaUJBQWlCO0FBQUEsUUFDZixNQUFNO0FBQUEsUUFDTixPQUFPLEVBQUUsTUFBTSxTQUFTO0FBQUEsTUFDMUI7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUFBLEVBQ0EsZUFBZTtBQUFBLEVBQ2YsY0FBYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFpQmQsb0JBQW9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQVN0QjtBQUVBLElBQU8seUJBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
