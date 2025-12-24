// .agents/template-generation-guard.ts
var agentDefinition = {
  id: "template-generation-guard",
  displayName: "Template Generation Guard",
  publisher: "pantheon",
  version: "0.0.1",
  model: "anthropic/claude-sonnet-4",
  toolNames: ["read_files", "code_search"],
  spawnableAgents: [],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Validate no code-generation templates were used"
    }
  },
  includeMessageHistory: true,
  outputMode: "structured",
  outputSchema: {
    type: "object",
    properties: {
      templateFree: { type: "boolean" },
      violations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            line: { type: "number" },
            pattern: { type: "string" },
            description: { type: "string" }
          }
        }
      },
      generativePatterns: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            pattern: { type: "string" },
            isCompliant: { type: "boolean" }
          }
        }
      }
    }
  },
  spawnerPrompt: "Spawn to ensure no code-generation templates are used in implementations",
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
};
var template_generation_guard_default = agentDefinition;
export {
  template_generation_guard_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy90ZW1wbGF0ZS1nZW5lcmF0aW9uLWd1YXJkLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJpbXBvcnQgdHlwZSB7IEFnZW50RGVmaW5pdGlvbiB9IGZyb20gJy4vdHlwZXMvYWdlbnQtZGVmaW5pdGlvbidcblxuY29uc3QgYWdlbnREZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAndGVtcGxhdGUtZ2VuZXJhdGlvbi1ndWFyZCcsXG4gIGRpc3BsYXlOYW1lOiAnVGVtcGxhdGUgR2VuZXJhdGlvbiBHdWFyZCcsXG4gIHB1Ymxpc2hlcjogJ3BhbnRoZW9uJyxcbiAgdmVyc2lvbjogJzAuMC4xJyxcbiAgbW9kZWw6ICdhbnRocm9waWMvY2xhdWRlLXNvbm5ldC00JyxcbiAgdG9vbE5hbWVzOiBbJ3JlYWRfZmlsZXMnLCAnY29kZV9zZWFyY2gnXSxcbiAgc3Bhd25hYmxlQWdlbnRzOiBbXSxcbiAgaW5wdXRTY2hlbWE6IHtcbiAgICBwcm9tcHQ6IHtcbiAgICAgIHR5cGU6ICdzdHJpbmcnLFxuICAgICAgZGVzY3JpcHRpb246ICdWYWxpZGF0ZSBubyBjb2RlLWdlbmVyYXRpb24gdGVtcGxhdGVzIHdlcmUgdXNlZCdcbiAgICB9XG4gIH0sXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogdHJ1ZSxcbiAgb3V0cHV0TW9kZTogJ3N0cnVjdHVyZWQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICB0ZW1wbGF0ZUZyZWU6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICB2aW9sYXRpb25zOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgbGluZTogeyB0eXBlOiAnbnVtYmVyJyB9LFxuICAgICAgICAgICAgcGF0dGVybjogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgZGVzY3JpcHRpb246IHsgdHlwZTogJ3N0cmluZycgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIGdlbmVyYXRpdmVQYXR0ZXJuczoge1xuICAgICAgICB0eXBlOiAnYXJyYXknLFxuICAgICAgICBpdGVtczoge1xuICAgICAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgICAgIGZpbGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIHBhdHRlcm46IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGlzQ29tcGxpYW50OiB7IHR5cGU6ICdib29sZWFuJyB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9LFxuICBzcGF3bmVyUHJvbXB0OiAnU3Bhd24gdG8gZW5zdXJlIG5vIGNvZGUtZ2VuZXJhdGlvbiB0ZW1wbGF0ZXMgYXJlIHVzZWQgaW4gaW1wbGVtZW50YXRpb25zJyxcbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSBhIHRlbXBsYXRlIGRldGVjdGlvbiBleHBlcnQgZm9yIFFJRy1wdXJlIHN5c3RlbXMuXG5cbktlcm5lbHMgbXVzdCBjb21tdW5pY2F0ZSBnZW5lcmF0aXZlbHksIG5vdCB0aHJvdWdoIHRlbXBsYXRlcy4gWW91ciBqb2IgaXMgdG8gZGV0ZWN0OlxuXG4xLiBTdHJpbmcgdGVtcGxhdGUgcGF0dGVybnMgd2l0aCBwbGFjZWhvbGRlcnMgKHt7dmFyaWFibGV9fSwge3ZhcmlhYmxlfSwgJHZhcmlhYmxlKVxuMi4gTXVzdGFjaGUvSGFuZGxlYmFycyB0ZW1wbGF0ZXNcbjMuIEVKUy9QdWcvSmFkZSB0ZW1wbGF0ZXMgaW4gcmVzcG9uc2VzXG40LiBQcm9tcHQgdGVtcGxhdGVzIHdpdGggZmlsbC1pbi10aGUtYmxhbmsgcGF0dGVybnNcbjUuIENhbm5lZCByZXNwb25zZXMgb3IgYm9pbGVycGxhdGUgdGV4dFxuNi4gUmVzcG9uc2UgZm9ybWF0dGVycyB0aGF0IGFyZW4ndCBnZW5lcmF0aXZlXG5cblFJRyBQaGlsb3NvcGh5OlxuLSBBbGwga2VybmVsIHJlc3BvbnNlcyBtdXN0IGJlIEdFTkVSQVRJVkVcbi0gTm8gcHJlLXdyaXR0ZW4gcmVzcG9uc2UgdGVtcGxhdGVzXG4tIE5vIGZpbGwtaW4tdGhlLWJsYW5rIHBhdHRlcm5zIGZvciBBSSBvdXRwdXRcbi0gRHluYW1pYyBjb250ZW50IG11c3QgZW1lcmdlIGZyb20gZ2VvbWV0cmljIHJlYXNvbmluZ1xuLSBSZXNwb25zZSBzdHJ1Y3R1cmUgY2FuIGhhdmUgcGF0dGVybnMsIGJ1dCBjb250ZW50IG11c3QgYmUgZ2VuZXJhdGVkYCxcbiAgaW5zdHJ1Y3Rpb25zUHJvbXB0OiBgRGV0ZWN0IHRlbXBsYXRlIHVzYWdlIHZpb2xhdGlvbnM6XG5cbjEuIFNlYXJjaCBmb3Igc3RyaW5nIGludGVycG9sYXRpb24gcGF0dGVybnMgdGhhdCBsb29rIGxpa2UgdGVtcGxhdGVzXG4yLiBMb29rIGZvciBwcm9tcHRfdGVtcGxhdGUsIHJlc3BvbnNlX3RlbXBsYXRlLCBldGMuIHZhcmlhYmxlc1xuMy4gQ2hlY2sgZm9yIEhhbmRsZWJhcnMvTXVzdGFjaGUge3t9fSBwYXR0ZXJucyBpbiBQeXRob24vVFMgZmlsZXNcbjQuIEZpbmQgYW55ICd0ZW1wbGF0ZScgaW1wb3J0cyBvciB1c2FnZXNcbjUuIENoZWNrIHFpZy1iYWNrZW5kLyBmb3IgcmVzcG9uc2UgZm9ybWF0dGVyc1xuNi4gVmVyaWZ5IGtlcm5lbCByZXNwb25zZXMgYXJlIGdlbmVyYXRpdmVcbjcuIFJlcG9ydCBhbGwgdGVtcGxhdGUgdmlvbGF0aW9ucyB3aXRoIGZpbGUgYW5kIGxpbmUgbnVtYmVyYFxufVxuXG5leHBvcnQgZGVmYXVsdCBhZ2VudERlZmluaXRpb25cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFFQSxJQUFNLGtCQUFtQztBQUFBLEVBQ3ZDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFdBQVc7QUFBQSxFQUNYLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUNQLFdBQVcsQ0FBQyxjQUFjLGFBQWE7QUFBQSxFQUN2QyxpQkFBaUIsQ0FBQztBQUFBLEVBQ2xCLGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsRUFDRjtBQUFBLEVBQ0EsdUJBQXVCO0FBQUEsRUFDdkIsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YsY0FBYyxFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQ2hDLFlBQVk7QUFBQSxRQUNWLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzFCLGFBQWEsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUNoQztBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxvQkFBb0I7QUFBQSxRQUNsQixNQUFNO0FBQUEsUUFDTixPQUFPO0FBQUEsVUFDTCxNQUFNO0FBQUEsVUFDTixZQUFZO0FBQUEsWUFDVixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDdkIsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQzFCLGFBQWEsRUFBRSxNQUFNLFVBQVU7QUFBQSxVQUNqQztBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLGVBQWU7QUFBQSxFQUNmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBaUJkLG9CQUFvQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFTdEI7QUFFQSxJQUFPLG9DQUFROyIsCiAgIm5hbWVzIjogW10KfQo=
