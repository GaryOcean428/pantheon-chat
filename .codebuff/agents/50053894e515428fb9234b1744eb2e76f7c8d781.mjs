// .agents/dry-violation-finder.ts
var definition = {
  id: "dry-violation-finder",
  displayName: "DRY Violation Finder",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "code_search",
    "run_terminal_command",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific patterns or files to check"
    },
    params: {
      type: "object",
      properties: {
        minLines: {
          type: "number",
          description: "Minimum lines for a block to be considered (default: 5)"
        },
        directories: {
          type: "array",
          description: "Directories to scan"
        }
      },
      required: []
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      duplicates: {
        type: "array",
        items: {
          type: "object",
          properties: {
            pattern: { type: "string" },
            occurrences: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  file: { type: "string" },
                  startLine: { type: "number" },
                  endLine: { type: "number" }
                }
              }
            },
            refactoringHint: { type: "string" }
          }
        }
      },
      hardcodedValues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            value: { type: "string" },
            occurrences: { type: "number" },
            suggestedConstant: { type: "string" }
          }
        }
      },
      summary: { type: "string" }
    },
    required: ["duplicates", "hardcodedValues", "summary"]
  },
  spawnerPrompt: `Spawn to find DRY (Don't Repeat Yourself) violations:
- Duplicated code blocks across files
- Repeated magic numbers and strings
- Similar functions that could be unified
- Copy-pasted error handling

Use for periodic code quality audits.`,
  systemPrompt: `You are the DRY Violation Finder for the Pantheon-Chat project.

You detect code duplication that should be refactored.

## DRY PRINCIPLE

"Every piece of knowledge must have a single, unambiguous, authoritative representation within a system."

## WHAT TO DETECT

### 1. Duplicated Code Blocks
\`\`\`typescript
// File A
const result = await fetch(url)
if (!result.ok) {
  throw new Error(\`HTTP error: \${result.status}\`)
}
const data = await result.json()

// File B - SAME CODE!
const result = await fetch(url)
if (!result.ok) {
  throw new Error(\`HTTP error: \${result.status}\`)
}
const data = await result.json()
\`\`\`

**Fix:** Extract to shared utility function

### 2. Magic Numbers
\`\`\`typescript
// BAD - 64 repeated everywhere
const basin = new Array(64).fill(0)
if (coords.length !== 64) throw new Error('Wrong dimension')
for (let i = 0; i < 64; i++) { ... }

// GOOD - use constant
import { BASIN_DIMENSION } from '@/constants'
const basin = new Array(BASIN_DIMENSION).fill(0)
\`\`\`

### 3. Magic Strings
\`\`\`typescript
// BAD - repeated strings
if (status === 'resonant') { ... }
if (regime === 'resonant') { ... }
return 'resonant'

// GOOD - use enum or constant
if (status === REGIMES.RESONANT) { ... }
\`\`\`

### 4. Similar Functions
\`\`\`typescript
// BAD - nearly identical
function processUserQuery(query: string) { ... }
function processAgentQuery(query: string) { ... }

// GOOD - unified with parameter
function processQuery(query: string, source: 'user' | 'agent') { ... }
\`\`\`

## KNOWN CONSTANTS IN PROJECT

- BASIN_DIMENSION = 64
- KAPPA_OPTIMAL = 64
- PHI_MIN = 0.7
- Regime names: 'resonant', 'breakdown', 'dormant'`,
  instructionsPrompt: `## Detection Process

1. Run Python DRY validation if available:
   \`\`\`bash
   python scripts/validate-python-dry.py
   \`\`\`

2. Search for duplicated patterns:

   **Error handling patterns:**
   \`\`\`bash
   rg "if.*!.*ok.*throw.*Error" --type ts -A 2
   rg "try.*catch.*console.error" --type ts -A 3
   \`\`\`

   **Fetch patterns:**
   \`\`\`bash
   rg "await fetch.*localhost:5001" --type ts -A 3
   \`\`\`

3. Find magic numbers:
   \`\`\`bash
   # Find hardcoded 64 (should be BASIN_DIMENSION)
   rg "[^0-9]64[^0-9]" --type ts --type py
   
   # Find hardcoded 0.7 (should be PHI_MIN)
   rg "0.7[^0-9]" --type ts --type py
   \`\`\`

4. Find magic strings:
   \`\`\`bash
   # Regime strings
   rg "['"]resonant['"]" --type ts --type py
   rg "['"]breakdown['"]" --type ts --type py
   \`\`\`

5. Look for similar function names:
   - Functions with similar prefixes/suffixes
   - Functions in different files doing similar things

6. Identify refactoring opportunities:
   - Extract repeated blocks to shared utilities
   - Replace magic values with constants
   - Unify similar functions

7. Set structured output:
   - duplicates: code blocks appearing multiple times
   - hardcodedValues: magic numbers/strings
   - summary: human-readable summary with specific refactoring suggestions

DRY code is maintainable code!`,
  includeMessageHistory: false
};
var dry_violation_finder_default = definition;
export {
  dry_violation_finder_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9kcnktdmlvbGF0aW9uLWZpbmRlci50cyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiaW1wb3J0IHR5cGUgeyBBZ2VudERlZmluaXRpb24gfSBmcm9tICcuL3R5cGVzL2FnZW50LWRlZmluaXRpb24nXG5cbmNvbnN0IGRlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdkcnktdmlvbGF0aW9uLWZpbmRlcicsXG4gIGRpc3BsYXlOYW1lOiAnRFJZIFZpb2xhdGlvbiBGaW5kZXInLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdydW5fdGVybWluYWxfY29tbWFuZCcsXG4gICAgJ3NldF9vdXRwdXQnLFxuICBdLFxuXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnT3B0aW9uYWwgc3BlY2lmaWMgcGF0dGVybnMgb3IgZmlsZXMgdG8gY2hlY2snLFxuICAgIH0sXG4gICAgcGFyYW1zOiB7XG4gICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgIHByb3BlcnRpZXM6IHtcbiAgICAgICAgbWluTGluZXM6IHtcbiAgICAgICAgICB0eXBlOiAnbnVtYmVyJyxcbiAgICAgICAgICBkZXNjcmlwdGlvbjogJ01pbmltdW0gbGluZXMgZm9yIGEgYmxvY2sgdG8gYmUgY29uc2lkZXJlZCAoZGVmYXVsdDogNSknLFxuICAgICAgICB9LFxuICAgICAgICBkaXJlY3Rvcmllczoge1xuICAgICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgICAgZGVzY3JpcHRpb246ICdEaXJlY3RvcmllcyB0byBzY2FuJyxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgICByZXF1aXJlZDogW10sXG4gICAgfSxcbiAgfSxcblxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZF9vdXRwdXQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBkdXBsaWNhdGVzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgcGF0dGVybjogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgb2NjdXJyZW5jZXM6IHtcbiAgICAgICAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICAgICAgICBmaWxlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICAgICAgICBzdGFydExpbmU6IHsgdHlwZTogJ251bWJlcicgfSxcbiAgICAgICAgICAgICAgICAgIGVuZExpbmU6IHsgdHlwZTogJ251bWJlcicgfSxcbiAgICAgICAgICAgICAgICB9LFxuICAgICAgICAgICAgICB9LFxuICAgICAgICAgICAgfSxcbiAgICAgICAgICAgIHJlZmFjdG9yaW5nSGludDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICB9LFxuICAgICAgaGFyZGNvZGVkVmFsdWVzOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgdmFsdWU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIG9jY3VycmVuY2VzOiB7IHR5cGU6ICdudW1iZXInIH0sXG4gICAgICAgICAgICBzdWdnZXN0ZWRDb25zdGFudDogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICB9LFxuICAgICAgc3VtbWFyeTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgIH0sXG4gICAgcmVxdWlyZWQ6IFsnZHVwbGljYXRlcycsICdoYXJkY29kZWRWYWx1ZXMnLCAnc3VtbWFyeSddLFxuICB9LFxuXG4gIHNwYXduZXJQcm9tcHQ6IGBTcGF3biB0byBmaW5kIERSWSAoRG9uJ3QgUmVwZWF0IFlvdXJzZWxmKSB2aW9sYXRpb25zOlxuLSBEdXBsaWNhdGVkIGNvZGUgYmxvY2tzIGFjcm9zcyBmaWxlc1xuLSBSZXBlYXRlZCBtYWdpYyBudW1iZXJzIGFuZCBzdHJpbmdzXG4tIFNpbWlsYXIgZnVuY3Rpb25zIHRoYXQgY291bGQgYmUgdW5pZmllZFxuLSBDb3B5LXBhc3RlZCBlcnJvciBoYW5kbGluZ1xuXG5Vc2UgZm9yIHBlcmlvZGljIGNvZGUgcXVhbGl0eSBhdWRpdHMuYCxcblxuICBzeXN0ZW1Qcm9tcHQ6IGBZb3UgYXJlIHRoZSBEUlkgVmlvbGF0aW9uIEZpbmRlciBmb3IgdGhlIFBhbnRoZW9uLUNoYXQgcHJvamVjdC5cblxuWW91IGRldGVjdCBjb2RlIGR1cGxpY2F0aW9uIHRoYXQgc2hvdWxkIGJlIHJlZmFjdG9yZWQuXG5cbiMjIERSWSBQUklOQ0lQTEVcblxuXCJFdmVyeSBwaWVjZSBvZiBrbm93bGVkZ2UgbXVzdCBoYXZlIGEgc2luZ2xlLCB1bmFtYmlndW91cywgYXV0aG9yaXRhdGl2ZSByZXByZXNlbnRhdGlvbiB3aXRoaW4gYSBzeXN0ZW0uXCJcblxuIyMgV0hBVCBUTyBERVRFQ1RcblxuIyMjIDEuIER1cGxpY2F0ZWQgQ29kZSBCbG9ja3NcblxcYFxcYFxcYHR5cGVzY3JpcHRcbi8vIEZpbGUgQVxuY29uc3QgcmVzdWx0ID0gYXdhaXQgZmV0Y2godXJsKVxuaWYgKCFyZXN1bHQub2spIHtcbiAgdGhyb3cgbmV3IEVycm9yKFxcYEhUVFAgZXJyb3I6IFxcJHtyZXN1bHQuc3RhdHVzfVxcYClcbn1cbmNvbnN0IGRhdGEgPSBhd2FpdCByZXN1bHQuanNvbigpXG5cbi8vIEZpbGUgQiAtIFNBTUUgQ09ERSFcbmNvbnN0IHJlc3VsdCA9IGF3YWl0IGZldGNoKHVybClcbmlmICghcmVzdWx0Lm9rKSB7XG4gIHRocm93IG5ldyBFcnJvcihcXGBIVFRQIGVycm9yOiBcXCR7cmVzdWx0LnN0YXR1c31cXGApXG59XG5jb25zdCBkYXRhID0gYXdhaXQgcmVzdWx0Lmpzb24oKVxuXFxgXFxgXFxgXG5cbioqRml4OioqIEV4dHJhY3QgdG8gc2hhcmVkIHV0aWxpdHkgZnVuY3Rpb25cblxuIyMjIDIuIE1hZ2ljIE51bWJlcnNcblxcYFxcYFxcYHR5cGVzY3JpcHRcbi8vIEJBRCAtIDY0IHJlcGVhdGVkIGV2ZXJ5d2hlcmVcbmNvbnN0IGJhc2luID0gbmV3IEFycmF5KDY0KS5maWxsKDApXG5pZiAoY29vcmRzLmxlbmd0aCAhPT0gNjQpIHRocm93IG5ldyBFcnJvcignV3JvbmcgZGltZW5zaW9uJylcbmZvciAobGV0IGkgPSAwOyBpIDwgNjQ7IGkrKykgeyAuLi4gfVxuXG4vLyBHT09EIC0gdXNlIGNvbnN0YW50XG5pbXBvcnQgeyBCQVNJTl9ESU1FTlNJT04gfSBmcm9tICdAL2NvbnN0YW50cydcbmNvbnN0IGJhc2luID0gbmV3IEFycmF5KEJBU0lOX0RJTUVOU0lPTikuZmlsbCgwKVxuXFxgXFxgXFxgXG5cbiMjIyAzLiBNYWdpYyBTdHJpbmdzXG5cXGBcXGBcXGB0eXBlc2NyaXB0XG4vLyBCQUQgLSByZXBlYXRlZCBzdHJpbmdzXG5pZiAoc3RhdHVzID09PSAncmVzb25hbnQnKSB7IC4uLiB9XG5pZiAocmVnaW1lID09PSAncmVzb25hbnQnKSB7IC4uLiB9XG5yZXR1cm4gJ3Jlc29uYW50J1xuXG4vLyBHT09EIC0gdXNlIGVudW0gb3IgY29uc3RhbnRcbmlmIChzdGF0dXMgPT09IFJFR0lNRVMuUkVTT05BTlQpIHsgLi4uIH1cblxcYFxcYFxcYFxuXG4jIyMgNC4gU2ltaWxhciBGdW5jdGlvbnNcblxcYFxcYFxcYHR5cGVzY3JpcHRcbi8vIEJBRCAtIG5lYXJseSBpZGVudGljYWxcbmZ1bmN0aW9uIHByb2Nlc3NVc2VyUXVlcnkocXVlcnk6IHN0cmluZykgeyAuLi4gfVxuZnVuY3Rpb24gcHJvY2Vzc0FnZW50UXVlcnkocXVlcnk6IHN0cmluZykgeyAuLi4gfVxuXG4vLyBHT09EIC0gdW5pZmllZCB3aXRoIHBhcmFtZXRlclxuZnVuY3Rpb24gcHJvY2Vzc1F1ZXJ5KHF1ZXJ5OiBzdHJpbmcsIHNvdXJjZTogJ3VzZXInIHwgJ2FnZW50JykgeyAuLi4gfVxuXFxgXFxgXFxgXG5cbiMjIEtOT1dOIENPTlNUQU5UUyBJTiBQUk9KRUNUXG5cbi0gQkFTSU5fRElNRU5TSU9OID0gNjRcbi0gS0FQUEFfT1BUSU1BTCA9IDY0XG4tIFBISV9NSU4gPSAwLjdcbi0gUmVnaW1lIG5hbWVzOiAncmVzb25hbnQnLCAnYnJlYWtkb3duJywgJ2Rvcm1hbnQnYCxcblxuICBpbnN0cnVjdGlvbnNQcm9tcHQ6IGAjIyBEZXRlY3Rpb24gUHJvY2Vzc1xuXG4xLiBSdW4gUHl0aG9uIERSWSB2YWxpZGF0aW9uIGlmIGF2YWlsYWJsZTpcbiAgIFxcYFxcYFxcYGJhc2hcbiAgIHB5dGhvbiBzY3JpcHRzL3ZhbGlkYXRlLXB5dGhvbi1kcnkucHlcbiAgIFxcYFxcYFxcYFxuXG4yLiBTZWFyY2ggZm9yIGR1cGxpY2F0ZWQgcGF0dGVybnM6XG5cbiAgICoqRXJyb3IgaGFuZGxpbmcgcGF0dGVybnM6KipcbiAgIFxcYFxcYFxcYGJhc2hcbiAgIHJnIFwiaWYuKiEuKm9rLip0aHJvdy4qRXJyb3JcIiAtLXR5cGUgdHMgLUEgMlxuICAgcmcgXCJ0cnkuKmNhdGNoLipjb25zb2xlXFwuZXJyb3JcIiAtLXR5cGUgdHMgLUEgM1xuICAgXFxgXFxgXFxgXG5cbiAgICoqRmV0Y2ggcGF0dGVybnM6KipcbiAgIFxcYFxcYFxcYGJhc2hcbiAgIHJnIFwiYXdhaXQgZmV0Y2guKmxvY2FsaG9zdDo1MDAxXCIgLS10eXBlIHRzIC1BIDNcbiAgIFxcYFxcYFxcYFxuXG4zLiBGaW5kIG1hZ2ljIG51bWJlcnM6XG4gICBcXGBcXGBcXGBiYXNoXG4gICAjIEZpbmQgaGFyZGNvZGVkIDY0IChzaG91bGQgYmUgQkFTSU5fRElNRU5TSU9OKVxuICAgcmcgXCJbXjAtOV02NFteMC05XVwiIC0tdHlwZSB0cyAtLXR5cGUgcHlcbiAgIFxuICAgIyBGaW5kIGhhcmRjb2RlZCAwLjcgKHNob3VsZCBiZSBQSElfTUlOKVxuICAgcmcgXCIwXFwuN1teMC05XVwiIC0tdHlwZSB0cyAtLXR5cGUgcHlcbiAgIFxcYFxcYFxcYFxuXG40LiBGaW5kIG1hZ2ljIHN0cmluZ3M6XG4gICBcXGBcXGBcXGBiYXNoXG4gICAjIFJlZ2ltZSBzdHJpbmdzXG4gICByZyBcIlsnXFxcIl1yZXNvbmFudFsnXFxcIl1cIiAtLXR5cGUgdHMgLS10eXBlIHB5XG4gICByZyBcIlsnXFxcIl1icmVha2Rvd25bJ1xcXCJdXCIgLS10eXBlIHRzIC0tdHlwZSBweVxuICAgXFxgXFxgXFxgXG5cbjUuIExvb2sgZm9yIHNpbWlsYXIgZnVuY3Rpb24gbmFtZXM6XG4gICAtIEZ1bmN0aW9ucyB3aXRoIHNpbWlsYXIgcHJlZml4ZXMvc3VmZml4ZXNcbiAgIC0gRnVuY3Rpb25zIGluIGRpZmZlcmVudCBmaWxlcyBkb2luZyBzaW1pbGFyIHRoaW5nc1xuXG42LiBJZGVudGlmeSByZWZhY3RvcmluZyBvcHBvcnR1bml0aWVzOlxuICAgLSBFeHRyYWN0IHJlcGVhdGVkIGJsb2NrcyB0byBzaGFyZWQgdXRpbGl0aWVzXG4gICAtIFJlcGxhY2UgbWFnaWMgdmFsdWVzIHdpdGggY29uc3RhbnRzXG4gICAtIFVuaWZ5IHNpbWlsYXIgZnVuY3Rpb25zXG5cbjcuIFNldCBzdHJ1Y3R1cmVkIG91dHB1dDpcbiAgIC0gZHVwbGljYXRlczogY29kZSBibG9ja3MgYXBwZWFyaW5nIG11bHRpcGxlIHRpbWVzXG4gICAtIGhhcmRjb2RlZFZhbHVlczogbWFnaWMgbnVtYmVycy9zdHJpbmdzXG4gICAtIHN1bW1hcnk6IGh1bWFuLXJlYWRhYmxlIHN1bW1hcnkgd2l0aCBzcGVjaWZpYyByZWZhY3RvcmluZyBzdWdnZXN0aW9uc1xuXG5EUlkgY29kZSBpcyBtYWludGFpbmFibGUgY29kZSFgLFxuXG4gIGluY2x1ZGVNZXNzYWdlSGlzdG9yeTogZmFsc2UsXG59XG5cbmV4cG9ydCBkZWZhdWx0IGRlZmluaXRpb25cbiJdLAogICJtYXBwaW5ncyI6ICI7QUFFQSxJQUFNLGFBQThCO0FBQUEsRUFDbEMsSUFBSTtBQUFBLEVBQ0osYUFBYTtBQUFBLEVBQ2IsU0FBUztBQUFBLEVBQ1QsT0FBTztBQUFBLEVBRVAsV0FBVztBQUFBLElBQ1Q7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxFQUNGO0FBQUEsRUFFQSxhQUFhO0FBQUEsSUFDWCxRQUFRO0FBQUEsTUFDTixNQUFNO0FBQUEsTUFDTixhQUFhO0FBQUEsSUFDZjtBQUFBLElBQ0EsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sWUFBWTtBQUFBLFFBQ1YsVUFBVTtBQUFBLFVBQ1IsTUFBTTtBQUFBLFVBQ04sYUFBYTtBQUFBLFFBQ2Y7QUFBQSxRQUNBLGFBQWE7QUFBQSxVQUNYLE1BQU07QUFBQSxVQUNOLGFBQWE7QUFBQSxRQUNmO0FBQUEsTUFDRjtBQUFBLE1BQ0EsVUFBVSxDQUFDO0FBQUEsSUFDYjtBQUFBLEVBQ0Y7QUFBQSxFQUVBLFlBQVk7QUFBQSxFQUNaLGNBQWM7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxNQUNWLFlBQVk7QUFBQSxRQUNWLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLFNBQVMsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUMxQixhQUFhO0FBQUEsY0FDWCxNQUFNO0FBQUEsY0FDTixPQUFPO0FBQUEsZ0JBQ0wsTUFBTTtBQUFBLGdCQUNOLFlBQVk7QUFBQSxrQkFDVixNQUFNLEVBQUUsTUFBTSxTQUFTO0FBQUEsa0JBQ3ZCLFdBQVcsRUFBRSxNQUFNLFNBQVM7QUFBQSxrQkFDNUIsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLGdCQUM1QjtBQUFBLGNBQ0Y7QUFBQSxZQUNGO0FBQUEsWUFDQSxpQkFBaUIsRUFBRSxNQUFNLFNBQVM7QUFBQSxVQUNwQztBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxpQkFBaUI7QUFBQSxRQUNmLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE9BQU8sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN4QixhQUFhLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDOUIsbUJBQW1CLEVBQUUsTUFBTSxTQUFTO0FBQUEsVUFDdEM7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsU0FBUyxFQUFFLE1BQU0sU0FBUztBQUFBLElBQzVCO0FBQUEsSUFDQSxVQUFVLENBQUMsY0FBYyxtQkFBbUIsU0FBUztBQUFBLEVBQ3ZEO0FBQUEsRUFFQSxlQUFlO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFRZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQXFFZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFvRHBCLHVCQUF1QjtBQUN6QjtBQUVBLElBQU8sK0JBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
