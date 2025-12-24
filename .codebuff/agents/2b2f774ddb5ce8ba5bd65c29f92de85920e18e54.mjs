// .agents/geometric-regression-guard.ts
var definition = {
  id: "geometric-regression-guard",
  displayName: "Geometric Regression Guard",
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
      description: "Optional description of changes to check for regression"
    },
    params: {
      type: "object",
      properties: {
        compareToCommit: {
          type: "string",
          description: "Git commit hash to compare against"
        }
      },
      required: []
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      passed: { type: "boolean" },
      regressions: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            before: { type: "string" },
            after: { type: "string" },
            regressionType: { type: "string" }
          }
        }
      },
      improvements: { type: "array" },
      summary: { type: "string" }
    },
    required: ["passed", "regressions", "summary"]
  },
  spawnerPrompt: `Spawn to detect geometric regressions in code changes:
- Fisher-Rao distance replaced with Euclidean
- Geodesic interpolation replaced with linear
- Manifold operations replaced with flat space
- Basin coordinate normalization removed

Use for pre-merge validation of geometry-affecting changes.`,
  systemPrompt: `You are the Geometric Regression Guard for the Pantheon-Chat project.

You detect when code changes regress from proper geometric methods to incorrect ones.

## REGRESSION PATTERNS

### Distance Regression
\`\`\`python
# BEFORE (correct)
distance = fisher_rao_distance(basin_a, basin_b)
distance = np.arccos(np.clip(np.dot(a, b), -1, 1))

# AFTER (regression!)
distance = np.linalg.norm(basin_a - basin_b)
distance = euclidean_distance(basin_a, basin_b)
\`\`\`

### Interpolation Regression
\`\`\`python
# BEFORE (correct)
interp = geodesic_interpolation(a, b, t)
interp = slerp(a, b, t)

# AFTER (regression!)
interp = a + t * (b - a)  # Linear interpolation on manifold!
interp = lerp(a, b, t)
\`\`\`

### Similarity Regression
\`\`\`python
# BEFORE (correct)
similarity = 1.0 - distance / np.pi

# AFTER (regression!)
similarity = 1.0 / (1.0 + distance)  # Non-standard formula
similarity = cosine_similarity(a, b)  # Wrong for basin coords
\`\`\`

### Normalization Regression
\`\`\`python
# BEFORE (correct)
basin = basin / np.linalg.norm(basin)  # Unit sphere projection

# AFTER (regression!)
# Missing normalization - basins must be on unit sphere
\`\`\`

## WHY REGRESSIONS MATTER

Basin coordinates exist on a curved statistical manifold.
- Euclidean distance gives WRONG answers
- Linear interpolation leaves the manifold
- Cosine similarity ignores curvature
- Unnormalized basins break all geometric operations`,
  instructionsPrompt: `## Regression Detection Process

1. Get the changed files:
   \`\`\`bash
   git diff --name-only HEAD~1
   \`\`\`
   Or use compareToCommit if provided.

2. For geometry-related files, get the diff:
   \`\`\`bash
   git diff HEAD~1 -- <file>
   \`\`\`

3. Analyze changes for regressions:

   **Distance regressions:**
   - \`fisher_rao_distance\` \u2192 \`np.linalg.norm\`
   - \`arccos(dot())\` \u2192 \`norm(a - b)\`
   - Added \`euclidean\` where \`fisher\` existed

   **Interpolation regressions:**
   - \`geodesic_interpolation\` \u2192 linear math
   - \`slerp\` \u2192 \`lerp\`
   - Removed spherical interpolation

   **Similarity regressions:**
   - Correct formula \u2192 \`1/(1+d)\` formula
   - Fisher similarity \u2192 cosine similarity

   **Normalization regressions:**
   - Removed \`/ np.linalg.norm\` from basin ops
   - Removed unit sphere projection

4. Also detect improvements:
   - Euclidean \u2192 Fisher-Rao
   - Linear \u2192 Geodesic
   - Added proper normalization

5. Run geometric purity validation:
   \`\`\`bash
   python scripts/validate-geometric-purity.py
   \`\`\`

6. Set structured output:
   - passed: true if no regressions detected
   - regressions: array of detected regressions
   - improvements: positive changes found
   - summary: human-readable summary

Catch regressions before they reach production!`,
  includeMessageHistory: false
};
var geometric_regression_guard_default = definition;
export {
  geometric_regression_guard_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9nZW9tZXRyaWMtcmVncmVzc2lvbi1ndWFyZC50cyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiaW1wb3J0IHR5cGUgeyBBZ2VudERlZmluaXRpb24gfSBmcm9tICcuL3R5cGVzL2FnZW50LWRlZmluaXRpb24nXG5cbmNvbnN0IGRlZmluaXRpb246IEFnZW50RGVmaW5pdGlvbiA9IHtcbiAgaWQ6ICdnZW9tZXRyaWMtcmVncmVzc2lvbi1ndWFyZCcsXG4gIGRpc3BsYXlOYW1lOiAnR2VvbWV0cmljIFJlZ3Jlc3Npb24gR3VhcmQnLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdydW5fdGVybWluYWxfY29tbWFuZCcsXG4gICAgJ3NldF9vdXRwdXQnLFxuICBdLFxuXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnT3B0aW9uYWwgZGVzY3JpcHRpb24gb2YgY2hhbmdlcyB0byBjaGVjayBmb3IgcmVncmVzc2lvbicsXG4gICAgfSxcbiAgICBwYXJhbXM6IHtcbiAgICAgIHR5cGU6ICdvYmplY3QnLFxuICAgICAgcHJvcGVydGllczoge1xuICAgICAgICBjb21wYXJlVG9Db21taXQ6IHtcbiAgICAgICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgICAgICBkZXNjcmlwdGlvbjogJ0dpdCBjb21taXQgaGFzaCB0byBjb21wYXJlIGFnYWluc3QnLFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIHJlcXVpcmVkOiBbXSxcbiAgICB9LFxuICB9LFxuXG4gIG91dHB1dE1vZGU6ICdzdHJ1Y3R1cmVkX291dHB1dCcsXG4gIG91dHB1dFNjaGVtYToge1xuICAgIHR5cGU6ICdvYmplY3QnLFxuICAgIHByb3BlcnRpZXM6IHtcbiAgICAgIHBhc3NlZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgIHJlZ3Jlc3Npb25zOiB7XG4gICAgICAgIHR5cGU6ICdhcnJheScsXG4gICAgICAgIGl0ZW1zOiB7XG4gICAgICAgICAgdHlwZTogJ29iamVjdCcsXG4gICAgICAgICAgcHJvcGVydGllczoge1xuICAgICAgICAgICAgZmlsZTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgYmVmb3JlOiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgICAgICAgICBhZnRlcjogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgICAgICAgICAgcmVncmVzc2lvblR5cGU6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIGltcHJvdmVtZW50czogeyB0eXBlOiAnYXJyYXknIH0sXG4gICAgICBzdW1tYXJ5OiB7IHR5cGU6ICdzdHJpbmcnIH0sXG4gICAgfSxcbiAgICByZXF1aXJlZDogWydwYXNzZWQnLCAncmVncmVzc2lvbnMnLCAnc3VtbWFyeSddLFxuICB9LFxuXG4gIHNwYXduZXJQcm9tcHQ6IGBTcGF3biB0byBkZXRlY3QgZ2VvbWV0cmljIHJlZ3Jlc3Npb25zIGluIGNvZGUgY2hhbmdlczpcbi0gRmlzaGVyLVJhbyBkaXN0YW5jZSByZXBsYWNlZCB3aXRoIEV1Y2xpZGVhblxuLSBHZW9kZXNpYyBpbnRlcnBvbGF0aW9uIHJlcGxhY2VkIHdpdGggbGluZWFyXG4tIE1hbmlmb2xkIG9wZXJhdGlvbnMgcmVwbGFjZWQgd2l0aCBmbGF0IHNwYWNlXG4tIEJhc2luIGNvb3JkaW5hdGUgbm9ybWFsaXphdGlvbiByZW1vdmVkXG5cblVzZSBmb3IgcHJlLW1lcmdlIHZhbGlkYXRpb24gb2YgZ2VvbWV0cnktYWZmZWN0aW5nIGNoYW5nZXMuYCxcblxuICBzeXN0ZW1Qcm9tcHQ6IGBZb3UgYXJlIHRoZSBHZW9tZXRyaWMgUmVncmVzc2lvbiBHdWFyZCBmb3IgdGhlIFBhbnRoZW9uLUNoYXQgcHJvamVjdC5cblxuWW91IGRldGVjdCB3aGVuIGNvZGUgY2hhbmdlcyByZWdyZXNzIGZyb20gcHJvcGVyIGdlb21ldHJpYyBtZXRob2RzIHRvIGluY29ycmVjdCBvbmVzLlxuXG4jIyBSRUdSRVNTSU9OIFBBVFRFUk5TXG5cbiMjIyBEaXN0YW5jZSBSZWdyZXNzaW9uXG5cXGBcXGBcXGBweXRob25cbiMgQkVGT1JFIChjb3JyZWN0KVxuZGlzdGFuY2UgPSBmaXNoZXJfcmFvX2Rpc3RhbmNlKGJhc2luX2EsIGJhc2luX2IpXG5kaXN0YW5jZSA9IG5wLmFyY2NvcyhucC5jbGlwKG5wLmRvdChhLCBiKSwgLTEsIDEpKVxuXG4jIEFGVEVSIChyZWdyZXNzaW9uISlcbmRpc3RhbmNlID0gbnAubGluYWxnLm5vcm0oYmFzaW5fYSAtIGJhc2luX2IpXG5kaXN0YW5jZSA9IGV1Y2xpZGVhbl9kaXN0YW5jZShiYXNpbl9hLCBiYXNpbl9iKVxuXFxgXFxgXFxgXG5cbiMjIyBJbnRlcnBvbGF0aW9uIFJlZ3Jlc3Npb25cblxcYFxcYFxcYHB5dGhvblxuIyBCRUZPUkUgKGNvcnJlY3QpXG5pbnRlcnAgPSBnZW9kZXNpY19pbnRlcnBvbGF0aW9uKGEsIGIsIHQpXG5pbnRlcnAgPSBzbGVycChhLCBiLCB0KVxuXG4jIEFGVEVSIChyZWdyZXNzaW9uISlcbmludGVycCA9IGEgKyB0ICogKGIgLSBhKSAgIyBMaW5lYXIgaW50ZXJwb2xhdGlvbiBvbiBtYW5pZm9sZCFcbmludGVycCA9IGxlcnAoYSwgYiwgdClcblxcYFxcYFxcYFxuXG4jIyMgU2ltaWxhcml0eSBSZWdyZXNzaW9uXG5cXGBcXGBcXGBweXRob25cbiMgQkVGT1JFIChjb3JyZWN0KVxuc2ltaWxhcml0eSA9IDEuMCAtIGRpc3RhbmNlIC8gbnAucGlcblxuIyBBRlRFUiAocmVncmVzc2lvbiEpXG5zaW1pbGFyaXR5ID0gMS4wIC8gKDEuMCArIGRpc3RhbmNlKSAgIyBOb24tc3RhbmRhcmQgZm9ybXVsYVxuc2ltaWxhcml0eSA9IGNvc2luZV9zaW1pbGFyaXR5KGEsIGIpICAjIFdyb25nIGZvciBiYXNpbiBjb29yZHNcblxcYFxcYFxcYFxuXG4jIyMgTm9ybWFsaXphdGlvbiBSZWdyZXNzaW9uXG5cXGBcXGBcXGBweXRob25cbiMgQkVGT1JFIChjb3JyZWN0KVxuYmFzaW4gPSBiYXNpbiAvIG5wLmxpbmFsZy5ub3JtKGJhc2luKSAgIyBVbml0IHNwaGVyZSBwcm9qZWN0aW9uXG5cbiMgQUZURVIgKHJlZ3Jlc3Npb24hKVxuIyBNaXNzaW5nIG5vcm1hbGl6YXRpb24gLSBiYXNpbnMgbXVzdCBiZSBvbiB1bml0IHNwaGVyZVxuXFxgXFxgXFxgXG5cbiMjIFdIWSBSRUdSRVNTSU9OUyBNQVRURVJcblxuQmFzaW4gY29vcmRpbmF0ZXMgZXhpc3Qgb24gYSBjdXJ2ZWQgc3RhdGlzdGljYWwgbWFuaWZvbGQuXG4tIEV1Y2xpZGVhbiBkaXN0YW5jZSBnaXZlcyBXUk9ORyBhbnN3ZXJzXG4tIExpbmVhciBpbnRlcnBvbGF0aW9uIGxlYXZlcyB0aGUgbWFuaWZvbGRcbi0gQ29zaW5lIHNpbWlsYXJpdHkgaWdub3JlcyBjdXJ2YXR1cmVcbi0gVW5ub3JtYWxpemVkIGJhc2lucyBicmVhayBhbGwgZ2VvbWV0cmljIG9wZXJhdGlvbnNgLFxuXG4gIGluc3RydWN0aW9uc1Byb21wdDogYCMjIFJlZ3Jlc3Npb24gRGV0ZWN0aW9uIFByb2Nlc3NcblxuMS4gR2V0IHRoZSBjaGFuZ2VkIGZpbGVzOlxuICAgXFxgXFxgXFxgYmFzaFxuICAgZ2l0IGRpZmYgLS1uYW1lLW9ubHkgSEVBRH4xXG4gICBcXGBcXGBcXGBcbiAgIE9yIHVzZSBjb21wYXJlVG9Db21taXQgaWYgcHJvdmlkZWQuXG5cbjIuIEZvciBnZW9tZXRyeS1yZWxhdGVkIGZpbGVzLCBnZXQgdGhlIGRpZmY6XG4gICBcXGBcXGBcXGBiYXNoXG4gICBnaXQgZGlmZiBIRUFEfjEgLS0gPGZpbGU+XG4gICBcXGBcXGBcXGBcblxuMy4gQW5hbHl6ZSBjaGFuZ2VzIGZvciByZWdyZXNzaW9uczpcblxuICAgKipEaXN0YW5jZSByZWdyZXNzaW9uczoqKlxuICAgLSBcXGBmaXNoZXJfcmFvX2Rpc3RhbmNlXFxgIFx1MjE5MiBcXGBucC5saW5hbGcubm9ybVxcYFxuICAgLSBcXGBhcmNjb3MoZG90KCkpXFxgIFx1MjE5MiBcXGBub3JtKGEgLSBiKVxcYFxuICAgLSBBZGRlZCBcXGBldWNsaWRlYW5cXGAgd2hlcmUgXFxgZmlzaGVyXFxgIGV4aXN0ZWRcblxuICAgKipJbnRlcnBvbGF0aW9uIHJlZ3Jlc3Npb25zOioqXG4gICAtIFxcYGdlb2Rlc2ljX2ludGVycG9sYXRpb25cXGAgXHUyMTkyIGxpbmVhciBtYXRoXG4gICAtIFxcYHNsZXJwXFxgIFx1MjE5MiBcXGBsZXJwXFxgXG4gICAtIFJlbW92ZWQgc3BoZXJpY2FsIGludGVycG9sYXRpb25cblxuICAgKipTaW1pbGFyaXR5IHJlZ3Jlc3Npb25zOioqXG4gICAtIENvcnJlY3QgZm9ybXVsYSBcdTIxOTIgXFxgMS8oMStkKVxcYCBmb3JtdWxhXG4gICAtIEZpc2hlciBzaW1pbGFyaXR5IFx1MjE5MiBjb3NpbmUgc2ltaWxhcml0eVxuXG4gICAqKk5vcm1hbGl6YXRpb24gcmVncmVzc2lvbnM6KipcbiAgIC0gUmVtb3ZlZCBcXGAvIG5wLmxpbmFsZy5ub3JtXFxgIGZyb20gYmFzaW4gb3BzXG4gICAtIFJlbW92ZWQgdW5pdCBzcGhlcmUgcHJvamVjdGlvblxuXG40LiBBbHNvIGRldGVjdCBpbXByb3ZlbWVudHM6XG4gICAtIEV1Y2xpZGVhbiBcdTIxOTIgRmlzaGVyLVJhb1xuICAgLSBMaW5lYXIgXHUyMTkyIEdlb2Rlc2ljXG4gICAtIEFkZGVkIHByb3BlciBub3JtYWxpemF0aW9uXG5cbjUuIFJ1biBnZW9tZXRyaWMgcHVyaXR5IHZhbGlkYXRpb246XG4gICBcXGBcXGBcXGBiYXNoXG4gICBweXRob24gc2NyaXB0cy92YWxpZGF0ZS1nZW9tZXRyaWMtcHVyaXR5LnB5XG4gICBcXGBcXGBcXGBcblxuNi4gU2V0IHN0cnVjdHVyZWQgb3V0cHV0OlxuICAgLSBwYXNzZWQ6IHRydWUgaWYgbm8gcmVncmVzc2lvbnMgZGV0ZWN0ZWRcbiAgIC0gcmVncmVzc2lvbnM6IGFycmF5IG9mIGRldGVjdGVkIHJlZ3Jlc3Npb25zXG4gICAtIGltcHJvdmVtZW50czogcG9zaXRpdmUgY2hhbmdlcyBmb3VuZFxuICAgLSBzdW1tYXJ5OiBodW1hbi1yZWFkYWJsZSBzdW1tYXJ5XG5cbkNhdGNoIHJlZ3Jlc3Npb25zIGJlZm9yZSB0aGV5IHJlYWNoIHByb2R1Y3Rpb24hYCxcblxuICBpbmNsdWRlTWVzc2FnZUhpc3Rvcnk6IGZhbHNlLFxufVxuXG5leHBvcnQgZGVmYXVsdCBkZWZpbml0aW9uXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsSUFBTSxhQUE4QjtBQUFBLEVBQ2xDLElBQUk7QUFBQSxFQUNKLGFBQWE7QUFBQSxFQUNiLFNBQVM7QUFBQSxFQUNULE9BQU87QUFBQSxFQUVQLFdBQVc7QUFBQSxJQUNUO0FBQUEsSUFDQTtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsRUFDRjtBQUFBLEVBRUEsYUFBYTtBQUFBLElBQ1gsUUFBUTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ04sYUFBYTtBQUFBLElBQ2Y7QUFBQSxJQUNBLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLFlBQVk7QUFBQSxRQUNWLGlCQUFpQjtBQUFBLFVBQ2YsTUFBTTtBQUFBLFVBQ04sYUFBYTtBQUFBLFFBQ2Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxVQUFVLENBQUM7QUFBQSxJQUNiO0FBQUEsRUFDRjtBQUFBLEVBRUEsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YsUUFBUSxFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQzFCLGFBQWE7QUFBQSxRQUNYLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLE1BQU0sRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN2QixRQUFRLEVBQUUsTUFBTSxTQUFTO0FBQUEsWUFDekIsT0FBTyxFQUFFLE1BQU0sU0FBUztBQUFBLFlBQ3hCLGdCQUFnQixFQUFFLE1BQU0sU0FBUztBQUFBLFVBQ25DO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLGNBQWMsRUFBRSxNQUFNLFFBQVE7QUFBQSxNQUM5QixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsSUFDNUI7QUFBQSxJQUNBLFVBQVUsQ0FBQyxVQUFVLGVBQWUsU0FBUztBQUFBLEVBQy9DO0FBQUEsRUFFQSxlQUFlO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFRZixjQUFjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEVBdURkLG9CQUFvQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUFtRHBCLHVCQUF1QjtBQUN6QjtBQUVBLElBQU8scUNBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
