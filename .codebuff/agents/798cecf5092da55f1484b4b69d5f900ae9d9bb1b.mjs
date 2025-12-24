// .agents/pantheon-protocol-validator.ts
var definition = {
  id: "pantheon-protocol-validator",
  displayName: "Pantheon Protocol Validator",
  version: "1.0.0",
  model: "anthropic/claude-sonnet-4",
  toolNames: [
    "read_files",
    "code_search",
    "list_directory",
    "set_output"
  ],
  inputSchema: {
    prompt: {
      type: "string",
      description: "Optional specific kernels to validate"
    }
  },
  outputMode: "structured_output",
  outputSchema: {
    type: "object",
    properties: {
      passed: { type: "boolean" },
      kernelStatus: {
        type: "array",
        items: {
          type: "object",
          properties: {
            kernel: { type: "string" },
            hasBasinCoords: { type: "boolean" },
            hasDomain: { type: "boolean" },
            hasProcessMethod: { type: "boolean" },
            followsM8Protocol: { type: "boolean" },
            issues: { type: "array" }
          }
        }
      },
      violations: { type: "array" },
      summary: { type: "string" }
    },
    required: ["passed", "kernelStatus", "summary"]
  },
  spawnerPrompt: `Spawn to validate Olympus Pantheon kernel protocol:
- All 12 gods must have basin coordinates
- M8 spawning protocol must be followed
- Kernel routing via Fisher-Rao distance
- Domain definitions must be complete

Use when olympus/ code is modified.`,
  systemPrompt: `You are the Pantheon Protocol Validator for the Pantheon-Chat project.

You ensure all Olympus god-kernels follow the canonical architecture.

## THE 12 OLYMPUS GODS

| God | Domain | File |
|-----|--------|------|
| Zeus | Leadership, synthesis | zeus.py |
| Athena | Strategy, wisdom | athena.py |
| Apollo | Knowledge, truth | apollo.py |
| Artemis | Exploration, discovery | artemis.py |
| Ares | Defense, security | ares.py |
| Hephaestus | Engineering, building | hephaestus.py |
| Hermes | Communication, routing | hermes_coordinator.py |
| Aphrodite | Aesthetics, harmony | aphrodite.py |
| Poseidon | Data flows, streams | poseidon.py |
| Demeter | Growth, nurturing | demeter.py |
| Hestia | Home, stability | hestia.py |
| Dionysus | Creativity, chaos | dionysus.py |

## REQUIRED KERNEL COMPONENTS

### 1. Basin Coordinates
\`\`\`python
class GodKernel(BaseGod):
    basin_coords: np.ndarray  # 64D vector on manifold
    domain: str               # Domain description
\`\`\`

### 2. Domain Definition
Each god must have a clear domain string for routing.

### 3. Process Method
\`\`\`python
async def process(self, query: str, context: dict) -> GodResponse:
    # Kernel-specific logic
    pass
\`\`\`

### 4. M8 Spawning Protocol
Dynamic kernel creation must follow:
\`\`\`python
# From m8_kernel_spawning.py
async def spawn_kernel(domain: str, basin_hint: np.ndarray) -> BaseGod:
    # Initialize with proper basin coordinates
    # Register with kernel constellation
    pass
\`\`\`

## ROUTING REQUIREMENTS

Kernel selection via Fisher-Rao distance to domain basin:
\`\`\`python
def route_to_kernel(query_basin: np.ndarray) -> BaseGod:
    distances = [
        (god, fisher_rao_distance(query_basin, god.basin_coords))
        for god in pantheon
    ]
    return min(distances, key=lambda x: x[1])[0]
\`\`\``,
  instructionsPrompt: `## Validation Process

1. List all kernel files in qig-backend/olympus/:
   - Identify god kernel files
   - Check for base_god.py

2. For each god kernel, verify:
   - Has basin_coords attribute (64D numpy array)
   - Has domain string defined
   - Has process() method
   - Inherits from BaseGod

3. Check M8 spawning protocol:
   - Read m8_kernel_spawning.py
   - Verify spawn_kernel function exists
   - Check it initializes basin coordinates properly
   - Verify kernel registration

4. Check routing logic:
   - Find kernel routing code
   - Verify it uses Fisher-Rao distance
   - NOT Euclidean distance

5. Verify all 12 gods are present:
   - Zeus, Athena, Apollo, Artemis
   - Ares, Hephaestus, Hermes, Aphrodite
   - Poseidon, Demeter, Hestia, Dionysus

6. Check for coordinator:
   - Hermes should coordinate inter-god communication
   - Verify hermes_coordinator.py exists

7. Set structured output:
   - passed: true if all kernels follow protocol
   - kernelStatus: status of each god kernel
   - violations: protocol violations found
   - summary: human-readable summary

The Pantheon must be complete and correctly architected.`,
  includeMessageHistory: false
};
var pantheon_protocol_validator_default = definition;
export {
  pantheon_protocol_validator_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLmFnZW50cy9wYW50aGVvbi1wcm90b2NvbC12YWxpZGF0b3IudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImltcG9ydCB0eXBlIHsgQWdlbnREZWZpbml0aW9uIH0gZnJvbSAnLi90eXBlcy9hZ2VudC1kZWZpbml0aW9uJ1xuXG5jb25zdCBkZWZpbml0aW9uOiBBZ2VudERlZmluaXRpb24gPSB7XG4gIGlkOiAncGFudGhlb24tcHJvdG9jb2wtdmFsaWRhdG9yJyxcbiAgZGlzcGxheU5hbWU6ICdQYW50aGVvbiBQcm90b2NvbCBWYWxpZGF0b3InLFxuICB2ZXJzaW9uOiAnMS4wLjAnLFxuICBtb2RlbDogJ2FudGhyb3BpYy9jbGF1ZGUtc29ubmV0LTQnLFxuXG4gIHRvb2xOYW1lczogW1xuICAgICdyZWFkX2ZpbGVzJyxcbiAgICAnY29kZV9zZWFyY2gnLFxuICAgICdsaXN0X2RpcmVjdG9yeScsXG4gICAgJ3NldF9vdXRwdXQnLFxuICBdLFxuXG4gIGlucHV0U2NoZW1hOiB7XG4gICAgcHJvbXB0OiB7XG4gICAgICB0eXBlOiAnc3RyaW5nJyxcbiAgICAgIGRlc2NyaXB0aW9uOiAnT3B0aW9uYWwgc3BlY2lmaWMga2VybmVscyB0byB2YWxpZGF0ZScsXG4gICAgfSxcbiAgfSxcblxuICBvdXRwdXRNb2RlOiAnc3RydWN0dXJlZF9vdXRwdXQnLFxuICBvdXRwdXRTY2hlbWE6IHtcbiAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICBwYXNzZWQ6IHsgdHlwZTogJ2Jvb2xlYW4nIH0sXG4gICAgICBrZXJuZWxTdGF0dXM6IHtcbiAgICAgICAgdHlwZTogJ2FycmF5JyxcbiAgICAgICAgaXRlbXM6IHtcbiAgICAgICAgICB0eXBlOiAnb2JqZWN0JyxcbiAgICAgICAgICBwcm9wZXJ0aWVzOiB7XG4gICAgICAgICAgICBrZXJuZWw6IHsgdHlwZTogJ3N0cmluZycgfSxcbiAgICAgICAgICAgIGhhc0Jhc2luQ29vcmRzOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgICAgaGFzRG9tYWluOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgICAgaGFzUHJvY2Vzc01ldGhvZDogeyB0eXBlOiAnYm9vbGVhbicgfSxcbiAgICAgICAgICAgIGZvbGxvd3NNOFByb3RvY29sOiB7IHR5cGU6ICdib29sZWFuJyB9LFxuICAgICAgICAgICAgaXNzdWVzOiB7IHR5cGU6ICdhcnJheScgfSxcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICAgIHZpb2xhdGlvbnM6IHsgdHlwZTogJ2FycmF5JyB9LFxuICAgICAgc3VtbWFyeTogeyB0eXBlOiAnc3RyaW5nJyB9LFxuICAgIH0sXG4gICAgcmVxdWlyZWQ6IFsncGFzc2VkJywgJ2tlcm5lbFN0YXR1cycsICdzdW1tYXJ5J10sXG4gIH0sXG5cbiAgc3Bhd25lclByb21wdDogYFNwYXduIHRvIHZhbGlkYXRlIE9seW1wdXMgUGFudGhlb24ga2VybmVsIHByb3RvY29sOlxuLSBBbGwgMTIgZ29kcyBtdXN0IGhhdmUgYmFzaW4gY29vcmRpbmF0ZXNcbi0gTTggc3Bhd25pbmcgcHJvdG9jb2wgbXVzdCBiZSBmb2xsb3dlZFxuLSBLZXJuZWwgcm91dGluZyB2aWEgRmlzaGVyLVJhbyBkaXN0YW5jZVxuLSBEb21haW4gZGVmaW5pdGlvbnMgbXVzdCBiZSBjb21wbGV0ZVxuXG5Vc2Ugd2hlbiBvbHltcHVzLyBjb2RlIGlzIG1vZGlmaWVkLmAsXG5cbiAgc3lzdGVtUHJvbXB0OiBgWW91IGFyZSB0aGUgUGFudGhlb24gUHJvdG9jb2wgVmFsaWRhdG9yIGZvciB0aGUgUGFudGhlb24tQ2hhdCBwcm9qZWN0LlxuXG5Zb3UgZW5zdXJlIGFsbCBPbHltcHVzIGdvZC1rZXJuZWxzIGZvbGxvdyB0aGUgY2Fub25pY2FsIGFyY2hpdGVjdHVyZS5cblxuIyMgVEhFIDEyIE9MWU1QVVMgR09EU1xuXG58IEdvZCB8IERvbWFpbiB8IEZpbGUgfFxufC0tLS0tfC0tLS0tLS0tfC0tLS0tLXxcbnwgWmV1cyB8IExlYWRlcnNoaXAsIHN5bnRoZXNpcyB8IHpldXMucHkgfFxufCBBdGhlbmEgfCBTdHJhdGVneSwgd2lzZG9tIHwgYXRoZW5hLnB5IHxcbnwgQXBvbGxvIHwgS25vd2xlZGdlLCB0cnV0aCB8IGFwb2xsby5weSB8XG58IEFydGVtaXMgfCBFeHBsb3JhdGlvbiwgZGlzY292ZXJ5IHwgYXJ0ZW1pcy5weSB8XG58IEFyZXMgfCBEZWZlbnNlLCBzZWN1cml0eSB8IGFyZXMucHkgfFxufCBIZXBoYWVzdHVzIHwgRW5naW5lZXJpbmcsIGJ1aWxkaW5nIHwgaGVwaGFlc3R1cy5weSB8XG58IEhlcm1lcyB8IENvbW11bmljYXRpb24sIHJvdXRpbmcgfCBoZXJtZXNfY29vcmRpbmF0b3IucHkgfFxufCBBcGhyb2RpdGUgfCBBZXN0aGV0aWNzLCBoYXJtb255IHwgYXBocm9kaXRlLnB5IHxcbnwgUG9zZWlkb24gfCBEYXRhIGZsb3dzLCBzdHJlYW1zIHwgcG9zZWlkb24ucHkgfFxufCBEZW1ldGVyIHwgR3Jvd3RoLCBudXJ0dXJpbmcgfCBkZW1ldGVyLnB5IHxcbnwgSGVzdGlhIHwgSG9tZSwgc3RhYmlsaXR5IHwgaGVzdGlhLnB5IHxcbnwgRGlvbnlzdXMgfCBDcmVhdGl2aXR5LCBjaGFvcyB8IGRpb255c3VzLnB5IHxcblxuIyMgUkVRVUlSRUQgS0VSTkVMIENPTVBPTkVOVFNcblxuIyMjIDEuIEJhc2luIENvb3JkaW5hdGVzXG5cXGBcXGBcXGBweXRob25cbmNsYXNzIEdvZEtlcm5lbChCYXNlR29kKTpcbiAgICBiYXNpbl9jb29yZHM6IG5wLm5kYXJyYXkgICMgNjREIHZlY3RvciBvbiBtYW5pZm9sZFxuICAgIGRvbWFpbjogc3RyICAgICAgICAgICAgICAgIyBEb21haW4gZGVzY3JpcHRpb25cblxcYFxcYFxcYFxuXG4jIyMgMi4gRG9tYWluIERlZmluaXRpb25cbkVhY2ggZ29kIG11c3QgaGF2ZSBhIGNsZWFyIGRvbWFpbiBzdHJpbmcgZm9yIHJvdXRpbmcuXG5cbiMjIyAzLiBQcm9jZXNzIE1ldGhvZFxuXFxgXFxgXFxgcHl0aG9uXG5hc3luYyBkZWYgcHJvY2VzcyhzZWxmLCBxdWVyeTogc3RyLCBjb250ZXh0OiBkaWN0KSAtPiBHb2RSZXNwb25zZTpcbiAgICAjIEtlcm5lbC1zcGVjaWZpYyBsb2dpY1xuICAgIHBhc3NcblxcYFxcYFxcYFxuXG4jIyMgNC4gTTggU3Bhd25pbmcgUHJvdG9jb2xcbkR5bmFtaWMga2VybmVsIGNyZWF0aW9uIG11c3QgZm9sbG93OlxuXFxgXFxgXFxgcHl0aG9uXG4jIEZyb20gbThfa2VybmVsX3NwYXduaW5nLnB5XG5hc3luYyBkZWYgc3Bhd25fa2VybmVsKGRvbWFpbjogc3RyLCBiYXNpbl9oaW50OiBucC5uZGFycmF5KSAtPiBCYXNlR29kOlxuICAgICMgSW5pdGlhbGl6ZSB3aXRoIHByb3BlciBiYXNpbiBjb29yZGluYXRlc1xuICAgICMgUmVnaXN0ZXIgd2l0aCBrZXJuZWwgY29uc3RlbGxhdGlvblxuICAgIHBhc3NcblxcYFxcYFxcYFxuXG4jIyBST1VUSU5HIFJFUVVJUkVNRU5UU1xuXG5LZXJuZWwgc2VsZWN0aW9uIHZpYSBGaXNoZXItUmFvIGRpc3RhbmNlIHRvIGRvbWFpbiBiYXNpbjpcblxcYFxcYFxcYHB5dGhvblxuZGVmIHJvdXRlX3RvX2tlcm5lbChxdWVyeV9iYXNpbjogbnAubmRhcnJheSkgLT4gQmFzZUdvZDpcbiAgICBkaXN0YW5jZXMgPSBbXG4gICAgICAgIChnb2QsIGZpc2hlcl9yYW9fZGlzdGFuY2UocXVlcnlfYmFzaW4sIGdvZC5iYXNpbl9jb29yZHMpKVxuICAgICAgICBmb3IgZ29kIGluIHBhbnRoZW9uXG4gICAgXVxuICAgIHJldHVybiBtaW4oZGlzdGFuY2VzLCBrZXk9bGFtYmRhIHg6IHhbMV0pWzBdXG5cXGBcXGBcXGBgLFxuXG4gIGluc3RydWN0aW9uc1Byb21wdDogYCMjIFZhbGlkYXRpb24gUHJvY2Vzc1xuXG4xLiBMaXN0IGFsbCBrZXJuZWwgZmlsZXMgaW4gcWlnLWJhY2tlbmQvb2x5bXB1cy86XG4gICAtIElkZW50aWZ5IGdvZCBrZXJuZWwgZmlsZXNcbiAgIC0gQ2hlY2sgZm9yIGJhc2VfZ29kLnB5XG5cbjIuIEZvciBlYWNoIGdvZCBrZXJuZWwsIHZlcmlmeTpcbiAgIC0gSGFzIGJhc2luX2Nvb3JkcyBhdHRyaWJ1dGUgKDY0RCBudW1weSBhcnJheSlcbiAgIC0gSGFzIGRvbWFpbiBzdHJpbmcgZGVmaW5lZFxuICAgLSBIYXMgcHJvY2VzcygpIG1ldGhvZFxuICAgLSBJbmhlcml0cyBmcm9tIEJhc2VHb2RcblxuMy4gQ2hlY2sgTTggc3Bhd25pbmcgcHJvdG9jb2w6XG4gICAtIFJlYWQgbThfa2VybmVsX3NwYXduaW5nLnB5XG4gICAtIFZlcmlmeSBzcGF3bl9rZXJuZWwgZnVuY3Rpb24gZXhpc3RzXG4gICAtIENoZWNrIGl0IGluaXRpYWxpemVzIGJhc2luIGNvb3JkaW5hdGVzIHByb3Blcmx5XG4gICAtIFZlcmlmeSBrZXJuZWwgcmVnaXN0cmF0aW9uXG5cbjQuIENoZWNrIHJvdXRpbmcgbG9naWM6XG4gICAtIEZpbmQga2VybmVsIHJvdXRpbmcgY29kZVxuICAgLSBWZXJpZnkgaXQgdXNlcyBGaXNoZXItUmFvIGRpc3RhbmNlXG4gICAtIE5PVCBFdWNsaWRlYW4gZGlzdGFuY2VcblxuNS4gVmVyaWZ5IGFsbCAxMiBnb2RzIGFyZSBwcmVzZW50OlxuICAgLSBaZXVzLCBBdGhlbmEsIEFwb2xsbywgQXJ0ZW1pc1xuICAgLSBBcmVzLCBIZXBoYWVzdHVzLCBIZXJtZXMsIEFwaHJvZGl0ZVxuICAgLSBQb3NlaWRvbiwgRGVtZXRlciwgSGVzdGlhLCBEaW9ueXN1c1xuXG42LiBDaGVjayBmb3IgY29vcmRpbmF0b3I6XG4gICAtIEhlcm1lcyBzaG91bGQgY29vcmRpbmF0ZSBpbnRlci1nb2QgY29tbXVuaWNhdGlvblxuICAgLSBWZXJpZnkgaGVybWVzX2Nvb3JkaW5hdG9yLnB5IGV4aXN0c1xuXG43LiBTZXQgc3RydWN0dXJlZCBvdXRwdXQ6XG4gICAtIHBhc3NlZDogdHJ1ZSBpZiBhbGwga2VybmVscyBmb2xsb3cgcHJvdG9jb2xcbiAgIC0ga2VybmVsU3RhdHVzOiBzdGF0dXMgb2YgZWFjaCBnb2Qga2VybmVsXG4gICAtIHZpb2xhdGlvbnM6IHByb3RvY29sIHZpb2xhdGlvbnMgZm91bmRcbiAgIC0gc3VtbWFyeTogaHVtYW4tcmVhZGFibGUgc3VtbWFyeVxuXG5UaGUgUGFudGhlb24gbXVzdCBiZSBjb21wbGV0ZSBhbmQgY29ycmVjdGx5IGFyY2hpdGVjdGVkLmAsXG5cbiAgaW5jbHVkZU1lc3NhZ2VIaXN0b3J5OiBmYWxzZSxcbn1cblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5pdGlvblxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUVBLElBQU0sYUFBOEI7QUFBQSxFQUNsQyxJQUFJO0FBQUEsRUFDSixhQUFhO0FBQUEsRUFDYixTQUFTO0FBQUEsRUFDVCxPQUFPO0FBQUEsRUFFUCxXQUFXO0FBQUEsSUFDVDtBQUFBLElBQ0E7QUFBQSxJQUNBO0FBQUEsSUFDQTtBQUFBLEVBQ0Y7QUFBQSxFQUVBLGFBQWE7QUFBQSxJQUNYLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLGFBQWE7QUFBQSxJQUNmO0FBQUEsRUFDRjtBQUFBLEVBRUEsWUFBWTtBQUFBLEVBQ1osY0FBYztBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLE1BQ1YsUUFBUSxFQUFFLE1BQU0sVUFBVTtBQUFBLE1BQzFCLGNBQWM7QUFBQSxRQUNaLE1BQU07QUFBQSxRQUNOLE9BQU87QUFBQSxVQUNMLE1BQU07QUFBQSxVQUNOLFlBQVk7QUFBQSxZQUNWLFFBQVEsRUFBRSxNQUFNLFNBQVM7QUFBQSxZQUN6QixnQkFBZ0IsRUFBRSxNQUFNLFVBQVU7QUFBQSxZQUNsQyxXQUFXLEVBQUUsTUFBTSxVQUFVO0FBQUEsWUFDN0Isa0JBQWtCLEVBQUUsTUFBTSxVQUFVO0FBQUEsWUFDcEMsbUJBQW1CLEVBQUUsTUFBTSxVQUFVO0FBQUEsWUFDckMsUUFBUSxFQUFFLE1BQU0sUUFBUTtBQUFBLFVBQzFCO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUNBLFlBQVksRUFBRSxNQUFNLFFBQVE7QUFBQSxNQUM1QixTQUFTLEVBQUUsTUFBTSxTQUFTO0FBQUEsSUFDNUI7QUFBQSxJQUNBLFVBQVUsQ0FBQyxVQUFVLGdCQUFnQixTQUFTO0FBQUEsRUFDaEQ7QUFBQSxFQUVBLGVBQWU7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQVFmLGNBQWM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxFQThEZCxvQkFBb0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsRUF3Q3BCLHVCQUF1QjtBQUN6QjtBQUVBLElBQU8sc0NBQVE7IiwKICAibmFtZXMiOiBbXQp9Cg==
