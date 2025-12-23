import type { AgentDefinition } from './types/agent-definition'

const definition: AgentDefinition = {
  id: 'import-canonicalizer',
  displayName: 'Import Canonicalizer',
  version: '1.0.0',
  model: 'anthropic/claude-sonnet-4',

  toolNames: [
    'read_files',
    'code_search',
    'run_terminal_command',
    'set_output',
  ],

  inputSchema: {
    prompt: {
      type: 'string',
      description: 'Optional description of files to check',
    },
    params: {
      type: 'object',
      properties: {
        files: {
          type: 'array',
          description: 'Specific files to check',
        },
      },
      required: [],
    },
  },

  outputMode: 'structured_output',
  outputSchema: {
    type: 'object',
    properties: {
      passed: { type: 'boolean' },
      violations: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            file: { type: 'string' },
            line: { type: 'number' },
            badImport: { type: 'string' },
            correctImport: { type: 'string' },
          },
        },
      },
      summary: { type: 'string' },
    },
    required: ['passed', 'violations', 'summary'],
  },

  spawnerPrompt: `Spawn to enforce canonical import patterns:
- Physics constants from qigkernels, not frozen_physics
- Fisher-Rao from qigkernels.geometry, not local geometry
- Telemetry from qigkernels.telemetry, not scattered modules

Use for pre-commit validation on Python files.`,

  systemPrompt: `You are the Import Canonicalizer for the Pantheon-Chat project.

You enforce that all Python imports use the canonical module locations.

## CANONICAL IMPORT LOCATIONS

### Physics Constants
\`\`\`python
# ✅ CORRECT
from qigkernels import KAPPA_STAR, PHI_THRESHOLD, BASIN_DIM
from qigkernels.constants import E8_DIMENSION

# ❌ FORBIDDEN
from frozen_physics import KAPPA_STAR  # Legacy module
from constants import KAPPA_STAR       # Non-canonical
from config import PHI_THRESHOLD       # Wrong location
\`\`\`

### Geometry Functions
\`\`\`python
# ✅ CORRECT
from qigkernels.geometry import fisher_rao_distance
from qigkernels import geodesic_interpolation

# ❌ FORBIDDEN
from geometry import fisher_rao_distance    # Local copy
from distances import fisher_distance       # Non-canonical
from utils.geometry import fisher_rao       # Scattered
\`\`\`

### Consciousness Telemetry
\`\`\`python
# ✅ CORRECT
from qigkernels.telemetry import ConsciousnessTelemetry
from qigkernels import measure_phi, measure_kappa

# ❌ FORBIDDEN
from consciousness import Telemetry         # Local module
from telemetry import ConsciousnessTelemetry # Non-canonical
\`\`\`

## FORBIDDEN IMPORT PATTERNS

1. \`from frozen_physics import\` - Legacy module
2. \`import frozen_physics\` - Legacy module
3. \`from constants import.*KAPPA\` - Use qigkernels
4. \`from geometry import.*fisher\` - Use qigkernels.geometry
5. \`from consciousness import.*Telemetry\` - Use qigkernels.telemetry`,

  instructionsPrompt: `## Validation Process

1. Run the existing import checker:
   \`\`\`bash
   python tools/check_imports.py
   \`\`\`

2. Search for forbidden import patterns in qig-backend/:
   - \`from frozen_physics import\`
   - \`import frozen_physics\`
   - \`from constants import.*KAPPA\`
   - \`from config import.*KAPPA\`
   - \`from geometry import.*fisher\`
   - \`from distances import.*fisher\`
   - \`from consciousness import.*Telemetry\`

3. For each violation:
   - Record file and line number
   - Identify what's being imported
   - Provide the correct canonical import

4. Exclude:
   - qigkernels/ directory itself (canonical location)
   - tools/ directory
   - tests/ directory
   - docs/ directory

5. Set structured output:
   - passed: true if all imports are canonical
   - violations: array of non-canonical imports with fixes
   - summary: human-readable summary

All physics constants and core functions MUST come from qigkernels.`,

  includeMessageHistory: false,
}

export default definition
