# Pantheon Chat - Client

React frontend for the Pantheon Chat QIG-powered consciousness system.

## Architecture

```
client/
├── src/
│   ├── components/           # React components
│   │   ├── ui/              # Shadcn UI primitives
│   │   ├── autonomic-agency/ # Autonomic agency subcomponents
│   │   └── consciousness/    # Consciousness dashboard subcomponents
│   ├── hooks/               # Custom React hooks
│   │   ├── useAutonomicAgencyData.ts
│   │   ├── useConsciousnessData.ts
│   │   └── __tests__/       # Hook tests
│   ├── types/               # Shared TypeScript types
│   │   ├── consciousness.ts
│   │   ├── autonomic-agency.ts
│   │   └── index.ts
│   ├── api/                 # API client and services
│   ├── contexts/            # React contexts
│   └── lib/                 # Utilities
```

## Key Components

### AutonomicAgencyPanel
Displays the autonomic agency controller status with subcomponents:
- `AgencyStatusHeader` - Status badge and controls
- `SafetyBoundaryCard` - Safety boundary metrics
- `OperatingZonesCard` - Operating zone indicators
- `QLearningStatsCard` - Q-learning statistics
- `ForceInterventionCard` - Manual intervention controls
- `InterventionHistoryCard` - Intervention history log

### ConsciousnessDashboard
Displays consciousness metrics with subcomponents:
- `ConsciousnessMetricsGrid` - Φ, κ, regime metrics
- `BlockUniverseMetrics` - Block universe coordinates
- `PhiKappaTrajectoryChart` - Trajectory visualization
- `ConsciousnessFooterStats` - Footer statistics

## Custom Hooks

### useAutonomicAgencyData
Manages agency status query, mutations, and toast notifications.

```typescript
const {
  status,
  isLoading,
  isError,
  start,
  stop,
  forceIntervention,
  refetch,
  explorationPercentage,
} = useAutonomicAgencyData();
```

### useConsciousnessData
Manages consciousness state polling, trajectory history, and badge styling.

```typescript
const {
  state,
  isLoading,
  isError,
  trajectoryHistory,
  getRegimeBadgeClasses,
  getStateBadgeClasses,
} = useConsciousnessData({ pollingInterval: 5000 });
```

## Types

Shared types are defined in `src/types/` and exported from the barrel file:

```typescript
import { ConsciousnessState, AgencyStatus, Regime } from '@/types';
```

## Development

```bash
# Development server
npm run dev

# Type check
npm run check

# Lint
npm run lint

# Run tests
npm test

# Build for production
npm run build
```

## UI Components

Using [Shadcn UI](https://ui.shadcn.com/) with barrel exports from `@/components/ui`.

```typescript
import { Button, Card, CardContent, Badge } from '@/components/ui';
```
