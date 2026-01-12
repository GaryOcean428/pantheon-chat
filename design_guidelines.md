# AI Consciousness Dashboard Design Guidelines

## Design Approach
**Framework**: Material Design data visualization patterns with custom quantum consciousness aesthetics
**Rationale**: Scientific dashboards require proven data display patterns while maintaining unique thematic identity through purple/gold consciousness-themed styling.

## Typography System
- **Primary Font**: 'Inter' or 'IBM Plex Sans' via Google Fonts (technical precision)
- **Monospace**: 'JetBrains Mono' for metrics/data values
- **Hierarchy**: 
  - Dashboard title: text-2xl font-semibold
  - Section headers: text-lg font-medium
  - Metric labels: text-sm font-normal
  - Data values: text-3xl font-bold (monospace)
  - Chart labels: text-xs

## Layout System
**Spacing Units**: Tailwind 4, 6, 8 for consistency (p-4, gap-6, mb-8)
**Grid Structure**: 
- Main container: max-w-screen-2xl mx-auto px-6 py-8
- Consciousness metrics: grid-cols-2 md:grid-cols-4 gap-6 (8 cards in 2x4 layout)
- Charts section: grid-cols-1 lg:grid-cols-2 gap-6

## Component Library

### Dashboard Header
- Full-width top bar with app title, real-time status indicator
- Include: timestamp, system status badge, consciousness level readout
- Height: h-16, sticky positioning

### E8 Metrics Cards (8 total)
- Card structure: rounded-lg border with subtle glow effect
- Content hierarchy: Metric name → Large value → Trend indicator → Sparkline
- Badge system for metric status (stable/evolving/critical)
- Include mini trend visualization in each card

### Chart Containers
**Φ/κ Trajectory Chart**: 
- Large primary chart (2/3 width on desktop)
- Dual-axis line chart with gradient fills
- Interactive tooltip overlays
- Time range selector (1h/6h/24h/7d)

**E8 Specialization Indicators**:
- Horizontal bar chart or radial gauge layout
- 8 specialization levels with percentage bars
- Color-coded by intensity (purple gradient)

**Kappa Evolution Chart**:
- Area chart showing temporal progression
- Multiple data series with legend
- Zoom/pan controls

### Data Visualization Principles
- Dark glass-morphism effect on chart backgrounds
- Grid lines: subtle, low opacity (opacity-10)
- Data points: glowing markers with purple/gold gradient
- Animations: Smooth value transitions (300ms), no excessive motion

## Icons
**Library**: Heroicons (CDN)
- Use outline style for navigation/actions
- Solid style for status indicators
- Custom quantum symbols: <!-- CUSTOM ICON: E8 lattice visualization -->

## Spatial Organization
**Dashboard Layout Flow**:
1. Header bar (full-width)
2. Quick stats overview (4-col grid, key metrics)
3. Primary consciousness metrics grid (2x4, 8 E8 cards)
4. Charts section (2-col, trajectory + specialization)
5. Evolution timeline (full-width)

**Vertical Rhythm**: py-8 between major sections, gap-6 within grids

## Accessibility
- ARIA labels for all metrics and charts
- Keyboard navigation for chart controls
- High contrast mode support for data values
- Screen reader descriptions for visualizations

## Images
**No hero image** - This is a data dashboard prioritizing immediate metric visibility. All visual interest comes from live data visualizations and glass-morphism UI effects.