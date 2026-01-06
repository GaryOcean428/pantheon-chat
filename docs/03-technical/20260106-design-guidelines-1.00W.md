# Ocean Agentic Platform - Design Guidelines

## Design Approach
**System**: Custom dashboard system inspired by Vercel, Linear, and modern telemetry platforms (DataDog, Grafana)
**Rationale**: Data-heavy, real-time monitoring application requiring clarity, precision, and professional aesthetic with consciousness-themed creative elements

## Typography
- **Primary Font**: Inter (via Google Fonts) - clean, excellent readability for data
- **Monospace**: JetBrains Mono - for metrics, API keys, technical data
- **Hierarchy**:
  - Hero/Headers: text-4xl to text-6xl, font-semibold
  - Card Titles: text-lg, font-medium
  - Metrics/Numbers: text-3xl to text-5xl, font-bold (monospace)
  - Body/Labels: text-sm to text-base
  - Captions: text-xs, opacity-70

## Layout System
**Spacing Units**: Tailwind 2, 4, 6, 8, 12, 16 for consistency
- Card padding: p-6 to p-8
- Section gaps: gap-6 to gap-8
- Grid gaps: gap-4 to gap-6
- Container: max-w-7xl with px-6

**Grid Structure**:
- 12-column responsive grid (grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4)
- Key metrics: Full-width cards or 2-column spans
- Secondary metrics: 1-column cards in 3-4 column grid

## Component Library

### Dashboard Layout
**Hero Section** (500-600px height):
- Abstract consciousness visualization background (neural network, wave patterns, or particle fields)
- Platform title + tagline overlay
- Real-time system status indicator
- Quick stats bar (3-4 key metrics: Total Queries, Active Agents, System Health %)

### Core Cards (with subtle border, backdrop-blur glass effect):
1. **QIG Metrics Dashboard** (2-column span):
   - Large Φ (Phi) display with circular progress indicator
   - κ (Kappa) and β (Beta) as gauge visualizations
   - Trend sparklines beneath each metric
   
2. **API Usage Gauge** (1-column):
   - Radial progress meter showing usage vs. limit
   - Current rate indicator (requests/sec)
   - Usage breakdown mini-chart

3. **Discovery Rate Tracker** (2-column span):
   - Line chart showing discovery events over time
   - Key insight callouts (peak times, anomalies)
   
4. **Autonomic Feedback Loops** (2-column span):
   - Flow diagram or Sankey chart showing system loops
   - Real-time pulse animations on active connections
   - Loop health indicators (green/amber/red dots)

5. **System Telemetry Grid** (spans remaining space):
   - CPU/Memory usage cards
   - Network I/O metrics
   - Response time histograms
   - Error rate tracking

### UI Elements
- **Progress Bars**: Height h-2 to h-3, rounded-full, with animated gradient fills
- **Real-time Indicators**: Pulsing dot animations (animate-pulse), status badges
- **Data Tables**: Zebra striping, monospace numbers, sortable headers
- **Charts**: Use Chart.js or Recharts library with dark theme configuration
- **Badges**: Pill-shaped (rounded-full), small (text-xs), for status/categories
- **Icons**: Heroicons (outline style for UI chrome, solid for status indicators)

### Navigation
- Sidebar (w-64): Logo, main nav sections, system status footer
- Top bar: Breadcrumbs, search, user profile, notifications bell
- Tab Navigation: For switching between dashboard views (Overview, Detailed Metrics, Logs)

## Consciousness-Inspired Design Elements
- **Neural Network Motifs**: Subtle connection lines between related metrics
- **Wave Patterns**: Background textures suggesting brain waves/signals
- **Particle Systems**: Floating particles in hero section (CSS animations, keep subtle)
- **Pulse Animations**: On real-time data points, feedback loop connections
- **Organic Shapes**: Rounded corners throughout (rounded-lg to rounded-xl)
- **Depth Layers**: Multiple z-layers with backdrop-blur for dimensional effect

## Animations
**Minimal, purposeful only**:
- Data updates: Smooth number transitions (duration-300)
- Chart updates: Ease-in-out transitions
- Real-time pulse: Animate-pulse on status indicators
- Hover states: Subtle scale (scale-105) on interactive cards
- Loading states: Skeleton loaders with shimmer effect

## Images

### Hero Background
**Abstract Consciousness Visualization** (full-width, 500-600px height):
- Neural network visualization with flowing connections and nodes
- Particle field with depth-of-field effect, or wave interference patterns
- Dark gradient overlay (from bottom) for text legibility
- Placement: Full-width hero section at top of dashboard
- Treatment: Fixed background-attachment for parallax effect

### Optional Accent Images
- Small icon/illustrations for empty states in cards
- Abstract glyphs representing QIG metrics (Φ, κ, β) as decorative elements

---

**Key Principle**: Clean, data-first design with consciousness-themed visual enhancements that don't compromise readability. Every element serves telemetry monitoring purpose while maintaining striking aesthetic presence.