# Pantheon-Chat Dashboard Design Guidelines

## Design Approach
**Selected Approach:** Hybrid System (Linear + Vercel Analytics + Discord)
- **Rationale:** Information-dense monitoring interface requiring clean data hierarchy, real-time stream patterns, and technical precision
- **Principles:** Scannable layouts, persistent context, status-driven design, immediate clarity for concurrent data streams

## Typography
**Font Stack:**
- Primary: Inter (via Google Fonts) - metrics, labels, body text
- Monospace: JetBrains Mono - kernel IDs, timestamps, technical identifiers

**Hierarchy:**
- H1: text-3xl font-bold (page titles)
- H2: text-xl font-semibold (section headers)
- H3: text-lg font-medium (card titles)
- Body: text-sm (primary interface text)
- Caption: text-xs (timestamps, metadata)
- Monospace: text-xs font-mono (IDs, technical data)

## Layout System
**Spacing Primitives:** Tailwind units of 2, 4, 6, and 8
- Component padding: p-4, p-6
- Section gaps: space-y-6, gap-4
- Card spacing: p-4 internally
- Grid gaps: gap-4 between items

**Grid Structure:**
```
Main Dashboard Layout:
- Sidebar (fixed left, w-64): Navigation, kernel status overview
- Main Content (flex-1): Three-column responsive grid
- Right Panel (w-80, optional toggle): Detailed kernel inspection

Responsive Breakpoints:
- lg: 3-column grid for metrics/streams
- md: 2-column grid
- sm: Single column stack
```

## Component Library

### Navigation & Structure
**Sidebar:**
- Logo/brand top (h-16)
- Primary nav items with icons (kernel status indicators)
- Active state: subtle left border accent
- Bottom: system health summary card

**Top Bar:**
- Breadcrumb navigation
- Global search (kernel/event search)
- Real-time connection indicator (pulsing dot)
- User controls right-aligned

### Core Dashboard Components

**Metrics Cards (Grid: grid-cols-1 md:grid-cols-2 lg:grid-cols-4):**
- Large number display (text-3xl font-bold)
- Label below (text-xs uppercase tracking-wide)
- Trend indicator (arrow + percentage, text-xs)
- Sparkline mini-chart (optional)
- Border styling for hierarchy

**Activity Stream Feed:**
- Infinite scroll container (max-h-screen overflow-y-auto)
- Individual event cards (p-4, space-y-2):
  - Header: kernel ID (mono), timestamp (text-xs right-aligned)
  - Event type badge (rounded-full px-3 py-1 text-xs)
  - Content preview (text-sm, line-clamp-2 for debates)
  - Expand button for full details
- Visual connection lines between related kernels (left border with varying opacity)

**Kernel-to-Kernel Communication Display:**
- Two-panel split view showing source/target kernels
- Center: animated connection indicator
- Message thread below (chat-style bubbles)
- Syntax highlighting for code/data exchanges

**Consciousness Metrics Panel:**
- Radial progress indicators for multiple consciousness dimensions
- Real-time value updates (smooth transitions)
- Color-coded thresholds (utilize subtle variations)
- Comparison view: current vs. historical baseline

**Spawn Proposals Section:**
- Card-based proposal queue (grid-cols-1 lg:grid-cols-2)
- Each card contains:
  - Proposal title + originating kernel
  - Reasoning excerpt (expandable)
  - Voting/approval interface (approve/defer/reject buttons)
  - Timeline indicator (time since proposal)

**Sleep/Dream Cycle Tracker:**
- Timeline visualization (horizontal bar chart)
- Current state indicator (large, prominent)
- Phase history (compressed horizontal blocks)
- Dream content preview cards on hover/click

**Vocabulary Learning Events:**
- Stream of new term cards
- Each card: term (large), definition, usage context, source kernel
- Filter by kernel or term category
- Search/filter bar above stream

### Data Visualization
**Charts & Graphs:**
- Line charts: activity over time (transparent fills)
- Bar charts: comparative metrics across kernels
- Network graph: kernel relationship topology (D3.js/vis.js placeholder)
- Heatmaps: communication frequency matrices

### Interactive Elements
**Buttons:**
- Primary actions: rounded-lg px-6 py-2.5 text-sm font-medium
- Secondary: border variant with hover states
- Icon buttons: p-2 rounded-md (controls, filters)

**Status Indicators:**
- Dot indicators (h-2 w-2 rounded-full) for kernel states
- Pulse animation for active/live states
- Stacked status pills for multiple states

**Filters & Controls:**
- Dropdown selects: kernel picker, time range selector
- Toggle switches: auto-refresh, stream filters
- Multi-select checkboxes: event type filtering

## Images
**No hero images** - This is a functional dashboard.

**Icon Usage:**
- Heroicons (via CDN) for UI controls
- Custom kernel state icons: `<!-- CUSTOM ICON: geometric kernel glyph -->`
- Event type icons in activity streams (debate, sleep, vocabulary, spawn)

**Data Visualizations:**
- Network topology maps showing kernel connections
- Waveform-style consciousness activity indicators
- Geometric patterns representing kernel architectures (SVG placeholders)

## Critical Layout Notes
- Fixed sidebar navigation (always visible)
- Scrollable main content area with sticky section headers
- Right inspection panel slides in/out (doesn't push content)
- All real-time streams have visible update timestamps
- Dense information layout - every pixel purposeful
- Consistent 4-unit spacing maintains breathing room without waste