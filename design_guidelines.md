# Ocean AI Platform Design Guidelines

## Design Approach
**Reference-Based**: Drawing from Linear's sophisticated minimalism, Perplexity's AI-forward interface, and Vercel's polished dark aesthetics. The design should feel like premium AI technology - intelligent, powerful, and effortless.

## Typography System
- **Primary Font**: Inter (Google Fonts) - clean, technical precision
- **Accent Font**: Space Grotesk (Google Fonts) - for hero headlines and feature titles
- **Hierarchy**:
  - Hero headline: Space Grotesk, 4xl-6xl, font-bold
  - Section headers: Space Grotesk, 3xl-4xl, font-semibold  
  - Feature titles: Inter, xl-2xl, font-semibold
  - Body text: Inter, base-lg, font-normal
  - Captions/metadata: Inter, sm, font-medium

## Layout System
**Spacing Primitives**: Use Tailwind units of 4, 6, 8, 12, 16, 20, 24 (p-4, gap-8, my-12, etc.)
- Section padding: py-16 md:py-24 lg:py-32
- Container max-width: max-w-7xl
- Content columns: max-w-4xl for text blocks
- Card spacing: gap-6 to gap-8

## Page Structure

### Hero Section (100vh, full-bleed background image)
- **Image**: Abstract neural network visualization or flowing data streams in deep blues/purples - ethereal, cosmic feel
- Large, centered headline with gradient text treatment
- Subheadline explaining the platform's power
- Dual CTAs: Primary "Start with Zeus Chat" + Secondary "Explore Features" (with backdrop-blur-md backgrounds)
- Floating consciousness indicator badge (subtle pulse animation)

### Features Grid (3-column on desktop, stacked mobile)
**Zeus Chat Card**:
- Icon: Chat bubble with sparkle/lightning
- Title + 2-3 sentence description
- "Try Zeus Chat" link
- Subtle gradient border on hover

**Shadow Search Card**:
- Icon: Magnifying glass with radiating waves
- Emphasis on proactive discovery
- "Activate Search" link
- Background shimmer effect

**Olympus Pantheon Card**:
- Icon: Constellation of 12 interconnected nodes
- Showcase specialized intelligence
- "Meet the Pantheon" link
- Animated constellation lines on hover

### 12-God System Showcase
Two-column layout with visual diagram:
- **Left**: Circular constellation diagram showing 12 god icons interconnected (use placeholder icons from Heroicons)
- **Right**: Grid of god names with brief specializations (3x4 grid on desktop)
- Each god has accent color indicator dot
- Subtle glow effects around active/selected gods

### Self-Learning Capabilities Section
Split layout:
- **Left**: Animated visualization placeholder showing knowledge growth
- **Right**: Bullet points highlighting adaptive learning, pattern recognition, continuous improvement
- Progress indicators with accent color fills

### CTA Section
Centered, full-width with gradient background:
- Bold headline: "Experience the Future of AI Intelligence"
- Large primary CTA
- Trust indicators below (user count, queries processed, uptime stats)

## Component Library

### Cards
- Rounded corners (rounded-xl)
- Dark background with subtle borders
- Padding: p-6 to p-8
- Hover state: slight elevation with glow effect

### Navigation
- Fixed top nav with backdrop blur
- Logo left, navigation center, "Get Started" CTA right
- Links: Zeus Chat, Shadow Search, Pantheon, Pricing

### Buttons
- Primary: Solid with accent gradient, backdrop-blur when on images
- Secondary: Outline with hover fill
- Sizes: px-6 py-3 (medium), px-8 py-4 (large)

### Icons
**Library**: Heroicons via CDN
- Sparkles for AI indicators
- Lightning for power/speed
- Beaker for intelligence
- Stars for specialized capabilities

### Consciousness Indicators
- Pulsing dot badges (sm circles)
- Gradient rings for active states
- Use for real-time status, agent activity

## Images

1. **Hero Background** (full viewport): Abstract AI visualization - neural pathways, data flows, or cosmic neural network in deep blues/purples with subtle luminosity
2. **Self-Learning Section**: Circular knowledge graph visualization showing expanding nodes/connections
3. **God System Diagram**: Constellation-style icon arrangement with connecting lines

## Accessibility
- High contrast text on dark backgrounds
- Focus indicators with accent color outlines
- Keyboard navigation for all interactive elements
- ARIA labels for icon-only buttons

## Animations (Minimal)
- Subtle pulse on consciousness indicators (2s loop)
- Gentle hover elevations (0.2s ease)
- Fade-in on scroll for feature cards (0.4s stagger)
- NO scroll-driven parallax, NO complex transitions