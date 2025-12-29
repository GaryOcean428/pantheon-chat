/**
 * COMPONENTS - Main Barrel Export
 * 
 * Re-exports all component modules.
 * Import from '@/components' for access to UI and custom components.
 */

// UI Components (shadcn/ui)
export * from './ui';

// Custom application components
export { AppSidebar } from './app-sidebar';
export { BasinCoordinateViewer } from './basin-coordinate-viewer';
export { BetaAttentionDisplay } from './BetaAttentionDisplay';
export { ThemeProvider } from './ThemeProvider';
export { ThemeToggle } from './ThemeToggle';
export { CapabilityTelemetryPanel, default as CapabilityTelemetryPanelDefault } from './capability-telemetry';
export { ConsciousnessDashboard } from './ConsciousnessDashboard';
export { ConsciousnessMonitoringDemo } from './consciousness-monitoring';
export { EmptyDebatesState } from './EmptyDebatesState';
export { AutonomicAgencyPanel } from './AutonomicAgencyPanel';
export { EmotionalStatePanel } from './EmotionalStatePanel';
export { KernelActivityStream } from './KernelActivityStream';
export { MarkdownUpload } from './MarkdownUpload';
export { ChatFileUpload } from './ChatFileUpload';
export { NodeRegistration } from './NodeRegistration';
export { StreamingMetricsPanel } from './StreamingMetricsPanel';
export { GeometricStreamingTelemetry } from './GeometricStreamingTelemetry';
export { ErrorBoundary } from './ErrorBoundary';
export { HealthIndicator } from './HealthIndicator';
export { InnateDrivesDisplay } from './InnateDrivesDisplay';
export { MarkdownRenderer, MarkdownExample } from './MarkdownRenderer';
export { MemoryFragmentSearch } from './MemoryFragmentSearch';
export { default as NeurochemistryAdminPanel } from './NeurochemistryAdminPanel';
export { default as NeurochemistryDisplay } from './NeurochemistryDisplay';
export { PhiVisualization } from './PhiVisualization';
export { SearchBudgetPanel } from './SearchBudgetPanel';
export { SessionExpirationModal } from './SessionExpirationModal';
export { WarStatusPanel } from './war-status-panel';
export { default as ZeusChat } from './ZeusChat';
