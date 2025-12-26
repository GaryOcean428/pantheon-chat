/**
 * Tool Factory Dashboard
 *
 * This file re-exports from the modular implementation for backwards compatibility.
 * New code should import from '@/pages/tool-factory'
 */

import { ToolFactoryDashboard } from './tool-factory';

export {
  ToolFactoryDashboard,
  useToolFactory,
  ToolCard,
  ToolList,
  ToolFilters,
  ToolStats,
  ToolDetailPanel,
  ToolForm,
  TOOL_CATEGORIES,
  TOOL_STATUSES,
  PARAMETER_TYPES,
  DASHBOARD_CONSTANTS,
} from './tool-factory';

export type {
  Tool,
  ToolCategory,
  ToolStatus,
  ToolParameter,
  ToolFormData,
  ViewMode,
  SortField,
  SortDirection,
} from './tool-factory';

// Default export for route compatibility
export default ToolFactoryDashboard;
