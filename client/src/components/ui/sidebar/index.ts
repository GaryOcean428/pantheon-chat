/**
 * Sidebar - barrel exports
 */

// Context and Provider
export {
  SidebarContext,
  SidebarProvider,
  useSidebar,
  SIDEBAR_WIDTH,
  SIDEBAR_WIDTH_MOBILE,
  SIDEBAR_WIDTH_ICON,
  SIDEBAR_KEYBOARD_SHORTCUT,
} from './context';
export type { SidebarState, SidebarContext as SidebarContextType, SidebarProviderProps } from './context';

// Core Components
export {
  Sidebar,
  SidebarTrigger,
  SidebarRail,
  SidebarInset,
  SidebarInput,
  SidebarSeparator,
  SidebarSkeleton,
} from './sidebar-core';
export type { SidebarProps, SidebarTriggerProps } from './sidebar-core';

// Structure Components
export {
  SidebarHeader,
  SidebarFooter,
  SidebarContent,
} from './sidebar-structure';

// Group Components
export {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupAction,
  SidebarGroupContent,
} from './sidebar-group';
export type { SidebarGroupLabelProps, SidebarGroupActionProps } from './sidebar-group';

// Menu Components
export {
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarMenuAction,
  SidebarMenuBadge,
  SidebarMenuSub,
  SidebarMenuSubItem,
  SidebarMenuSubButton,
  SidebarMenuCollapsible,
  SidebarMenuCollapsibleContent,
  SidebarMenuCollapsibleTrigger,
} from './sidebar-menu';
export type {
  SidebarMenuButtonProps,
  SidebarMenuActionProps,
  SidebarMenuSubButtonProps,
  SidebarMenuCollapsibleTriggerProps,
} from './sidebar-menu';
