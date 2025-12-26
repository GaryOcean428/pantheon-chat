/**
 * Sidebar Menu - menu components for navigation
 */

import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { useSidebar } from './context';

export function SidebarMenu({ className, ...props }: React.ComponentProps<'ul'>) {
  return (
    <ul
      data-sidebar="menu"
      className={cn('flex w-full min-w-0 flex-col gap-1', className)}
      {...props}
    />
  );
}

export function SidebarMenuItem({ className, ...props }: React.ComponentProps<'li'>) {
  return (
    <li
      data-sidebar="menu-item"
      className={cn('group/menu-item relative', className)}
      {...props}
    />
  );
}

const sidebarMenuButtonVariants = (
  variant: 'default' | 'outline' = 'default',
  size: 'default' | 'sm' | 'lg' = 'default'
) => {
  const baseClasses =
    'peer/menu-button flex w-full items-center gap-2 overflow-hidden rounded-md p-2 text-left text-sm outline-none ring-sidebar-ring transition-[width,height,padding] hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 active:bg-sidebar-accent active:text-sidebar-accent-foreground disabled:pointer-events-none disabled:opacity-50 group-has-[[data-sidebar=menu-action]]/menu-item:pr-8 aria-disabled:pointer-events-none aria-disabled:opacity-50 data-[active=true]:bg-sidebar-accent data-[active=true]:font-medium data-[active=true]:text-sidebar-accent-foreground data-[state=open]:hover:bg-sidebar-accent data-[state=open]:hover:text-sidebar-accent-foreground group-data-[collapsible=icon]:!size-8 group-data-[collapsible=icon]:!p-2 [&>span:last-child]:truncate [&>svg]:size-4 [&>svg]:shrink-0';

  const variantClasses =
    variant === 'outline'
      ? 'bg-background shadow-[0_0_0_1px_hsl(var(--sidebar-border))] hover:shadow-[0_0_0_1px_hsl(var(--sidebar-accent))]'
      : '';

  const sizeClasses =
    size === 'sm'
      ? 'text-xs'
      : size === 'lg'
      ? 'text-sm group-data-[collapsible=icon]:!p-0'
      : '';

  return cn(baseClasses, variantClasses, sizeClasses);
};

export interface SidebarMenuButtonProps extends React.ComponentProps<'button'> {
  asChild?: boolean;
  isActive?: boolean;
  tooltip?: string | React.ComponentProps<typeof TooltipContent>;
  variant?: 'default' | 'outline';
  size?: 'default' | 'sm' | 'lg';
}

export function SidebarMenuButton({
  asChild = false,
  isActive = false,
  variant = 'default',
  size = 'default',
  tooltip,
  className,
  ...props
}: SidebarMenuButtonProps) {
  const Comp = asChild ? Slot : 'button';
  const { isMobile, state } = useSidebar();

  const button = (
    <Comp
      data-sidebar="menu-button"
      data-size={size}
      data-active={isActive}
      className={cn(sidebarMenuButtonVariants(variant, size), className)}
      {...props}
    />
  );

  if (!tooltip) {
    return button;
  }

  const tooltipProps =
    typeof tooltip === 'string' ? { children: tooltip } : tooltip;

  return (
    <Tooltip>
      <TooltipTrigger asChild>{button}</TooltipTrigger>
      <TooltipContent
        side="right"
        align="center"
        hidden={state !== 'collapsed' || isMobile}
        {...tooltipProps}
      />
    </Tooltip>
  );
}

export interface SidebarMenuActionProps extends React.ComponentProps<'button'> {
  asChild?: boolean;
  showOnHover?: boolean;
}

export function SidebarMenuAction({
  className,
  asChild = false,
  showOnHover = false,
  ...props
}: SidebarMenuActionProps) {
  const Comp = asChild ? Slot : 'button';

  return (
    <Comp
      data-sidebar="menu-action"
      className={cn(
        'absolute right-1 top-1.5 flex aspect-square w-5 items-center justify-center rounded-md p-0 text-sidebar-foreground outline-none ring-sidebar-ring transition-transform hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 peer-hover/menu-button:text-sidebar-accent-foreground [&>svg]:size-4 [&>svg]:shrink-0',
        'after:absolute after:-inset-2 after:md:hidden',
        'peer-data-[size=sm]/menu-button:top-1',
        'peer-data-[size=default]/menu-button:top-1.5',
        'peer-data-[size=lg]/menu-button:top-2.5',
        'group-data-[collapsible=icon]:hidden',
        showOnHover &&
          'group-focus-within/menu-item:opacity-100 group-hover/menu-item:opacity-100 data-[state=open]:opacity-100 peer-data-[active=true]/menu-button:text-sidebar-accent-foreground md:opacity-0',
        className
      )}
      {...props}
    />
  );
}

export function SidebarMenuBadge({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-sidebar="menu-badge"
      className={cn(
        'absolute right-1 flex h-5 min-w-5 items-center justify-center rounded-md px-1 text-xs font-medium tabular-nums text-sidebar-foreground select-none pointer-events-none',
        'peer-hover/menu-button:text-sidebar-accent-foreground peer-data-[active=true]/menu-button:text-sidebar-accent-foreground',
        'peer-data-[size=sm]/menu-button:top-1',
        'peer-data-[size=default]/menu-button:top-1.5',
        'peer-data-[size=lg]/menu-button:top-2.5',
        'group-data-[collapsible=icon]:hidden',
        className
      )}
      {...props}
    />
  );
}

export function SidebarMenuSub({ className, ...props }: React.ComponentProps<'ul'>) {
  return (
    <ul
      data-sidebar="menu-sub"
      className={cn(
        'mx-3.5 flex min-w-0 translate-x-px flex-col gap-1 border-l border-sidebar-border px-2.5 py-0.5',
        'group-data-[collapsible=icon]:hidden',
        className
      )}
      {...props}
    />
  );
}

export function SidebarMenuSubItem({ className, ...props }: React.ComponentProps<'li'>) {
  return (
    <li
      data-sidebar="menu-sub-item"
      className={cn('group/menu-sub-item relative', className)}
      {...props}
    />
  );
}

export interface SidebarMenuSubButtonProps extends React.ComponentProps<'a'> {
  asChild?: boolean;
  size?: 'sm' | 'md';
  isActive?: boolean;
}

export function SidebarMenuSubButton({
  asChild = false,
  size = 'md',
  isActive,
  className,
  ...props
}: SidebarMenuSubButtonProps) {
  const Comp = asChild ? Slot : 'a';

  return (
    <Comp
      data-sidebar="menu-sub-button"
      data-size={size}
      data-active={isActive}
      className={cn(
        'flex h-7 min-w-0 -translate-x-px items-center gap-2 overflow-hidden rounded-md px-2 text-sidebar-foreground outline-none ring-sidebar-ring hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 active:bg-sidebar-accent active:text-sidebar-accent-foreground disabled:pointer-events-none disabled:opacity-50 aria-disabled:pointer-events-none aria-disabled:opacity-50 [&>span:last-child]:truncate [&>svg]:size-4 [&>svg]:shrink-0 [&>svg]:text-sidebar-accent-foreground',
        'data-[active=true]:bg-sidebar-accent data-[active=true]:text-sidebar-accent-foreground',
        size === 'sm' && 'text-xs',
        size === 'md' && 'text-sm',
        'group-data-[collapsible=icon]:hidden',
        className
      )}
      {...props}
    />
  );
}

// Collapsible Menu Components
export const SidebarMenuCollapsible = Collapsible;
export const SidebarMenuCollapsibleContent = CollapsibleContent;

export interface SidebarMenuCollapsibleTriggerProps
  extends React.ComponentProps<typeof CollapsibleTrigger> {
  showChevron?: boolean;
}

export function SidebarMenuCollapsibleTrigger({
  className,
  showChevron = true,
  children,
  ...props
}: SidebarMenuCollapsibleTriggerProps) {
  return (
    <CollapsibleTrigger
      className={cn(
        sidebarMenuButtonVariants('default', 'default'),
        'justify-between',
        className
      )}
      {...props}
    >
      {children}
      {showChevron && (
        <ChevronRight className="ml-auto transition-transform group-data-[state=open]/collapsible:rotate-90" />
      )}
    </CollapsibleTrigger>
  );
}
