/**
 * Sidebar Group - grouping components for sidebar items
 */

import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cn } from '@/lib/utils';

export function SidebarGroup({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-sidebar="group"
      className={cn('relative flex w-full min-w-0 flex-col p-2', className)}
      {...props}
    />
  );
}

export interface SidebarGroupLabelProps extends React.ComponentProps<'div'> {
  asChild?: boolean;
}

export function SidebarGroupLabel({
  className,
  asChild = false,
  ...props
}: SidebarGroupLabelProps) {
  const Comp = asChild ? Slot : 'div';

  return (
    <Comp
      data-sidebar="group-label"
      className={cn(
        'duration-200 flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium text-sidebar-foreground/70 outline-none ring-sidebar-ring transition-[margin,opa] ease-linear focus-visible:ring-2 [&>svg]:size-4 [&>svg]:shrink-0',
        'group-data-[collapsible=icon]:-mt-8 group-data-[collapsible=icon]:opacity-0',
        className
      )}
      {...props}
    />
  );
}

export interface SidebarGroupActionProps extends React.ComponentProps<'button'> {
  asChild?: boolean;
}

export function SidebarGroupAction({
  className,
  asChild = false,
  ...props
}: SidebarGroupActionProps) {
  const Comp = asChild ? Slot : 'button';

  return (
    <Comp
      data-sidebar="group-action"
      className={cn(
        'absolute right-3 top-3.5 flex aspect-square w-5 items-center justify-center rounded-md p-0 text-sidebar-foreground outline-none ring-sidebar-ring transition-transform hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 [&>svg]:size-4 [&>svg]:shrink-0',
        'after:absolute after:-inset-2 after:md:hidden',
        'group-data-[collapsible=icon]:hidden',
        className
      )}
      {...props}
    />
  );
}

export function SidebarGroupContent({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-sidebar="group-content"
      className={cn('w-full text-sm', className)}
      {...props}
    />
  );
}
