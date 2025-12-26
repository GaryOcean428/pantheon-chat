/**
 * Sidebar Structure - header, footer, and content components
 */

import * as React from 'react';
import { cn } from '@/lib/utils';

export function SidebarHeader({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-sidebar="header"
      className={cn('flex flex-col gap-2 p-2', className)}
      {...props}
    />
  );
}

export function SidebarFooter({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-sidebar="footer"
      className={cn('flex flex-col gap-2 p-2', className)}
      {...props}
    />
  );
}

export function SidebarContent({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-sidebar="content"
      className={cn(
        'flex min-h-0 flex-1 flex-col gap-2 overflow-auto group-data-[collapsible=icon]:overflow-hidden',
        className
      )}
      {...props}
    />
  );
}
