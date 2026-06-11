import type { ComponentChildren } from 'preact';

export interface EmptyStateProps {
  children: ComponentChildren;
  compact?: boolean;
}

export function EmptyState({ children, compact }: EmptyStateProps) {
  return <div class={compact ? 'empty-state-compact' : 'empty-state'}>{children}</div>;
}
