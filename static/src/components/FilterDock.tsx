import type { JSX } from 'preact';
import { filterDrawerOpenSignal } from '../state/store';

export interface FilterDockProps {
  onOpen: () => void;
  activeCount: number;
}

const FILTER_ICON = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>`;

export function FilterDock({ onOpen, activeCount }: FilterDockProps): JSX.Element | null {
  if (filterDrawerOpenSignal.value) return null;
  return (
    <button
      type="button"
      class="filter-dock"
      onClick={onOpen}
      aria-label="打开筛选"
      title="筛选"
    >
      <span class="filter-dock-icon" dangerouslySetInnerHTML={{ __html: FILTER_ICON }} />
      <span>筛选</span>
      {activeCount > 0 && <span class="filter-dock-badge">{activeCount}</span>}
    </button>
  );
}
