import { useEffect } from 'preact/hooks';
import { Field } from './Field';
import { Toggle } from './Toggle';
import type { LogFilter } from '../state/logs';
import type { FilterOptions } from '../api/types';
import { esc } from '../utils/format';

const PAGE_SIZE_PRESETS = [50, 200, 500] as const;

export interface FilterDrawerProps {
  open: boolean;
  onClose: () => void;
  options: FilterOptions | null;
  filter: LogFilter;
  limit: number;
  merge: boolean;
  onFilterChange: (field: keyof LogFilter, value: string) => void;
  onLimitChange: (newLimit: number) => void;
  onToggleMerge: (enabled: boolean) => void;
  onClear: () => void;
  hasActiveFilters: boolean;
}

export function FilterDrawer({
  open, onClose, options, filter, limit, merge,
  onFilterChange, onLimitChange, onToggleMerge, onClear, hasActiveFilters,
}: FilterDrawerProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      <div class="drawer-backdrop" onClick={onClose} />
      <aside class="drawer" role="dialog" aria-label="筛选">
        <header class="drawer-header">
          <h3>筛选</h3>
          <button type="button" class="icon" onClick={onClose} aria-label="关闭">×</button>
        </header>
        <div class="drawer-body">
          <Field label="端点">
            <select
              class="text-base"
              value={filter.endpoint_id}
              onChange={(e: Event) => onFilterChange('endpoint_id', (e.target as HTMLSelectElement).value)}
            >
              <option value="">所有端点</option>
              {options?.endpoints.map((e) => <option value={e.id}>{esc(e.name || e.id)}</option>)}
            </select>
          </Field>
          <Field label="模型">
            <select
              class="text-base"
              value={filter.model_id}
              onChange={(e: Event) => onFilterChange('model_id', (e.target as HTMLSelectElement).value)}
            >
              <option value="">所有模型</option>
              {options?.models.map((m) => <option value={m}>{esc(m)}</option>)}
            </select>
          </Field>
          <Field label="状态">
            <select
              class="text-base"
              value={filter.status}
              onChange={(e: Event) => onFilterChange('status', (e.target as HTMLSelectElement).value)}
            >
              <option value="">所有状态</option>
              {options?.statuses.map((s) => <option value={s}>{esc(s)}</option>)}
            </select>
          </Field>
          <Field label="起始时间" hint="本地时间">
            <input
              type="datetime-local"
              class="text-base"
              value={filter.since}
              onInput={(e: Event) => onFilterChange('since', (e.target as HTMLInputElement).value)}
            />
          </Field>
          <Field label="结束时间" hint="本地时间">
            <input
              type="datetime-local"
              class="text-base"
              value={filter.until}
              onInput={(e: Event) => onFilterChange('until', (e.target as HTMLInputElement).value)}
            />
          </Field>
          <Field label="每页条数">
            <div class="pagination-page-size" role="group" aria-label="每页显示条数">
              {PAGE_SIZE_PRESETS.map((n) => (
                <button
                  type="button"
                  key={n}
                  class={n === limit ? 'active' : ''}
                  aria-pressed={n === limit}
                  onClick={() => onLimitChange(n)}
                >
                  {n}
                </button>
              ))}
            </div>
          </Field>
          <Toggle checked={merge} onChange={onToggleMerge} label="合并连续请求" />
        </div>
        <footer class="drawer-actions">
          {hasActiveFilters && (
            <button type="button" onClick={onClear}>清除全部</button>
          )}
          <button type="button" class="primary" onClick={onClose}>完成</button>
        </footer>
      </aside>
    </>
  );
}
