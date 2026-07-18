import { useState, useRef, useEffect } from 'preact/hooks';
import { endpointsSignal } from '../state/endpoints';
import { modelsSignal } from '../state/config';
import {
  usageModeSignal, usageEndpointFilterSignal, usageModelFilterSignal,
  usageCustomTimeRangeSignal,
  setUsageMode, setUsageEndpointFilter, setUsageModelFilter, setUsageCustomTimeRange,
  type UsageMode, type UsageCustomTimeRange,
} from '../state/usage';
import { esc } from '../utils/format';
import type { JSX } from 'preact';

type DropdownKind = 'model' | 'endpoint' | 'time' | null;

export function UsageFilterBar(): JSX.Element {
  const [openDropdown, setOpenDropdown] = useState<DropdownKind>(null);
  const barRef = useRef<HTMLDivElement | null>(null);

  const modelFilter = usageModelFilterSignal.value;
  const endpointFilter = usageEndpointFilterSignal.value;
  const usageMode = usageModeSignal.value;
  const customRange = usageCustomTimeRangeSignal.value;

  const endpoints = endpointsSignal.value;
  const models = Object.keys(modelsSignal.value);

  // 点击外部关闭下拉
  useEffect(() => {
    if (!openDropdown) return;
    function handleClickOutside(e: MouseEvent): void {
      if (barRef.current && !barRef.current.contains(e.target as Node)) {
        setOpenDropdown(null);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [openDropdown]);

  function toggleDropdown(kind: DropdownKind): void {
    setOpenDropdown(openDropdown === kind ? null : kind);
  }

  const modelLabel = modelFilter ? modelFilter : '全部模型';
  const endpointLabel = endpointFilter
    ? (endpoints.find(ep => ep.endpoint_id === endpointFilter)?.name || endpointFilter)
    : '全部端点';
  const timeLabel = getTimeLabel(usageMode, customRange);

  const hasActiveFilter = Boolean(modelFilter || endpointFilter || usageMode === 'custom');

  function handleClear(): void {
    setUsageModelFilter('');
    setUsageEndpointFilter('');
    setUsageMode('30d');
  }

  return (
    <div class="endpoint-dock usage-filter-bar" ref={barRef}>
      {/* 模型胶囊 */}
      <div class="usage-filter-chip-wrap">
        <button
          type="button"
          class={`protocol-chip${modelFilter ? ' active' : ''}`}
          onClick={() => toggleDropdown('model')}
        >
          {esc(modelLabel)}
          <span class="usage-filter-caret">▾</span>
        </button>
        {openDropdown === 'model' && (
          <div class="usage-filter-dropdown">
            <button
              class={`usage-filter-option${!modelFilter ? ' active' : ''}`}
              onClick={() => { setUsageModelFilter(''); setOpenDropdown(null); }}
            >全部模型</button>
            {models.map(m => (
              <button
                key={m}
                class={`usage-filter-option${modelFilter === m ? ' active' : ''}`}
                onClick={() => { setUsageModelFilter(m); setOpenDropdown(null); }}
              >{esc(m)}</button>
            ))}
          </div>
        )}
      </div>

      {/* 端点胶囊 */}
      <div class="usage-filter-chip-wrap">
        <button
          type="button"
          class={`protocol-chip${endpointFilter ? ' active' : ''}`}
          onClick={() => toggleDropdown('endpoint')}
        >
          {esc(endpointLabel)}
          <span class="usage-filter-caret">▾</span>
        </button>
        {openDropdown === 'endpoint' && (
          <div class="usage-filter-dropdown">
            <button
              class={`usage-filter-option${!endpointFilter ? ' active' : ''}`}
              onClick={() => { setUsageEndpointFilter(''); setOpenDropdown(null); }}
            >全部端点</button>
            {endpoints.map(ep => (
              <button
                key={ep.endpoint_id}
                class={`usage-filter-option${endpointFilter === ep.endpoint_id ? ' active' : ''}`}
                onClick={() => { setUsageEndpointFilter(ep.endpoint_id); setOpenDropdown(null); }}
              >{esc(ep.name || ep.endpoint_id)}</button>
            ))}
          </div>
        )}
      </div>

      {/* 时间胶囊 */}
      <div class="usage-filter-chip-wrap">
        <button
          type="button"
          class={`protocol-chip${usageMode === 'custom' ? ' active' : ''}`}
          onClick={() => toggleDropdown('time')}
        >
          {esc(timeLabel)}
          <span class="usage-filter-caret">▾</span>
        </button>
        {openDropdown === 'time' && (
          <div class="usage-filter-dropdown">
            <button
              class={`usage-filter-option${usageMode === '30d' ? ' active' : ''}`}
              onClick={() => { setUsageMode('30d'); setOpenDropdown(null); }}
            >近 30 天</button>
            <button
              class={`usage-filter-option${usageMode === '7d' ? ' active' : ''}`}
              onClick={() => { setUsageMode('7d'); setOpenDropdown(null); }}
            >近 7 天</button>
            <button
              class={`usage-filter-option${usageMode === '1h' ? ' active' : ''}`}
              onClick={() => { setUsageMode('1h'); setOpenDropdown(null); }}
            >逐小时</button>
            <div class="usage-filter-divider" />
            <div class="usage-filter-custom">
              <div class="usage-filter-custom-label">自定义范围</div>
              <input
                type="date"
                class="usage-filter-date"
                value={customRange?.since || ''}
                onChange={(e: Event) => {
                  const since = (e.target as HTMLInputElement).value;
                  const until = customRange?.until || since;
                  setUsageCustomTimeRange({ since, until });
                }}
              />
              <input
                type="date"
                class="usage-filter-date"
                value={customRange?.until || ''}
                onChange={(e: Event) => {
                  const until = (e.target as HTMLInputElement).value;
                  const since = customRange?.since || until;
                  setUsageCustomTimeRange({ since, until });
                }}
              />
              <button
                type="button"
                class="usage-filter-apply"
                disabled={!customRange?.since || !customRange?.until}
                onClick={() => {
                  if (customRange?.since && customRange?.until) {
                    setUsageMode('custom');
                    setOpenDropdown(null);
                  }
                }}
              >应用</button>
            </div>
          </div>
        )}
      </div>

      {/* 清除筛选（仅在有筛选时显示） */}
      {hasActiveFilter && (
        <button
          type="button"
          class="protocol-chip usage-filter-clear"
          onClick={handleClear}
          title="清除筛选"
        >✕</button>
      )}
    </div>
  );
}

function getTimeLabel(mode: UsageMode, range: UsageCustomTimeRange | null): string {
  switch (mode) {
    case '30d': return '近 30 天';
    case '7d': return '近 7 天';
    case '1h': return '逐小时';
    case 'custom': return range ? `${range.since}~${range.until}` : '自定义';
  }
}
