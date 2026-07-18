import { signal } from '@preact/signals';

export type UsageMode = '30d' | '7d' | '1h' | 'custom';
export type UsageGroupBy = 'model' | 'endpoint';
export type UsageSplitMode = 'merged' | 'split';

export interface UsageCustomTimeRange {
  since: string;
  until: string;
}

export const usageModeSignal = signal<UsageMode>('30d');
export const usageGroupBySignal = signal<UsageGroupBy>('model');
export const hourlyDayOffsetSignal = signal(0);
export const heatmapDaysSignal = signal(365);
export const usageEndpointFilterSignal = signal('');
export const usageModelFilterSignal = signal('');
export const usageCustomTimeRangeSignal = signal<UsageCustomTimeRange | null>(null);
export const usageSplitModeSignal = signal<UsageSplitMode>('split');

export const usageChartRef = signal<unknown>(null);
export const endpointUsageChartRef = signal<unknown>(null);

export const usageRefreshTrigger = signal(0);
export const heatmapRefreshTrigger = signal(0);

export const HEATMAP_DAYS_OPTIONS = [90, 180, 365] as const;

export function loadHeatmapDaysFromLocalStorage(): void {
  try {
    const saved = localStorage.getItem('heatmapDays');
    if (saved !== null) {
      const n = parseInt(saved, 10);
      if ((HEATMAP_DAYS_OPTIONS as readonly number[]).includes(n)) {
        heatmapDaysSignal.value = n as 90 | 180 | 365;
        return;
      }
    }
  } catch {
    // localStorage unavailable
  }
  heatmapDaysSignal.value = 365;
}

export function setHeatmapDays(days: 90 | 180 | 365): void {
  heatmapDaysSignal.value = days;
  try {
    localStorage.setItem('heatmapDays', String(days));
  } catch {
    // ignore
  }
  heatmapRefreshTrigger.value++;
}

export function setUsageMode(mode: UsageMode): void {
  usageModeSignal.value = mode;
  if (mode !== '1h') {
    hourlyDayOffsetSignal.value = 0;
  }
  usageRefreshTrigger.value++;
  heatmapRefreshTrigger.value++;
}

export function setUsageGroupBy(group: UsageGroupBy): void {
  usageGroupBySignal.value = group;
  usageRefreshTrigger.value++;
}

export function setHourlyDayOffset(offset: number): void {
  hourlyDayOffsetSignal.value = Math.max(0, Math.min(6, offset));
  usageRefreshTrigger.value++;
}

export function setUsageEndpointFilter(endpointId: string): void {
  usageEndpointFilterSignal.value = endpointId;
  usageRefreshTrigger.value++;
  heatmapRefreshTrigger.value++;
}

export function setUsageModelFilter(modelId: string): void {
  usageModelFilterSignal.value = modelId;
  usageRefreshTrigger.value++;
  heatmapRefreshTrigger.value++;
}

export function setUsageCustomTimeRange(range: UsageCustomTimeRange | null): void {
  usageCustomTimeRangeSignal.value = range;
  usageRefreshTrigger.value++;
  heatmapRefreshTrigger.value++;
}

export function setUsageSplitMode(mode: UsageSplitMode): void {
  usageSplitModeSignal.value = mode;
}
