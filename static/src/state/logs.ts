import { signal, computed } from '@preact/signals';
import type { LogRecord, RequestStatus } from '../api/types';

export interface LogFilter {
  endpoint_id: string;
  model_id: string;
  status: RequestStatus | '';
  since: string;
  until: string;
}

export const logsSignal = signal<LogRecord[]>([]);
export const logsTotalSignal = signal(0);
export const logsOffsetSignal = signal(0);
export const logsLimitSignal = signal(50);
export const logsLoadingSignal = signal(false);
export const logFilterSignal = signal<LogFilter>({
  endpoint_id: '',
  model_id: '',
  status: '',
  since: '',
  until: '',
});
export const mergeConsecutiveSignal = signal(true);

const MERGE_KEY = 'logs.mergeConsecutive';

export function loadMergeConsecutiveFromLocalStorage(): void {
  try {
    const saved = localStorage.getItem(MERGE_KEY);
    if (saved !== null) {
      mergeConsecutiveSignal.value = saved === '1';
    }
  } catch {
    // localStorage unavailable
  }
}

export function setMergeConsecutive(enabled: boolean): void {
  mergeConsecutiveSignal.value = enabled;
  try {
    localStorage.setItem(MERGE_KEY, enabled ? '1' : '0');
  } catch {
    // ignore
  }
}

export const hasActiveFiltersSignal = computed(() => {
  const f = logFilterSignal.value;
  return !!(f.endpoint_id || f.model_id || f.status || f.since || f.until);
});

export const filterDrawerOpenSignal = signal(false);

export const activeFilterCountSignal = computed(() => {
  const f = logFilterSignal.value;
  let n = 0;
  if (f.endpoint_id) n++;
  if (f.model_id) n++;
  if (f.status) n++;
  if (f.since) n++;
  if (f.until) n++;
  return n;
});
