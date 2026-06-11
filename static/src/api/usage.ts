import { api } from './client';
import type { UsageResponse, UsageSummary, HeatmapResponse } from './types';

export interface UsageParams {
  days?: number;
  group_by?: string;
  granularity?: string;
  endpoint_id?: string;
}

function buildQuery(params: Record<string, unknown>): URLSearchParams {
  return new URLSearchParams(
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null && v !== '')
      .map(([k, v]) => [k, String(v)]),
  );
}

export function fetchUsage(params: UsageParams = {}): Promise<UsageResponse> {
  return api<UsageResponse>(`/api/usage?${buildQuery(params as Record<string, unknown>)}`);
}

export function fetchUsageSummary(): Promise<UsageSummary> {
  return api<UsageSummary>('/api/usage/summary');
}

export function fetchUsageHeatmap(params: { days?: number; endpoint_id?: string } = {}): Promise<HeatmapResponse> {
  return api<HeatmapResponse>(`/api/usage?${buildQuery({ ...params, view: 'heatmap' })}`);
}
