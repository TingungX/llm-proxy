import { api } from './client';
import type { LogsListResponse, LogsSummary, FilterOptions } from './types';

export interface LogsListParams {
  since?: string;
  until?: string;
  endpoint_id?: string;
  model_id?: string;
  status?: string;
  limit?: number;
  offset?: number;
}

export interface LogsSummaryParams {
  since?: string;
  until?: string;
  endpoint_id?: string;
  model_id?: string;
}

function buildQuery(params: Record<string, unknown>): URLSearchParams {
  return new URLSearchParams(
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null && v !== '')
      .map(([k, v]) => [k, String(v)]),
  );
}

export function fetchLogs(params: LogsListParams = {}): Promise<LogsListResponse> {
  return api<LogsListResponse>(`/api/logs/list?${buildQuery(params as Record<string, unknown>)}`);
}

export function fetchLogsSummary(params: LogsSummaryParams = {}): Promise<LogsSummary> {
  return api<LogsSummary>(`/api/logs/summary?${buildQuery(params as Record<string, unknown>)}`);
}

export function fetchFilterOptions(): Promise<FilterOptions> {
  return api<FilterOptions>('/api/logs/filter-options');
}
