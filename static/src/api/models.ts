import { api } from './client';
import type { ModelConfig, LatencyResult } from './types';

export function saveModel(modelId: string, data: Partial<ModelConfig>): Promise<void> {
  return api<void>(`/api/models/${encodeURIComponent(modelId)}`, {
    method: 'PUT',
    json: data,
  });
}

export function deleteModel(modelId: string): Promise<void> {
  return api<void>(`/api/models/${encodeURIComponent(modelId)}`, { method: 'DELETE' });
}

export function testLatency(model: string, rounds = 2): Promise<LatencyResult> {
  return api<LatencyResult>('/api/latency', {
    method: 'POST',
    json: { model, rounds },
  });
}
