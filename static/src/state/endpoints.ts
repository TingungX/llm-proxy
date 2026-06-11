import { signal, computed } from '@preact/signals';
import type { Endpoint } from '../api/types';

export const endpointsSignal = signal<Endpoint[]>([]);

export const endpointNameMapSignal = computed<Record<string, string>>(() => {
  const map: Record<string, string> = {};
  for (const ep of endpointsSignal.value) {
    map[ep.endpoint_id] = ep.name || ep.endpoint_id;
  }
  return map;
});

export function setEndpoints(eps: Endpoint[]): void {
  endpointsSignal.value = eps;
}

export function getEndpoint(id: string): Endpoint | undefined {
  return endpointsSignal.value.find(ep => ep.endpoint_id === id);
}
