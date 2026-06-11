import { signal } from '@preact/signals';

export const currentEndpointIdSignal = signal<string | null>(null);

export const epModelFilterSignal = signal('');

export type EpModelSort = 'name-asc' | 'name-desc' | 'mapping';
export const epModelSortSignal = signal<EpModelSort>('name-asc');

export function selectEndpoint(id: string | null): void {
  currentEndpointIdSignal.value = id;
  epModelFilterSignal.value = '';
  epModelSortSignal.value = 'name-asc';
}

export function setEpModelFilter(v: string): void {
  epModelFilterSignal.value = v;
}

export function setEpModelSort(s: EpModelSort): void {
  epModelSortSignal.value = s;
}
