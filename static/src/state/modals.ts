import { signal } from '@preact/signals';

export const modelModalOpen = signal(false);
export const modelModalEditing = signal<string | null>(null);

export function openModelModal(modelName?: string): void {
  modelModalEditing.value = modelName ?? null;
  modelModalOpen.value = true;
}

export function closeModelModal(): void {
  modelModalEditing.value = null;
  modelModalOpen.value = false;
}

export const endpointModalOpen = signal(false);
export const endpointModalEditing = signal<string | null>(null);

export function openEndpointModal(endpointId?: string): void {
  endpointModalEditing.value = endpointId ?? null;
  endpointModalOpen.value = true;
}

export function closeEndpointModal(): void {
  endpointModalEditing.value = null;
  endpointModalOpen.value = false;
}

export const mappingModalOpen = signal(false);
export const mappingModalEndpointId = signal<string | null>(null);

export function openMappingModal(endpointId: string): void {
  mappingModalEndpointId.value = endpointId;
  mappingModalOpen.value = true;
}

export function closeMappingModal(): void {
  mappingModalEndpointId.value = null;
  mappingModalOpen.value = false;
}
