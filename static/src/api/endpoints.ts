import { api } from './client';
import type { Endpoint, CreateEndpointRequest, ProtocolDetectionResult } from './types';

export function fetchEndpoints(): Promise<Endpoint[]> {
  return api<Endpoint[]>('/api/endpoints');
}

export function fetchEndpoint(id: string): Promise<Endpoint> {
  return api<Endpoint>(`/api/endpoints/${id}`);
}

export function saveEndpoint(id: string, data: Partial<CreateEndpointRequest>): Promise<void> {
  return api<void>(`/api/endpoints/${id}`, { method: 'PUT', json: data });
}

export function createEndpoint(data: CreateEndpointRequest): Promise<Endpoint> {
  return api<Endpoint>('/api/endpoints', { method: 'POST', json: data });
}

export function deleteEndpoint(id: string): Promise<void> {
  return api<void>(`/api/endpoints/${id}`, { method: 'DELETE' });
}

export function detectProtocol(apiBase: string, apiKey: string): Promise<ProtocolDetectionResult> {
  return api<ProtocolDetectionResult>('/api/detect-protocol', {
    method: 'POST',
    json: { api_base: apiBase, api_key: apiKey },
  });
}
