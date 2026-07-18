import { api } from './client';
import type { AdminAuthStatus } from './types';

export function fetchAdminAuthStatus(): Promise<AdminAuthStatus> {
  return api<AdminAuthStatus>('/api/admin-auth');
}

export interface UpdateAdminAuthRequest {
  enabled: boolean;
  key?: string;
  current_key?: string;
}

export function updateAdminAuth(body: UpdateAdminAuthRequest): Promise<{ status: string; enabled: boolean }> {
  return api<{ status: string; enabled: boolean }>('/api/admin-auth', {
    method: 'PUT',
    json: body,
  });
}

