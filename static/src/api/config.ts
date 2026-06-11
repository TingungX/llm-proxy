import { api } from './client';
import type { Config } from './types';

export function fetchConfig(): Promise<Config> {
  return api<Config>('/api/config');
}

export function saveConfig(config: Config): Promise<void> {
  return api<void>('/api/config', { method: 'PUT', json: config });
}
