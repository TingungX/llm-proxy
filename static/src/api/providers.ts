import { api } from './client';
import type { ProviderProfileInfo, ThinkingEffortMapping } from './types';

export function fetchProviderProfiles(): Promise<Record<string, ProviderProfileInfo>> {
  return api<Record<string, ProviderProfileInfo>>('/api/provider-profiles');
}

export function fetchThinkingEffortDefaults(): Promise<ThinkingEffortMapping> {
  return api<ThinkingEffortMapping>('/api/thinking-effort-defaults');
}
