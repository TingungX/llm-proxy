import { signal, computed } from '@preact/signals';
import type { Config, ErrorHandlingConfig, AdminAuthStatus } from '../api/types';

export const configSignal = signal<Config | null>(null);

export const modelsSignal = computed(() => configSignal.value?.models ?? {});

export const errorHandlingSignal = computed<ErrorHandlingConfig>(() => {
  const cfg = configSignal.value;
  return cfg?.error_handling ?? { failover_enabled: false, no_retry_enabled: false };
});

export const adminAuthSignal = signal<AdminAuthStatus>({ enabled: false, source: null });

export function getMappingName(models: string[]): string {
  const fr = (configSignal.value as (Config & { family_routing?: Record<string, string> }) | null)?.family_routing ?? {};
  const names: string[] = [];
  Object.entries(fr).forEach(([claudeName, targetModel]) => {
    if (models.includes(targetModel)) names.push(claudeName);
  });
  return names.join(', ');
}

export function setConfig(c: Config): void {
  configSignal.value = c;
  // 从 config 提取 admin_auth 状态
  const aa = (c as Config & { admin_auth?: AdminAuthStatus }).admin_auth;
  if (aa) {
    adminAuthSignal.value = aa;
  }
}
