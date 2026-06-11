import { describe, it, expect, beforeEach } from 'vitest';
import { configSignal, modelsSignal, errorHandlingSignal, setConfig } from './config';
import type { Config } from '../api/types';

describe('config state', () => {
  beforeEach(() => {
    configSignal.value = null;
  });

  it('setConfig updates configSignal', () => {
    const cfg: Config = {
      models: { 'opus-4-7': { api_base: 'https://api.anthropic.com', api_key: 'sk-test', upstream_model: 'opus-4-7' } },
      error_handling: { failover_enabled: true, no_retry_enabled: false },
    };
    setConfig(cfg);
    expect(configSignal.value).toEqual(cfg);
  });

  it('modelsSignal returns models map when config is set', () => {
    const cfg: Config = {
      models: { 'gpt-5': { api_base: 'https://api.openai.com', api_key: 'sk-test', upstream_model: 'gpt-5' } },
      error_handling: { failover_enabled: false, no_retry_enabled: false },
    };
    setConfig(cfg);
    expect(modelsSignal.value).toEqual(cfg.models);
  });

  it('modelsSignal returns empty object when config is null', () => {
    expect(modelsSignal.value).toEqual({});
  });

  it('errorHandlingSignal returns defaults when config is null', () => {
    expect(errorHandlingSignal.value).toEqual({ failover_enabled: false, no_retry_enabled: false });
  });
});
