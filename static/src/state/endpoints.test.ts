import { describe, it, expect, beforeEach } from 'vitest';
import { endpointsSignal, endpointNameMapSignal, setEndpoints, getEndpoint } from './endpoints';
import type { Endpoint } from '../api/types';

const mockEndpoint: Endpoint = {
  endpoint_id: 'a1b2c3d4',
  name: 'Claude Desktop',
  api_key: 'sk-test',
  enabled: true,
  models: ['opus-4-7'],
  settings: { failover_enabled: false, no_retry_enabled: false },
  family_routing: {},
  accept_protocols: ['anthropic', 'openai'],
  last_used: null,
};

describe('endpoints state', () => {
  beforeEach(() => {
    endpointsSignal.value = [];
  });

  it('setEndpoints updates endpointsSignal', () => {
    setEndpoints([mockEndpoint]);
    expect(endpointsSignal.value).toHaveLength(1);
    expect(endpointsSignal.value[0].endpoint_id).toBe('a1b2c3d4');
  });

  it('endpointNameMapSignal maps ID to name', () => {
    setEndpoints([mockEndpoint]);
    expect(endpointNameMapSignal.value['a1b2c3d4']).toBe('Claude Desktop');
  });

  it('endpointNameMapSignal falls back to ID when name is empty', () => {
    const ep = { ...mockEndpoint, name: '' };
    setEndpoints([ep]);
    expect(endpointNameMapSignal.value['a1b2c3d4']).toBe('a1b2c3d4');
  });

  it('getEndpoint returns endpoint by ID', () => {
    setEndpoints([mockEndpoint]);
    expect(getEndpoint('a1b2c3d4')).toEqual(mockEndpoint);
  });

  it('getEndpoint returns undefined for unknown ID', () => {
    setEndpoints([mockEndpoint]);
    expect(getEndpoint('unknown')).toBeUndefined();
  });
});
