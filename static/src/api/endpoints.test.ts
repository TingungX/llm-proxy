import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ApiCallError } from './client';
import { createEndpoint } from './endpoints';

describe('createEndpoint error propagation', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('throws ApiCallError with server error detail on conflict', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 409,
      statusText: 'Conflict',
      json: () => Promise.resolve({ error: 'API key already in use by endpoint abc123' }),
    } as Response);

    try {
      await createEndpoint({ name: 'test', api_key: 'sk-duplicate' });
      expect.unreachable('Should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ApiCallError);
      expect((e as ApiCallError).status).toBe(409);
      expect((e as ApiCallError).detail).toBe('API key already in use by endpoint abc123');
    }
  });

  it('returns endpoint data on success', async () => {
    const mockEndpoint = {
      endpoint_id: 'a1b2c3d4',
      name: 'test',
      api_key: 'sk-test',
      enabled: true,
      models: [],
      settings: { failover_enabled: false, no_retry_enabled: false },
      family_routing: {},
      accept_protocols: ['anthropic', 'openai'],
      last_used: null,
    };
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockEndpoint),
    } as Response);

    const result = await createEndpoint({ name: 'test', api_key: 'sk-test' });
    expect(result.endpoint_id).toBe('a1b2c3d4');
  });
});
