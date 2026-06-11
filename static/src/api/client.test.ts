import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ApiCallError, api } from './client';

describe('ApiCallError', () => {
  it('has status and detail fields', () => {
    const err = new ApiCallError(400, 'API key already exists');
    expect(err.status).toBe(400);
    expect(err.detail).toBe('API key already exists');
    expect(err.message).toBe('400: API key already exists');
    expect(err).toBeInstanceOf(Error);
  });
});

describe('api()', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('returns parsed JSON on success', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ models: {} }),
    } as Response);
    const result = await api<{ models: Record<string, unknown> }>('/api/config');
    expect(result).toEqual({ models: {} });
  });

  it('throws ApiCallError with .detail from response body', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 409,
      statusText: 'Conflict',
      json: () => Promise.resolve({ error: 'API key already in use' }),
    } as Response);
    try {
      await api('/api/endpoints', { method: 'POST', json: {} });
      expect.unreachable('Should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ApiCallError);
      expect((e as ApiCallError).status).toBe(409);
      expect((e as ApiCallError).detail).toBe('API key already in use');
    }
  });

  it('falls back to statusText when response body has no error field', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: () => Promise.resolve({}),
    } as Response);
    try {
      await api('/api/config', { method: 'PUT', json: {} });
      expect.unreachable('Should have thrown');
    } catch (e) {
      expect((e as ApiCallError).detail).toBe('Internal Server Error');
    }
  });

  it('returns undefined for 204 No Content', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 204,
    } as Response);
    const result = await api<void>('/api/endpoints/abc', { method: 'DELETE' });
    expect(result).toBeUndefined();
  });

  it('sends JSON body when json option is provided', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({}),
    } as Response);
    await api('/api/endpoints', { method: 'POST', json: { name: 'test' } });
    expect(fetchSpy).toHaveBeenCalledWith('/api/endpoints', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ name: 'test' }),
      headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
    }));
  });
});
