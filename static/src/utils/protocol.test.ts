import { describe, it, expect } from 'vitest';
import { type ProtocolConfig, toUpstreamProtocols, getModelProtocols, DEFAULT_PATHS } from './protocol';

describe('toUpstreamProtocols', () => {
  it('builds entries for all chips, including disabled ones', () => {
    const p: ProtocolConfig = {
      enabled: { anthropic: true, openai_chat: true, openai_responses: true },
      paths: { anthropic: '/v1/messages', openai_chat: '/v1/custom-chat', openai_responses: '/v1/custom-responses' },
    };
    const result = toUpstreamProtocols(p);
    expect(result).toHaveLength(3);
    expect(result[0]).toEqual({ protocol: 'anthropic', enabled: true, path: '/v1/messages' });
    expect(result[1]).toEqual({ protocol: 'openai/chat-completions', enabled: true, path: '/v1/custom-chat' });
    expect(result[2]).toEqual({ protocol: 'openai/responses', enabled: true, path: '/v1/custom-responses' });
  });

  it('uses defaults for enabled chips with empty paths', () => {
    const p: ProtocolConfig = { enabled: { anthropic: true, openai_chat: false, openai_responses: true }, paths: { anthropic: '', openai_chat: '', openai_responses: '' } };
    const result = toUpstreamProtocols(p);
    expect(result[0].path).toBe(DEFAULT_PATHS['anthropic/messages']);
    expect(result[0].enabled).toBe(true);
    expect(result[1].enabled).toBe(false);
    expect(result[2].path).toBe(DEFAULT_PATHS['openai/responses']);
    expect(result[2].enabled).toBe(true);
  });

  it('includes disabled chips', () => {
    const p: ProtocolConfig = { enabled: { anthropic: false, openai_chat: false, openai_responses: false }, paths: { anthropic: '/v1/messages', openai_chat: '', openai_responses: '' } };
    const result = toUpstreamProtocols(p);
    expect(result).toHaveLength(3);
    expect(result.every(e => e.enabled === false)).toBe(true);
  });

  it('trims user paths', () => {
    const p: ProtocolConfig = {
      enabled: { anthropic: true, openai_chat: true, openai_responses: false },
      paths: { anthropic: '  /v1/messages  ', openai_chat: '  /v1/chat/completions  ', openai_responses: '' },
    };
    const result = toUpstreamProtocols(p);
    expect(result[0].path).toBe('/v1/messages');
    expect(result[1].path).toBe('/v1/chat/completions');
  });
});

describe('DEFAULT_PATHS', () => {
  it('uses anthropic/v1/messages default', () => {
    expect(DEFAULT_PATHS['anthropic/messages']).toBe('anthropic/v1/messages');
  });
  it('uses /v1/chat/completions default', () => {
    expect(DEFAULT_PATHS['openai/chat-completions']).toBe('/v1/chat/completions');
  });
  it('uses /v1/responses default', () => {
    expect(DEFAULT_PATHS['openai/responses']).toBe('/v1/responses');
  });
});

describe('getModelProtocols', () => {
  it('returns Anthropic when that protocol is enabled', () => {
    const m = { api_base: '', api_key: '', upstream_model: '', upstream_protocols: [{ protocol: 'anthropic', enabled: true, path: '/v1/messages' }] };
    expect(getModelProtocols(m)).toEqual(['Anthropic']);
  });
  it('returns Chat when chat-completions is enabled', () => {
    const m = { api_base: '', api_key: '', upstream_model: '', upstream_protocols: [{ protocol: 'openai/chat-completions', enabled: true, path: '/v1/chat/completions' }] };
    expect(getModelProtocols(m)).toEqual(['Chat']);
  });
  it('returns empty when no protocols', () => {
    const m = { api_base: '', api_key: '', upstream_model: '', upstream_protocols: [] };
    expect(getModelProtocols(m)).toEqual([]);
  });
  it('returns empty when all disabled', () => {
    const m = { api_base: '', api_key: '', upstream_model: '', upstream_protocols: [
      { protocol: 'openai/chat-completions', enabled: false, path: '' },
    ]};
    expect(getModelProtocols(m)).toEqual([]);
  });
});