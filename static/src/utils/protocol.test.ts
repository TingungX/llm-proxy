import { describe, it, expect } from 'vitest';
import { type ProtocolConfig, toUpstreamProtocol, toUpstreamPaths, DEFAULT_PATHS } from './protocol';

describe('toUpstreamProtocol', () => {
  it('returns "anthropic" when only Anthropic chip is enabled', () => {
    const p: ProtocolConfig = { enabled: { anthropic: true, openai_chat: false, openai_responses: false }, paths: { anthropic: '', openai_chat: '', openai_responses: '' } };
    expect(toUpstreamProtocol(p)).toBe('anthropic');
  });
  it('returns "openai" when only OpenAI Chat chip is enabled', () => {
    const p: ProtocolConfig = { enabled: { anthropic: false, openai_chat: true, openai_responses: false }, paths: { anthropic: '', openai_chat: '', openai_responses: '' } };
    expect(toUpstreamProtocol(p)).toBe('openai');
  });
  it('returns "openai" when only OpenAI Responses chip is enabled', () => {
    const p: ProtocolConfig = { enabled: { anthropic: false, openai_chat: false, openai_responses: true }, paths: { anthropic: '', openai_chat: '', openai_responses: '' } };
    expect(toUpstreamProtocol(p)).toBe('openai');
  });
  it('returns "openai" when both OpenAI chips are enabled but Anthropic is not', () => {
    const p: ProtocolConfig = { enabled: { anthropic: false, openai_chat: true, openai_responses: true }, paths: { anthropic: '', openai_chat: '', openai_responses: '' } };
    expect(toUpstreamProtocol(p)).toBe('openai');
  });
  it('returns "" when all chips are disabled (auto detect)', () => {
    const p: ProtocolConfig = { enabled: { anthropic: false, openai_chat: false, openai_responses: false }, paths: { anthropic: '', openai_chat: '', openai_responses: '' } };
    expect(toUpstreamProtocol(p)).toBe('');
  });
  it('returns "" when multiple protocol types are enabled (auto detect)', () => {
    const p: ProtocolConfig = { enabled: { anthropic: true, openai_chat: true, openai_responses: false }, paths: { anthropic: '', openai_chat: '', openai_responses: '' } };
    expect(toUpstreamProtocol(p)).toBe('');
  });
});

describe('toUpstreamPaths', () => {
  it('builds paths for all enabled chips with user paths', () => {
    const p: ProtocolConfig = {
      enabled: { anthropic: true, openai_chat: true, openai_responses: true },
      paths: { anthropic: '/v1/messages', openai_chat: '/v1/custom-chat', openai_responses: '/v1/custom-responses' },
    };
    expect(toUpstreamPaths(p)).toEqual({
      'anthropic/messages': '/v1/messages',
      'openai/chat-completions': '/v1/custom-chat',
      'openai/responses': '/v1/custom-responses',
    });
  });
  it('uses defaults for enabled chips with empty paths', () => {
    const p: ProtocolConfig = { enabled: { anthropic: true, openai_chat: false, openai_responses: true }, paths: { anthropic: '', openai_chat: '', openai_responses: '' } };
    expect(toUpstreamPaths(p)).toEqual({
      'anthropic/messages': DEFAULT_PATHS['anthropic/messages'],
      'openai/responses': DEFAULT_PATHS['openai/responses'],
    });
  });
  it('uses anthropic/v1/messages default', () => {
    expect(DEFAULT_PATHS['anthropic/messages']).toBe('anthropic/v1/messages');
  });
  it('uses /v1/chat/completions default', () => {
    expect(DEFAULT_PATHS['openai/chat-completions']).toBe('/v1/chat/completions');
  });
  it('uses /v1/responses default', () => {
    expect(DEFAULT_PATHS['openai/responses']).toBe('/v1/responses');
  });
  it('returns {} when no chips enabled', () => {
    const p: ProtocolConfig = { enabled: { anthropic: false, openai_chat: false, openai_responses: false }, paths: { anthropic: '/v1/messages', openai_chat: '', openai_responses: '' } };
    expect(toUpstreamPaths(p)).toEqual({});
  });
  it('trims user paths', () => {
    const p: ProtocolConfig = {
      enabled: { anthropic: true, openai_chat: true, openai_responses: false },
      paths: { anthropic: '  /v1/messages  ', openai_chat: '  /v1/chat/completions  ', openai_responses: '' },
    };
    const result = toUpstreamPaths(p);
    expect(result['anthropic/messages']).toBe('/v1/messages');
    expect(result['openai/chat-completions']).toBe('/v1/chat/completions');
  });
});
