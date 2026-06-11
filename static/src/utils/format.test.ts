import { describe, it, expect } from 'vitest';
import { formatNumber, maskApiKey, esc } from './format';

describe('formatNumber', () => {
  it('formats millions', () => {
    expect(formatNumber(1500000)).toBe('1.5M');
  });

  it('formats thousands', () => {
    expect(formatNumber(2500)).toBe('2.5K');
  });

  it('formats small numbers as-is', () => {
    expect(formatNumber(42)).toBe('42');
  });

  it('formats zero', () => {
    expect(formatNumber(0)).toBe('0');
  });
});

describe('maskApiKey', () => {
  it('masks long keys', () => {
    expect(maskApiKey('sk-1234567890abcdef')).toBe('sk-1***cdef');
  });

  it('returns *** for short keys', () => {
    expect(maskApiKey('sk-12')).toBe('***');
  });

  it('returns empty string for empty input', () => {
    expect(maskApiKey('')).toBe('');
  });
});

describe('esc', () => {
  it('escapes HTML entities', () => {
    expect(esc('<script>alert("xss")</script>')).toBe('&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;');
  });

  it('escapes ampersands', () => {
    expect(esc('foo & bar')).toBe('foo &amp; bar');
  });

  it('handles non-string input', () => {
    expect(esc(42 as unknown as string)).toBe('42');
  });
});
