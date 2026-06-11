import { describe, it, expect } from 'vitest';
import { localDateTimeToBackend, localDateToBackendPrefix, formatDateTime, toLocalDateString } from './date';

describe('localDateTimeToBackend', () => {
  it('returns empty for empty input', () => {
    expect(localDateTimeToBackend('')).toBe('');
  });

  it('returns empty for invalid input', () => {
    expect(localDateTimeToBackend('not-a-date')).toBe('');
  });

  it('formats a local datetime as Beijing-time string', () => {
    const out = localDateTimeToBackend('2026-06-01T00:00:00');
    expect(out).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/);
  });
});

describe('localDateToBackendPrefix', () => {
  it('returns YYYY-MM-DD for a Date', () => {
    const d = new Date(2026, 5, 1, 12, 0, 0);
    expect(localDateToBackendPrefix(d)).toBe('2026-06-01');
  });

  it('zero-pads single-digit months and days', () => {
    const d = new Date(2026, 0, 5, 0, 0, 0);
    expect(localDateToBackendPrefix(d)).toBe('2026-01-05');
  });
});

describe('formatDateTime', () => {
  it('renders Beijing-time string as MM/DD HH:MM', () => {
    expect(formatDateTime('2026-06-06 22:59:43')).toBe('06/06 22:59');
  });

  it('returns input unchanged for non-Beijing string', () => {
    expect(formatDateTime('garbage')).toBe('garbage');
  });
});

describe('toLocalDateString', () => {
  it('extracts date prefix from Beijing-time string', () => {
    expect(toLocalDateString('2026-06-01 12:34:56')).toBe('2026-06-01');
  });

  it('passes through date-only strings', () => {
    expect(toLocalDateString('2026-06-01')).toBe('2026-06-01');
  });
});
