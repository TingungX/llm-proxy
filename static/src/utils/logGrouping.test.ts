import { describe, it, expect } from 'vitest';
import { aggregateRecords } from './logGrouping';
import type { LogRecord } from '../api/types';

const rec = (over: Partial<LogRecord> & { id: number; endpoint_id: string; model_id: string; request_status: 'success' | 'error'; timestamp: string; input_tokens: number; output_tokens: number; }): LogRecord => ({
  request_id: null,
  latency_ms: null,
  error_type: null,
  client_ip: null,
  user_agent: null,
  ...over,
});

describe('aggregateRecords', () => {
  it('returns single entries when no consecutive same key', () => {
    const records = [
      rec({ id: 1, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:00Z', input_tokens: 10, output_tokens: 5 }),
      rec({ id: 2, endpoint_id: 'b', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:01Z', input_tokens: 10, output_tokens: 5 }),
    ];
    const items = aggregateRecords(records);
    expect(items).toHaveLength(2);
    expect(items[0].kind).toBe('single');
    expect(items[1].kind).toBe('single');
  });

  it('merges 2+ consecutive same key (endpoint+model+status)', () => {
    const records = [
      rec({ id: 3, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:02Z', input_tokens: 100, output_tokens: 50, latency_ms: 100 }),
      rec({ id: 2, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:01Z', input_tokens: 200, output_tokens: 80, latency_ms: 200 }),
      rec({ id: 1, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:00Z', input_tokens: 50, output_tokens: 20, latency_ms: 50 }),
    ];
    const items = aggregateRecords(records);
    expect(items).toHaveLength(1);
    expect(items[0].kind).toBe('merged');
    if (items[0].kind !== 'merged') throw new Error('expected merged');
    expect(items[0].count).toBe(3);
    expect(items[0].totalInputTokens).toBe(350);
    expect(items[0].totalOutputTokens).toBe(150);
    expect(items[0].avgLatencyMs).toBe(117);
  });

  it('does not merge across different status', () => {
    const records = [
      rec({ id: 2, endpoint_id: 'a', model_id: 'model-a', request_status: 'error', timestamp: '2026-06-05T10:00:01Z', input_tokens: 10, output_tokens: 0, error_type: '4xx' }),
      rec({ id: 1, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:00Z', input_tokens: 10, output_tokens: 0 }),
    ];
    const items = aggregateRecords(records);
    expect(items.every((i) => i.kind === 'single')).toBe(true);
  });

  it('skips null latencies in avg', () => {
    const records = [
      rec({ id: 2, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:01Z', input_tokens: 0, output_tokens: 0, latency_ms: null }),
      rec({ id: 1, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:00Z', input_tokens: 0, output_tokens: 0, latency_ms: 100 }),
    ];
    const items = aggregateRecords(records);
    if (items[0].kind !== 'merged') throw new Error('expected merged');
    expect(items[0].avgLatencyMs).toBe(100);
  });

  it('avgLatencyMs is null when all latencies null', () => {
    const records = [
      rec({ id: 2, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:01Z', input_tokens: 0, output_tokens: 0, latency_ms: null }),
      rec({ id: 1, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:00Z', input_tokens: 0, output_tokens: 0, latency_ms: null }),
    ];
    const items = aggregateRecords(records);
    if (items[0].kind !== 'merged') throw new Error('expected merged');
    expect(items[0].avgLatencyMs).toBeNull();
  });

  it('deduplicates error_types in merged group', () => {
    const records = [
      rec({ id: 3, endpoint_id: 'a', model_id: 'model-a', request_status: 'error', timestamp: '2026-06-05T10:00:02Z', input_tokens: 0, output_tokens: 0, error_type: '5xx' }),
      rec({ id: 2, endpoint_id: 'a', model_id: 'model-a', request_status: 'error', timestamp: '2026-06-05T10:00:01Z', input_tokens: 0, output_tokens: 0, error_type: '4xx' }),
      rec({ id: 1, endpoint_id: 'a', model_id: 'model-a', request_status: 'error', timestamp: '2026-06-05T10:00:00Z', input_tokens: 0, output_tokens: 0, error_type: '4xx' }),
    ];
    const items = aggregateRecords(records);
    if (items[0].kind !== 'merged') throw new Error('expected merged');
    expect(items[0].errorTypes.sort()).toEqual(['4xx', '5xx']);
  });

  it('merges only same key within 120s window, breaks on different endpoint', () => {
    const records = [
      rec({ id: 4, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:03Z', input_tokens: 1, output_tokens: 1 }),
      rec({ id: 3, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:02Z', input_tokens: 1, output_tokens: 1 }),
      rec({ id: 2, endpoint_id: 'b', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:01Z', input_tokens: 1, output_tokens: 1 }),
      rec({ id: 1, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:00Z', input_tokens: 1, output_tokens: 1 }),
    ];
    const items = aggregateRecords(records);
    expect(items).toHaveLength(2);
    expect(items[0].kind).toBe('merged');
    if (items[0].kind !== 'merged') throw new Error('expected merged');
    expect(items[0].endpoint_id).toBe('a');
    expect(items[0].count).toBe(3);
    expect(items[1].kind).toBe('single');
    if (items[1].kind !== 'single') throw new Error('expected single');
    expect(items[1].record.endpoint_id).toBe('b');
  });

  it('handles empty input', () => {
    expect(aggregateRecords([])).toEqual([]);
  });

  it('does NOT merge records > 120s apart even with same key', () => {
    const records = [
      rec({ id: 2, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:03:00Z', input_tokens: 10, output_tokens: 5 }),
      rec({ id: 1, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:00Z', input_tokens: 10, output_tokens: 5 }),
    ];
    const items = aggregateRecords(records);
    expect(items).toHaveLength(2);
    expect(items.every((i) => i.kind === 'single')).toBe(true);
  });

  it('merges records at exactly 120s boundary', () => {
    const records = [
      rec({ id: 2, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:02:00Z', input_tokens: 10, output_tokens: 5 }),
      rec({ id: 1, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:00Z', input_tokens: 10, output_tokens: 5 }),
    ];
    const items = aggregateRecords(records);
    expect(items[0].kind).toBe('merged');
  });

  it('keeps different models in separate groups even when alternating rapidly', () => {
    const records = [
      rec({ id: 6, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:10Z', input_tokens: 1, output_tokens: 1 }),
      rec({ id: 5, endpoint_id: 'a', model_id: 'model-b', request_status: 'success', timestamp: '2026-06-05T10:00:08Z', input_tokens: 1, output_tokens: 1 }),
      rec({ id: 4, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:06Z', input_tokens: 1, output_tokens: 1 }),
      rec({ id: 3, endpoint_id: 'a', model_id: 'model-b', request_status: 'success', timestamp: '2026-06-05T10:00:04Z', input_tokens: 1, output_tokens: 1 }),
      rec({ id: 2, endpoint_id: 'a', model_id: 'model-a', request_status: 'success', timestamp: '2026-06-05T10:00:02Z', input_tokens: 1, output_tokens: 1 }),
      rec({ id: 1, endpoint_id: 'a', model_id: 'model-b', request_status: 'success', timestamp: '2026-06-05T10:00:00Z', input_tokens: 1, output_tokens: 1 }),
    ];
    const items = aggregateRecords(records);
    expect(items).toHaveLength(2);
    const m1 = items.find((i) => i.kind === 'merged' && i.model_id === 'model-a');
    const m2 = items.find((i) => i.kind === 'merged' && i.model_id === 'model-b');
    expect(m1).toBeDefined();
    expect(m2).toBeDefined();
    if (m1?.kind !== 'merged' || m2?.kind !== 'merged') throw new Error('expected merged');
    expect(m1.count).toBe(3);
    expect(m2.count).toBe(3);
    const first = items[0];
    if (first.kind !== 'merged') throw new Error('expected merged first');
    expect(first.model_id).toBe('model-a');
  });
});
