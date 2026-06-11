import type { LogRecord } from '../api/types';

export interface MergedLogGroup {
  kind: 'merged';
  key: string;
  records: LogRecord[];
  count: number;
  endpoint_id: string;
  model_id: string;
  request_status: LogRecord['request_status'];
  firstTimestamp: string;
  lastTimestamp: string;
  totalInputTokens: number;
  totalOutputTokens: number;
  avgLatencyMs: number | null;
  errorTypes: string[];
}

export interface SingleLogEntry {
  kind: 'single';
  record: LogRecord;
}

export type LogTimelineItem = MergedLogGroup | SingleLogEntry;

const formatHM = (ts: string): string => {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
};

const MERGE_WINDOW_MS = 120_000;

interface GroupKey {
  endpoint_id: string;
  model_id: string;
  request_status: LogRecord['request_status'];
}

const keyOf = (r: LogRecord): GroupKey => ({
  endpoint_id: r.endpoint_id,
  model_id: r.model_id,
  request_status: r.request_status,
});

const keyStr = (k: GroupKey): string => `${k.endpoint_id}|${k.model_id}|${k.request_status}`;

export function aggregateRecords(records: LogRecord[]): LogTimelineItem[] {
  const tsOf = (r: LogRecord): number => new Date(r.timestamp).getTime();

  const buckets = new Map<string, LogRecord[]>();
  for (const r of records) {
    const k = keyStr(keyOf(r));
    const arr = buckets.get(k);
    if (arr) arr.push(r);
    else buckets.set(k, [r]);
  }
  for (const arr of buckets.values()) {
    arr.sort((a, b) => tsOf(b) - tsOf(a));
  }

  const heads: Array<{ key: string; idx: number; list: LogRecord[] }> = [];
  for (const [k, list] of buckets) {
    heads.push({ key: k, idx: 0, list });
  }

  const result: LogTimelineItem[] = [];
  while (heads.length > 0) {
    let bestI = 0;
    for (let i = 1; i < heads.length; i++) {
      if (tsOf(heads[i].list[heads[i].idx]) > tsOf(heads[bestI].list[heads[bestI].idx])) {
        bestI = i;
      }
    }
    const head = heads[bestI];
    const first = head.list[head.idx];
    const group: LogRecord[] = [first];
    head.idx++;
    while (head.idx < head.list.length) {
      const next = head.list[head.idx];
      if (tsOf(first) - tsOf(next) <= MERGE_WINDOW_MS) {
        group.push(next);
        head.idx++;
      } else {
        break;
      }
    }
    if (head.idx >= head.list.length) heads.splice(bestI, 1);

    if (group.length >= 2) {
      let totalLatency = 0;
      let latencyCount = 0;
      let totalIn = 0;
      let totalOut = 0;
      const errTypes = new Set<string>();
      for (const r of group) {
        totalIn += r.input_tokens;
        totalOut += r.output_tokens;
        if (r.latency_ms != null) {
          totalLatency += r.latency_ms;
          latencyCount++;
        }
        if (r.error_type) errTypes.add(r.error_type);
      }
      result.push({
        kind: 'merged',
        key: `m${group[0].id}-${group[group.length - 1].id}`,
        records: group,
        count: group.length,
        endpoint_id: group[0].endpoint_id,
        model_id: group[0].model_id,
        request_status: group[0].request_status,
        firstTimestamp: formatHM(group[group.length - 1].timestamp),
        lastTimestamp: formatHM(group[0].timestamp),
        totalInputTokens: totalIn,
        totalOutputTokens: totalOut,
        avgLatencyMs: latencyCount > 0 ? Math.round(totalLatency / latencyCount) : null,
        errorTypes: Array.from(errTypes),
      });
    } else {
      result.push({ kind: 'single', record: group[0] });
    }
  }
  return result;
}
