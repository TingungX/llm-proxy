import { useEffect, useMemo, useState } from 'preact/hooks';
import {
  logsSignal, logsTotalSignal, logsOffsetSignal, logsLimitSignal,
  logsLoadingSignal, logFilterSignal, hasActiveFiltersSignal,
  mergeConsecutiveSignal, setMergeConsecutive,
  endpointsSignal, setEndpoints,
  loadMergeConsecutiveFromLocalStorage,
  filterDrawerOpenSignal, activeFilterCountSignal,
} from '../state/store';
import type { LogFilter } from '../state/logs';
import { fetchLogs, fetchLogsSummary, fetchFilterOptions } from '../api/logs';
import { fetchEndpoints } from '../api/endpoints';
import { showToast } from '../components/Toast';
import { esc, formatNumber } from '../utils/format';
import { formatDateTime, localDateTimeToBackend } from '../utils/date';
import { Modal } from '../components/Modal';
import { EmptyState } from '../components/EmptyState';
import { FilterDock } from '../components/FilterDock';
import { FilterDrawer } from '../components/FilterDrawer';
import { CHEVRON_RIGHT_ICON, CHEVRON_DOWN_ICON } from '../utils/icons';
import { aggregateRecords, type LogTimelineItem, type MergedLogGroup } from '../utils/logGrouping';
import type { LogRecord, LogsSummary, FilterOptions } from '../api/types';

function MergedGroupRow({
  group, epMap, expanded, onToggle, onSelectChild,
}: {
  group: MergedLogGroup;
  epMap: Map<string, string>;
  expanded: boolean;
  onToggle: () => void;
  onSelectChild: (r: LogRecord) => void;
}) {
  const isError = group.request_status === 'error';
  const timeRange = group.firstTimestamp === group.lastTimestamp
    ? group.lastTimestamp
    : `${group.firstTimestamp} → ${group.lastTimestamp}`;
  const epName = epMap.get(group.endpoint_id) || group.endpoint_id.slice(0, 12);
  return (
    <div class={`log-entry log-group log-${group.request_status}`}>
      <div class="log-group-header" onClick={onToggle}>
        <div class="log-group-row1">
          <span class={`badge badge-${isError ? 'err' : 'ok'} log-group-badge`}>
            <span>×{group.count} {group.request_status}</span>
            <span class="log-group-chevron" dangerouslySetInnerHTML={{ __html: expanded ? CHEVRON_DOWN_ICON : CHEVRON_RIGHT_ICON }} />
          </span>
          <span class="log-group-meta">
            <code>{esc(epName)}</code> · {esc(group.model_id)} · {formatNumber(group.totalInputTokens)}↑ / {formatNumber(group.totalOutputTokens)}↓
            {group.avgLatencyMs != null && <span class="log-entry-latency"> · avg {group.avgLatencyMs}ms</span>}
            {group.errorTypes.length > 0 && <span class="log-entry-error"> · {group.errorTypes.map(esc).join(' · ')}</span>}
          </span>
          <span class="text-xs text-muted log-group-time">{timeRange}</span>
        </div>
      </div>
      {expanded && (
        <div class="log-group-children">
          {group.records.map(r => (
            <div
              key={r.id}
              class="log-entry log-group-child"
              onClick={(e: Event) => { e.stopPropagation(); onSelectChild(r); }}
            >
              <div class="log-entry-header">
                <span class={`badge badge-${isError ? 'err' : 'ok'}`}>{r.request_status}</span>
                <span class="text-xs text-muted">{formatDateTime(r.timestamp)}</span>
              </div>
              <div class="log-entry-meta">
                <span>
                  <code>{esc(epName)}</code> · {esc(r.model_id)} · {r.input_tokens}↑ / {r.output_tokens}↓
                  {r.latency_ms != null && <span class="log-entry-latency"> · {r.latency_ms}ms</span>}
                  {r.error_type && <span class="log-entry-error"> · {esc(r.error_type)}</span>}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function LogTimeline({
  items, epMap, expandedKeys, onToggleExpand, onSelect,
}: {
  items: LogTimelineItem[];
  epMap: Map<string, string>;
  expandedKeys: Set<string>;
  onToggleExpand: (key: string) => void;
  onSelect: (r: LogRecord) => void;
}) {
  if (items.length === 0) {
    return <EmptyState>暂无日志记录</EmptyState>;
  }
  return (
    <div class="log-timeline">
      {items.map((it) => {
        if (it.kind === 'merged') {
          return (
            <MergedGroupRow
              key={it.key}
              group={it}
              epMap={epMap}
              expanded={expandedKeys.has(it.key)}
              onToggle={() => onToggleExpand(it.key)}
              onSelectChild={onSelect}
            />
          );
        }
        const r = it.record;
        return (
          <div
            key={r.id}
            class={`log-entry log-${r.request_status}`}
            onClick={() => onSelect(r)}
          >
            <div class="log-entry-header">
              <span class={`badge badge-${r.request_status === 'error' ? 'err' : 'ok'}`}>{r.request_status}</span>
              <span class="text-xs text-muted">{formatDateTime(r.timestamp)}</span>
            </div>
            <div class="log-entry-meta">
              <span>
                <code>{esc(epMap.get(r.endpoint_id) || r.endpoint_id.slice(0, 12))}</code> · {esc(r.model_id)} · {r.input_tokens}↑ / {r.output_tokens}↓
                {r.latency_ms != null && <span class="log-entry-latency"> · {r.latency_ms}ms</span>}
                {r.error_type && <span class="log-entry-error"> · {esc(r.error_type)}</span>}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Pagination({
  total, limit, offset, onChange,
}: {
  total: number;
  limit: number;
  offset: number;
  onChange: (newOffset: number) => void;
}) {
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);
  return (
    <div class="pagination">
      <span>显示 {start}-{end} 共 {total}</span>
      <div class="pagination-buttons">
        <button disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - limit))}>上一页</button>
        <button disabled={end >= total} onClick={() => onChange(offset + limit)}>下一页</button>
      </div>
    </div>
  );
}

function LogDetailModal({ record, epMap, onClose }: { record: LogRecord | null; epMap: Map<string, string>; onClose: () => void }) {
  if (!record) return null;
  return (
    <Modal onClose={onClose} size="sm">
      <h2 class="modal-title">日志详情</h2>
      <table style="margin-top: 16px;">
        <tbody>
          <tr><td class="k">ID</td><td>{record.id}</td></tr>
          <tr><td class="k">时间</td><td>{formatDateTime(record.timestamp)}</td></tr>
          <tr><td class="k">端点</td><td><code>{esc(epMap.get(record.endpoint_id) || record.endpoint_id)}</code></td></tr>
          <tr><td class="k">模型</td><td>{esc(record.model_id)}</td></tr>
          <tr><td class="k">状态</td><td>{record.request_status}</td></tr>
          <tr><td class="k">输入 Tokens</td><td>{formatNumber(record.input_tokens)}</td></tr>
          <tr><td class="k">输出 Tokens</td><td>{formatNumber(record.output_tokens)}</td></tr>
          <tr><td class="k">延迟</td><td>{record.latency_ms != null ? `${record.latency_ms}ms` : '—'}</td></tr>
          <tr><td class="k">错误类型</td><td>{record.error_type ?? '—'}</td></tr>
          <tr><td class="k">Request ID</td><td><code>{record.request_id ?? '—'}</code></td></tr>
          <tr><td class="k">客户端 IP</td><td>{record.client_ip ?? '—'}</td></tr>
          <tr><td class="k">User-Agent</td><td>{record.user_agent ?? '—'}</td></tr>
        </tbody>
      </table>
      <div class="modal-actions">
        <button onClick={onClose}>关闭</button>
      </div>
    </Modal>
  );
}

export function LogsPage() {
  const [summary, setSummary] = useState<LogsSummary | null>(null);
  const [options, setOptions] = useState<FilterOptions | null>(null);
  const [selectedRecord, setSelectedRecord] = useState<LogRecord | null>(null);
  const [expandedKeys, setExpandedKeys] = useState<Set<string>>(new Set());

  const endpoints = endpointsSignal.value;
  const epMap = new Map(endpoints.map(e => [e.endpoint_id, e.name || e.endpoint_id.slice(0, 8)]));
  const records = logsSignal.value;
  const total = logsTotalSignal.value;
  const offset = logsOffsetSignal.value;
  const limit = logsLimitSignal.value;
  const loading = logsLoadingSignal.value;
  const f = logFilterSignal.value;
  const merge = mergeConsecutiveSignal.value;

  useEffect(() => {
    loadMergeConsecutiveFromLocalStorage();
  }, []);

  const items: LogTimelineItem[] = useMemo(
    () => (merge ? aggregateRecords(records) : records.map((r) => ({ kind: 'single' as const, record: r }))),
    [records, merge],
  );

  const loadData = async () => {
    logsLoadingSignal.value = true;
    try {
      const params: Record<string, string | number> = { limit, offset };
      if (f.endpoint_id) params.endpoint_id = f.endpoint_id;
      if (f.model_id) params.model_id = f.model_id;
      if (f.status) params.status = f.status;
      const sinceUtc = localDateTimeToBackend(f.since);
      const untilUtc = localDateTimeToBackend(f.until);
      if (sinceUtc) params.since = sinceUtc;
      if (untilUtc) params.until = untilUtc;
      const summaryParams: { endpoint_id?: string; model_id?: string; since?: string; until?: string } = {};
      if (f.endpoint_id) summaryParams.endpoint_id = f.endpoint_id;
      if (f.model_id) summaryParams.model_id = f.model_id;
      if (sinceUtc) summaryParams.since = sinceUtc;
      if (untilUtc) summaryParams.until = untilUtc;
      const [resp, sum, opts] = await Promise.all([
        fetchLogs(params),
        fetchLogsSummary(summaryParams),
        fetchFilterOptions(),
      ]);
      logsSignal.value = resp.records;
      logsTotalSignal.value = resp.total;
      setSummary(sum);
      setOptions(opts);
      setExpandedKeys(new Set());
    } catch {
      showToast('加载日志失败', 'err');
    } finally {
      logsLoadingSignal.value = false;
    }
  };

  useEffect(() => {
    void loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offset, limit, f.endpoint_id, f.model_id, f.status, f.since, f.until]);

  useEffect(() => {
    fetchEndpoints().then(setEndpoints).catch(() => {});
  }, []);

  const onFilterChange = (field: keyof LogFilter, value: string) => {
    logFilterSignal.value = { ...logFilterSignal.value, [field]: value };
    logsOffsetSignal.value = 0;
  };

  const onLimitChange = (newLimit: number) => {
    logsLimitSignal.value = newLimit;
    logsOffsetSignal.value = 0;
  };

  const onClearAll = () => {
    logFilterSignal.value = { endpoint_id: '', model_id: '', status: '', since: '', until: '' };
    logsOffsetSignal.value = 0;
  };

  const toggleExpand = (key: string) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <div id="tab-logs">
      <div class="card">
        <h2>日志概览</h2>
        {summary ? (
          <div class="usage-stats">
            <div class="stat-box">
              <div class="value">{formatNumber(summary.total_requests)}</div>
              <div class="label">总请求</div>
            </div>
            <div class="stat-box">
              <div class="value">{formatNumber(summary.error_count)}</div>
              <div class="label">错误数</div>
            </div>
          </div>
        ) : (
          <p class="text-muted">加载中...</p>
        )}
      </div>

      <div class="card">
        <div class="card-header">
          <h2>请求日志</h2>
        </div>
        {loading ? (
          <div class="loading-state">加载中...</div>
        ) : (
          <>
            <LogTimeline
              items={items}
              epMap={epMap}
              expandedKeys={expandedKeys}
              onToggleExpand={toggleExpand}
              onSelect={setSelectedRecord}
            />
            <Pagination
              total={total}
              limit={limit}
              offset={offset}
              onChange={(o) => { logsOffsetSignal.value = o; }}
            />
          </>
        )}
      </div>

      <FilterDock
        onOpen={() => { filterDrawerOpenSignal.value = true; }}
        activeCount={activeFilterCountSignal.value}
      />
      <FilterDrawer
        open={filterDrawerOpenSignal.value}
        onClose={() => { filterDrawerOpenSignal.value = false; }}
        options={options}
        filter={f}
        limit={limit}
        merge={merge}
        onFilterChange={onFilterChange}
        onLimitChange={onLimitChange}
        onToggleMerge={setMergeConsecutive}
        onClear={onClearAll}
        hasActiveFilters={hasActiveFiltersSignal.value}
      />

      <LogDetailModal record={selectedRecord} epMap={epMap} onClose={() => setSelectedRecord(null)} />
    </div>
  );
}
