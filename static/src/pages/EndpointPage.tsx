import { useEffect, useState } from 'preact/hooks';
import {
  endpointsSignal,
  modelsSignal,
  currentEndpointIdSignal,
  epModelFilterSignal,
  epModelSortSignal,
  setEpModelFilter,
  setEpModelSort,
} from '../state/store';
import { fetchUsage } from '../api/usage';
import { formatNumber, esc } from '../utils/format';
import { ChartCanvas } from '../components/ChartCanvas';
import { EndpointDock } from '../components/EndpointDock';
import { ProtocolBadge } from '../components/ProtocolChip';
import { getModelProtocols } from '../utils/protocol';
import type { UsageDataPoint } from '../api/types';
import type { ChartConfiguration } from 'chart.js';

function buildUsageChart(records: UsageDataPoint[]): ChartConfiguration<'bar'> {
  const byDay: Record<string, number> = {};
  for (const r of records) {
    byDay[r.time] = (byDay[r.time] ?? 0) + r.total_tokens;
  }
  const labels = Object.keys(byDay).sort();
  return {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: '合计', data: labels.map(l => byDay[l]), backgroundColor: '#4f8cff' },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true },
        y: {
          stacked: true,
          beginAtZero: true,
          ticks: { callback: (v: number | string) => formatNumber(Number(v)) },
        },
      },
      plugins: { legend: { display: false } },
    },
  };
}

export function EndpointPage() {
  const epId = currentEndpointIdSignal.value;
  const ep = epId ? endpointsSignal.value.find(e => e.endpoint_id === epId) : null;
  const filterText = epModelFilterSignal.value;
  const sortMode = epModelSortSignal.value;

  const [usageChart, setUsageChart] = useState<ChartConfiguration<'bar'> | null>(null);
  const [usageRecords, setUsageRecords] = useState<UsageDataPoint[]>([]);

  useEffect(() => {
    if (!ep) { setUsageChart(null); setUsageRecords([]); return; }
    (async () => {
      try {
        const r = await fetchUsage({ days: 7, group_by: 'model', endpoint_id: ep.endpoint_id });
        const data = r.data ?? [];
        setUsageRecords(data);
        setUsageChart(buildUsageChart(data));
      } catch {
        setUsageRecords([]);
        setUsageChart(buildUsageChart([]));
      }
    })();
  }, [ep?.endpoint_id]);

  if (!ep) {
    return (
      <div class="card">
        <p style={{ color: 'var(--text-muted)' }}>请选择端点标签查看详情。</p>
      </div>
    );
  }

  const models = Object.entries(modelsSignal.value);
  const filtered = models.filter(([_, m]) => {
    if (!filterText) return true;
    const q = filterText.toLowerCase();
    return (m.display_name ?? '').toLowerCase().includes(q) || _.toLowerCase().includes(q);
  });
  const sorted = [...filtered].sort(([aName, aCfg], [bName, bCfg]) => {
    if (sortMode === 'name-asc') return (aCfg.display_name ?? aName).localeCompare(bCfg.display_name ?? bName);
    if (sortMode === 'name-desc') return (bCfg.display_name ?? bName).localeCompare(bCfg.display_name ?? aName);
    return 0;
  });

  const totalIn = usageRecords.reduce((s, r) => s + r.input_tokens, 0);
  const totalOut = usageRecords.reduce((s, r) => s + r.output_tokens, 0);
  const totalReq = usageRecords.reduce((s, r) => s + r.count, 0);

  return (
    <div>
      <EndpointDock endpoint={ep} />

      <div class="card">
        <h2>7 天用量</h2>
        <div class="usage-stats">
          <div class="stat-box">
            <div class="value">{formatNumber(totalIn)}</div>
            <div class="label">总输入</div>
          </div>
          <div class="stat-box">
            <div class="value">{formatNumber(totalOut)}</div>
            <div class="label">总输出</div>
          </div>
          <div class="stat-box">
            <div class="value">{formatNumber(totalReq)}</div>
            <div class="label">请求数</div>
          </div>
        </div>
        {usageChart ? <ChartCanvas config={usageChart} height={220} /> : <p>暂无数据</p>}
      </div>

      <div class="card">
        <div class="card-header">
          <h2>模型清单</h2>
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              type="search"
              placeholder="搜索模型..."
              value={filterText}
              onInput={(e: Event) => setEpModelFilter((e.target as HTMLInputElement).value)}
            />
            <select
              value={sortMode}
              onChange={(e: Event) => setEpModelSort((e.target as HTMLSelectElement).value as typeof sortMode)}
            >
              <option value="name-asc">名称 A-Z</option>
              <option value="name-desc">名称 Z-A</option>
              <option value="mapping">按映射</option>
            </select>
          </div>
        </div>
        <table>
          <thead>
            <tr>
              <th>模型</th>
              <th>协议</th>
              <th>上游</th>
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colspan={3} style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '12px' }}>
                  无匹配模型
                </td>
              </tr>
            ) : (
              sorted.map(([name, m]) => (
                <tr key={name}>
                  <td>
                    <div class="value">{esc(m.display_name ?? name)}</div>
                    <div class="hint">{esc(name)}</div>
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
                      {getModelProtocols(m).length > 0
                        ? getModelProtocols(m).map(t => <ProtocolBadge key={t} label={t} />)
                        : <span style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)' }}>—</span>}
                    </div>
                  </td>
                  <td><code>{esc(m.upstream_model)}</code></td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
