import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks';
import {
  usageModeSignal, usageGroupBySignal, hourlyDayOffsetSignal,
  heatmapDaysSignal, usageEndpointFilterSignal, heatmapEndpointFilterSignal,
  usageSplitModeSignal,
  usageRefreshTrigger, heatmapRefreshTrigger,
  setHeatmapDays, setUsageMode, setUsageGroupBy, setHourlyDayOffset,
  setUsageEndpointFilter, setHeatmapEndpointFilter, setUsageSplitMode,
  loadHeatmapDaysFromLocalStorage,
  HEATMAP_DAYS_OPTIONS,
  type UsageSplitMode,
} from '../state/usage';
import { endpointsSignal, endpointNameMapSignal } from '../state/endpoints';
import { fetchUsage, fetchUsageSummary, fetchUsageHeatmap } from '../api/usage';
import { showToast } from '../components/Toast';
import { ChartCanvas } from '../components/ChartCanvas';
import type { ChartHoverPayload } from '../components/ChartCanvas';
import { formatNumber, esc } from '../utils/format';
import { localDateToBackendPrefix } from '../utils/date';
import type { UsageDataPoint, HeatmapDataPoint, UsageSummary } from '../api/types';
import type { ChartConfiguration } from 'chart.js';
import type { JSX } from 'preact';

// 每个模型分配一个"主色"，输入/输出用对比明显的两种色相：
//   输入 = 蓝色族（同模型不同色）   输出 = 绿色族（同模型不同色）
// 这样 stacked bar 上下两段能直接用色相区分（蓝/绿），同时同一种角色内不同模型用亮度区分。
const USAGE_INPUT_COLORS = ['#93c5fd', '#60a5fa', '#3b82f6', '#1d4ed8', '#1e3a8a', '#172554'];
const USAGE_OUTPUT_COLORS = ['#86efac', '#4ade80', '#22c55e', '#16a34a', '#15803d', '#14532d'];
const HEATMAP_LEVELS = ['heatmap-level-0', 'heatmap-level-1', 'heatmap-level-2', 'heatmap-level-3', 'heatmap-level-4'];
const MONTH_NAMES = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];

function computeLevels(data: HeatmapDataPoint[]): { p25: number; p50: number; p75: number } {
  const values = data.filter(d => d.total_tokens > 0).map(d => d.total_tokens);
  if (values.length === 0) return { p25: 0, p50: 0, p75: 0 };
  values.sort((a, b) => a - b);
  const pct = (p: number) => values[Math.min(Math.floor(values.length * p), values.length - 1)];
  return { p25: pct(0.25), p50: pct(0.50), p75: pct(0.75) };
}

function getLevel(tokens: number, t: { p25: number; p50: number; p75: number }): number {
  if (tokens === 0) return 0;
  if (tokens <= t.p25) return 1;
  if (tokens <= t.p50) return 2;
  if (tokens <= t.p75) return 3;
  return 4;
}

function formatDateStr(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function getTodayDate(): Date {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d;
}

function buildUsageChartConfig(
  data: UsageDataPoint[],
  groupBy: string,
  endpointNameMap: Record<string, string>,
  usageMode: string,
  splitMode: UsageSplitMode,
): ChartConfiguration<'bar'> {
  type Slot = Record<string, { in: number; out: number; total: number }>;
  const timeMap = new Map<string, Slot>();
  const groupKeys = new Set<string>();

  for (const item of data) {
    const displayKey = (groupBy === 'endpoint' && endpointNameMap[item.group_key])
      ? endpointNameMap[item.group_key]
      : item.group_key;
    groupKeys.add(displayKey);
    if (!timeMap.has(item.time)) {
      timeMap.set(item.time, {} as Slot);
    }
    const slot = timeMap.get(item.time)!;
    if (!(displayKey in slot)) slot[displayKey] = { in: 0, out: 0, total: 0 };
    slot[displayKey].in += item.input_tokens;
    slot[displayKey].out += item.output_tokens;
    slot[displayKey].total += item.total_tokens;
  }

  const times = Array.from(timeMap.keys()).sort();
  const groups = Array.from(groupKeys).sort();

  type Dataset = {
    label: string;
    data: number[];
    backgroundColor: string;
    stack?: string;
    borderWidth: number;
    borderRadius: number;
    barPercentage: number;
    categoryPercentage: number;
  };
  const datasets: Dataset[] = [];

  if (splitMode === 'merged') {
    // 合并：每组 1 个 dataset（total_tokens），不设 stack
    // scales.x/y.stacked: true 让 Chart.js 自动堆叠所有 dataset 到一根柱子
    for (let i = 0; i < groups.length; i++) {
      const group = groups[i];
      datasets.push({
        label: group,
        data: times.map(t => timeMap.get(t)?.[group]?.total ?? 0),
        backgroundColor: USAGE_INPUT_COLORS[i % USAGE_INPUT_COLORS.length],
        borderWidth: 0,
        borderRadius: 0,
        barPercentage: 0.60,
        categoryPercentage: 0.82,
      });
    }
  } else {
    // 分条：每天 2 根柱子（输入/输出），柱内按模型堆叠
    for (let i = 0; i < groups.length; i++) {
      const group = groups[i];
      datasets.push({
        label: `${group} · 输入`,
        data: times.map(t => timeMap.get(t)?.[group]?.in ?? 0),
        backgroundColor: USAGE_INPUT_COLORS[i % USAGE_INPUT_COLORS.length],
        stack: 'in',
        borderWidth: 0,
        borderRadius: 0,
        barPercentage: 0.60,
        categoryPercentage: 0.82,
      });
      datasets.push({
        label: `${group} · 输出`,
        data: times.map(t => timeMap.get(t)?.[group]?.out ?? 0),
        backgroundColor: USAGE_OUTPUT_COLORS[i % USAGE_OUTPUT_COLORS.length],
        stack: 'out',
        borderWidth: 0,
        borderRadius: 0,
        barPercentage: 0.60,
        categoryPercentage: 0.82,
      });
    }
  }

  return {
    type: 'bar',
    data: { labels: times, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        // 隐藏图例：颜色信息只用于内部堆叠，不直接展示
        legend: { display: false },
        // 关闭 chart.js 内置 tooltip；用 UsageChartTooltip 自定义（按角色/模型分卡片）
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          // 两种模式都用 stacked: true
          // - 合并：所有 dataset 堆叠到一根柱子
          // - 分条：stack: 'in' 和 stack: 'out' 自动分两组并排
          stacked: true,
          ticks: {
            color: '#7a8ba6',
            font: { size: 10 },
            maxRotation: 0,
            callback(this: { getLabelForValue?: (v: number) => string }, val: number | string, index: number, ticks: unknown[]) {
              const label = this.getLabelForValue ? this.getLabelForValue(val as number) : String(val);
              if (usageMode === '1h') {
                const tArr = ticks as Array<{ label: string }>;
                if (index === 0 || index === tArr.length - 1) return label;
                const parts = label.split(' ');
                return parts[1] || label;
              }
              return label;
            },
          },
          grid: { display: false },
          border: { color: '#1e3355', width: 1 },
        },
        y: {
          stacked: true,
          ticks: {
            color: '#7a8ba6',
            font: { size: 10 },
            callback: (v: number | string) => formatNumber(Number(v)),
            padding: 8,
          },
          grid: { display: false },
          border: { color: '#1e3355', width: 1 },
        },
      },
    },
  };
}

interface HeatmapRenderResult {
  monthsVNodes: JSX.Element[];
  gridVNodes: JSX.Element[];
  totalWeeks: number;
}

let tooltipEl: HTMLDivElement | null = null;

function showHeatmapTooltip(e: MouseEvent, date: string, tokens: number): void {
  if (!tooltipEl) {
    tooltipEl = document.createElement('div');
    tooltipEl.className = 'heatmap-tooltip';
    document.body.appendChild(tooltipEl);
  }
  tooltipEl.textContent = `${date}: ${formatNumber(tokens)} tokens`;
  tooltipEl.style.left = `${e.clientX + 12}px`;
  tooltipEl.style.top = `${e.clientY - 30}px`;
  tooltipEl.style.display = 'block';
}

function hideHeatmapTooltip(): void {
  if (tooltipEl) tooltipEl.style.display = 'none';
}

function renderHeatmapData(
  data: HeatmapDataPoint[],
  heatmapDays: number,
  onCellClick: (dateStr: string) => void,
): HeatmapRenderResult {
  const thresholds = computeLevels(data);
  const tokenMap = new Map(data.map(d => [d.date, d.total_tokens]));

  const today = getTodayDate();
  const startDate = new Date(today);
  startDate.setDate(startDate.getDate() - heatmapDays + 1);
  const startDay = startDate.getDay();
  startDate.setDate(startDate.getDate() - startDay);

  const endDate = new Date(today);
  endDate.setDate(endDate.getDate() + (6 - endDate.getDay()));

  const totalWeeks = Math.ceil((endDate.getTime() - startDate.getTime()) / (7 * 86400000));

  const weekPositions: { week: number; month: number }[] = [];
  let currentMonth = -1;
  let d = new Date(startDate);
  for (let w = 0; w < totalWeeks; w++) {
    const cellDate = new Date(d);
    cellDate.setDate(cellDate.getDate() + 1);
    if (cellDate.getMonth() !== currentMonth && cellDate.getDate() <= 7) {
      weekPositions.push({ week: w, month: cellDate.getMonth() });
      currentMonth = cellDate.getMonth();
    }
    d.setDate(d.getDate() + 7);
  }

  // 月份标签：用 grid-column-start 精确定位到对应周
  // CSS grid: 32px 留白 + repeat(N, 18px) 每周。column 1 = 第 0 周, column 2 = 第 1 周, ...
  // 月份从 week w 开始 → grid-column-start = w + 2（+1 是 32px 留白列，+1 是 1-based）
  const monthsVNodes: JSX.Element[] = [
    <span class="heatmap-month-label" key="spacer" />,
    ...weekPositions.map((wp) => (
      <span
        class="heatmap-month-label"
        key={`m-${wp.week}-${wp.month}`}
        style={{ gridColumnStart: wp.week + 2 }}
      >{MONTH_NAMES[wp.month]}</span>
    )),
  ];

  const gridVNodes: JSX.Element[] = [];
  for (let col = 0; col < totalWeeks; col++) {
    for (let row = 0; row < 7; row++) {
      const cellDate = new Date(startDate);
      cellDate.setDate(cellDate.getDate() + col * 7 + row);
      const dateStr = formatDateStr(cellDate);
      const tokens = tokenMap.get(dateStr) || 0;
      const level = getLevel(tokens, thresholds);
      const isFuture = cellDate > today;

      const cellStyle: JSX.CSSProperties | undefined = isFuture ? { visibility: 'hidden' } : undefined;
      const cellProps: Record<string, unknown> = {
        class: `heatmap-cell ${HEATMAP_LEVELS[level]}`,
        key: `hm-${dateStr}`,
        'data-date': dateStr,
        'data-tokens': String(tokens),
      };
      if (cellStyle) cellProps.style = cellStyle;
      if (!isFuture) {
        cellProps.onMouseEnter = (e: MouseEvent) => showHeatmapTooltip(e, dateStr, tokens);
        cellProps.onMouseLeave = hideHeatmapTooltip;
        cellProps.onClick = () => onCellClick(dateStr);
      }
      gridVNodes.push(<span {...(cellProps as JSX.HTMLAttributes<HTMLSpanElement>)} />);
    }
  }

  return { monthsVNodes, gridVNodes, totalWeeks };
}

export function UsagePage() {
  const usageMode = usageModeSignal.value;
  const usageGroupBy = usageGroupBySignal.value;
  const hourlyDayOffset = hourlyDayOffsetSignal.value;
  const heatmapDays = heatmapDaysSignal.value;
  const usageEpFilter = usageEndpointFilterSignal.value;
  const heatmapEpFilter = heatmapEndpointFilterSignal.value;
  const usageSplitMode = usageSplitModeSignal.value;
  const refreshTick = usageRefreshTrigger.value;
  const heatmapTick = heatmapRefreshTrigger.value;

  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [usageData, setUsageData] = useState<UsageDataPoint[]>([]);
  const [heatmapData, setHeatmapData] = useState<HeatmapDataPoint[]>([]);
  const [endpoints, setLocalEndpoints] = useState<Array<{ id: string; name: string }>>([]);
  const [usageLoading, setUsageLoading] = useState(false);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [chartHover, setChartHover] = useState<ChartHoverPayload | null>(null);
  const heatmapWrapperRef = useRef<HTMLDivElement | null>(null);
  const initialized = useRef(false);

  useEffect(() => {
    if (!initialized.current) {
      loadHeatmapDaysFromLocalStorage();
      initialized.current = true;
    }
  }, []);

  useEffect(() => {
    setLocalEndpoints(endpointsSignal.value.map(ep => ({ id: ep.endpoint_id, name: ep.name || ep.endpoint_id })));
  }, [endpointsSignal.value.length]);

  useEffect(() => {
    let cancelled = false;
    fetchUsageSummary()
      .then(s => { if (!cancelled) setSummary(s); })
      .catch(() => { /* silently fail */ });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setUsageLoading(true);

    const days = usageMode === '30d' ? 30 : 7;
    const granularity = usageMode === '1h' ? 'hour' : 'day';

    const params: Record<string, string | number> = {
      days,
      group_by: usageGroupBy,
      granularity,
    };
    if (usageEpFilter) params.endpoint_id = usageEpFilter;

    fetchUsage(params)
      .then(resp => {
        if (cancelled) return;
        let data = resp.data;

        if (usageMode === '1h') {
          const targetDate = new Date();
          targetDate.setDate(targetDate.getDate() - hourlyDayOffset);
          const datePrefix = localDateToBackendPrefix(targetDate);
          data = data.filter(item => item.time.startsWith(datePrefix));
        }

        setUsageData(data);
        setUsageLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setUsageLoading(false);
        showToast('加载用量数据失败', 'err');
      });

    return () => { cancelled = true; };
  }, [usageMode, usageGroupBy, hourlyDayOffset, usageEpFilter, refreshTick]);

  useEffect(() => {
    let cancelled = false;
    setHeatmapLoading(true);

    const params: Record<string, string | number> = {
      days: heatmapDays,
      view: 'heatmap',
    };
    if (heatmapEpFilter) params.endpoint_id = heatmapEpFilter;

    fetchUsageHeatmap(params)
      .then(resp => {
        if (cancelled) return;
        setHeatmapData(resp.data || []);
        setHeatmapLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setHeatmapLoading(false);
        showToast('加载热力图数据失败', 'err');
      });

    return () => { cancelled = true; };
  }, [heatmapDays, heatmapEpFilter, heatmapTick]);

  function handleHeatmapCellClick(dateStr: string): void {
    const today = getTodayDate();
    const clicked = new Date(dateStr + 'T00:00:00');
    const diffDays = Math.floor((today.getTime() - clicked.getTime()) / 86400000);
    if (diffDays < 0 || diffDays > 6) return;
    setUsageMode('1h');
    setHourlyDayOffset(diffDays);
    const chartEl = document.getElementById('usage-chart-section');
    if (chartEl) chartEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function handlePrevDay(): void { setHourlyDayOffset(hourlyDayOffsetSignal.value + 1); }
  function handleNextDay(): void { setHourlyDayOffset(hourlyDayOffsetSignal.value - 1); }

  function getDaySliderLabel(): string {
    const targetDate = new Date();
    targetDate.setDate(targetDate.getDate() - hourlyDayOffset);
    const localStr = formatDateStr(targetDate);
    return localStr.replace(/-/g, '/');
  }

  const chartConfig = usageData.length > 0
    ? buildUsageChartConfig(usageData, usageGroupBy, endpointNameMapSignal.value, usageMode, usageSplitMode)
    : null;

  const heatmapResult = heatmapData.length > 0
    ? renderHeatmapData(heatmapData, heatmapDays, handleHeatmapCellClick)
    : null;

  // 热力图渲染后：把"今天"列滚到容器 0.618 位置（黄金分割），右侧留 0.382 空白
  useEffect(() => {
    if (!heatmapResult) return;
    requestAnimationFrame(() => {
      const wrap = heatmapWrapperRef.current;
      if (!wrap) return;
      const todayColIdx = heatmapResult.totalWeeks - 1;
      const cellStride = 18 + 3;  // cell 18px + gap 3px
      const todayColX = todayColIdx * cellStride;
      const targetScroll = todayColX - wrap.clientWidth * 0.618;
      wrap.scrollLeft = Math.max(0, targetScroll);
    });
  }, [heatmapResult]);

  return (
    <div id="tab-usage">
      <div class="usage-overview-row">
        <div class="card usage-overview-card">
          <h2 class="section-title">用量概览</h2>
          <div class="usage-stats usage-stats-vertical">
            <div class="stat-box">
              <div class="value" id="stat-total">{summary ? formatNumber(summary.total_tokens) : '—'}</div>
              <div class="label">总 Token</div>
            </div>
            <div class="stat-box">
              <div class="value" id="stat-today">{summary ? formatNumber(summary.today_tokens) : '—'}</div>
              <div class="label">今日 Token</div>
            </div>
            <div class="stat-box">
              <div class="value" id="stat-requests">{summary ? formatNumber(summary.total_requests) : '—'}</div>
              <div class="label">总请求</div>
            </div>
            <div class="stat-box">
              <div class="value" id="stat-endpoints">{summary ? String(summary.active_endpoints) : '—'}</div>
              <div class="label">活跃端点</div>
            </div>
          </div>
        </div>

        <div class="card usage-heatmap-card">
          <div class="usage-toolbar">
            <h2 class="section-title">用量热力图</h2>
            <span style={{ flex: 1 }} />
            <select
              class="usage-ep-select"
              value={heatmapEpFilter}
              onChange={(e: Event) => setHeatmapEndpointFilter((e.target as HTMLSelectElement).value)}
            >
              <option value="">全部端点</option>
              {endpoints.map(ep => <option value={ep.id}>{esc(ep.name || ep.id)}</option>)}
            </select>
            <select
              class="usage-ep-select"
              value={String(heatmapDays)}
              onChange={(e: Event) => setHeatmapDays(Number((e.target as HTMLSelectElement).value) as 90 | 180 | 365)}
            >
              {HEATMAP_DAYS_OPTIONS.map(d => <option value={String(d)}>{d} 天</option>)}
            </select>
          </div>
          {heatmapLoading ? (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-muted)' }}>加载中...</div>
          ) : heatmapResult ? (
            <div class="heatmap-wrapper" ref={heatmapWrapperRef}>
              <div class="heatmap-months" style={`--week-count: ${heatmapResult.totalWeeks}`}>{heatmapResult.monthsVNodes}</div>
              <div class="heatmap-grid" style={`--week-count: ${heatmapResult.totalWeeks}`}>{heatmapResult.gridVNodes}</div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-muted)' }}>暂无热力图数据</div>
          )}
          <div class="heatmap-footer">
            <span class="heatmap-legend-label">少</span>
            {HEATMAP_LEVELS.map(l => <span class={`heatmap-cell ${l}`} />)}
            <span class="heatmap-legend-label">多</span>
          </div>
        </div>
      </div>

      <div class="card" id="usage-chart-section">
        <div class="usage-toolbar">
          <div class="seg-group" id="usage-group-btns">
            <button
              class={`seg-btn${usageGroupBy === 'model' ? ' active' : ''}`}
              onClick={() => setUsageGroupBy('model')}
            >按模型</button>
            <button
              class={`seg-btn${usageGroupBy === 'endpoint' ? ' active' : ''}`}
              onClick={() => setUsageGroupBy('endpoint')}
            >按端点</button>
          </div>
          <div class="seg-group" id="usage-time-btns">
            <button
              class={`seg-btn${usageMode === '30d' ? ' active' : ''}`}
              onClick={() => setUsageMode('30d')}
            >近 30 天</button>
            <button
              class={`seg-btn${usageMode === '7d' ? ' active' : ''}`}
              onClick={() => setUsageMode('7d')}
            >近 7 天</button>
            <button
              class={`seg-btn${usageMode === '1h' ? ' active' : ''}`}
              onClick={() => setUsageMode('1h')}
            >逐小时</button>
          </div>
          <div class="seg-group" id="usage-split-btns" title="选择输入/输出的显示方式">
            <button
              class={`seg-btn${usageSplitMode === 'merged' ? ' active' : ''}`}
              onClick={() => setUsageSplitMode('merged')}
            >合并</button>
            <button
              class={`seg-btn${usageSplitMode === 'split' ? ' active' : ''}`}
              onClick={() => setUsageSplitMode('split')}
            >分条</button>
          </div>
          <span style={{ flex: 1 }} />
          <select
            class="usage-ep-select"
            value={usageEpFilter}
            onChange={(e: Event) => setUsageEndpointFilter((e.target as HTMLSelectElement).value)}
          >
            <option value="">全部端点</option>
            {endpoints.map(ep => <option value={ep.id}>{esc(ep.name || ep.id)}</option>)}
          </select>
        </div>
        <div
          class={`hourly-day-slider${usageMode === '1h' ? ' visible' : ''}`}
          id="usage-day-slider"
        >
          <button class="day-arrow" id="usage-day-prev" disabled={hourlyDayOffset >= 6} onClick={handlePrevDay}>← 前一天</button>
          <span class="day-label" id="usage-day-label">{getDaySliderLabel()}</span>
          <button class="day-arrow" id="usage-day-next" disabled={hourlyDayOffset <= 0} onClick={handleNextDay}>后一天 →</button>
        </div>
        <div class="chart-container">
          {usageLoading ? (
            <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>加载中...</div>
          ) : chartConfig ? (
            <>
              <ChartCanvas config={chartConfig} height={400} onHover={setChartHover} />
              <UsageChartTooltip payload={chartHover} splitMode={usageSplitMode} />
            </>
          ) : (
            <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>暂无用量数据</div>
          )}
        </div>
      </div>
    </div>
  );
}

interface UsageChartTooltipProps {
  payload: ChartHoverPayload | null;
  splitMode: UsageSplitMode;
}

function UsageChartTooltip({ payload, splitMode }: UsageChartTooltipProps): JSX.Element | null {
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  // 渲染后用 ref 测量真实尺寸，做边界检测
  useLayoutEffect(() => {
    if (!payload || !tooltipRef.current) {
      setPos(null);
      return;
    }
    const el = tooltipRef.current;
    const W = el.offsetWidth;
    const H = el.offsetHeight;
    const pad = 12;  // 距视口边缘的内边距
    // 默认：tooltip 显示在鼠标右上方（不遮挡鼠标）
    // - left = clientX + 16
    // - top  = clientY - H - 16（显示在鼠标上方）
    let left = payload.clientX + 16;
    let top = payload.clientY - H - 16;

    // 右边超出 → 翻到鼠标左侧
    if (left + W > window.innerWidth - pad) {
      left = payload.clientX - W - 16;
    }
    // 左边超出 → 夹紧到 pad
    if (left < pad) {
      left = pad;
    }
    // 上方超出 → 翻到鼠标下方
    if (top < pad) {
      top = payload.clientY + 16;
    }
    // 下方超出 → 夹紧
    if (top + H > window.innerHeight - pad) {
      top = window.innerHeight - H - pad;
    }
    setPos({ left, top });
  }, [payload, splitMode]);

  if (!payload) return null;
  const { rows, label } = payload;
  // 合并模式：rows 形如 "model-name" → 直接显示
  // 分条模式：rows 形如 "model-name · 输入" / "model-name · 输出" → 按模型聚合
  type ModelRow = { model: string; in: number; out: number; total: number };
  const byModel = new Map<string, ModelRow>();
  for (const r of rows) {
    const sepIdx = r.label.lastIndexOf(' · ');
    if (sepIdx < 0) {
      // 合并模式
      const m = r.label;
      if (!byModel.has(m)) byModel.set(m, { model: m, in: 0, out: 0, total: 0 });
      byModel.get(m)!.total += r.value;
    } else {
      const model = r.label.slice(0, sepIdx);
      const role = r.label.slice(sepIdx + 3); // "输入" or "输出"
      if (!byModel.has(model)) byModel.set(model, { model, in: 0, out: 0, total: 0 });
      if (role === '输入') byModel.get(model)!.in += r.value;
      else if (role === '输出') byModel.get(model)!.out += r.value;
    }
  }
  const modelRows = Array.from(byModel.values()).sort((a, b) =>
    (b.in + b.out + b.total) - (a.in + a.out + a.total),
  );
  const fmtIn = (n: number) => n === 0 ? '—' : formatNumber(n);
  const fmtOut = (n: number) => n === 0 ? '—' : formatNumber(n);
  const fmtTotal = (n: number) => n === 0 ? '—' : formatNumber(n);

  return (
    <div
      ref={tooltipRef}
      class="usage-chart-tooltip"
      style={pos ? { left: `${pos.left}px`, top: `${pos.top}px` } : { left: '-9999px', top: '-9999px' }}
    >
      <div class="usage-chart-tooltip-title">{label}</div>
      <div class="usage-chart-tooltip-grid">
        {modelRows.map((m) => (
          <div class="usage-chart-tooltip-card" key={m.model}>
            <div class="usage-chart-tooltip-model">{m.model}</div>
            {splitMode === 'split' ? (
              <>
                <div class="usage-chart-tooltip-row">
                  <span class="role-tag role-in">输入</span>
                  <span class="val">{fmtIn(m.in)}</span>
                </div>
                <div class="usage-chart-tooltip-row">
                  <span class="role-tag role-out">输出</span>
                  <span class="val">{fmtOut(m.out)}</span>
                </div>
              </>
            ) : (
              <div class="usage-chart-tooltip-row single">
                <span class="role-tag">合计</span>
                <span class="val">{fmtTotal(m.total)}</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
