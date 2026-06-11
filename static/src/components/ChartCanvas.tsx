import { useRef, useEffect } from 'preact/hooks';
import {
  Chart,
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import type { ChartConfiguration } from 'chart.js';
import type { JSX } from 'preact';

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend, Filler);

export interface ChartCanvasProps {
  config: ChartConfiguration<'bar'>;
  height?: number;
  /** 鼠标 hover 柱子时触发（payload: {index, datasets, label, x, y} | null 表示离开） */
  onHover?: (payload: ChartHoverPayload | null) => void;
}

export interface ChartHoverPayload {
  /** 该 X 索引对应所有 dataset 的值（已和 datasets 配对） */
  rows: Array<{ label: string; value: number }>;
  /** X 轴标签（如 "2026-05-30"） */
  label: string;
  /** 鼠标 clientX/Y 用于定位 tooltip */
  clientX: number;
  clientY: number;
}

export function ChartCanvas({ config, height = 220, onHover }: ChartCanvasProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<Chart<'bar'> | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (chartRef.current) {
      chartRef.current.destroy();
    }
    const chart = new Chart(canvasRef.current, config);
    chartRef.current = chart;

    if (onHover) {
      const handleHover = (event: unknown, active: Array<{ index: number }>, _chart: unknown) => {
        if (!active || active.length === 0) {
          onHover(null);
          return;
        }
        const idx = active[0].index;
        const datasets = chart.data.datasets;
        const rows = datasets.map((ds, i) => ({
          label: String(ds.label ?? `Series ${i}`),
          value: Number(ds.data[idx] ?? 0),
        }));
        const native = (event as { native?: MouseEvent })?.native;
        onHover({
          rows,
          label: String(chart.data.labels?.[idx] ?? ''),
          clientX: native?.clientX ?? 0,
          clientY: native?.clientY ?? 0,
        });
      };
      chart.options.onHover = handleHover as unknown as (this: unknown, event: unknown, active: Array<{ index: number }>, chart: unknown) => void;
      chart.update();
    }

    return () => {
      chart.destroy();
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(config.data)]);

  return (
    <div
      style={{ position: 'relative', height: `${height}px` }}
      onMouseLeave={onHover ? () => onHover(null) : undefined}
    >
      <canvas ref={canvasRef} />
    </div>
  );
}
