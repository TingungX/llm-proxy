import { modelsSignal, errorHandlingSignal, configSignal, setConfig } from '../state/store';
import { deleteModel, testLatency } from '../api/models';
import { saveConfig, fetchConfig } from '../api/config';
import { ApiCallError } from '../api/client';
import { esc } from '../utils/format';
import { showToast } from '../components/Toast';
import { Toggle } from '../components/Toggle';
import { ProtocolBadge } from '../components/ProtocolChip';
import { getModelProtocols } from '../utils/protocol';
import { openModelModal } from '../modals/ModelModal';
import { EDIT_ICON, DELETE_ICON } from '../utils/icons';
import type { ModelConfig } from '../api/types';

function formatContextWindow(cw?: number): string {
  if (!cw) return '—';
  if (cw >= 1_000_000) return (cw / 1_000_000).toFixed(0) + 'M';
  if (cw >= 1_000) return (cw / 1_000).toFixed(0) + 'K';
  return String(cw);
}

async function handleTestLatency(name: string, cellEl: HTMLElement): Promise<void> {
  cellEl.textContent = '⏳';
  cellEl.style.color = '#f39c12';
  try {
    const d = await testLatency(name);
    if (d.error) {
      cellEl.textContent = '失败';
      cellEl.style.color = 'var(--danger)';
      showToast(d.error, 'err');
      return;
    }
    const avg = (d.avg * 1000).toFixed(0);
    cellEl.textContent = avg + 'ms';
    cellEl.style.color = Number(avg) > 3000 ? '#f39c12' : 'var(--accent)';
    showToast(`${name} 平均 ${avg}ms`, 'ok');
  } catch {
    cellEl.textContent = '失败';
    cellEl.style.color = 'var(--danger)';
    showToast('测试失败', 'err');
  }
}

async function handleDeleteModel(name: string): Promise<void> {
  if (!confirm(`删除模型 ${name}？`)) return;
  try {
    await deleteModel(name);
    const cfg = await fetchConfig();
    setConfig(cfg);
    showToast('模型已删除', 'ok');
  } catch (e) {
    const msg = e instanceof ApiCallError ? e.detail : '删除失败';
    showToast(msg, 'err');
  }
}

async function handleErrorHandlingChange(field: string, value: boolean): Promise<void> {
  const cfg = configSignal.value;
  if (!cfg) return;
  const newConfig = {
    ...cfg,
    error_handling: {
      ...cfg.error_handling,
      [field]: value,
    },
  };
  try {
    await saveConfig(newConfig);
    setConfig(newConfig);
    showToast('配置已保存', 'ok');
  } catch (e) {
    const msg = e instanceof ApiCallError ? e.detail : '保存失败';
    showToast(msg, 'err');
  }
}

declare global {
  interface Window {
    openModelModal?: (modelId?: string) => void;
  }
}

function openAddModel() { openModelModal(); }
function openEditModel(name: string) { openModelModal(name); }

export function ConfigPage() {
  const models = Object.entries(modelsSignal.value);
  const eh = errorHandlingSignal.value;

  return (
    <div class="two-col">
      <div class="card">
        <div class="card-header">
          <h2>已有模型</h2>
          <button onClick={() => openAddModel()}>添加模型</button>
        </div>
        <table>
          <thead>
            <tr>
              <th>模型</th>
              <th>协议</th>
              <th>上下文</th>
              <th>延迟</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {models.length === 0 ? (
              <tr>
                <td colspan={5} style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
                  暂无模型
                </td>
              </tr>
            ) : (
              models.map(([name, v]: [string, ModelConfig]) => {
                const displayName = v.display_name || v.upstream_model || name;
                return (
                  <tr key={name}>
                    <td>
                      <div class="value">{esc(displayName)}</div>
                      <div class="hint">{esc(name)}</div>
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
                        {getModelProtocols(v).length > 0
                          ? getModelProtocols(v).map(t => <ProtocolBadge key={t} label={t} />)
                          : <span style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)' }}>自动</span>}
                      </div>
                    </td>
                    <td>{formatContextWindow(v.context_window)}</td>
                    <td>
                      <span
                        class="value latency-cell"
                        style={{ cursor: 'pointer', color: 'var(--accent-dim)' }}
                        data-model={name}
                        onClick={(e: MouseEvent) => handleTestLatency(name, e.currentTarget as HTMLElement)}
                      >
                        测延迟
                      </span>
                    </td>
                    <td style={{ whiteSpace: 'nowrap' }}>
                      <button
                        class="icon"
                        onClick={() => openEditModel(name)}
                        title="编辑"
                        dangerouslySetInnerHTML={{ __html: EDIT_ICON }}
                      />
                      <button
                        class="icon"
                        style={{ color: 'var(--danger)' }}
                        onClick={() => handleDeleteModel(name)}
                        title="删除"
                        dangerouslySetInnerHTML={{ __html: DELETE_ICON }}
                      />
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <div class="card">
        <h2>错误处理配置</h2>
        <div style={{ marginTop: '8px' }}>
          <Toggle
            checked={eh.failover_enabled}
            onChange={(v: boolean) => handleErrorHandlingChange('failover_enabled', v)}
            label="自动转移"
          />
          <Toggle
            checked={eh.no_retry_enabled}
            onChange={(v: boolean) => handleErrorHandlingChange('no_retry_enabled', v)}
            label="取消重试"
          />
        </div>
      </div>
    </div>
  );
}
