import { useEffect } from 'preact/hooks';
import { Fragment } from 'preact';
import {
  endpointModalOpen, endpointModalEditing, closeEndpointModal,
  endpointsSignal, modelsSignal, setEndpoints, setConfig,
} from '../state/store';
import { openEndpointModal } from '../state/store';
import { fetchEndpoints, saveEndpoint, createEndpoint, deleteEndpoint } from '../api/endpoints';
import { fetchConfig } from '../api/config';
import { showToast } from '../components/Toast';
import { Toggle } from '../components/Toggle';
import { ApiCallError } from '../api/client';
import { useFormState } from '../hooks/useFormState';
import { togglePasswordVisibility } from '../utils/clipboard';
import { Modal } from '../components/Modal';
import { Field } from '../components/Field';
import { ProtocolChip, ProtocolBadge } from '../components/ProtocolChip';
import { getModelProtocols } from '../utils/protocol';
import { EmptyState } from '../components/EmptyState';
import { esc } from '../utils/format';

const COMPRESSION_STRATEGY_KEYS = ['drop_progress', 'truncate', 'collapse', 'shorten_paths'] as const;
type CompressionStrategyKey = typeof COMPRESSION_STRATEGY_KEYS[number];

const COMPRESSION_STRATEGY_LABELS: Record<CompressionStrategyKey, string> = {
  drop_progress: '丢弃进度条',
  truncate: '截断长代码',
  collapse: '折叠空行',
  shorten_paths: '相对路径',
};

interface Route {
  family: string;
  model: string;
}

interface EndpointFormValues extends Record<string, unknown> {
  name: string;
  apiKey: string;
  enabled: boolean;
  acceptAnthropic: boolean;
  acceptOpenaiChat: boolean;
  acceptOpenaiResponses: boolean;
  failoverEnabled: boolean;
  noRetryEnabled: boolean;
  compression: {
    enabled: boolean;
    strategies: Record<CompressionStrategyKey, boolean>;
  };
  selectedModels: Set<string>;
  filterText: string;
  routes: Route[];
}

function defaultStrategies(): Record<CompressionStrategyKey, boolean> {
  return { drop_progress: true, truncate: true, collapse: true, shorten_paths: true };
}

function getInitialValues(id: string | null): EndpointFormValues {
  if (id) {
    const ep = endpointsSignal.value.find(e => e.endpoint_id === id);
    if (ep) {
      const saved = new Set<string>(ep.settings?.compression?.strategies ?? []);
      const strategies = Object.fromEntries(
        COMPRESSION_STRATEGY_KEYS.map(k => [k, saved.has(k)]),
      ) as Record<CompressionStrategyKey, boolean>;
      const ap = ep.accept_protocols ?? [];
      return {
        name: ep.name ?? '',
        apiKey: ep.api_key ?? '',
        enabled: ep.enabled,
        acceptAnthropic: ap.includes('anthropic'),
        acceptOpenaiChat: ap.includes('openai'),
        acceptOpenaiResponses: ap.includes('openai'),
        failoverEnabled: ep.settings?.failover_enabled ?? false,
        noRetryEnabled: ep.settings?.no_retry_enabled ?? false,
        compression: { enabled: ep.settings?.compression?.enabled ?? false, strategies },
        selectedModels: new Set(ep.models ?? []),
        filterText: '',
        routes: Object.entries(ep.family_routing ?? {}).map(([family, model]) => ({ family, model })),
      };
    }
  }
  return {
    name: '', apiKey: '', enabled: true,
    acceptAnthropic: true, acceptOpenaiChat: true, acceptOpenaiResponses: true,
    failoverEnabled: false, noRetryEnabled: false,
    compression: { enabled: false, strategies: defaultStrategies() },
    selectedModels: new Set(), filterText: '', routes: [],
  };
}

export function EndpointModal() {
  const isOpen = endpointModalOpen.value;
  const editing = endpointModalEditing.value;
  const form = useFormState<EndpointFormValues>(getInitialValues(editing));

  useEffect(() => {
    if (isOpen) form.reset(getInitialValues(editing));
  }, [isOpen, editing]);

  if (!isOpen) return null;

  const models = Object.entries(modelsSignal.value);
  const filtered = models.filter(([n, m]) => {
    if (!form.values.value.filterText) return true;
    const q = form.values.value.filterText.toLowerCase();
    return n.toLowerCase().includes(q) || (m.display_name ?? '').toLowerCase().includes(q);
  });

  const toggleModel = (name: string) => {
    const next = new Set(form.values.value.selectedModels);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    form.setField('selectedModels', next);
  };

  const selectAll = () => {
    form.setField('selectedModels', new Set(filtered.map(([n]) => n)));
  };
  const deselectAll = () => {
    form.setField('selectedModels', new Set());
  };

  const onSave = async () => {
    const v = form.values.value;
    if (!v.name) { form.setErrors({ name: '必填' }); return; }
    const acceptProtocols: string[] = [];
    if (v.acceptAnthropic) acceptProtocols.push('anthropic');
    if (v.acceptOpenaiChat || v.acceptOpenaiResponses) acceptProtocols.push('openai');
    const strategies: string[] = [];
    if (v.compression.enabled) {
      for (const k of COMPRESSION_STRATEGY_KEYS) {
        if (v.compression.strategies[k]) strategies.push(k);
      }
    }
    const familyRouting: Record<string, string> = {};
    for (const r of v.routes) {
      if (r.family && r.model) familyRouting[r.family] = r.model;
    }
    const data = {
      name: v.name,
      api_key: v.apiKey || undefined,
      enabled: v.enabled,
      accept_protocols: acceptProtocols,
      models: Array.from(v.selectedModels),
      family_routing: familyRouting,
      settings: {
        failover_enabled: v.failoverEnabled,
        no_retry_enabled: v.noRetryEnabled,
        compression: v.compression.enabled ? { enabled: true, strategies } : undefined,
      },
    };
    try {
      await form.handleSubmit(async () => {
        if (editing) {
          await saveEndpoint(editing, data);
        } else {
          await createEndpoint(data);
        }
        const [eps, cfg] = await Promise.all([fetchEndpoints(), fetchConfig()]);
        setEndpoints(eps);
        setConfig(cfg);
        showToast('端点已保存', 'ok');
        closeEndpointModal();
      });
    } catch {
      // error surfaced via form.generalError.value
    }
  };

  const onDelete = async () => {
    if (!editing || editing === 'default') return;
    if (!confirm('确定删除此端点？')) return;
    try {
      await deleteEndpoint(editing);
      const eps = await fetchEndpoints();
      setEndpoints(eps);
      showToast('端点已删除', 'ok');
      closeEndpointModal();
    } catch (e) {
      showToast(e instanceof ApiCallError ? e.detail : '删除失败', 'err');
    }
  };

  const v = form.values.value;

  return (
    <Modal onClose={closeEndpointModal} size="xl">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
        <h2 class="modal-title">{editing ? '编辑端点' : '添加端点'}</h2>
        <Toggle checked={v.enabled} onChange={(val: boolean) => form.setField('enabled', val)} label="启用" />
      </div>
      {form.generalError.value && <div class="error-banner">{form.generalError.value}</div>}
      <div class="modal-cols-3">
        <div>
          <Field label="名称" hint={editing ? '不可修改' : ''} required error={form.errors.value.name}>
            <input class="w-full" value={v.name} disabled={!!editing} onInput={(e: Event) => form.setField('name', (e.target as HTMLInputElement).value)} />
          </Field>
          <Field label="API Key">
            <div style="display: flex; gap: 6px;">
              <input type="password" value={v.apiKey} placeholder={editing ? '(保留现有)' : ''} onInput={(e: Event) => form.setField('apiKey', (e.target as HTMLInputElement).value)} style="flex: 1;" />
              <button type="button" class="ghost" onClick={(e: MouseEvent) => togglePasswordVisibility((e.currentTarget as HTMLButtonElement).previousElementSibling as HTMLInputElement)}>显示</button>
            </div>
          </Field>
          <fieldset style="margin-bottom: 12px;">
            <legend>接受协议</legend>
            <div style="display: flex; flex-direction: column; gap: 6px; margin-top: 4px;">
              <ProtocolChip checked={v.acceptAnthropic} onToggle={() => form.setField('acceptAnthropic', !v.acceptAnthropic)} label="Anthropic" />
              <ProtocolChip checked={v.acceptOpenaiChat} onToggle={() => form.setField('acceptOpenaiChat', !v.acceptOpenaiChat)} label="OpenAI Chat" />
              <ProtocolChip checked={v.acceptOpenaiResponses} onToggle={() => form.setField('acceptOpenaiResponses', !v.acceptOpenaiResponses)} label="OpenAI Responses" />
            </div>
          </fieldset>
          <fieldset>
            <legend>容错</legend>
            <Toggle checked={v.failoverEnabled} onChange={(val: boolean) => form.setField('failoverEnabled', val)} label="故障转移" />
            <Toggle checked={v.noRetryEnabled} onChange={(val: boolean) => form.setField('noRetryEnabled', val)} label="不重试" />
          </fieldset>
        </div>
        <div>
          <fieldset style="margin-bottom: 12px;">
            <legend>压缩</legend>
            <Toggle
              checked={v.compression.enabled}
              onChange={(val: boolean) => form.setField('compression', { ...v.compression, enabled: val })}
              label="启用压缩"
            />
            {v.compression.enabled && (
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; margin-top: 6px;">
                {COMPRESSION_STRATEGY_KEYS.map((k) => (
                  <label key={k} style="display: flex; align-items: center; gap: 6px; font-size: var(--fs-sm);">
                    <input
                      type="checkbox"
                      checked={v.compression.strategies[k]}
                      onChange={(e: Event) => form.setField('compression', {
                        ...v.compression,
                        strategies: { ...v.compression.strategies, [k]: (e.target as HTMLInputElement).checked },
                      })}
                    />
                    {COMPRESSION_STRATEGY_LABELS[k]}
                  </label>
                ))}
              </div>
            )}
          </fieldset>
          <fieldset>
            <legend>允许模型 ({v.selectedModels.size}/{models.length})</legend>
            <div style="display: flex; gap: 6px; margin-bottom: 8px;">
              <input type="search" placeholder="搜索模型..." value={v.filterText} onInput={(e: Event) => form.setField('filterText', (e.target as HTMLInputElement).value)} style="flex: 1;" />
              <button type="button" class="ghost text-xs" onClick={selectAll}>全选</button>
              <button type="button" class="ghost text-xs" onClick={deselectAll}>清空</button>
            </div>
            <div style="max-height: 240px; overflow: auto; border: 1px solid var(--border); border-radius: 4px;">
              {filtered.map(([name, m]) => {
                const tags = getModelProtocols(m);
                const checked = v.selectedModels.has(name);
                return (
                  <div
                    key={name}
                    onClick={() => toggleModel(name)}
                    style={`display: flex; align-items: center; gap: 8px; padding: 7px 10px; cursor: pointer; border-bottom: 1px solid var(--border); background: ${checked ? 'rgba(78,204,163,0.06)' : 'transparent'};`}
                  >
                    <input type="checkbox" checked={checked} readOnly style="flex-shrink: 0;" />
                    <span class="text-sm" style="flex: 1; min-width: 0;">{m.display_name || name}</span>
                    <div style="display: flex; gap: 3px; flex-shrink: 0;">
                      {tags.map(t => <ProtocolBadge key={t} label={t} />)}
                    </div>
                  </div>
                );
              })}
              {filtered.length === 0 && <EmptyState compact>无匹配模型</EmptyState>}
            </div>
          </fieldset>
        </div>
        <div>
          <fieldset>
            <legend>模型映射</legend>
            <p class="text-xs text-muted" style="margin-bottom: 8px;">客户端请求的模型名 → 实际转发的模型</p>
            {v.routes.length > 0 && (
              <div style="display: grid; grid-template-columns: 1fr 1fr auto; gap: 4px 6px; align-items: center; margin-bottom: 8px;">
                <span class="text-xs text-muted" style="font-weight: 600;">客户端名称</span>
                <span class="text-xs text-muted" style="font-weight: 600;">实际模型</span>
                <span />
                {v.routes.map((r, i) => (
                  <Fragment key={i}>
                    <input
                      class="text-sm"
                      value={r.family}
                      placeholder="opus-4-7"
                      onInput={(e: Event) => {
                        const next = [...v.routes];
                        next[i] = { ...next[i], family: (e.target as HTMLInputElement).value };
                        form.setField('routes', next);
                      }}
                    />
                    <select
                      class="text-sm"
                      value={r.model}
                      onChange={(e: Event) => {
                        const next = [...v.routes];
                        next[i] = { ...next[i], model: (e.target as HTMLSelectElement).value };
                        form.setField('routes', next);
                      }}
                    >
                      {models.map(([n, m]) => <option value={n}>{esc(m.display_name || n)}</option>)}
                    </select>
                    <button
                      type="button"
                      class="icon icon-danger"
                      onClick={() => form.setField('routes', v.routes.filter((_, idx) => idx !== i))}
                    >×</button>
                  </Fragment>
                ))}
              </div>
            )}
            <button
              type="button"
              class="ghost text-sm"
              onClick={() => form.setField('routes', [...v.routes, { family: '', model: models[0]?.[0] ?? '' }])}
            >+ 添加映射</button>
          </fieldset>
        </div>
      </div>
      <div class="modal-actions between">
        <div>{editing && editing !== 'default' && <button class="danger" onClick={onDelete}>删除</button>}</div>
        <div style="display: flex; gap: 8px;">
          <button class="ghost" onClick={closeEndpointModal}>取消</button>
          <button class="primary" onClick={onSave} disabled={form.submitting.value}>{form.submitting.value ? '保存中...' : '保存'}</button>
        </div>
      </div>
    </Modal>
  );
}

export { openEndpointModal };
