import { useEffect, useState } from 'preact/hooks';
import { modelsSignal, configSignal, setConfig, modelModalOpen, modelModalEditing, closeModelModal } from '../state/store';
import { saveModel } from '../api/models';
import { detectProtocol as apiDetectProtocol } from '../api/endpoints';
import { fetchConfig } from '../api/config';
import { fetchProviderProfiles } from '../api/providers';
import { showToast } from '../components/Toast';
import { ApiCallError } from '../api/client';
import { useFormState } from '../hooks/useFormState';
import { togglePasswordVisibility } from '../utils/clipboard';
import { toUpstreamProtocols, extractProtocolConfig } from '../utils/protocol';
import { Modal } from '../components/Modal';
import { Field } from '../components/Field';
import { ProtocolChip } from '../components/ProtocolChip';
import { Toggle } from '../components/Toggle';
import { ComboBox } from '../components/ComboBox';
import { EffortStrategyFieldset } from '../components/EffortStrategyFieldset';
import type { ModelConfig, ProviderProfileInfo } from '../api/types';

interface ModelFormValues extends Record<string, unknown> {
  name: string;
  displayName: string;
  apiBase: string;
  apiKey: string;
  contextWindow: string;
  contextWindowUnit: string;
  chipAnthropic: boolean;
  chipOpenaiChat: boolean;
  chipOpenaiResponses: boolean;
  pathAnthropic: string;
  pathOpenaiChat: string;
  pathOpenaiResponses: string;
  visionSupport: boolean;
  allowProxy: boolean;
  provider: string;
  thinkingEffortMode: 'default' | 'provider' | 'custom';
  thinkingEffortCustomType: string;
  thinkingEffortCustomRules: Record<string, string>;
}

function emptyRules(): Record<string, string> {
  return { low: 'low', medium: 'medium', high: 'high', '*': 'medium' };
}

function getInitialValues(name: string | null): ModelFormValues {
  const base: ModelFormValues = {
    name: '', displayName: '', apiBase: '', apiKey: '',
    contextWindow: '', contextWindowUnit: '1000',
    chipAnthropic: false, chipOpenaiChat: false, chipOpenaiResponses: false,
    pathAnthropic: '', pathOpenaiChat: '', pathOpenaiResponses: '',
    visionSupport: false, allowProxy: false, provider: '',
    thinkingEffortMode: 'default',
    thinkingEffortCustomType: 'any_to_any',
    thinkingEffortCustomRules: emptyRules(),
  };
  if (name) {
    const m: ModelConfig | undefined = (modelsSignal.value as Record<string, ModelConfig>)[name];
    if (m) {
      let cw = '';
      let cwUnit = '1000';
      if (m.context_window) {
        if (m.context_window >= 1_000_000) { cw = String(m.context_window / 1_000_000); cwUnit = '1000000'; }
        else { cw = String(m.context_window / 1000); cwUnit = '1000'; }
      }
      const pc = extractProtocolConfig(m);

      // 解析 thinking effort 模式
      let mode = m.thinking_effort_mode;
      const presetVal = m.thinking_effort_preset;
      if (!mode) {
        // 向后兼容：老配置无 mode 但有 preset → 视为 custom
        if (presetVal) {
          mode = 'custom';
        } else if (m.provider) {
          // 有厂商的模型默认使用厂商映射
          mode = 'provider';
        } else {
          mode = 'default';
        }
      }
      let customType = 'any_to_any';
      let customRules: Record<string, string> = emptyRules();
      if (mode === 'custom') {
        if (typeof presetVal === 'object' && presetVal && 'rules' in presetVal) {
          customType = presetVal.type ?? 'any_to_any';
          customRules = { ...presetVal.rules };
        } else if (typeof presetVal === 'string' && presetVal) {
          // 命名引用 → 从全局 presets 复制
          const globalPreset = configSignal.value?.thinking_effort_mapping?.presets.find(p => p.name === presetVal);
          if (globalPreset) {
            customType = globalPreset.type;
            customRules = { ...globalPreset.rules };
          }
        }
      }

      return {
        ...base,
        name,
        displayName: m.display_name ?? '',
        apiBase: m.api_base ?? '',
        apiKey: m.api_key ?? '',
        contextWindow: cw,
        contextWindowUnit: cwUnit,
        chipAnthropic: pc.enabled.anthropic,
        chipOpenaiChat: pc.enabled.openai_chat,
        chipOpenaiResponses: pc.enabled.openai_responses,
        pathAnthropic: pc.paths.anthropic,
        pathOpenaiChat: pc.paths.openai_chat,
        pathOpenaiResponses: pc.paths.openai_responses,
        visionSupport: m.vision_support ?? false,
        allowProxy: m.allow_proxy ?? false,
        provider: m.provider ?? '',
        thinkingEffortMode: mode as 'default' | 'provider' | 'custom',
        thinkingEffortCustomType: customType,
        thinkingEffortCustomRules: customRules,
      };
    }
  }
  return base;
}

export function ModelModal() {
  const isOpen = modelModalOpen.value;
  const editing = modelModalEditing.value;
  const form = useFormState<ModelFormValues>(getInitialValues(editing));
  const [profiles, setProfiles] = useState<Record<string, ProviderProfileInfo>>({});

  useEffect(() => {
    if (isOpen) form.reset(getInitialValues(editing));
  }, [isOpen, editing]);

  useEffect(() => {
    if (!isOpen) return;
    fetchProviderProfiles().then(setProfiles).catch(() => {
      // 失败时静默，provider 字段仍可手动输入
    });
  }, [isOpen]);

  if (!isOpen) return null;

  // 订阅 form.values：必须读 .value 才能在 setField 后重渲染
  const v = form.values.value;
  const pc = { enabled: { anthropic: v.chipAnthropic, openai_chat: v.chipOpenaiChat, openai_responses: v.chipOpenaiResponses }, paths: { anthropic: v.pathAnthropic, openai_chat: v.pathOpenaiChat, openai_responses: v.pathOpenaiResponses } };

  const onSave = async () => {
    if (!v.name) { form.setErrors({ name: '必填' }); return; }
    const data: Record<string, unknown> = {
      display_name: v.displayName || undefined,
      api_base: v.apiBase,
      upstream_model: v.name,
      upstream_protocols: toUpstreamProtocols(pc),
      context_window: v.contextWindow ? Math.round(Number(v.contextWindow) * Number(v.contextWindowUnit)) : undefined,
      vision_support: v.visionSupport || undefined,
      allow_proxy: v.allowProxy || undefined,
      provider: v.provider || undefined,
    };
    if (v.apiKey) {
      data.api_key = v.apiKey;
    } else if (!editing) {
      // 新建模型：必须显式发送空串，否则后端 build_model_map 缺 api_key 会 500
      data.api_key = '';
    }

    // Thinking effort：按模式写入（type 字段已废弃，统一用 'any_to_any'）
    if (v.thinkingEffortMode === 'custom') {
      data.thinking_effort_mode = 'custom';
      data.thinking_effort_preset = {
        type: 'any_to_any',
        rules: { ...v.thinkingEffortCustomRules },
      };
    } else {
      data.thinking_effort_mode = v.thinkingEffortMode;
      data.thinking_effort_preset = null; // 清理旧的内联值
    }
    try {
      await form.handleSubmit(async () => {
        await saveModel(v.name, data);
        const cfg = await fetchConfig();
        setConfig(cfg);
        showToast('模型已保存', 'ok');
        closeModelModal();
      });
    } catch {
      // error surfaced via form.generalError.value
    }
  };

  const onDetect = async () => {
    if (!v.apiBase) { showToast('请先填写 API Base', 'err'); return; }
    if (!v.apiKey) { showToast('请先填写 API Key', 'err'); return; }
    try {
      const result = await apiDetectProtocol(v.apiBase, v.apiKey);
      const entries = result.upstream_protocols ?? [];
      for (const e of entries) {
        if (e.protocol === 'anthropic') {
          form.setField('chipAnthropic', e.enabled);
          if (e.path) form.setField('pathAnthropic', e.path);
        } else if (e.protocol === 'openai/chat-completions') {
          form.setField('chipOpenaiChat', e.enabled);
          if (e.path) form.setField('pathOpenaiChat', e.path);
        } else if (e.protocol === 'openai/responses') {
          form.setField('chipOpenaiResponses', e.enabled);
          if (e.path) form.setField('pathOpenaiResponses', e.path);
        }
      }
      const labels: string[] = [];
      if (form.values.value.chipAnthropic) labels.push('Anthropic');
      if (form.values.value.chipOpenaiChat) labels.push('OpenAI Chat');
      if (form.values.value.chipOpenaiResponses) labels.push('OpenAI Responses');
      showToast(labels.length ? `已检测: ${labels.join('、')}` : '未检测到协议', labels.length ? 'ok' : 'err');
    } catch (e) {
      showToast(e instanceof ApiCallError ? e.detail : '检测失败', 'err');
    }
  };

  return (
    <Modal onClose={closeModelModal} size="lg">
      <h2 class="modal-title">{editing ? '编辑模型' : '添加模型'}</h2>
      {form.generalError.value && <div class="error-banner">{form.generalError.value}</div>}
      <div class="modal-cols-2" style="margin-top: 20px;">
        <div>
          <Field label="配置键" hint="不可修改（编辑模式下）" required error={form.errors.value.name}>
            <input class="w-full" value={v.name} disabled={!!editing} onInput={(e: Event) => form.setField('name', (e.target as HTMLInputElement).value)} />
          </Field>
          <Field label="显示名称" hint="留空则使用配置键">
            <input class="w-full" value={v.displayName} onInput={(e: Event) => form.setField('displayName', (e.target as HTMLInputElement).value)} />
          </Field>
          <Field label="厂商 (Provider)" hint="选择或输入厂商标识，用于自动适配 thinking/reasoning 格式">
            <ComboBox
              value={v.provider}
              options={Object.entries(profiles).map(([key, p]) => ({ value: key, label: p.display_name }))}
              onInput={(value) => {
                form.setField('provider', value);
                const profile = profiles[value];
                if (profile && !v.apiBase) {
                  form.setField('apiBase', profile.default_api_base);
                }
              }}
              placeholder="如 deepseek、minimax、moonshot-k3"
            />
          </Field>

          <Field label="API Base" hint="如 https://api.example.com">
            <input class="w-full" value={v.apiBase} onInput={(e: Event) => form.setField('apiBase', (e.target as HTMLInputElement).value)} placeholder="https://..." />
          </Field>
          <Field label="API Key">
            <div style="display: flex; gap: 6px;">
              <input type="password" class="w-full" value={v.apiKey} onInput={(e: Event) => form.setField('apiKey', (e.target as HTMLInputElement).value)} style="flex: 1;" />
              <button type="button" class="ghost" onClick={(e: MouseEvent) => togglePasswordVisibility((e.currentTarget as HTMLButtonElement).previousElementSibling as HTMLInputElement)}>显示</button>
            </div>
          </Field>
          <Field label="上下文窗口">
            <div style="display: flex; gap: 6px;">
              <input type="number" value={v.contextWindow} onInput={(e: Event) => form.setField('contextWindow', (e.target as HTMLInputElement).value)} style="flex: 1;" />
              <select value={v.contextWindowUnit} onChange={(e: Event) => form.setField('contextWindowUnit', (e.target as HTMLSelectElement).value)} style="width: 80px;">
                <option value="1000">K</option>
                <option value="1000000">M</option>
              </select>
            </div>
          </Field>
        </div>
        <div>
          <fieldset style="margin-bottom: 12px;">
            <legend>协议支持</legend>
            <div style="display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px;">
              <ProtocolChip checked={v.chipAnthropic} onToggle={() => form.setField('chipAnthropic', !v.chipAnthropic)} label="Anthropic" />
              <ProtocolChip checked={v.chipOpenaiChat} onToggle={() => form.setField('chipOpenaiChat', !v.chipOpenaiChat)} label="OpenAI Chat" />
              <ProtocolChip checked={v.chipOpenaiResponses} onToggle={() => form.setField('chipOpenaiResponses', !v.chipOpenaiResponses)} label="OpenAI Responses" />
              <button type="button" class="secondary" onClick={onDetect}>自动检测</button>
            </div>
            <div style="display: grid; grid-template-columns: auto 1fr; gap: 6px 10px; align-items: center;">
              <span class="text-xs text-muted">Anthropic</span>
              <input class="text-sm" value={v.pathAnthropic} onInput={(e: Event) => form.setField('pathAnthropic', (e.target as HTMLInputElement).value)} placeholder="anthropic/v1/messages" disabled={!v.chipAnthropic} />
              <span class="text-xs text-muted">Chat</span>
              <input class="text-sm" value={v.pathOpenaiChat} onInput={(e: Event) => form.setField('pathOpenaiChat', (e.target as HTMLInputElement).value)} placeholder="/v1/chat/completions" disabled={!v.chipOpenaiChat} />
              <span class="text-xs text-muted">Responses</span>
              <input class="text-sm" value={v.pathOpenaiResponses} onInput={(e: Event) => form.setField('pathOpenaiResponses', (e.target as HTMLInputElement).value)} placeholder="/v1/responses" disabled={!v.chipOpenaiResponses} />
            </div>
          </fieldset>
          <label style="display: flex; align-items: center; gap: 8px; font-size: var(--fs-sm);">
            <input type="checkbox" checked={v.visionSupport} onChange={(e: Event) => form.setField('visionSupport', (e.target as HTMLInputElement).checked)} />
            支持图像输入
          </label>
          <div style="margin-top: 12px;">
            <Toggle
              checked={v.allowProxy}
              onChange={(val) => form.setField('allowProxy', val)}
              label="允许走系统代理（HTTPS_PROXY）"
            />
            <div class="text-xs text-muted" style="margin-top: 4px;">
              默认直连上游。开启后此模型出站会走系统代理。
            </div>
          </div>
        </div>
      </div>

      <EffortStrategyFieldset
        mode={v.thinkingEffortMode as 'default' | 'provider' | 'custom'}
        provider={v.provider || ''}
        customRules={v.thinkingEffortCustomRules}
        profiles={profiles}
        onModeChange={(mode) => form.setField('thinkingEffortMode', mode)}
        onRulesChange={(rules) => form.setField('thinkingEffortCustomRules', rules)}
      />

      <div class="modal-actions">
        <button class="ghost" onClick={closeModelModal}>取消</button>
        <button class="primary" onClick={onSave} disabled={form.submitting.value}>{form.submitting.value ? '保存中...' : '保存'}</button>
      </div>
    </Modal>
  );
}

export { openModelModal } from '../state/store';
