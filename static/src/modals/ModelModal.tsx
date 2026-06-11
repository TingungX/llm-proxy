import { useEffect } from 'preact/hooks';
import { modelsSignal, setConfig, modelModalOpen, modelModalEditing, closeModelModal } from '../state/store';
import { saveModel } from '../api/models';
import { detectProtocol as apiDetectProtocol } from '../api/endpoints';
import { fetchConfig } from '../api/config';
import { showToast } from '../components/Toast';
import { ApiCallError } from '../api/client';
import { useFormState } from '../hooks/useFormState';
import { togglePasswordVisibility } from '../utils/clipboard';
import { toUpstreamProtocols, toUpstreamPaths } from '../utils/protocol';
import { Modal } from '../components/Modal';
import { Field } from '../components/Field';
import { ProtocolChip } from '../components/ProtocolChip';
import { Toggle } from '../components/Toggle';
import type { ModelConfig } from '../api/types';

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
}

function getInitialValues(name: string | null): ModelFormValues {
  const base: ModelFormValues = {
    name: '', displayName: '', apiBase: '', apiKey: '',
    contextWindow: '', contextWindowUnit: '1000',
    chipAnthropic: false, chipOpenaiChat: false, chipOpenaiResponses: false,
    pathAnthropic: '', pathOpenaiChat: '', pathOpenaiResponses: '',
    visionSupport: false, allowProxy: false,
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
      const paths = m.upstream_paths ?? {};
      // 优先读上游协议列表（新格式）；缺失时回退到标量（迁移期兼容）
      const protocols = new Set<string>(
        m.upstream_protocols ?? (m.upstream_protocol ? [m.upstream_protocol] : [])
      );
      // 旧标量 'openai' 等价于 'openai/chat-completions'（向后兼容映射）
      const openaiChat = protocols.has('openai') || protocols.has('openai/chat-completions') || !!paths['openai/chat-completions'];
      return {
        ...base,
        name,
        displayName: m.display_name ?? '',
        apiBase: m.api_base ?? '',
        apiKey: m.api_key ?? '',
        contextWindow: cw,
        contextWindowUnit: cwUnit,
        chipAnthropic: protocols.has('anthropic') || !!paths['anthropic/messages'],
        chipOpenaiChat: openaiChat,
        chipOpenaiResponses: protocols.has('openai/responses') || !!paths['openai/responses'],
        pathAnthropic: paths['anthropic/messages'] ?? '',
        pathOpenaiChat: paths['openai/chat-completions'] ?? '',
        pathOpenaiResponses: paths['openai/responses'] ?? '',
        visionSupport: m.vision_support ?? false,
        allowProxy: m.allow_proxy ?? false,
      };
    }
  }
  return base;
}

export function ModelModal() {
  const isOpen = modelModalOpen.value;
  const editing = modelModalEditing.value;
  const form = useFormState<ModelFormValues>(getInitialValues(editing));

  useEffect(() => {
    if (isOpen) form.reset(getInitialValues(editing));
  }, [isOpen, editing]);

  if (!isOpen) return null;

  // 订阅 form.values：必须读 .value 才能在 setField 后重渲染
  const v = form.values.value;
  const pc = { enabled: { anthropic: v.chipAnthropic, openai_chat: v.chipOpenaiChat, openai_responses: v.chipOpenaiResponses }, paths: { anthropic: v.pathAnthropic, openai_chat: v.pathOpenaiChat, openai_responses: v.pathOpenaiResponses } };

  const onSave = async () => {
    if (!v.name) { form.setErrors({ name: '必填' }); return; }
    const data: Partial<ModelConfig> = {
      display_name: v.displayName || undefined,
      api_base: v.apiBase,
      api_key: v.apiKey,
      upstream_model: v.name,
      upstream_protocols: toUpstreamProtocols(pc),
      context_window: v.contextWindow ? Math.round(Number(v.contextWindow) * Number(v.contextWindowUnit)) : undefined,
      vision_support: v.visionSupport || undefined,
      allow_proxy: v.allowProxy || undefined,
    };
    const paths = toUpstreamPaths(pc);
    if (Object.keys(paths).length > 0) data.upstream_paths = paths;
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
      // 后端返回 upstream_protocols: ["anthropic" | "openai/chat-completions" | "openai/responses", ...]
      // 分别映射到三个 chip
      const protocols = result.upstream_protocols ?? (result.upstream_protocol ? [result.upstream_protocol] : []);
      form.setField('chipAnthropic', protocols.includes('anthropic'));
      form.setField('chipOpenaiChat', protocols.includes('openai/chat-completions'));
      form.setField('chipOpenaiResponses', protocols.includes('openai/responses'));
      const labels = protocols.map(p => p === 'anthropic' ? 'Anthropic' : p === 'openai/chat-completions' ? 'OpenAI Chat' : 'OpenAI Responses');
      showToast(protocols.length ? `已检测: ${labels.join('、')}` : '未检测到协议', protocols.length ? 'ok' : 'err');
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
      <div class="modal-actions">
        <button class="ghost" onClick={closeModelModal}>取消</button>
        <button class="primary" onClick={onSave} disabled={form.submitting.value}>{form.submitting.value ? '保存中...' : '保存'}</button>
      </div>
    </Modal>
  );
}

export { openModelModal } from '../state/store';
