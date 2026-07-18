import { useState, useEffect } from 'preact/hooks';
import { configSignal, setConfig } from '../state/store';
import { saveConfig } from '../api/config';
import { fetchThinkingEffortDefaults } from '../api/providers';
import { ApiCallError } from '../api/client';
import { showToast } from './Toast';
import { Field } from './Field';
import type { ThinkingEffortMapping, ProviderProfileInfo } from '../api/types';

interface Props {
  providers: Record<string, ProviderProfileInfo>;
}

const FORMAT_LABELS: Record<string, string> = {
  thinking_type_plus_reasoning_effort: 'type + effort',
  thinking_enabled_disabled: '开关',
  thinking_adaptive_disabled: 'adaptive',
  reasoning_effort_only: 'effort only',
  reasoning_effort_fixed: '固定值',
  enable_thinking_boolean: '布尔',
  chat_template_kwargs_enable_thinking: 'chat template',
};

function formatLabel(fmt: string | null | undefined): string {
  if (!fmt) return '—';
  return FORMAT_LABELS[fmt] ?? fmt;
}

export function ThinkingEffortMappingCard({ providers }: Props) {
  const [saving, setSaving] = useState(false);
  const [formatRefOpen, setFormatRefOpen] = useState(false);
  const [sysDefaults, setSysDefaults] = useState<ThinkingEffortMapping | null>(null);

  const mapping = configSignal.value?.thinking_effort_mapping;
  const [defaultPreset, setDefaultPreset] = useState(mapping?.default_preset ?? 'default');

  useEffect(() => {
    fetchThinkingEffortDefaults().then(setSysDefaults).catch(() => {});
  }, []);

  // 收集可选 preset：系统 default + 厂商推荐
  const allPresetNames = new Set<string>();
  allPresetNames.add('default');
  for (const p of Object.values(providers)) {
    if (p.default_thinking_effort_preset) allPresetNames.add(p.default_thinking_effort_preset);
  }

  async function handleSave() {
    const cfg = configSignal.value;
    if (!cfg) return;

    const existingPresets = cfg.thinking_effort_mapping?.presets ?? [];
    // 保留所有非 default 的已有 preset（如厂商推荐），确保 default 存在
    const hasDefault = existingPresets.some(p => p.name === 'default');
    let presets = existingPresets.filter(p => p.name !== 'default');
    if (hasDefault) {
      // 保留原有的 default preset（用户可能通过其他方式配置过）
      const oldDefault = existingPresets.find(p => p.name === 'default')!;
      presets = [oldDefault, ...presets];
    }

    const nextMapping: ThinkingEffortMapping = {
      presets,
      default_preset: defaultPreset,
    };

    const newConfig = { ...cfg, thinking_effort_mapping: nextMapping };
    setSaving(true);
    try {
      await saveConfig(newConfig);
      setConfig(newConfig);
      showToast('Thinking Effort 映射已保存', 'ok');
    } catch (e) {
      showToast(e instanceof ApiCallError ? e.detail : '保存失败', 'err');
    } finally {
      setSaving(false);
    }
  }

  const providerEntries = Object.entries(providers);

  return (
    <div class="card">
      <div class="card-header">
        <h2>Thinking Effort 映射</h2>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={handleSave} disabled={saving}>{saving ? '保存中…' : '保存'}</button>
        </div>
      </div>

      <div class="hint mb-12">
        将请求中的 reasoning effort（low/medium/high/...）按规则映射为目标 effort。
        规则键 <code>*</code> 作为兜底。映射规则由系统内置及厂商预置，不可在此编辑。
      </div>

      {/* 厂商格式参考（可折叠） */}
      {providerEntries.length > 0 && (
        <div class="format-ref-section">
          <div class="collapse-toggle mb-8" onClick={() => setFormatRefOpen(!formatRefOpen)}>
            <span class={`arrow ${formatRefOpen ? 'open' : ''}`}>▶</span>
            <span>厂商格式参考（{providerEntries.length} 个厂商）</span>
            <span class="badge badge-system" style="margin-left: 8px;">只读参考</span>
          </div>
          {formatRefOpen && (
            <div class="collapse-body mb-12">
              <p class="text-xs text-muted mb-8">
                不同厂商接受 thinking / reasoning 的字段和值域不同。修改需编辑
                <code>provider_profiles.json</code>。
              </p>
              <table>
                <thead>
                  <tr>
                    <th>厂商</th>
                    <th>格式类型</th>
                    <th>请求字段</th>
                    <th>接受值</th>
                    <th>预设</th>
                  </tr>
                </thead>
                <tbody>
                  {providerEntries.map(([key, p]) => (
                    <tr key={key}>
                      <td><strong>{p.display_name}</strong></td>
                      <td><span class="badge-format">{formatLabel(p.thinking_format)}</span></td>
                      <td class="text-xs font-mono">
                        {p.thinking_format === 'thinking_type_plus_reasoning_effort' && (
                          <><span>thinking.type</span><br /><span>reasoning_effort</span></>
                        )}
                        {p.thinking_format === 'thinking_enabled_disabled' && 'thinking.type'}
                        {p.thinking_format === 'thinking_adaptive_disabled' && 'thinking.type'}
                        {p.thinking_format === 'reasoning_effort_only' && 'reasoning_effort'}
                        {p.thinking_format === 'reasoning_effort_fixed' && 'reasoning_effort'}
                        {p.thinking_format === 'enable_thinking_boolean' && (p.effort_field ?? 'enable_thinking')}
                        {p.thinking_format === 'chat_template_kwargs_enable_thinking' && 'chat_template_kwargs'}
                        {!p.thinking_format && '—'}
                      </td>
                      <td class="text-xs">
                        {p.thinking_format === 'thinking_type_plus_reasoning_effort' && (
                          <><span class="badge-format">enabled</span> <span class="badge-format">disabled</span> · <span class="badge-format">high</span> <span class="badge-format">max</span></>
                        )}
                        {p.thinking_format === 'thinking_enabled_disabled' && (
                          <><span class="badge-format">{p.default_thinking_type ?? 'enabled'}</span> <span class="badge-format">{p.disable_thinking_value ?? 'disabled'}</span></>
                        )}
                        {p.thinking_format === 'thinking_adaptive_disabled' && (
                          <><span class="badge-format">adaptive</span> <span class="badge-format">disabled</span></>
                        )}
                        {p.thinking_format === 'reasoning_effort_only' && (
                          <><span class="badge-format">low</span> <span class="badge-format">medium</span> <span class="badge-format">high</span></>
                        )}
                        {p.thinking_format === 'reasoning_effort_fixed' && (
                          <><span class="badge-format">{p.fixed_effort ?? 'max'}</span> <span class="text-muted">固定</span></>
                        )}
                        {p.thinking_format === 'enable_thinking_boolean' && (
                          <><span class="badge-format">true</span> <span class="badge-format">false</span></>
                        )}
                        {p.thinking_format === 'chat_template_kwargs_enable_thinking' && (
                          <><span class="badge-format">true</span> <span class="badge-format">false</span></>
                        )}
                        {!p.thinking_format && '—'}
                      </td>
                      <td class="text-xs">
                        {p.default_thinking_effort_preset ? (
                          <code>{p.default_thinking_effort_preset}</code>
                        ) : (
                          <span class="text-muted">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* 系统内置默认 */}
      {sysDefaults && sysDefaults.presets.length > 0 && (
        <div class="effort-preset effort-preset-system mb-16">
          <div class="effort-preset-head">
            <span class="badge badge-system">系统内置</span>
            <strong>{sysDefaults.presets[0].name}</strong>
            <span class="badge-format">{sysDefaults.presets[0].type}</span>
            <span class="text-xs text-muted">不可编辑，作为兜底</span>
          </div>
          <div class="effort-rules-ro">
            <div class="effort-rules-head text-xs text-muted"><span>输入</span><span /><span>输出</span><span /></div>
            {Object.entries(sysDefaults.presets[0].rules).map(([k, v]) => (
              <div class="effort-rule-row" key={k}>
                <span class="effort-cell">{k}</span>
                <span class="effort-arrow">→</span>
                <span class="effort-cell">{v}</span>
                <span />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 厂商推荐预设（只读） */}
      {providerEntries.map(([key, p]) => {
        const presetName = p.default_thinking_effort_preset;
        if (!presetName) return null;
        const preset = mapping?.presets?.find(pr => pr.name === presetName);
        if (!preset?.rules) return null;
        const entries = Object.entries(preset.rules);
        if (entries.length === 0) return null;

        return (
          <div class="effort-preset effort-preset-ro mb-16" key={key}>
            <div class="effort-preset-head">
              <span class="badge badge-vendor">厂商推荐</span>
              <strong>{presetName}</strong>
              <span class="badge-format">{preset.type}</span>
              <span class="text-xs text-muted">适用于 {p.display_name} · 只读</span>
            </div>
            <div class="effort-rules-ro">
              <div class="effort-rules-head text-xs text-muted"><span>输入</span><span /><span>输出</span><span /></div>
              {entries.map(([k, v]) => (
                <div class="effort-rule-row" key={k}>
                  <span class="effort-cell">{k}</span>
                  <span class="effort-arrow">→</span>
                  <span class="effort-cell">{v}</span>
                  <span />
                </div>
              ))}
            </div>
            <div class="text-xs text-muted mt-8">
              映射后经 {p.display_name} 格式 <code>{formatLabel(p.thinking_format)}</code> 编码为最终请求字段。
            </div>
          </div>
        );
      })}

      <Field label="默认 Preset" hint="未显式指定时使用的映射预设">
        <select
          value={defaultPreset}
          onChange={(e: Event) => setDefaultPreset((e.target as HTMLSelectElement).value)}
          disabled={saving}
        >
          {[...allPresetNames].map(name => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </Field>
    </div>
  );
}
