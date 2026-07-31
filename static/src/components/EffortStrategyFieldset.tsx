import { Fragment } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { configSignal } from '../state/store';
import { fetchThinkingEffortDefaults } from '../api/providers';
import { Toggle } from './Toggle';
import type { ProviderProfileInfo, ThinkingEffortMapping } from '../api/types';

const COMMON_EFFORT_KEYS = ['none', 'auto', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', '*'];

interface Props {
  mode: 'default' | 'provider' | 'custom';
  provider: string;
  customRules: Record<string, string>;
  profiles: Record<string, ProviderProfileInfo>;
  mappingEnabled: boolean;
  passthroughNoMap: boolean;
  onModeChange: (mode: 'default' | 'provider' | 'custom') => void;
  onRulesChange: (rules: Record<string, string>) => void;
  onMappingEnabledChange: (enabled: boolean) => void;
  onPassthroughNoMapChange: (noMap: boolean) => void;
}

/** format 中文标签 */
const FORMAT_LABELS: Record<string, string> = {
  thinking_type_plus_reasoning_effort: 'thinking.type + reasoning_effort',
  thinking_enabled_disabled: 'thinking.type（开关）',
  thinking_adaptive_disabled: 'thinking.type（adaptive）',
  reasoning_effort_only: 'reasoning_effort',
  reasoning_effort_fixed: 'reasoning_effort（固定）',
  enable_thinking_boolean: 'enable_thinking',
  chat_template_kwargs_enable_thinking: 'chat_template_kwargs',
};

function formatLabel(fmt: string | null | undefined): string {
  return FORMAT_LABELS[fmt ?? ''] ?? fmt ?? '—';
}

function formatAcceptValues(p: ProviderProfileInfo): string {
  const fmt = p.thinking_format;
  if (fmt === 'thinking_type_plus_reasoning_effort')
    return `type: ${p.default_thinking_type ?? 'enabled'} / ${p.disable_thinking_value ?? 'disabled'} · effort: high / max`;
  if (fmt === 'thinking_enabled_disabled')
    return `${p.default_thinking_type ?? 'enabled'} / ${p.disable_thinking_value ?? 'disabled'}`;
  if (fmt === 'thinking_adaptive_disabled') return 'adaptive / disabled';
  if (fmt === 'reasoning_effort_only') return 'low / medium / high';
  if (fmt === 'reasoning_effort_fixed') return `固定 ${p.fixed_effort ?? 'max'}`;
  if (fmt === 'enable_thinking_boolean' || fmt === 'chat_template_kwargs_enable_thinking')
    return 'true / false';
  return '—';
}

export function EffortStrategyFieldset({
  mode, provider, customRules, profiles,
  mappingEnabled, passthroughNoMap,
  onModeChange, onRulesChange,
  onMappingEnabledChange, onPassthroughNoMapChange,
}: Props) {
  const [sysDefaults, setSysDefaults] = useState<ThinkingEffortMapping | null>(null);
  const [formatOpen, setFormatOpen] = useState(false);
  const profile = provider ? profiles[provider] : null;

  useEffect(() => {
    fetchThinkingEffortDefaults().then(setSysDefaults).catch(() => {});
  }, []);

  // 二元开关 / 固定值厂商：生成实际行为展示表，不套用 any_to_any 规则
  function buildBinaryDisplay(p: ProviderProfileInfo):
    | { source: string; presetName: string; rules: Record<string, string> }
    | null
  {
    const fmt = p.thinking_format;
    if (fmt === 'thinking_enabled_disabled') {
      return {
        source: `厂商格式 (${p.display_name})`,
        presetName: '开关',
        rules: {
          'none': `off (${p.disable_thinking_value || 'disabled'})`,
          '* (any)': `on (${p.default_thinking_type || 'enabled'})`,
        },
      };
    }
    if (fmt === 'thinking_adaptive_disabled') {
      return {
        source: `厂商格式 (${p.display_name})`,
        presetName: '开关',
        rules: {
          'none': `off (${p.disable_thinking_value || 'disabled'})`,
          '* (any)': `on (${p.default_thinking_type || 'adaptive'})`,
        },
      };
    }
    if (fmt === 'enable_thinking_boolean' || fmt === 'chat_template_kwargs_enable_thinking') {
      return {
        source: `厂商格式 (${p.display_name})`,
        presetName: '开关',
        rules: { 'none': 'off', '* (any)': 'on' },
      };
    }
    if (fmt === 'reasoning_effort_fixed') {
      return {
        source: `厂商格式 (${p.display_name})`,
        presetName: '固定值',
        rules: { '* (any)': p.fixed_effort || '—' },
      };
    }
    if ((fmt === 'thinking_type_plus_reasoning_effort' || fmt === 'reasoning_effort_only')
        && p.effort_aliases && Object.keys(p.effort_aliases).length > 0) {
      const rules: Record<string, string> = {};
      for (const [k, v] of Object.entries(p.effort_aliases)) {
        rules[k] = v ?? '(remove)';
      }
      return {
        source: `厂商格式 (${p.display_name})`,
        presetName: '别名映射',
        rules,
      };
    }
    return null;
  }

  // 解析当前生效的规则（用于 default / provider 模式的只读展示）
  function resolveReadonlyRules():
    | { source: string; presetName: string; rules: Record<string, string> }
    | null
  {
    if (mode === 'default') {
      // 优先从 API 获取系统内置默认
      if (sysDefaults?.presets?.[0]?.rules) {
        return {
          source: '系统内置',
          presetName: sysDefaults.presets[0].name,
          rules: sysDefaults.presets[0].rules,
        };
      }
      // 回退：全局 config 的 default_preset
      const gm = configSignal.value?.thinking_effort_mapping;
      const dp = gm?.presets?.find(p => p.name === gm?.default_preset);
      if (dp?.rules) return { source: '全局默认', presetName: dp.name, rules: dp.rules };
      return null;
    }
    if (mode === 'provider' && profile) {
      // 二元开关 / 固定值厂商：直接展示实际行为，不套 any_to_any 规则表
      const binary = buildBinaryDisplay(profile);
      if (binary) return binary;

      const pn = profile.default_thinking_effort_preset;
      if (pn) {
        const gm = configSignal.value?.thinking_effort_mapping;
        let pp = gm?.presets?.find(p => p.name === pn);
        // 全局 config 未保存映射时，回退到系统内置默认
        if (!pp?.rules && sysDefaults?.presets) {
          pp = sysDefaults.presets.find(p => p.name === pn);
        }
        if (pp?.rules) return { source: `厂商推荐 (${profile.display_name})`, presetName: pp.name, rules: pp.rules };
      }
      // 无预设名 → 展示厂商 aliases
      if (profile.effort_aliases && Object.keys(profile.effort_aliases).length > 0) {
        return {
          source: `厂商内置 (${profile.display_name})`,
          presetName: 'effort_aliases',
          rules: Object.fromEntries(
            Object.entries(profile.effort_aliases).map(([k, v]) => [k, v ?? '(移除)'])
          ),
        };
      }
      return null;
    }
    return null;
  }

  const ro = resolveReadonlyRules();
  const showProviderRadio = !!profile;
  const modes = showProviderRadio
    ? (['default', 'provider', 'custom'] as const)
    : (['default', 'custom'] as const);

  return (
    <fieldset style="margin-top: 16px;">
      <legend>Thinking Effort 策略</legend>

      <div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 14px;">
        <div>
          <Toggle
            checked={mappingEnabled}
            onChange={onMappingEnabledChange}
            label="effort 映射开启"
          />
          <div class="text-xs text-muted" style="margin-top: 4px;">
            关闭后该模型任何路径都不做 effort 值映射。
          </div>
        </div>
        <div>
          <Toggle
            checked={passthroughNoMap}
            onChange={onPassthroughNoMapChange}
            label="透传模式不映射 effort"
          />
          <div class="text-xs text-muted" style="margin-top: 4px;">
            开启后同协议透传保持客户端原始 effort；默认关闭（透传也映射）。
          </div>
        </div>
      </div>

      {!mappingEnabled ? (
        <div class="text-xs text-muted" style="padding: 12px; border: 1px dashed var(--border); border-radius: 6px; text-align: center;">
          effort 映射已关闭，策略预设不生效。
        </div>
      ) : (
        <>
      {/* 未指定厂商提示 */}
      {!profile && (
        <div class="model-format-info model-format-info-warn mb-12">
          <span class="text-xs text-muted">
            未指定厂商，effort 将以标准 <code>reasoning_effort</code> 字段直接透传。
            如需厂商格式编码，请在上方「厂商」字段选择对应厂商。
          </span>
        </div>
      )}

      {/* Radio 选择 */}
      <div class="thinking-effort-modes">
        {modes.map((m) => (
          <label key={m} class="thinking-effort-mode">
            <input
              type="radio"
              name="thinkingEffortMode"
              checked={mode === m}
              onChange={() => onModeChange(m)}
            />
            <span>
              {m === 'default' ? '全局默认' : m === 'provider' ? '厂商默认' : '自定义'}
            </span>
          </label>
        ))}
      </div>

{/* default / provider：只读展示 */}
      {(mode === 'default' || mode === 'provider') && ro && (
        <div>
          <div class="text-xs text-muted mb-8">{ro.source} · 预设 <strong>{ro.presetName}</strong></div>
          <div style="display: grid; grid-template-columns: 1fr 20px 1fr; gap: 4px 8px; align-items: center; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px;">
            <span class="text-xs text-muted" style="font-weight: 600;">输入</span>
            <span />
            <span class="text-xs text-muted" style="font-weight: 600;">→ 规范 effort</span>
            {Object.entries(ro.rules).map(([k, val]) => (
              <Fragment key={k}>
                <span class="effort-cell">{k}</span>
                <span class="effort-arrow">→</span>
                <span class="effort-cell">{val}</span>
              </Fragment>
            ))}
          </div>
          {profile && (
            <div class="text-xs text-muted mt-8">
              映射后的 effort 经 {profile.display_name} 格式编码：<code>{formatLabel(profile.thinking_format)}</code>
            </div>
          )}
        </div>
      )}

      {/* default / provider：无规则时的回退提示 */}
      {(mode === 'default' || mode === 'provider') && !ro && (
        <div class="text-xs text-muted" style="padding: 12px; border: 1px dashed var(--border); border-radius: 6px; text-align: center;">
          {mode === 'default'
            ? '系统内置默认规则加载中…'
            : !profile
              ? '请先选择厂商'
              : `${profile.display_name} 未配置推荐预设，将回退全局默认`}
        </div>
      )}

      {/* 自定义：可编辑 sheet */}
      {mode === 'custom' && (
        <div>
          <div class="text-xs text-muted mb-8">规则键 <code>*</code> 作为兜底，匹配所有未列出的 effort。</div>

          {Object.keys(customRules).length > 0 ? (
            <div style="display: grid; grid-template-columns: 1fr 20px 1fr 36px; gap: 4px 8px; align-items: center; margin-bottom: 8px;">
              <span class="text-xs text-muted" style="font-weight: 600;">输入</span>
              <span />
              <span class="text-xs text-muted" style="font-weight: 600;">→ 规范 effort</span>
              <span />
              {Object.entries(customRules).map(([key, val], rIdx) => (
                <Fragment key={rIdx}>
                  <span class="effort-cell editable" contenteditable="true"
                    onBlur={(e: Event) => {
                      const newKey = (e.target as HTMLSpanElement).textContent || key;
                      if (newKey === key) return;
                      const rules = { ...customRules };
                      delete rules[key];
                      rules[newKey] = val;
                      onRulesChange(rules);
                    }}
                  >{key}</span>
                  <span class="effort-arrow">→</span>
                  <span class="effort-cell editable" contenteditable="true"
                    onBlur={(e: Event) => {
                      const newVal = (e.target as HTMLSpanElement).textContent || val;
                      if (newVal === val) return;
                      onRulesChange({ ...customRules, [key]: newVal });
                    }}
                  >{val}</span>
                  <button
                    type="button"
                    class="icon-sm"
                    onClick={() => {
                      const rules = { ...customRules };
                      delete rules[key];
                      onRulesChange(rules);
                    }}
                    title="删除规则"
                  >✕</button>
                </Fragment>
              ))}
            </div>
          ) : (
            <div class="text-xs text-muted mb-8" style="padding: 12px; border: 1px dashed var(--border); border-radius: 6px; text-align: center;">
              暂无自定义规则，点击下方添加
            </div>
          )}
          <button
            type="button"
            class="ghost"
            onClick={() => {
              let newKey = 'low';
              for (const c of COMMON_EFFORT_KEYS) {
                if (!(c in customRules)) { newKey = c; break; }
              }
              onRulesChange({ ...customRules, [newKey]: 'low' });
            }}
          >+ 添加规则</button>
          {profile ? (
            <div class="text-xs text-muted mt-8">
              映射后的 effort 经 {profile.display_name} 格式编码：<code>{formatLabel(profile.thinking_format)}</code>
            </div>
          ) : (
            <div class="text-xs text-muted mt-8">
              未指定厂商，映射后的 effort 直接作为 <code>reasoning_effort</code> 透传。
            </div>
          )}
        </div>
      )}

      {/* 厂商格式参考（可折叠，默认折叠，置于底部） */}
      {profile && (
        <div class="format-ref-section" style="margin-top: 16px;">
          <div class="collapse-toggle" onClick={() => setFormatOpen(!formatOpen)}>
            <span class={`arrow ${formatOpen ? 'open' : ''}`}>▶</span>
            <span>厂商格式参考</span>
            <span class="text-xs text-muted">（{profile.display_name}）</span>
          </div>
          {formatOpen && (
            <div class="collapse-body">
              <div class="model-format-head">
                <span>厂商格式</span>
                <span class="badge-format">{formatLabel(profile.thinking_format)}</span>
                <span class="text-xs text-muted">{profile.display_name}</span>
              </div>
              <div class="text-xs text-muted">
                字段：<code>{formatLabel(profile.thinking_format)}</code>
                &nbsp;·&nbsp; 接受值：{formatAcceptValues(profile)}
              </div>
            </div>
          )}
        </div>
      )}
        </>
      )}
    </fieldset>
  );
}
