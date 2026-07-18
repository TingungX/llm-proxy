import { useState, useEffect } from 'preact/hooks';
import { adminAuthSignal } from '../state/config';
import { showToast } from './Toast';
import { fetchAdminAuthStatus, updateAdminAuth } from '../api/adminAuth';
import { ApiCallError } from '../api/client';
import { Modal } from './Modal';
import { Field } from './Field';
import { Toggle } from './Toggle';

interface AdminAuthModalProps {
  onClose: () => void;
}

export function AdminAuthModal({ onClose }: AdminAuthModalProps) {
  const [key, setKey] = useState('');
  const [confirmKey, setConfirmKey] = useState('');
  const [currentKey, setCurrentKey] = useState('');
  const [saving, setSaving] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const auth = adminAuthSignal.value;
  const isEnabled = auth.enabled;
  const isEnvManaged = auth.source === 'env';

  // 刷新状态
  const refresh = async () => {
    try {
      const status = await fetchAdminAuthStatus();
      adminAuthSignal.value = status;
    } catch {
      // 静默失败
    }
  };

  useEffect(() => { refresh(); }, []);

  const handleSave = async (enabled: boolean) => {
    if (enabled) {
      if (!key || key.length < 6) {
        showToast('密钥至少需要 6 个字符', 'err');
        return;
      }
      if (key !== confirmKey) {
        showToast('两次输入的密钥不一致', 'err');
        return;
      }
    } else {
      if (!currentKey) {
        showToast('请输入当前密钥以禁用保护', 'err');
        return;
      }
    }

    setSaving(true);
    try {
      await updateAdminAuth({
        enabled,
        key: enabled ? key : undefined,
        current_key: enabled ? undefined : currentKey,
      });
      if (enabled) {
        localStorage.setItem('adminKey', key);
        showToast('高级数据保护已启用');
      } else {
        localStorage.removeItem('adminKey');
        showToast('高级数据保护已禁用');
      }
      setKey('');
      setConfirmKey('');
      setCurrentKey('');
      await refresh();
    } catch (e) {
      const msg = e instanceof ApiCallError ? e.detail : '操作失败';
      showToast(msg, 'err');
    } finally {
      setSaving(false);
    }
  };

  const handleChangeKey = async () => {
    if (!currentKey) {
      showToast('请输入当前密钥', 'err');
      return;
    }
    if (!key || key.length < 6) {
      showToast('新密钥至少需要 6 个字符', 'err');
      return;
    }
    if (key !== confirmKey) {
      showToast('两次输入的新密钥不一致', 'err');
      return;
    }

    setSaving(true);
    try {
      // 修改密钥：先禁用再启用（用 PUT 同时传 current_key + 新 key）
      // 当前实现：直接重新启用，current_key 验证在 handler 中
      await updateAdminAuth({
        enabled: true,
        key,
        current_key: currentKey,
      });
      localStorage.setItem('adminKey', key);
      showToast('密钥已更新');
      setKey('');
      setConfirmKey('');
      setCurrentKey('');
      await refresh();
    } catch (e) {
      const msg = e instanceof ApiCallError ? e.detail : '操作失败';
      showToast(msg, 'err');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal onClose={onClose} size="md">
      <h2>高级数据保护</h2>

      {isEnvManaged ? (
        <div class="info-banner" style="margin-bottom: 16px;">
          已通过环境变量 <code>LLM_PROXY_ADMIN_KEY</code> 启用认证，无法在此面板修改。
        </div>
      ) : (
        <>
          <Toggle
            checked={isEnabled}
            onChange={(checked) => {
              if (!checked && isEnabled) return; // 通过按钮禁用
              if (checked && !isEnabled) return;  // 通过按钮启用
            }}
            label="启用管理面板认证"
          />

          <div style="margin-top: 12px;">
            {!isEnabled ? (
              /* 首次启用：输入新密钥 */
              <div style="display: flex; flex-direction: column; gap: 12px;">
                <Field label="管理密钥" hint="至少 6 个字符。设置后管理面板需要此密钥。">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={key}
                    onInput={(e) => setKey((e.target as HTMLInputElement).value)}
                    placeholder="输入管理密钥"
                    class="w-full"
                  />
                </Field>
                <Field label="确认密钥">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={confirmKey}
                    onInput={(e) => setConfirmKey((e.target as HTMLInputElement).value)}
                    placeholder="再次输入以确认"
                    class="w-full"
                  />
                </Field>
                <label style="display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: var(--fs-sm); color: var(--text-secondary);">
                  <input
                    type="checkbox"
                    checked={showPassword}
                    onChange={(e) => setShowPassword((e.target as HTMLInputElement).checked)}
                  />
                  显示密钥
                </label>
                <button
                  class="btn btn-secondary"
                  disabled={saving}
                  onClick={() => handleSave(true)}
                  style="align-self: flex-start;"
                >
                  {saving ? '保存中...' : '启用保护'}
                </button>
              </div>
            ) : (
              /* 已启用：显示状态 + 修改/禁用 */
              <div style="display: flex; flex-direction: column; gap: 12px;">
                <div class="info-banner">管理面板认证已启用。密钥已保存在浏览器本地。</div>

                <Field label="当前密钥" hint="输入当前密钥以修改或禁用。">
                  <input
                    type="password"
                    value={currentKey}
                    onInput={(e) => setCurrentKey((e.target as HTMLInputElement).value)}
                    placeholder="输入当前密钥"
                    class="w-full"
                  />
                </Field>
                <Field label="新密钥（可选）" hint="留空则仅禁用保护。">
                  <input
                    type="password"
                    value={key}
                    onInput={(e) => setKey((e.target as HTMLInputElement).value)}
                    placeholder="新密钥"
                    class="w-full"
                  />
                </Field>
                <Field label="确认新密钥">
                  <input
                    type="password"
                    value={confirmKey}
                    onInput={(e) => setConfirmKey((e.target as HTMLInputElement).value)}
                    placeholder="确认新密钥"
                    class="w-full"
                  />
                </Field>
                <div style="display: flex; gap: 8px;">
                  <button
                    class="btn btn-secondary"
                    disabled={saving}
                    onClick={handleChangeKey}
                  >
                    {saving ? '保存中...' : '更新密钥'}
                  </button>
                  <button
                    class="btn btn-secondary"
                    disabled={saving || !currentKey}
                    onClick={() => handleSave(false)}
                  >
                    禁用保护
                  </button>
                </div>
                <p class="hint">忘记密钥？设置环境变量 <code>LLM_PROXY_ADMIN_KEY</code> 可覆盖。</p>
              </div>
            )}
          </div>
        </>
      )}
    </Modal>
  );
}
