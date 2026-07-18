import { signal } from '@preact/signals';
import { useEffect, useState } from 'preact/hooks';
import { initStore, endpointsSignal, selectEndpoint, adminAuthSignal } from './state/store';
import { ConfigPage } from './pages/ConfigPage';
import { EndpointPage } from './pages/EndpointPage';
import { UsagePage } from './pages/UsagePage';
import { LogsPage } from './pages/LogsPage';
import { ToastContainer } from './components/Toast';
import { EmptyState } from './components/EmptyState';
import { ModelModal } from './modals/ModelModal';
import { EndpointModal } from './modals/EndpointModal';
import { MappingModal } from './modals/MappingModal';
import { AdminAuthModal } from './components/AdminAuthModal';
import { openEndpointModal as openEndpointModalFn } from './modals/EndpointModal';
import { ApiCallError } from './api/client';

type TabId = 'config' | 'usage' | 'logs' | `ep:${string}`;

const activeTab = signal<TabId>('config');
const appLoading = signal(true);
const appError = signal<string | null>(null);
const adminAuthModalOpen = signal(false);
const keyPromptOpen = signal(false);

function TabIcon({ name }: { name: string }) {
  const cls = 'tab-icon';
  switch (name) {
    case 'config':
      return <svg class={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>;
    case 'usage':
      return <svg class={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/></svg>;
    case 'endpoint':
      return <svg class={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>;
    case 'logs':
      return <svg class={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>;
    case 'plus':
      return <svg class={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>;
    default:
      return null;
  }
}

function TabButton({ id, label, isEndpoint }: { id: TabId; label: string; isEndpoint?: boolean }) {
  const isActive = activeTab.value === id;
  const iconKey = id.startsWith('ep:') ? 'endpoint' : id;
  return (
    <button
      class={`tab-btn${isActive ? ' active' : ''}${isEndpoint ? ' endpoint' : ''}`}
      onClick={() => { activeTab.value = id; if (id.startsWith('ep:')) selectEndpoint(id.slice(3)); }}
    >
      <TabIcon name={iconKey} />
      {label}
    </button>
  );
}

function SettingsIcon() {
  return (
    <svg class="header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function LockIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14" style="margin-right: 4px; vertical-align: middle;">
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

function KeyPrompt({ onSuccess }: { onSuccess: () => void }) {
  const [key, setKey] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: Event) => {
    e.preventDefault();
    if (!key) { setError('请输入管理密钥'); return; }
    setLoading(true);
    setError('');

    // 保存到 localStorage 并重试
    localStorage.setItem('adminKey', key);
    try {
      await initStore();
      onSuccess();
    } catch (err) {
      localStorage.removeItem('adminKey');
      if (err instanceof ApiCallError && err.status === 401) {
        setError('密钥不正确，请重试');
      } else {
        setError(err instanceof Error ? err.message : '连接失败');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style="max-width: 400px; margin: 40px auto;">
      <div class="card">
        <h2><LockIcon /> 需要管理密钥</h2>
        <p class="hint" style="margin-bottom: 12px;">此管理面板已启用高级数据保护，请输入密钥以继续。</p>
        <form onSubmit={handleSubmit}>
          <input
            type="password"
            value={key}
            onInput={(e) => { setKey((e.target as HTMLInputElement).value); setError(''); }}
            placeholder="输入管理密钥"
            class="w-full"
            style="margin-bottom: 8px;"
            autofocus
          />
          {error && <p style="color: var(--danger); font-size: var(--fs-sm); margin-bottom: 8px;">{error}</p>}
          <button type="submit" class="btn btn-secondary" disabled={loading}>
            {loading ? '验证中...' : '确认'}
          </button>
        </form>
      </div>
    </div>
  );
}

function LoadingIndicator() {
  return <EmptyState>加载中...</EmptyState>;
}

function ErrorDisplay({ message }: { message: string }) {
  const onRetry = async () => {
    appLoading.value = true;
    appError.value = null;
    try {
      await initStore();
      appLoading.value = false;
    } catch (e) {
      appError.value = e instanceof Error ? e.message : '初始化失败，请刷新页面';
      appLoading.value = false;
    }
  };
  return (
    <div class="card error-banner" style="text-align: center;">
      <p>{message}</p>
      <button onClick={onRetry}>重试</button>
    </div>
  );
}

declare global {
  interface Window {
    openEndpointModal?: () => void;
  }
}

export function App() {
  useEffect(() => {
    (async () => {
      try {
        await initStore();
        appLoading.value = false;
      } catch (e) {
        if (e instanceof ApiCallError && e.status === 401) {
          // 管理面板需要认证，弹出 key 输入框
          keyPromptOpen.value = true;
        } else {
          appError.value = e instanceof Error ? e.message : '初始化失败，请刷新页面';
        }
        appLoading.value = false;
      }
    })();
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem('activeTab') as TabId | null;
    if (saved && (saved === 'config' || saved === 'usage' || saved === 'logs' || saved.startsWith('ep:'))) {
      activeTab.value = saved;
      if (saved.startsWith('ep:')) selectEndpoint(saved.slice(3));
    }
  }, []);

  useEffect(() => {
    const unsubscribe = activeTab.subscribe((tab) => {
      localStorage.setItem('activeTab', tab);
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        await initStore();
      } catch {
        // Silently fail on background refresh
      }
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  if (appLoading.value) {
    return (
      <div>
        <h1>LLM Proxy 控制面板</h1>
        <LoadingIndicator />
      </div>
    );
  }

  if (keyPromptOpen.value) {
    return (
      <div>
        <h1>LLM Proxy 控制面板</h1>
        <KeyPrompt onSuccess={() => { keyPromptOpen.value = false; }} />
      </div>
    );
  }

  if (appError.value) {
    return (
      <div>
        <h1>LLM Proxy 控制面板</h1>
        <ErrorDisplay message={appError.value} />
        <ToastContainer />
      </div>
    );
  }

  const tab = activeTab.value;
  const endpoints = endpointsSignal.value;

  const tabContent = (() => {
    if (tab === 'config') return <ConfigPage />;
    if (tab === 'usage') return <UsagePage />;
    if (tab === 'logs') return <LogsPage />;
    if (tab.startsWith('ep:')) return <EndpointPage />;
    return null;
  })();

  return (
    <div>
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
        <h1 style="margin-bottom: 0;">LLM Proxy 控制面板</h1>
        <div style="display: flex; align-items: center; gap: 8px;">
          {adminAuthSignal.value.enabled && (
            <span title="高级数据保护已启用" style="color: var(--accent); font-size: var(--fs-sm); display: flex; align-items: center;">
              <LockIcon />
            </span>
          )}
          <button
            class="tab-btn"
            onClick={() => { adminAuthModalOpen.value = true; }}
            title="高级数据保护设置"
            style="padding: 4px 8px;"
          >
            <SettingsIcon />
          </button>
        </div>
      </div>
      <div class="tabs">
        <TabButton id="config" label="配置" />
        <TabButton id="usage" label="用量" />
        <TabButton id="logs" label="日志" />
        {endpoints.map(ep => (
          <TabButton
            key={ep.endpoint_id}
            id={`ep:${ep.endpoint_id}` as TabId}
            label={ep.name || ep.endpoint_id.slice(0, 6)}
            isEndpoint
          />
        ))}
        <button class="tab-btn add-btn" onClick={() => openEndpointModalFn()} title="添加端点">
          <TabIcon name="plus" />
          端点
        </button>
      </div>
      <div class="tab-content active">
        {tabContent}
      </div>
      <ModelModal />
      <EndpointModal />
      <MappingModal />
      {adminAuthModalOpen.value && (
        <AdminAuthModal onClose={() => { adminAuthModalOpen.value = false; }} />
      )}
      <ToastContainer />
    </div>
  );
}
