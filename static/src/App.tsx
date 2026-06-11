import { signal } from '@preact/signals';
import { useEffect } from 'preact/hooks';
import { initStore, endpointsSignal, selectEndpoint } from './state/store';
import { ConfigPage } from './pages/ConfigPage';
import { EndpointPage } from './pages/EndpointPage';
import { UsagePage } from './pages/UsagePage';
import { LogsPage } from './pages/LogsPage';
import { ToastContainer } from './components/Toast';
import { EmptyState } from './components/EmptyState';
import { ModelModal } from './modals/ModelModal';
import { EndpointModal } from './modals/EndpointModal';
import { MappingModal } from './modals/MappingModal';
import { openEndpointModal as openEndpointModalFn } from './modals/EndpointModal';

type TabId = 'config' | 'usage' | 'logs' | `ep:${string}`;

const activeTab = signal<TabId>('config');
const appLoading = signal(true);
const appError = signal<string | null>(null);

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
        appError.value = e instanceof Error ? e.message : '初始化失败，请刷新页面';
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
      <h1>LLM Proxy 控制面板</h1>
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
      <ToastContainer />
    </div>
  );
}
