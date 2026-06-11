import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/preact';
import { ConfigPage } from './ConfigPage';
import { configSignal, setConfig } from '../state/store';
import type { Config } from '../api/types';

const mockConfirm = vi.fn();
window.confirm = mockConfirm;

vi.mock('../components/Toast', () => ({
  showToast: vi.fn(),
  ToastContainer: () => null,
}));

const mockOpenModelModal = vi.fn();
vi.mock('../modals/ModelModal', () => ({
  openModelModal: (...args: unknown[]) => mockOpenModelModal(...args),
  ModelModal: () => null,
}));

const sampleConfig: Config = {
  models: {
    'opus-4-7': {
      api_base: 'https://api.anthropic.com',
      api_key: 'sk-test-1',
      upstream_model: 'opus-4-7',
      display_name: 'Claude Opus 4.7',
      upstream_protocol: 'anthropic',
      context_window: 200000,
    },
    'gpt-5': {
      api_base: 'https://api.openai.com',
      api_key: 'sk-test-2',
      upstream_model: 'gpt-5',
      display_name: 'GPT-5',
      upstream_paths: { 'openai/chat-completions': '/v1/chat/completions' },
    },
    'deepseek-v4': {
      api_base: 'https://api.deepseek.com',
      api_key: 'sk-test-3',
      upstream_model: 'deepseek-v4',
      context_window: 128000,
    },
  },
  error_handling: {
    failover_enabled: true,
    no_retry_enabled: false,
  },
};

describe('ConfigPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    configSignal.value = null;
  });

  it('renders models in a table', () => {
    setConfig(sampleConfig);
    render(<ConfigPage />);

    expect(screen.getByText('已有模型')).toBeTruthy();
    expect(screen.getByText('Claude Opus 4.7')).toBeTruthy();
    expect(screen.getByText('GPT-5')).toBeTruthy();
    expect(screen.getByText('opus-4-7')).toBeTruthy();
    expect(screen.getByText('gpt-5')).toBeTruthy();
  });

  it('shows protocol labels correctly', () => {
    setConfig(sampleConfig);
    render(<ConfigPage />);

    expect(screen.getByText('Anthropic')).toBeTruthy();
    expect(screen.getByText('Chat')).toBeTruthy();
  });

  it('renders error handling toggles', () => {
    setConfig(sampleConfig);
    render(<ConfigPage />);

    const toggles = document.querySelectorAll('.toggle-switch input[type="checkbox"]');
    expect(toggles.length).toBe(2);
    expect((toggles[0] as HTMLInputElement).checked).toBe(true);
    expect((toggles[1] as HTMLInputElement).checked).toBe(false);
  });

  it('shows empty state when no models', () => {
    setConfig({ models: {}, error_handling: { failover_enabled: false, no_retry_enabled: false } });
    render(<ConfigPage />);

    expect(screen.getByText('暂无模型')).toBeTruthy();
  });

  it('calls openModelModal when add model button clicked', () => {
    setConfig(sampleConfig);
    render(<ConfigPage />);

    const addBtn = screen.getByText('添加模型');
    fireEvent.click(addBtn);
    expect(mockOpenModelModal).toHaveBeenCalledWith();
  });

  it('calls openModelModal with model name when edit button clicked', () => {
    setConfig(sampleConfig);
    render(<ConfigPage />);

    const editBtns = document.querySelectorAll('button.icon[title="编辑"]');
    expect(editBtns.length).toBe(3);

    fireEvent.click(editBtns[0]);
    expect(mockOpenModelModal).toHaveBeenCalledWith('opus-4-7');
  });

  it('shows confirm dialog when delete button clicked', () => {
    mockConfirm.mockReturnValueOnce(false);
    setConfig(sampleConfig);
    render(<ConfigPage />);

    const deleteBtns = document.querySelectorAll('button.icon[title="删除"]');
    expect(deleteBtns.length).toBe(3);

    fireEvent.click(deleteBtns[0]);
    expect(mockConfirm).toHaveBeenCalledWith('删除模型 opus-4-7？');
  });

  it('context window formats correctly', () => {
    setConfig(sampleConfig);
    render(<ConfigPage />);

    expect(screen.getByText('200K')).toBeTruthy();
  });

  it('renders — for models without context_window', () => {
    setConfig(sampleConfig);
    render(<ConfigPage />);

    const dashes = document.querySelectorAll('td');
    const dashCells = Array.from(dashes).filter(td => td.textContent === '—');
    expect(dashCells.length).toBeGreaterThan(0);
  });
});
