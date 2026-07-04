import type { ModelConfig, ProtocolEntry } from '../api/types';

export const DEFAULT_PATHS = {
  'anthropic/messages': 'anthropic/v1/messages',
  'openai/chat-completions': '/v1/chat/completions',
  'openai/responses': '/v1/responses',
} as const;

export interface ProtocolConfig {
  enabled: {
    anthropic: boolean;
    openai_chat: boolean;
    openai_responses: boolean;
  };
  paths: {
    anthropic: string;
    openai_chat: string;
    openai_responses: string;
  };
}

/** 从模型配置提取三个协议的启用状态和路径 */
export function extractProtocolConfig(m: ModelConfig): ProtocolConfig {
  const entries = m.upstream_protocols ?? [];

  const getEntry = (proto: string) => entries.find(e => e.protocol === proto);

  const eAnthropic = getEntry('anthropic');
  const eChat = getEntry('openai/chat-completions');
  const eResponses = getEntry('openai/responses');

  return {
    enabled: {
      anthropic: eAnthropic?.enabled ?? false,
      openai_chat: eChat?.enabled ?? false,
      openai_responses: eResponses?.enabled ?? false,
    },
    paths: {
      anthropic: eAnthropic?.path || DEFAULT_PATHS['anthropic/messages'],
      openai_chat: eChat?.path || DEFAULT_PATHS['openai/chat-completions'],
      openai_responses: eResponses?.path || DEFAULT_PATHS['openai/responses'],
    },
  };
}

/** 构建 upstream_protocols 数组（所有三个协议都在，含 enabled 和 path） */
export function toUpstreamProtocols(p: ProtocolConfig): ProtocolEntry[] {
  return [
    { protocol: 'anthropic', enabled: p.enabled.anthropic, path: p.paths.anthropic.trim() || DEFAULT_PATHS['anthropic/messages'] },
    { protocol: 'openai/chat-completions', enabled: p.enabled.openai_chat, path: p.paths.openai_chat.trim() || DEFAULT_PATHS['openai/chat-completions'] },
    { protocol: 'openai/responses', enabled: p.enabled.openai_responses, path: p.paths.openai_responses.trim() || DEFAULT_PATHS['openai/responses'] },
  ];
}

/** 从模型配置推导支持的协议标签列表（显示用，仅启用的） */
export function getModelProtocols(m: ModelConfig): string[] {
  const config = extractProtocolConfig(m);
  const tags: string[] = [];
  if (config.enabled.anthropic) tags.push('Anthropic');
  if (config.enabled.openai_chat) tags.push('Chat');
  if (config.enabled.openai_responses) tags.push('Responses');
  return tags;
}