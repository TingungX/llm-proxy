import type { ModelConfig } from '../api/types';

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

export function toUpstreamProtocol(p: ProtocolConfig): string {
  const { anthropic, openai_chat, openai_responses } = p.enabled;
  const openaiEnabled = openai_chat || openai_responses;
  // Mixed (anthropic + openai) = auto
  if (anthropic && openaiEnabled) return '';
  if (!anthropic && !openaiEnabled) return '';
  if (anthropic) return 'anthropic';
  return 'openai';
}

export function toUpstreamProtocols(p: ProtocolConfig): string[] {
  const out: string[] = [];
  if (p.enabled.anthropic) out.push('anthropic');
  if (p.enabled.openai_chat) out.push('openai/chat-completions');
  if (p.enabled.openai_responses) out.push('openai/responses');
  return out;
}

export function toUpstreamPaths(p: ProtocolConfig): Record<string, string> {
  const out: Record<string, string> = {};
  if (p.enabled.anthropic) {
    const v = p.paths.anthropic.trim();
    out['anthropic/messages'] = v || DEFAULT_PATHS['anthropic/messages'];
  }
  if (p.enabled.openai_chat) {
    const v = p.paths.openai_chat.trim();
    out['openai/chat-completions'] = v || DEFAULT_PATHS['openai/chat-completions'];
  }
  if (p.enabled.openai_responses) {
    const v = p.paths.openai_responses.trim();
    out['openai/responses'] = v || DEFAULT_PATHS['openai/responses'];
  }
  return out;
}

/** 从模型配置推导支持的协议标签列表 */
export function getModelProtocols(m: ModelConfig): string[] {
  const paths = m.upstream_paths ?? {};
  // 优先读上游协议列表（新格式）；缺失时回退到标量（迁移期兼容）
  const protocols = new Set<string>(
    m.upstream_protocols ?? (m.upstream_protocol ? [m.upstream_protocol] : [])
  );
  const tags: string[] = [];
  if (protocols.has('anthropic') || paths['anthropic/messages']) tags.push('Anthropic');
  // 旧标量 'openai' 等价于 'openai/chat-completions'（向后兼容映射）
  if (protocols.has('openai') || protocols.has('openai/chat-completions') || paths['openai/chat-completions']) tags.push('Chat');
  if (protocols.has('openai/responses') || paths['openai/responses']) tags.push('Responses');
  return tags;
}
