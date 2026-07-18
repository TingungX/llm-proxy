// ===== Config =====
export interface ProtocolEntry {
  protocol: string;
  enabled: boolean;
  path?: string;
}

export interface ModelConfig {
  api_base: string;
  api_key: string;
  upstream_model: string;
  display_name?: string;
  upstream_protocols?: ProtocolEntry[];
  context_window?: number;
  vision_support?: boolean;
  allow_proxy?: boolean;
  provider?: string;
  thinking_effort_mode?: 'default' | 'provider' | 'custom';
  thinking_effort_preset?: string | EffortPreset;
}

export interface ProviderProfileInfo {
  display_name: string;
  default_api_base: string;
  default_thinking_effort_preset?: string | null;
  /** thinking format 类型：thinking_type_plus_reasoning_effort / thinking_enabled_disabled / ... */
  thinking_format?: string | null;
  /** 默认 thinking type（如 enabled / adaptive） */
  default_thinking_type?: string | null;
  /** 禁用 thinking 的值（如 disabled） */
  disable_thinking_value?: string | null;
  /** 厂商专属字段名（如 enable_thinking） */
  effort_field?: string | null;
  /** 固定 effort 值（如 Kimi K3 的 max） */
  fixed_effort?: string | null;
  /** effort 别名映射（如 low→high） */
  effort_aliases?: Record<string, string | null>;
  supports_reasoning_split?: boolean;
  preserve_reasoning_content?: boolean;
}

export interface ErrorHandlingConfig {
  failover_enabled: boolean;
  no_retry_enabled: boolean;
}

export type EffortPresetType = 'any_to_any' | 'thinking_on' | string;

export interface EffortPreset {
  name: string;
  type: EffortPresetType;
  rules: Record<string, string>;
}

export interface ThinkingEffortMapping {
  presets: EffortPreset[];
  default_preset: string;
}

export interface Config {
  models: Record<string, ModelConfig>;
  error_handling: ErrorHandlingConfig;
  thinking_effort_mapping?: ThinkingEffortMapping;
  sidecar?: {
    bin_path: string;
    start_port: number;
  };
}

// ===== Endpoints =====
export interface EndpointSettings {
  failover_enabled: boolean;
  no_retry_enabled: boolean;
  compression?: {
    enabled: boolean;
    strategies?: string[];
  };
}

export interface Endpoint {
  endpoint_id: string;
  name: string;
  api_key?: string;
  is_default?: boolean;
  enabled: boolean;
  models: string[];
  settings: EndpointSettings;
  family_routing: Record<string, string>;
  accept_protocols: string[];
  last_used: string | null;
}

export interface CreateEndpointRequest {
  name: string;
  api_key?: string;
  enabled?: boolean;
  models?: string[];
  settings?: Partial<EndpointSettings>;
  accept_protocols?: string[];
  family_routing?: Record<string, string>;
}

// ===== Usage =====
export interface UsageDataPoint {
  time: string;
  group_key: string;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  count: number;
}

export interface UsageResponse {
  data: UsageDataPoint[];
}

export interface UsageSummary {
  total_tokens: number;
  today_tokens: number;
  total_requests: number;
  active_endpoints: number;
}

export interface HeatmapDataPoint {
  date: string;
  total_tokens: number;
}

export interface HeatmapResponse {
  data: HeatmapDataPoint[];
}

// ===== Logs =====
export type ErrorType = 'timeout' | '4xx' | '5xx' | 'failover' | 'parse_error';
export type RequestStatus = 'success' | 'error';

export interface LogRecord {
  id: number;
  timestamp: string;
  endpoint_id: string;
  model_id: string;
  input_tokens: number;
  output_tokens: number;
  request_status: RequestStatus;
  request_id: string | null;
  latency_ms: number | null;
  error_type: ErrorType | null;
  client_ip: string | null;
  user_agent: string | null;
}

export interface LogsListResponse {
  records: LogRecord[];
  total: number;
  limit: number;
  offset: number;
}

export interface LogsSummary {
  total_requests: number;
  error_count: number;
  avg_latency_ms: number | null;
  p95_latency_ms: number | null;
  total_input_tokens: number;
  total_output_tokens: number;
}

export interface FilterOptions {
  endpoints: Array<{ id: string; name: string }>;
  models: string[];
  statuses: RequestStatus[];
  error_types: ErrorType[];
}

// ===== Latency =====
export interface LatencyResult {
  model: string;
  rounds: number;
  avg: number;
  min: number;
  max: number;
  error?: string;
}

// ===== Admin Auth =====
export interface AdminAuthStatus {
  enabled: boolean;
  source: 'env' | 'config' | null;
}

// ===== Protocol Detection =====
export interface ProtocolDetectionResult {
  upstream_protocols?: ProtocolEntry[];
  error?: string;
}
