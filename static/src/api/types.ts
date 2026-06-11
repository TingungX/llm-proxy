// ===== Config =====
export interface ModelConfig {
  api_base: string;
  api_key: string;
  upstream_model: string;
  display_name?: string;
  upstream_protocol?: string;
  upstream_protocols?: string[];
  upstream_paths?: Record<string, string>;
  context_window?: number;
  vision_support?: boolean;
  allow_proxy?: boolean;
}

export interface ErrorHandlingConfig {
  failover_enabled: boolean;
  no_retry_enabled: boolean;
}

export interface Config {
  models: Record<string, ModelConfig>;
  error_handling: ErrorHandlingConfig;
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

// ===== Protocol Detection =====
export interface ProtocolDetectionResult {
  upstream_protocol: string;
  upstream_protocols?: string[];
  error?: string;
}
