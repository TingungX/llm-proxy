export class ApiCallError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
  ) {
    super(`${status}: ${detail}`);
    this.name = 'ApiCallError';
  }
}

export async function api<T>(
  path: string,
  init?: RequestInit & { json?: unknown },
): Promise<T> {
  const { json, ...rest } = init ?? {};
  const r = await fetch(path, {
    ...rest,
    headers: {
      'Content-Type': 'application/json',
      ...rest?.headers,
    },
    body: json !== undefined ? JSON.stringify(json) : rest?.body,
  });

  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = (j as { error?: string; detail?: string }).error
        ?? (j as { error?: string; detail?: string }).detail
        ?? detail;
    } catch {
      // Response body is not JSON; use statusText
    }
    throw new ApiCallError(r.status, detail);
  }

  if (r.status === 204) {
    return undefined as T;
  }

  return r.json() as Promise<T>;
}
