import { useSignal } from '@preact/signals';
import type { ApiCallError } from '../api/client';

export interface UseFetchResult<T> {
  readonly data: T | null;
  readonly loading: boolean;
  readonly error: string | null;
  refetch: () => Promise<void>;
}

export function useFetch<T>(
  fetcher: () => Promise<T>,
  options: { immediate?: boolean } = {},
): UseFetchResult<T> {
  const { immediate = true } = options;
  const data = useSignal<T | null>(null);
  const loading = useSignal(immediate);
  const error = useSignal<string | null>(null);
  const started = useSignal(false);

  async function refetch(): Promise<void> {
    loading.value = true;
    error.value = null;
    try {
      data.value = await fetcher();
    } catch (e) {
      if (e && typeof e === 'object' && 'detail' in e) {
        error.value = (e as ApiCallError).detail;
      } else if (e instanceof Error) {
        error.value = e.message;
      } else {
        error.value = String(e);
      }
    } finally {
      loading.value = false;
      started.value = true;
    }
  }

  if (immediate && !started.value) {
    void refetch();
  }

  return {
    get data() { return data.value; },
    get loading() { return loading.value; },
    get error() { return error.value; },
    refetch,
  };
}
