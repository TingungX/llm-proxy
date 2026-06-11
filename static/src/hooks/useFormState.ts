import { useSignal } from '@preact/signals';
import type { Signal } from '@preact/signals';
import type { ApiCallError } from '../api/client';

export interface FormState<T> {
  /** Signal 形式，组件读 `form.values.value` 自动订阅 */
  readonly values: Signal<T>;
  readonly errors: Signal<Partial<Record<keyof T, string>>>;
  readonly submitting: Signal<boolean>;
  readonly generalError: Signal<string | null>;
  setField: <K extends keyof T>(key: K, value: T[K]) => void;
  setErrors: (errors: Partial<Record<keyof T, string>>) => void;
  handleSubmit: (onSubmit: (values: T) => Promise<void>) => Promise<void>;
  reset: (initialValues: T) => void;
}

export function useFormState<T extends Record<string, unknown>>(initialValues: T): FormState<T> {
  const values = useSignal<T>({ ...initialValues });
  const errors = useSignal<Partial<Record<keyof T, string>>>({});
  const submitting = useSignal(false);
  const generalError = useSignal<string | null>(null);

  function setField<K extends keyof T>(key: K, value: T[K]): void {
    values.value = { ...values.value, [key]: value };
    if (errors.value[key]) {
      const newErrors = { ...errors.value };
      delete newErrors[key];
      errors.value = newErrors;
    }
  }

  function setErrors(newErrors: Partial<Record<keyof T, string>>): void {
    errors.value = newErrors;
  }

  async function handleSubmit(onSubmit: (values: T) => Promise<void>): Promise<void> {
    submitting.value = true;
    generalError.value = null;
    try {
      await onSubmit(values.value);
    } catch (e) {
      if (e && typeof e === 'object' && 'detail' in e) {
        generalError.value = (e as ApiCallError).detail;
      } else if (e instanceof Error) {
        generalError.value = e.message;
      } else {
        generalError.value = String(e);
      }
      throw e;
    } finally {
      submitting.value = false;
    }
  }

  function reset(newInitialValues: T): void {
    values.value = { ...newInitialValues };
    errors.value = {};
    generalError.value = null;
    submitting.value = false;
  }

  return {
    values,
    errors,
    submitting,
    generalError,
    setField,
    setErrors,
    handleSubmit,
    reset,
  };
}
