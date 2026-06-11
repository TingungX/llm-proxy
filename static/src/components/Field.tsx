import type { ComponentChildren } from 'preact';

export interface FieldProps {
  label: string;
  hint?: string;
  error?: string;
  children: ComponentChildren;
  required?: boolean;
}

export function Field({ label, hint, error, children, required }: FieldProps) {
  return (
    <label class="field">
      <span class="field-label">{label}{required ? ' *' : ''}</span>
      {children}
      {hint && <span class="field-hint">{hint}</span>}
      {error && <span class="field-error">{error}</span>}
    </label>
  );
}
