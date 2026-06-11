import type { JSX } from 'preact';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}

export function Toggle({ checked, onChange, label }: ToggleProps): JSX.Element {
  return (
    <div class="toggle-row">
      <span class="label">{label}</span>
      <label class="toggle-switch">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e: Event) => onChange((e.target as HTMLInputElement).checked)}
        />
        <span class="toggle-slider" />
      </label>
    </div>
  );
}
