export interface ProtocolChipProps {
  checked: boolean;
  onToggle: () => void;
  label: string;
}

export function ProtocolChip({ checked, onToggle, label }: ProtocolChipProps) {
  return (
    <button
      type="button"
      onClick={onToggle}
      class={`protocol-chip${checked ? ' active' : ''}`}
    >
      {label}
    </button>
  );
}

const BADGE_VARIANT: Record<string, string> = {
  Anthropic: 'anthropic',
  Chat: 'chat',
  Responses: 'responses',
};

export interface ProtocolBadgeProps {
  label: string;
}

export function ProtocolBadge({ label }: ProtocolBadgeProps) {
  const variant = BADGE_VARIANT[label] ?? '';
  return <span class={`protocol-chip-badge ${variant}`}>{label}</span>;
}
