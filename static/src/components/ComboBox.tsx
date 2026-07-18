import { useEffect, useRef, useState } from 'preact/hooks';

export interface ComboBoxOption {
  value: string;
  label: string;
}

interface ComboBoxProps {
  value: string;
  options: ComboBoxOption[];
  onInput: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
}

export function ComboBox({ value, options, onInput, placeholder, disabled }: ComboBoxProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState(value);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setInputValue(value);
  }, [value]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const filtered = options.filter(
    (o) =>
      o.value.toLowerCase().includes(inputValue.toLowerCase()) ||
      o.label.toLowerCase().includes(inputValue.toLowerCase()),
  );

  const handleSelect = (v: string) => {
    setInputValue(v);
    onInput(v);
    setOpen(false);
  };

  return (
    <div class="combobox" ref={containerRef}>
      <input
        type="text"
        class="combobox-input w-full"
        value={inputValue}
        placeholder={placeholder}
        disabled={disabled}
        onInput={(e: Event) => {
          const v = (e.target as HTMLInputElement).value;
          setInputValue(v);
          onInput(v);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e: KeyboardEvent) => {
          if (e.key === 'Escape') setOpen(false);
        }}
      />
      <button
        type="button"
        class="combobox-toggle"
        disabled={disabled}
        onClick={() => setOpen(!open)}
        aria-label="展开选项"
      >
        <svg width="10" height="6" viewBox="0 0 10 6" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M1 1L5 5L9 1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>
      {open && filtered.length > 0 && (
        <ul class="combobox-dropdown">
          {filtered.map((o) => (
            <li
              key={o.value}
              class={o.value === value ? 'active' : ''}
              onClick={() => handleSelect(o.value)}
            >
              <span class="combobox-label">{o.label}</span>
              <span class="combobox-value">{o.value}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
