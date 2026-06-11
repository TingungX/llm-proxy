import { describe, it, expect, vi } from 'vitest';
import { signal, computed, effect } from '@preact/signals';
import { render, screen } from '@testing-library/preact';

describe('Preact + Signals Technology Validation', () => {
  it('signal: basic get/set works', () => {
    const count = signal(0);
    expect(count.value).toBe(0);
    count.value = 1;
    expect(count.value).toBe(1);
  });

  it('computed: derived signal updates when deps change', () => {
    const a = signal(1);
    const b = signal(2);
    const sum = computed(() => a.value + b.value);
    expect(sum.value).toBe(3);
    a.value = 10;
    expect(sum.value).toBe(12);
  });

  it('effect: side effects run on signal change', () => {
    const count = signal(0);
    const sideEffect = vi.fn();
    const dispose = effect(() => { sideEffect(count.value); });
    expect(sideEffect).toHaveBeenCalledTimes(1);
    expect(sideEffect).toHaveBeenCalledWith(0);
    count.value = 5;
    expect(sideEffect).toHaveBeenCalledTimes(2);
    expect(sideEffect).toHaveBeenCalledWith(5);
    dispose();
  });

  it('preact: renders component with signal', () => {
    const count = signal(42);
    function Counter() {
      return <div data-testid="count">{count.value}</div>;
    }
    render(<Counter />);
    expect(screen.getByTestId('count').textContent).toBe('42');
  });

  it('preact: component re-renders when signal changes', async () => {
    const count = signal(0);
    function Counter() {
      return <div data-testid="count">{count.value}</div>;
    }
    render(<Counter />);
    expect(screen.getByTestId('count').textContent).toBe('0');
    count.value = 100;
    await new Promise(r => setTimeout(r, 0));
    expect(screen.getByTestId('count').textContent).toBe('100');
  });
});
