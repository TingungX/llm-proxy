import { useState, useEffect } from 'preact/hooks';
import type { JSX } from 'preact';

interface ToastMessage {
  id: number;
  text: string;
  type: 'ok' | 'err';
}

let nextId = 0;
let setToastsFn: ((updater: (prev: ToastMessage[]) => ToastMessage[]) => void) | null = null;

export function showToast(text: string, type: 'ok' | 'err' = 'ok'): void {
  const id = ++nextId;
  setToastsFn?.((prev) => [...prev, { id, text, type }]);
  setTimeout(() => {
    setToastsFn?.((prev) => prev.filter((t) => t.id !== id));
  }, 2000);
}

export function ToastContainer(): JSX.Element | null {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  useEffect(() => {
    setToastsFn = setToasts;
    return () => {
      setToastsFn = null;
    };
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div class="toast-container">
      {toasts.map((t) => (
        <div key={t.id} class={`toast toast-${t.type}`}>
          {t.text}
        </div>
      ))}
    </div>
  );
}
