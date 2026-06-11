import type { ComponentChildren } from 'preact';

type ModalSize = 'sm' | 'md' | 'lg' | 'xl';

export interface ModalProps {
  onClose: () => void;
  children: ComponentChildren;
  size?: ModalSize;
  className?: string;
  closeOnBackdrop?: boolean;
}

export function Modal({ onClose, children, size = 'md', className = '', closeOnBackdrop = true }: ModalProps) {
  const handleBackdrop = (e: MouseEvent) => {
    if (closeOnBackdrop && e.target === e.currentTarget) onClose();
  };
  const stop = (e: MouseEvent) => e.stopPropagation();
  return (
    <div class="modal" onClick={handleBackdrop}>
      <div class={`modal-content ${size} modal-body-pad ${className}`} onClick={stop}>
        {children}
      </div>
    </div>
  );
}
