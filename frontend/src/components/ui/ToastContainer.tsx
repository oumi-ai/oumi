"use client";

import React from 'react';
import { onToast, ToastVariant } from '@/lib/toastBus';

type ToastItem = {
  id: string;
  message: string;
  variant: ToastVariant;
  expiresAt: number;
};

function clsx(...parts: Array<string | false | undefined>) {
  return parts.filter(Boolean).join(' ');
}

export default function ToastContainer() {
  const [items, setItems] = React.useState<ToastItem[]>([]);

  React.useEffect(() => {
    const unsubscribe = onToast(({ message, variant = 'info', durationMs = 3500 }) => {
      const id = Math.random().toString(36).slice(2);
      const expiresAt = Date.now() + Math.max(1500, durationMs);
      setItems((prev) => [...prev, { id, message, variant, expiresAt }]);
    });
    return () => unsubscribe();
  }, []);

  // Auto-prune expired items
  React.useEffect(() => {
    const t = setInterval(() => {
      const now = Date.now();
      setItems((prev) => prev.filter((it) => it.expiresAt > now));
    }, 250);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="fixed bottom-4 right-4 z-[9999] space-y-2 pointer-events-none">
      {items.map((it) => (
        <div
          key={it.id}
          className={clsx(
            'min-w-[260px] max-w-[420px] px-3 py-2 rounded shadow-lg text-sm text-white pointer-events-auto',
            it.variant === 'success' && 'bg-green-600',
            it.variant === 'warning' && 'bg-amber-600',
            it.variant === 'error' && 'bg-red-600',
            it.variant === 'info' && 'bg-slate-800'
          )}
          style={{ opacity: 1 }}
        >
          {it.message}
        </div>
      ))}
    </div>
  );
}

