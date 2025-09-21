"use client";

/**
 * Minimal event-based toast bus for lightweight notifications.
 */

export type ToastVariant = 'info' | 'success' | 'warning' | 'error';

export interface ToastEventDetail {
  message: string;
  variant?: ToastVariant;
  durationMs?: number; // default 3500ms
}

const emitter: EventTarget = typeof window !== 'undefined' ? new EventTarget() : ({} as any);

export function showToast(detail: ToastEventDetail) {
  if (typeof window === 'undefined' || !('dispatchEvent' in emitter)) return;
  const event = new CustomEvent<ToastEventDetail>('toast', { detail });
  (emitter as EventTarget).dispatchEvent(event);
}

export function onToast(handler: (d: ToastEventDetail) => void) {
  if (typeof window === 'undefined' || !('addEventListener' in emitter)) return () => {};
  const wrapped = (e: Event) => handler((e as CustomEvent<ToastEventDetail>).detail);
  (emitter as EventTarget).addEventListener('toast', wrapped);
  return () => (emitter as EventTarget).removeEventListener('toast', wrapped);
}

