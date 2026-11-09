import { html, useEffect, useLayoutEffect, useRef, useState } from '../lib.js';
import { closeModal, confirmDialog, notify, state } from '../store.js';

const PATHS = {
  plus: 'M12 5v14M5 12h14',
  search: 'M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM20 20l-3.6-3.6',
  archive: 'M3 5h18v4H3zM5 9v10h14V9M10 13h4',
  restore: 'M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5',
  sliders: 'M4 7h9M17 7h3M4 17h3M11 17h9M15 5v4M9 15v4',
  chip: 'M6 6h12v12H6zM10 10h4v4h-4zM9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4',
  chevronDown: 'M6 9l6 6 6-6',
  chevronRight: 'M9 6l6 6-6 6',
  chevronLeft: 'M15 6l-6 6 6 6',
  copy: 'M9 9h11v11H9zM5 15H4V4h11v1',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  edit: 'M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4',
  refresh: 'M20 11a8 8 0 0 0-14.3-4L4 9M4 4v5h5M4 13a8 8 0 0 0 14.3 4l1.7-2M20 20v-5h-5',
  trash: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v5M14 11v5',
  more: 'M5 12h.01M12 12h.01M19 12h.01',
  pin: 'M12 17v5M8 3h8l-1 7 3 4H6l3-4z',
  send: 'M12 19V5M6 11l6-6 6 6',
  arrowDown: 'M12 5v14M6 13l6 6 6-6',
  clip: 'M20 11.5l-8.2 8.2a5 5 0 0 1-7.1-7.1l8.5-8.4a3.3 3.3 0 0 1 4.7 4.7l-8.5 8.4a1.7 1.7 0 0 1-2.4-2.4l7.8-7.7',
  panelLeft: 'M4 5h16v14H4zM9 5v14',
  panelRight: 'M4 5h16v14H4zM15 5v14',
  think: 'M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.4 1 2.5h6c0-1.1.3-1.8 1-2.5A6 6 0 0 0 12 3z',
  x: 'M6 6l12 12M18 6L6 18',
  download: 'M12 4v11M7 11l5 5 5-5M4 20h16',
  file: 'M7 3h7l5 5v13H7zM14 3v5h5',
  play: 'M8 5l11 7-11 7z',
  compress: 'M4 12h16M12 3v5M9.5 5.5L12 8l2.5-2.5M12 21v-5M9.5 18.5L12 16l2.5 2.5',
  eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
  book: 'M5 4h12a2 2 0 0 1 2 2v14H7a2 2 0 0 1-2-2zM5 18a2 2 0 0 1 2-2h12',
  alert: 'M12 4l9 16H3zM12 10v5M12 18h.01',
  list: 'M5 5h6v6H5zM14 8h6M5 14h6v6H5zM14 17h6',
  chart: 'M4 20V4M4 20h16M9 16v-5M14 16V8M19 16V6',
  link: 'M10 14a4 4 0 0 0 6 .4l3-3a4.2 4.2 0 0 0-6-6l-1.5 1.5M14 10a4 4 0 0 0-6-.4l-3 3a4.2 4.2 0 0 0 6 6l1.5-1.5',
  stop: 'M7 7h10v10H7z',
  globe: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM3 12h18M12 3c2.5 2.6 3.8 5.7 3.8 9s-1.300 6.400-3.800 9c-2.500-2.600-3.800-5.700-3.800-9S9.500 5.600 12 3z',
  heart: 'M12 20s-7-4.600-7-10a4 4 0 0 1 7-2.600A4 4 0 0 1 19 10c0 5.400-7 10-7 10z',
  lock: 'M6 11h12v9H6zM9 11V8a3 3 0 0 1 6 0v3',
  shield: 'M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6z',
};

export function Icon({ name, size = 16 }) {
  const filled = name === 'stop';
  const wide = name === 'more';
  return html`<svg class="icon" width=${size} height=${size} viewBox="0 0 24 24" aria-hidden="true"
    fill=${filled ? 'currentColor' : 'none'} stroke=${filled ? 'none' : 'currentColor'}
    stroke-width=${wide ? 3 : 1.8} stroke-linecap="round" stroke-linejoin="round">
    <path d=${PATHS[name]} />
  </svg>`;
}

export const IconButton = ({ icon, tip, onClick, size = 16, class: cls = '', disabled, active }) => html`
  <button type="button" class="icon-btn ${active ? 'active' : ''} ${cls}" data-tip=${tip} aria-label=${tip}
    onClick=${onClick} disabled=${disabled}>
    <${Icon} name=${icon} size=${size} />
  </button>`;

export function Spinner({ size = 14 }) {
  return html`<span class="spinner" style=${{ width: `${size}px`, height: `${size}px` }} role="status" aria-label="加载中"></span>`;
}

/** 固定定位的浮层，自动避开视口边缘；不受父元素 overflow 裁剪。 */
export function Popover({ anchor, onClose, placement = 'bottom-start', width, class: cls = '', children }) {
  const ref = useRef(null);
  const [position, setPosition] = useState({ top: -9999, left: -9999 });

  useLayoutEffect(() => {
    if (!anchor || !ref.current) return;
    const box = anchor.getBoundingClientRect();
    const element = ref.current;
    const w = element.offsetWidth;
    const h = element.offsetHeight;
    const gap = 6;
    let top = placement.startsWith('top') ? box.top - h - gap : box.bottom + gap;
    let left = placement.endsWith('end') ? box.right - w : box.left;
    if (top + h > window.innerHeight - 8) top = Math.max(8, box.top - h - gap);
    if (top < 8) top = Math.min(box.bottom + gap, window.innerHeight - h - 8);
    left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
    setPosition({ top, left });
  }, [anchor]);

  useEffect(() => {
    const onPointer = (event) => {
      if (ref.current?.contains(event.target) || anchor?.contains(event.target)) return;
      onClose();
    };
    const onKey = (event) => {
      if (event.key === 'Escape') {
        event.stopPropagation();
        onClose();
      }
    };
    document.addEventListener('pointerdown', onPointer, true);
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('pointerdown', onPointer, true);
      document.removeEventListener('keydown', onKey, true);
    };
  }, [anchor, onClose]);

  return html`<div ref=${ref} class="popover ${cls}"
    style=${{ top: `${position.top}px`, left: `${position.left}px`, width: width ? `${width}px` : undefined }}>
    ${children}
  </div>`;
}

export const MenuItem = ({ icon, label, onClick, danger }) => html`
  <button type="button" class="menu-item ${danger ? 'danger' : ''}" onClick=${onClick}>
    ${icon && html`<${Icon} name=${icon} />`}<span>${label}</span>
  </button>`;

export function Modal({ title, onClose = closeModal, width = 760, class: cls = '', children }) {
  useEffect(() => {
    const onKey = (event) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);
  return html`<div class="modal-backdrop" onPointerDown=${(e) => e.target === e.currentTarget && onClose()}>
    <div class="modal ${cls}" role="dialog" aria-modal="true" aria-label=${title} style=${{ width: `${width}px` }}>
      <header class="modal-head">
        <h2>${title}</h2>
        <${IconButton} icon="x" tip="关闭" onClick=${onClose} />
      </header>
      ${children}
    </div>
  </div>`;
}

export function ConfirmHost() {
  const dialog = state.confirm;
  const okRef = useRef(null);
  useEffect(() => {
    okRef.current?.focus();
  }, [dialog]);
  if (!dialog) return null;
  const close = () => {
    state.confirm = null;
    notify();
  };
  const accept = () => {
    close();
    dialog.onConfirm();
  };
  return html`<${Modal} title=${dialog.title} width=${420} onClose=${close} class="confirm">
    <div class="confirm-body">${dialog.body}</div>
    <footer class="modal-foot">
      <button type="button" class="btn" onClick=${close}>取消</button>
      <button type="button" ref=${okRef} class="btn ${dialog.danger ? 'danger' : 'primary'}" onClick=${accept}>
        ${dialog.confirmLabel || '确定'}
      </button>
    </footer>
  <//>`;
}

export const askDelete = (title, body, onConfirm) =>
  confirmDialog({ title, body, confirmLabel: '删除', danger: true, onConfirm });

export const Toasts = () => html`<div class="toasts" aria-live="polite">
  ${state.toasts.map((t) => html`<div key=${t.id} class="toast ${t.kind}">
    ${t.kind === 'error' && html`<${Icon} name="alert" />`}
    ${t.kind === 'success' && html`<${Icon} name="check" />`}
    <span>${t.message}</span>
  </div>`)}
</div>`;

export const Toggle = ({ checked, onChange, label }) => html`
  <button type="button" role="switch" aria-checked=${checked} aria-label=${label}
    class="toggle ${checked ? 'on' : ''}" onClick=${() => onChange(!checked)}><span></span></button>`;

export const Segmented = ({ value, options, onChange }) => html`
  <div class="segmented" role="radiogroup">
    ${options.map(([key, label]) => html`
      <button type="button" key=${key} role="radio" aria-checked=${value === key}
        class=${value === key ? 'on' : ''} onClick=${() => onChange(key)}>${label}</button>`)}
  </div>`;

/** 全局提示层：任何带 data-tip 的元素悬停时显示，比原生 title 响应快。 */
export function installTooltips() {
  const tip = document.getElementById('tip');
  let current = null;
  const place = (x, y) => {
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    tip.style.left = `${Math.max(6, Math.min(x - w / 2, window.innerWidth - w - 6))}px`;
    tip.style.top = `${y - h - 10 < 6 ? y + 16 : y - h - 10}px`;
  };
  document.addEventListener('pointerover', (event) => {
    const element = event.target.closest?.('[data-tip]');
    if (element === current) return;
    current = element;
    const text = element?.getAttribute('data-tip');
    if (!text) {
      tip.classList.remove('show');
      return;
    }
    tip.textContent = text;
    tip.classList.add('show');
    const box = element.getBoundingClientRect();
    place(box.left + box.width / 2, box.top);
  });
  document.addEventListener('pointerdown', () => {
    current = null;
    tip.classList.remove('show');
  });
}

export function showTip(text, x, y) {
  const tip = document.getElementById('tip');
  tip.textContent = text;
  tip.classList.add('show');
  tip.style.left = `${Math.max(6, Math.min(x - tip.offsetWidth / 2, window.innerWidth - tip.offsetWidth - 6))}px`;
  tip.style.top = `${y - tip.offsetHeight - 12}px`;
}

export function hideTip() {
  document.getElementById('tip').classList.remove('show');
}
