import { html, render, useEffect, useState } from './lib.js';
import { addAttachments, boot, newChat, openModal, setUi, state, stop, subscribe } from './store.js';
import { Sidebar } from './components/sidebar.js';
import { MessageList } from './components/chat.js';
import { Composer } from './components/composer.js';
import { Topbar } from './components/topbar.js';
import { Inspector } from './components/inspector.js';
import { ModalHost } from './components/modals.js';
import { ConfirmHost, Icon, installTooltips, Toasts } from './components/ui.js';
import { VaultScreen } from './components/vault.js';

function useShortcuts() {
  useEffect(() => {
    const onKey = (event) => {
      const command = event.metaKey || event.ctrlKey;
      const key = event.key.toLowerCase();
      if (command && key === 'k') {
        event.preventDefault();
        if (state.modal?.name === 'search') return;
        openModal('search');
      } else if (command && event.shiftKey && key === 'o') {
        event.preventDefault();
        newChat();
        document.querySelector('.composer textarea')?.focus();
      } else if (command && key === 'b') {
        event.preventDefault();
        setUi({ sidebar_open: !state.settings.ui.sidebar_open });
      } else if (command && key === 'i') {
        event.preventDefault();
        setUi({ inspector_open: !state.settings.ui.inspector_open });
      } else if (event.key === 'Escape' && state.job && !state.modal && !state.confirm) {
        stop();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}

function Main() {
  const [dragging, setDragging] = useState(false);
  return html`<main class="main"
    onDragOver=${(e) => { if (e.dataTransfer?.types?.includes('Files')) { e.preventDefault(); setDragging(true); } }}
    onDragLeave=${(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false); }}
    onDrop=${(e) => { e.preventDefault(); setDragging(false); addAttachments([...e.dataTransfer.files]); }}>
    <${Topbar} />
    <${MessageList} />
    <${Composer} />
    ${dragging && html`<div class="drop-hint"><${Icon} name="clip" size=${28} /><span>松开以添加文件</span></div>`}
  </main>`;
}

function App() {
  const [, setVersion] = useState(0);
  useEffect(() => subscribe(() => setVersion((v) => v + 1)), []);
  useShortcuts();

  if (state.vault && state.vault.state !== 'unlocked') return html`<${VaultScreen} />`;
  if (!state.ready) {
    return html`<div class="boot">
      ${state.offline
        ? html`<p>无法连接本地服务。请确认 MLX UI 正在运行，然后刷新页面。</p>`
        : html`<p>正在启动…</p>`}
    </div>`;
  }
  const ui = state.settings.ui;
  return html`<div class="app">
    ${state.offline && html`<div class="offline" role="alert">与本地服务的连接已断开，正在重试…</div>`}
    ${ui.sidebar_open && html`<${Sidebar} />`}
    ${ui.sidebar_open && html`<div class="scrim side" onClick=${() => setUi({ sidebar_open: false })}></div>`}
    <${Main} />
    ${ui.inspector_open && html`<div class="scrim ins" onClick=${() => setUi({ inspector_open: false })}></div>`}
    ${ui.inspector_open && html`<${Inspector} />`}
    <${ModalHost} />
    <${ConfirmHost} />
    <${Toasts} />
  </div>`;
}

installTooltips();
render(html`<${App} />`, document.getElementById('app'));
boot();
