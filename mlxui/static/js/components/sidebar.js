import { html, useRef, useState } from '../lib.js';
import { bytes, groupByDate, percent } from '../format.js';
import {
  bulkAction, busyConversationIds, exportConversation, newChat, notify, openConversation,
  openModal, removeConversation, renameConversation, setArchived, setPinned, setUi, state,
} from '../store.js';
import { askDelete, Icon, IconButton, MenuItem, Popover, Spinner } from './ui.js';

function Row({ item, busy }) {
  const [menu, setMenu] = useState(null);
  const [renaming, setRenaming] = useState(false);
  const inputRef = useRef(null);
  const active = item.id === state.currentId;
  const checked = state.selected.has(item.id);

  const close = () => setMenu(null);
  const commit = (value) => {
    setRenaming(false);
    const title = value.trim();
    if (title && title !== item.title) renameConversation(item.id, title);
  };
  const click = () => {
    if (state.selecting) {
      if (checked) state.selected.delete(item.id);
      else state.selected.add(item.id);
      notify();
    } else {
      openConversation(item.id);
      if (window.innerWidth < 760) setUi({ sidebar_open: false });
    }
  };

  if (renaming) {
    return html`<div class="conv-row active">
      <input ref=${(el) => { inputRef.current = el; el?.focus(); }} class="rename" value=${item.title} aria-label="对话标题"
        onBlur=${(e) => commit(e.target.value)}
        onKeyDown=${(e) => {
          if (e.key === 'Enter' && !e.isComposing) e.target.blur();
          if (e.key === 'Escape') setRenaming(false);
        }} />
    </div>`;
  }

  return html`<div class="conv-row ${active ? 'active' : ''} ${menu ? 'menu-open' : ''}">
    <button type="button" class="conv-main" onClick=${click} title=${item.title}>
      ${state.selecting && html`<span class="check ${checked ? 'on' : ''}">${checked && html`<${Icon} name="check" size=${12} />`}</span>`}
      ${item.pinned && !state.selecting && html`<span class="conv-pin"><${Icon} name="pin" size=${12} /></span>`}
      <span class="conv-title">${item.title}</span>
      ${busy && html`<${Spinner} size=${12} />`}
    </button>
    ${!state.selecting && html`<button type="button" class="conv-more" aria-label="更多操作"
      onClick=${(e) => setMenu(menu ? null : e.currentTarget)}><${Icon} name="more" /></button>`}
    ${menu && html`<${Popover} anchor=${menu} onClose=${close} width=${176}>
      <${MenuItem} icon="edit" label="重命名" onClick=${() => { close(); setRenaming(true); }} />
      <${MenuItem} icon="pin" label=${item.pinned ? '取消置顶' : '置顶'} onClick=${() => { close(); setPinned(item.id, !item.pinned); }} />
      <${MenuItem} icon="archive" label="归档" onClick=${() => { close(); setArchived(item.id, true); }} />
      <div class="menu-sep"></div>
      <${MenuItem} icon="download" label="导出 Markdown" onClick=${() => { close(); exportConversation(item.id, 'md'); }} />
      <${MenuItem} icon="download" label="导出 JSON" onClick=${() => { close(); exportConversation(item.id, 'json'); }} />
      <div class="menu-sep"></div>
      <${MenuItem} icon="trash" label="删除" danger=${true} onClick=${() => {
        close();
        askDelete('彻底删除这个对话？', `“${item.title}”和由它产生的记忆都会被清除，无法恢复。`, () => removeConversation(item.id));
      }} />
    <//>`}
  </div>`;
}

function SelectBar() {
  const ids = [...state.selected];
  const finish = () => {
    state.selecting = false;
    state.selected = new Set();
    notify();
  };
  const run = (action) => bulkAction(ids, action).then(finish);
  return html`<div class="select-bar">
    <span>已选 ${ids.length} 项</span>
    <div class="grow"></div>
    <${IconButton} icon="pin" tip="置顶" disabled=${!ids.length} onClick=${() => run('pin')} />
    <${IconButton} icon="archive" tip="归档" disabled=${!ids.length} onClick=${() => run('archive')} />
    <${IconButton} icon="trash" tip="删除" disabled=${!ids.length} onClick=${() =>
      askDelete(`彻底删除 ${ids.length} 个对话？`, '选中的对话和由它们产生的记忆都会被清除，无法恢复。', () => run('delete'))} />
    <${IconButton} icon="x" tip="取消选择" onClick=${finish} />
  </div>`;
}

function MemoryFooter() {
  const monitor = state.monitor;
  if (!monitor) return null;
  const active = monitor.mlx.active;
  const limit = monitor.mlx.working_set_limit;
  const share = percent(active, limit);
  return html`<button type="button" class="mem-foot" onClick=${() => setUi({ inspector_open: true })}
    data-tip="MLX 当前占用 / 系统建议的显存上限">
    <div class="mem-foot-row">
      <span>显存</span>
      <span class="num">${bytes(active)} / ${bytes(limit)}</span>
    </div>
    <div class="meter"><div style=${{ width: `${share}%` }} class=${share > 90 ? 'hot' : ''}></div></div>
  </button>`;
}

export function Sidebar() {
  const groups = groupByDate(state.conversations);
  const busy = busyConversationIds();
  return html`<aside class="sidebar">
    <div class="side-head">
      <div class="brand"><span class="brand-mark">M</span><span>MLX UI</span></div>
      <${IconButton} icon="panelLeft" tip="收起侧栏 ⌘B" onClick=${() => setUi({ sidebar_open: false })} />
    </div>

    <div class="side-actions">
      <button type="button" class="new-chat" onClick=${newChat}>
        <${Icon} name="plus" /><span>新对话</span><kbd>⌘⇧O</kbd>
      </button>
      <button type="button" class="side-search" onClick=${() => openModal('search')}>
        <${Icon} name="search" /><span>搜索对话</span><kbd>⌘K</kbd>
      </button>
    </div>

    ${state.selecting && html`<${SelectBar} />`}

    <nav class="conv-list" aria-label="对话列表">
      ${groups.map((group) => html`<section key=${group.key}>
        <div class="group-label">
          <span>${group.label}</span>
          ${group === groups[0] && !state.selecting && html`
            <button type="button" class="link-btn" onClick=${() => { state.selecting = true; notify(); }}>选择</button>`}
        </div>
        ${group.items.map((item) => html`<${Row} key=${item.id} item=${item} busy=${busy.has(item.id)} />`)}
      </section>`)}
      ${!groups.length && html`<div class="side-empty">还没有对话。<br/>发出第一条消息后会出现在这里。</div>`}
    </nav>

    <div class="side-foot">
      <${MemoryFooter} />
      <div class="foot-nav">
        <button type="button" onClick=${() => openModal('models')}><${Icon} name="chip" /><span>模型</span></button>
        <button type="button" class="nav-memory" onClick=${() => openModal('memory')}>
          <${Icon} name="heart" /><span>记忆</span>
          ${state.memoryStatus?.unseen > 0 && html`<i class="nav-badge" aria-label=${`${state.memoryStatus.unseen} 条新记忆`}>${state.memoryStatus.unseen}</i>`}
        </button>
        <button type="button" onClick=${() => openModal('archive')}><${Icon} name="archive" /><span>归档</span></button>
        <button type="button" onClick=${() => openModal('prompts')}><${Icon} name="book" /><span>提示词</span></button>
        <button type="button" onClick=${() => openModal('settings')}><${Icon} name="sliders" /><span>设置</span></button>
      </div>
    </div>
  </aside>`;
}
