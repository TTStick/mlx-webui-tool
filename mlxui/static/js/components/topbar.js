import { html, useState } from '../lib.js';
import { bytes, contextLabel, number } from '../format.js';
import {
  currentModelId, effectiveParams, exportConversation, kvBytesPerToken, loadedModelId, loadModel, modelInfo,
  openModal, renameConversation, selectModel, setParam, setUi, state,
} from '../store.js';
import { Icon, IconButton, MenuItem, Popover, Spinner } from './ui.js';

const CONTEXT_STEPS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];

function FitBadge({ fit }) {
  if (fit === 'too_large') return html`<span class="badge danger">超出显存</span>`;
  if (fit === 'tight') return html`<span class="badge warn">偏紧</span>`;
  return null;
}

function ModelPicker() {
  const [anchor, setAnchor] = useState(null);
  const id = currentModelId();
  const model = modelInfo(id);
  const loaded = loadedModelId();
  const engine = state.monitor?.engine;
  const loading = engine?.state === 'loading';
  const close = () => setAnchor(null);

  return html`<div class="picker">
    <button type="button" class="picker-btn" aria-haspopup="listbox" onClick=${(e) => setAnchor(anchor ? null : e.currentTarget)}>
      ${loading && loaded !== id ? html`<${Spinner} size=${12} />` : html`<span class="dot ${loaded === id && id ? 'on' : ''}"></span>`}
      <span class="picker-name">${model ? model.name : id ? `${id.split('/').pop()}（不在本地）` : '选择模型'}</span>
      <${Icon} name="chevronDown" size=${14} />
    </button>
    ${anchor && html`<${Popover} anchor=${anchor} onClose=${close} width=${380} class="model-pop">
      <div class="pop-scroll">
        ${state.models.map((item) => html`
          <div key=${item.id} class="model-row ${item.id === id ? 'on' : ''}">
            <button type="button" class="model-main" onClick=${() => { selectModel(item.id); close(); }}>
              <span class="model-title">
                <span class="dot ${item.id === loaded ? 'on' : ''}"></span>
                <span class="model-name">${item.name}</span>
                ${item.id === id && html`<${Icon} name="check" size=${14} />`}
              </span>
              <span class="model-meta num">${bytes(item.weights_bytes)} · ${item.quantization}${item.max_context ? ` · 最长 ${contextLabel(item.max_context)}` : ''}${item.source === 'local' ? ' · 本地文件夹' : ''}</span>
            </button>
            <${FitBadge} fit=${item.fit} />
            ${item.id !== loaded && html`<button type="button" class="link-btn" onClick=${() => { loadModel(item.id); close(); }}>预加载</button>`}
          </div>`)}
        ${!state.models.length && html`<div class="pop-empty">还没有本地模型</div>`}
      </div>
      <div class="menu-sep"></div>
      <${MenuItem} icon="plus" label="手动添加本地模型…" onClick=${() => { close(); openModal('models', { tab: 'local' }); }} />
      <${MenuItem} icon="chip" label="管理与下载模型…" onClick=${() => { close(); openModal('models'); }} />
    <//>`}
  </div>`;
}

function ContextPicker() {
  const [anchor, setAnchor] = useState(null);
  const params = effectiveParams();
  const model = modelInfo(currentModelId());
  const perToken = kvBytesPerToken(model);
  const budget = model ? state.workingSetLimit - model.weights_bytes : null;
  const maximum = model?.max_context || Infinity;
  const steps = CONTEXT_STEPS.filter((value) => value <= maximum);
  if (model?.max_context && !steps.includes(model.max_context) && model.max_context <= 2_000_000) {
    steps.push(model.max_context);
  }
  const close = () => setAnchor(null);
  const current = params.context_length;
  const measured = model && state.monitor?.engine?.kv_bytes_per_token?.[model.id];

  const row = (value) => {
    const cost = perToken ? perToken * value : null;
    const level = cost == null || budget == null ? '' : cost > budget ? 'danger' : cost > budget * 0.7 ? 'warn' : '';
    return html`<button type="button" key=${value} class="ctx-option ${value === current ? 'on' : ''}"
      onClick=${() => { setParam('context_length', value); close(); }}>
      <span class="ctx-size num">${contextLabel(value)}</span>
      <span class="ctx-tokens num">${number(value)} tokens</span>
      <span class="grow"></span>
      ${level === 'danger' && html`<span class="badge danger">超出显存</span>`}
      ${level === 'warn' && html`<span class="badge warn">偏紧</span>`}
      ${cost != null && html`<span class="ctx-cost num">${measured ? '' : '约 '}+${bytes(cost)}</span>`}
      ${value === current && html`<${Icon} name="check" size=${14} />`}
    </button>`;
  };

  return html`<div class="picker">
    <button type="button" class="picker-btn" aria-haspopup="listbox" onClick=${(e) => setAnchor(anchor ? null : e.currentTarget)}>
      <span class="picker-label">上下文</span>
      <span class="picker-name num">${contextLabel(current)}</span>
      <${Icon} name="chevronDown" size=${14} />
    </button>
    ${anchor && html`<${Popover} anchor=${anchor} onClose=${close} width=${360}>
      <div class="pop-note">上下文越大，能记住的内容越多，KV 缓存占用的显存也越多。右侧是窗口占满时的额外显存${measured ? '（按实测）' : '（按模型结构估算）'}。</div>
      <div class="pop-scroll">${steps.map(row)}</div>
      <div class="menu-sep"></div>
      <label class="ctx-custom">
        <span>自定义</span>
        <input type="number" min="256" step="256" value=${current} aria-label="自定义上下文长度"
          onChange=${(e) => {
            const value = Math.max(parseInt(e.target.value, 10) || 256, 256);
            setParam('context_length', Number.isFinite(maximum) ? Math.min(value, maximum) : value);
          }} />
        <span>tokens</span>
      </label>
    <//>`}
  </div>`;
}

function ConversationTitle() {
  const [editing, setEditing] = useState(false);
  const conversation = state.conversation;
  if (!conversation) return html`<div class="top-title"></div>`;
  if (editing) {
    return html`<div class="top-title">
      <input class="rename" ref=${(el) => el?.focus()} value=${conversation.title} aria-label="对话标题"
        onBlur=${(e) => {
          setEditing(false);
          const title = e.target.value.trim();
          if (title && title !== conversation.title) renameConversation(conversation.id, title);
        }}
        onKeyDown=${(e) => {
          if (e.key === 'Enter' && !e.isComposing) e.target.blur();
          if (e.key === 'Escape') setEditing(false);
        }} />
    </div>`;
  }
  return html`<div class="top-title">
    <button type="button" data-tip="点击重命名" onClick=${() => setEditing(true)}>${conversation.title}</button>
  </div>`;
}

export function Topbar() {
  const ui = state.settings.ui;
  const [menu, setMenu] = useState(null);
  const conversation = state.conversation;
  return html`<header class="topbar">
    ${!ui.sidebar_open && html`<${IconButton} icon="panelLeft" tip="展开侧栏 ⌘B" onClick=${() => setUi({ sidebar_open: true })} />`}
    <${ModelPicker} />
    <${ContextPicker} />
    <${ConversationTitle} />
    ${conversation && html`<${IconButton} icon="download" tip="导出对话" onClick=${(e) => setMenu(menu ? null : e.currentTarget)} />`}
    ${menu && html`<${Popover} anchor=${menu} onClose=${() => setMenu(null)} placement="bottom-end" width=${176}>
      <${MenuItem} icon="download" label="导出 Markdown" onClick=${() => { setMenu(null); exportConversation(conversation.id, 'md'); }} />
      <${MenuItem} icon="download" label="导出 JSON" onClick=${() => { setMenu(null); exportConversation(conversation.id, 'json'); }} />
    <//>`}
    <${IconButton} icon="panelRight" tip="上下文与显存 ⌘I" active=${ui.inspector_open} onClick=${() => setUi({ inspector_open: !ui.inspector_open })} />
  </header>`;
}
