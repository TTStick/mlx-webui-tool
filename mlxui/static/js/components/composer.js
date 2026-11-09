import { html, useEffect, useLayoutEffect, useRef, useState } from '../lib.js';
import { compact, contextLabel, percent } from '../format.js';
import {
  addAttachments, currentModelId, effectiveParams, modelInfo, notify, openModal, removeAttachment,
  scheduleContext, send, setParam, setUi, state, stop,
} from '../store.js';
import { Icon, IconButton, Popover } from './ui.js';

function TokenRing({ used, limit }) {
  const share = percent(used, limit);
  const radius = 8;
  const circumference = 2 * Math.PI * radius;
  const level = share > 95 ? 'critical' : share > 80 ? 'warn' : '';
  return html`<svg class="ring ${level}" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
    <circle cx="10" cy="10" r=${radius} class="ring-track" />
    <circle cx="10" cy="10" r=${radius} class="ring-fill" transform="rotate(-90 10 10)"
      stroke-dasharray=${`${(share / 100) * circumference} ${circumference}`} />
  </svg>`;
}

const PROVIDER_NAMES = { wikipedia: '维基百科', searxng: 'SearXNG', brave: 'Brave Search', tavily: 'Tavily' };

export function Composer() {
  const draft = state.draft;
  const search = state.settings.search;
  const textarea = useRef(null);
  const fileInput = useRef(null);
  const [slashIndex, setSlashIndex] = useState(0);
  const [anchor, setAnchor] = useState(null);
  const params = effectiveParams();
  const model = modelInfo(currentModelId());
  const context = state.context;

  useLayoutEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, 240)}px`;
  }, [draft.text]);

  useEffect(() => {
    textarea.current?.focus();
  }, [state.currentId]);

  const slash = draft.text.startsWith('/') && !draft.text.includes('\n') ? draft.text.slice(1).toLowerCase() : null;
  const matches = slash === null ? [] : state.prompts.filter((p) => p.name.toLowerCase().includes(slash));
  const showSlash = matches.length > 0;
  const canSend = (draft.text.trim() || draft.attachments.length) && !state.job;

  useEffect(() => {
    setAnchor(showSlash ? textarea.current : null);
  }, [showSlash]);

  const applyPrompt = (prompt) => {
    draft.text = prompt.content;
    setSlashIndex(0);
    scheduleContext(0);
    notify();
    textarea.current?.focus();
  };

  const onKeyDown = (event) => {
    if (event.isComposing || event.keyCode === 229) return; // 中文输入法选词时的回车不算发送
    if (showSlash) {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        const step = event.key === 'ArrowDown' ? 1 : -1;
        setSlashIndex((slashIndex + step + matches.length) % matches.length);
        return;
      }
      if (event.key === 'Enter' || event.key === 'Tab') {
        event.preventDefault();
        applyPrompt(matches[Math.min(slashIndex, matches.length - 1)]);
        return;
      }
    }
    if (event.key !== 'Enter') return;
    const sendKey = state.settings.ui.send_key;
    const modified = event.metaKey || event.ctrlKey;
    const shouldSend = sendKey === 'enter' ? !event.shiftKey && !modified : modified;
    if (shouldSend) {
      event.preventDefault();
      if (canSend) send();
    }
  };

  const onInput = (event) => {
    draft.text = event.target.value;
    setSlashIndex(0);
    scheduleContext(250);
    notify();
  };

  const onPaste = (event) => {
    const files = [...(event.clipboardData?.files || [])];
    if (files.length) {
      event.preventDefault();
      addAttachments(files);
    }
  };

  const used = context?.total ?? 0;
  const limit = context?.limit ?? params.context_length;
  const overLimit = context?.over_limit;
  const excludedCount = context?.excluded?.length ?? 0;

  return html`<div class="composer-wrap">
    <div class="composer ${overLimit ? 'over' : ''}">
      ${draft.attachments.length > 0 && html`<div class="composer-files">
        ${draft.attachments.map((a) => html`<span class="file-chip" key=${a.id}>
          <${Icon} name="file" size=${14} /><span class="file-name">${a.name}</span>
          <button type="button" aria-label=${`移除 ${a.name}`} onClick=${() => removeAttachment(a.id)}><${Icon} name="x" size=${12} /></button>
        </span>`)}
      </div>`}
      <textarea ref=${textarea} rows="1" value=${draft.text} aria-label="输入消息"
        placeholder=${state.settings.ui.send_key === 'enter' ? '输入消息，Enter 发送，Shift+Enter 换行，/ 调用提示词' : '输入消息，⌘Enter 发送，/ 调用提示词'}
        onInput=${onInput} onKeyDown=${onKeyDown} onPaste=${onPaste}></textarea>
      <div class="composer-row">
        <${IconButton} icon="clip" tip="添加文件（文本、代码、PDF、Word）" onClick=${() => fileInput.current.click()} />
        <input ref=${fileInput} type="file" multiple hidden onChange=${(e) => { addAttachments([...e.target.files]); e.target.value = ''; }} />
        ${model?.thinking_toggle && html`
          <button type="button" class="chip-toggle ${params.thinking ? 'on' : ''}" aria-pressed=${!!params.thinking}
            data-tip="让模型先推理再回答，更慢但通常更准" onClick=${() => setParam('thinking', !params.thinking)}>
            <${Icon} name="think" size=${14} /><span>思考</span>
          </button>`}
        <button type="button" class="chip-toggle ${params.web_search ? 'on' : ''}" aria-pressed=${!!params.web_search}
          data-tip=${`联网搜索（${PROVIDER_NAMES[search.provider]}）\n开启后，模型写的检索词会发送给该服务，其余内容不会离开这台 Mac`}
          onClick=${() => setParam('web_search', !params.web_search)}>
          <${Icon} name="globe" size=${14} /><span>联网</span>
        </button>
        ${state.settings.memory.enabled && html`
          <button type="button" class="chip-toggle ${params.memory ? 'on' : ''}" aria-pressed=${!!params.memory}
            data-tip=${params.memory ? '这段对话会用到并更新长期记忆，点击关闭' : '这段对话不读取也不写入长期记忆'}
            onClick=${() => setParam('memory', !params.memory)}>
            <${Icon} name="heart" size=${14} /><span>记忆</span>
          </button>`}
        <div class="grow"></div>
        ${excludedCount > 0 && html`<span class="trim-note" data-tip="最早的消息放不进上下文窗口，发送时会被跳过">已截断 ${excludedCount} 条</span>`}
        <button type="button" class="token-meter" onClick=${() => setUi({ inspector_open: !state.settings.ui.inspector_open })}
          data-tip=${`上下文已用 ${used.toLocaleString()} / ${limit.toLocaleString()} tokens${context?.estimated ? '（估算）' : ''}`}>
          <${TokenRing} used=${used} limit=${limit} />
          <span class="num">${context?.estimated ? '≈' : ''}${compact(used)} / ${contextLabel(limit)}</span>
        </button>
        ${state.job
          ? html`<button type="button" class="send stop" aria-label="停止生成" data-tip="停止 Esc" onClick=${stop}><${Icon} name="stop" size=${14} /></button>`
          : html`<button type="button" class="send" aria-label="发送" disabled=${!canSend} onClick=${() => send()}><${Icon} name="send" size=${16} /></button>`}
      </div>
    </div>
    ${overLimit && html`<div class="composer-warn" role="alert">
      <${Icon} name="alert" size=${14} />这条消息放不进当前上下文窗口，请调大窗口或缩短内容
    </div>`}
    ${anchor && showSlash && html`<${Popover} anchor=${anchor} placement="top-start" width=${420} onClose=${() => setAnchor(null)}>
      <div class="slash-head">提示词</div>
      ${matches.map((prompt, index) => html`
        <button type="button" key=${prompt.id} class="slash-item ${index === slashIndex ? 'on' : ''}" onClick=${() => applyPrompt(prompt)}>
          <strong>${prompt.name}</strong><span>${prompt.content}</span>
        </button>`)}
      <button type="button" class="slash-manage" onClick=${() => { setAnchor(null); openModal('prompts'); }}>管理提示词…</button>
    <//>`}
  </div>`;
}
