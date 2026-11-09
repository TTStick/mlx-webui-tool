import { html, useEffect, useLayoutEffect, useMemo, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { compact, number, rate, seconds, splitThinking } from '../format.js';
import { renderMarkdown } from '../markdown.js';
import {
  clearSummary, continueMessage, currentModelId, deleteMessage, editMessage, modelInfo, notify, openModal,
  regenerate, setUi, state, switchBranch, toast,
} from '../store.js';
import { askDelete, Icon, IconButton, Modal, Popover, Spinner } from './ui.js';

const Markdown = ({ text }) => html`<div class="md" dangerouslySetInnerHTML=${{ __html: renderMarkdown(text) }}></div>`;

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    toast('复制失败，浏览器没有授予剪贴板权限', 'error');
    return false;
  }
}

function CopyButton({ text }) {
  const [done, setDone] = useState(false);
  return html`<${IconButton} icon=${done ? 'check' : 'copy'} tip="复制" onClick=${async () => {
    if (await copyText(text)) {
      setDone(true);
      setTimeout(() => setDone(false), 1400);
    }
  }} />`;
}

function Branches({ message }) {
  if (!message.siblings || message.siblings.length < 2) return null;
  const index = message.sibling_index;
  return html`<div class="branches">
    <${IconButton} icon="chevronLeft" tip="上一个版本" size=${14} disabled=${index === 0 || state.job}
      onClick=${() => switchBranch(message, -1)} />
    <span class="num">${index + 1} / ${message.siblings.length}</span>
    <${IconButton} icon="chevronRight" tip="下一个版本" size=${14} disabled=${index === message.siblings.length - 1 || state.job}
      onClick=${() => switchBranch(message, 1)} />
  </div>`;
}

function UserMessage({ message }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(message.content);
  const ref = useRef(null);

  useLayoutEffect(() => {
    if (!editing || !ref.current) return;
    ref.current.style.height = 'auto';
    ref.current.style.height = `${ref.current.scrollHeight + 2}px`;
  }, [editing, value]);

  const remove = () => askDelete('删除这条消息？', '这条消息和它之后的所有回复都会被删除。', () => deleteMessage(message));
  const submit = () => {
    setEditing(false);
    editMessage(message, value);
  };

  if (editing) {
    return html`<article class="msg user editing">
      <textarea ref=${(el) => { ref.current = el; }} value=${value} aria-label="编辑消息"
        onInput=${(e) => setValue(e.target.value)}
        onKeyDown=${(e) => {
          if (e.key === 'Escape') setEditing(false);
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && !e.isComposing) submit();
        }}></textarea>
      <div class="edit-actions">
        <span class="hint">保存后会基于新内容重新生成，原回复保留为另一个版本</span>
        <button type="button" class="btn" onClick=${() => setEditing(false)}>取消</button>
        <button type="button" class="btn primary" disabled=${!value.trim() || state.job} onClick=${submit}>保存并发送</button>
      </div>
    </article>`;
  }

  return html`<article class="msg user">
    ${message.attachments.length > 0 && html`<div class="msg-files">
      ${message.attachments.map((a) => html`<span class="file-chip" key=${a.id || a.name}><${Icon} name="file" size=${14} />${a.name}</span>`)}
    </div>`}
    ${message.content && html`<div class="bubble">${message.content}</div>`}
    <div class="msg-tools">
      ${message.tokens != null && state.settings.ui.show_stats && html`<span class="stat num">${number(message.tokens)} tokens</span>`}
      <${Branches} message=${message} />
      <${CopyButton} text=${message.content} />
      <${IconButton} icon="edit" tip="编辑" disabled=${state.job} onClick=${() => { setValue(message.content); setEditing(true); }} />
      <${IconButton} icon="trash" tip="删除" disabled=${state.job} onClick=${remove} />
    </div>
  </article>`;
}

function Thinking({ text, open: streaming }) {
  const [open, setOpen] = useState(null);
  const shown = open ?? streaming; // 用户没有手动切换时，思考中展开、结束后收起
  return html`<div class="think ${shown ? 'open' : ''}">
    <button type="button" class="think-head" onClick=${() => setOpen(!shown)} aria-expanded=${shown}>
      <${Icon} name="think" size=${14} />
      <span>${streaming ? '思考中…' : '思考过程'}</span>
      <span class="think-len num">${compact(text.length)} 字</span>
      <${Icon} name=${shown ? 'chevronDown' : 'chevronRight'} size=${14} />
    </button>
    ${shown && html`<div class="think-body"><${Markdown} text=${text} /></div>`}
  </div>`;
}

function statusLine(live) {
  switch (live.state) {
    case 'queued':
      return live.position > 0 ? `排队中，前面还有 ${live.position} 个任务` : '正在准备…';
    case 'loading':
      return '正在加载模型到显存…';
    case 'searching':
      return live.search?.query ? `正在搜索：${live.search.query}` : '正在想该搜什么…';
    case 'reading':
      return live.count > 0 ? `正在阅读 ${live.count} 个网页…` : '正在整理搜索结果…';
    case 'preparing':
      return '正在组装上下文…';
    case 'prompt':
      if (live.progress && live.progress.total > 0) {
        return `正在处理上下文 ${number(live.progress.done)} / ${number(live.progress.total)}`;
      }
      return live.cached > 0 ? `复用了 ${number(live.cached)} tokens 缓存…` : '正在处理上下文…';
    default:
      return '';
  }
}

function StatsLine({ stats }) {
  if (!stats || stats.completion_tokens == null || !state.settings.ui.show_stats) return null;
  const detail = [
    `Prompt ${number(stats.prompt_tokens)} tokens（缓存命中 ${number(stats.cached_tokens)}）`,
    `回复 ${number(stats.completion_tokens)} tokens`,
    `首字延迟 ${seconds(stats.ttft)}`,
    `总耗时 ${seconds(stats.duration)}`,
    `模型 ${stats.model}`,
  ].join('\n');
  return html`<span class="stat num" data-tip=${detail}>
    ${number(stats.completion_tokens)} tokens${stats.generation_tps != null ? ` · ${rate(stats.generation_tps)}` : ''} · 首字 ${seconds(stats.ttft)}
  </span>`;
}

function Sources({ search }) {
  const [open, setOpen] = useState(false);
  if (!search?.query) return null;
  if (search.error) {
    return html`<div class="msg-note warn"><${Icon} name="alert" size=${13} />联网搜索失败：${search.error}。这次回答没有用到网络信息。</div>`;
  }
  const sources = search.sources || [];
  if (!sources.length) return null;
  const domain = (url) => {
    try {
      return new URL(url).hostname.replace(/^www\./, '');
    } catch {
      return url;
    }
  };
  return html`<div class="sources">
    <button type="button" class="sources-head" aria-expanded=${open} onClick=${() => setOpen(!open)}>
      <${Icon} name="globe" size=${13} />
      <span>${sources.length} 个来源 · 检索词：${search.query}</span>
      <${Icon} name=${open ? 'chevronDown' : 'chevronRight'} size=${13} />
    </button>
    ${open && html`<ol>
      ${sources.map((source) => html`<li key=${source.url}>
        <a href=${source.url} target="_blank" rel="noopener noreferrer">${source.title}</a>
        <span>${domain(source.url)}</span>
      </li>`)}
    </ol>`}
  </div>`;
}

function MemoryChip({ ids }) {
  const [anchor, setAnchor] = useState(null);
  const [items, setItems] = useState(null);
  const show = async (event) => {
    const target = event.currentTarget;
    if (anchor) {
      setAnchor(null);
      return;
    }
    setAnchor(target);
    try {
      const { memories } = await api.get('/api/memory');
      setItems(memories.filter((m) => ids.includes(m.id)));
    } catch {
      setItems([]);
    }
  };
  return html`<button type="button" class="memory-chip" onClick=${show} data-tip="这次回答参考了哪些记忆">
    <${Icon} name="heart" size=${12} /><span class="num">${ids.length}</span>
  </button>
  ${anchor && html`<${Popover} anchor=${anchor} onClose=${() => setAnchor(null)} width=${320} placement="top-start">
    <div class="slash-head">这次用到的记忆</div>
    ${items === null && html`<div class="pop-empty"><${Spinner} /></div>`}
    ${items && !items.length && html`<div class="pop-empty">这些记忆已经被删除了</div>`}
    ${items && items.map((m) => html`<div class="memory-pop-item" key=${m.id}>${m.content}</div>`)}
    <button type="button" class="slash-manage" onClick=${() => { setAnchor(null); openModal('memory'); }}>管理记忆…</button>
  <//>`}`;
}

function AssistantMessage({ message, isLast }) {
  const streaming = message.status === 'streaming' || message.status === 'pending';
  const { thinking, answer, open } = useMemo(() => splitThinking(message.content), [message.content]);
  const reason = message.stats?.finish_reason;
  const canContinue = !streaming && message.content && (message.status === 'stopped' || message.status === 'interrupted' || reason === 'length');
  const remove = () => askDelete('删除这条回复？', '这条回复和它之后的所有消息都会被删除。', () => deleteMessage(message));
  const line = isLast && streaming ? statusLine(state.live) : '';
  const hasThinking = thinking != null && thinking.length > 0;

  const search = message.stats?.search ?? (isLast && streaming ? state.live.search : null);

  return html`<article class="msg assistant ${streaming ? 'streaming' : ''}">
    <${Sources} search=${search} />
    ${hasThinking && html`<${Thinking} text=${thinking} open=${open && streaming} />`}
    ${answer && html`<${Markdown} text=${answer} />`}
    ${streaming && !answer && !open && html`<div class="status-line">
      <span class="dots"><i></i><i></i><i></i></span>${line && html`<span>${line}</span>`}
    </div>`}

    ${message.status === 'error' && html`<div class="msg-error" role="alert">
      <${Icon} name="alert" />
      <div class="grow">
        <div>${message.error || '生成失败'}</div>
        <div class="msg-error-actions">
          <button type="button" class="btn small" onClick=${() => regenerate(message)}>重试</button>
          ${message.error_code === 'model_too_large' && html`
            <button type="button" class="btn small danger" onClick=${() => regenerate(message, true)}>仍然加载</button>`}
          ${message.error_code === 'model_missing' && html`
            <button type="button" class="btn small" onClick=${() => openModal('models')}>管理模型</button>`}
        </div>
      </div>
    </div>`}
    ${message.status === 'interrupted' && html`<div class="msg-note">服务重启，这条回复被中断了</div>`}
    ${message.status === 'stopped' && !streaming && html`<div class="msg-note">已手动停止</div>`}
    ${reason === 'length' && !streaming && html`<div class="msg-note">已达到长度上限，回复可能不完整</div>`}

    ${!streaming && html`<div class="msg-tools ${isLast ? 'always' : ''}">
      <${CopyButton} text=${answer || message.content} />
      <${IconButton} icon="refresh" tip="重新生成" disabled=${state.job} onClick=${() => regenerate(message)} />
      ${canContinue && html`<${IconButton} icon="play" tip="继续生成" disabled=${state.job} onClick=${() => continueMessage(message)} />`}
      <${IconButton} icon="trash" tip="删除" disabled=${state.job} onClick=${remove} />
      <${Branches} message=${message} />
      ${message.stats?.memory?.length > 0 && html`<${MemoryChip} ids=${message.stats.memory} />`}
      <${StatsLine} stats=${message.stats} />
    </div>`}
  </article>`;
}

function SummaryModal({ onClose }) {
  return html`<${Modal} title="压缩摘要" width=${620} onClose=${onClose}>
    <div class="modal-scroll pad"><${Markdown} text=${state.conversation.summary || ''} /></div>
  <//>`;
}

function Divider({ kind, count }) {
  const [viewing, setViewing] = useState(false);
  if (kind === 'summary') {
    return html`<div class="ctx-divider">
      <span>以上 ${count} 条消息已压缩成摘要发送</span>
      <button type="button" class="link-btn" onClick=${() => setViewing(true)}>查看摘要</button>
      <button type="button" class="link-btn" onClick=${clearSummary}>撤销</button>
      ${viewing && html`<${SummaryModal} onClose=${() => setViewing(false)} />`}
    </div>`;
  }
  return html`<div class="ctx-divider warn">
    <span>以上 ${count} 条消息超出上下文窗口，不会发送给模型</span>
    <button type="button" class="link-btn" onClick=${() => setUi({ inspector_open: true })}>查看上下文</button>
  </div>`;
}

const SUGGESTIONS = [
  ['解释一个概念', '用通俗的语言解释一下什么是 KV 缓存，以及它为什么占显存'],
  ['写一段代码', '用 Python 写一个带超时和重试的 HTTP 请求函数'],
  ['润色文字', '帮我把下面这段话改得更自然：'],
  ['制定计划', '帮我制定一个两周的 Rust 入门学习计划'],
];

function Welcome() {
  const hasModel = state.models.length > 0;
  const model = modelInfo(currentModelId());
  return html`<div class="welcome">
    <h1>${hasModel ? '有什么可以帮你？' : '先装一个模型吧'}</h1>
    ${hasModel ? html`
      <p class="welcome-sub">所有内容都在这台 Mac 上运行，不会发送到任何服务器${model ? `。当前模型 ${model.name}` : ''}</p>
      <div class="suggestions">
        ${SUGGESTIONS.map(([title, text]) => html`
          <button type="button" key=${title} onClick=${() => { state.draft.text = text; notify(); document.querySelector('.composer textarea')?.focus(); }}>
            <strong>${title}</strong><span>${text}</span>
          </button>`)}
      </div>` : html`
      <p class="welcome-sub">这台 Mac 上还没有可用的 MLX 模型。下载一个，或添加你已有的模型文件夹，全程在本地运行。</p>
      <div class="welcome-actions">
        <button type="button" class="btn primary large" onClick=${() => openModal('models', { tab: 'recommended' })}>
          <${Icon} name="download" />选择并下载模型
        </button>
        <button type="button" class="btn large" onClick=${() => openModal('models', { tab: 'local' })}>
          <${Icon} name="plus" />添加本地模型
        </button>
      </div>`}
  </div>`;
}

export function MessageList() {
  const ref = useRef(null);
  const stick = useRef(true);
  const [showJump, setShowJump] = useState(false);

  const onScroll = () => {
    const element = ref.current;
    const near = element.scrollHeight - element.scrollTop - element.clientHeight < 90;
    stick.current = near;
    if (showJump === near) setShowJump(!near);
  };
  const toBottom = () => {
    stick.current = true;
    ref.current.scrollTop = ref.current.scrollHeight;
    setShowJump(false);
  };

  useEffect(() => {
    stick.current = true;
  }, [state.currentId, state.job?.id]);

  useLayoutEffect(() => {
    if (stick.current && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  });

  const onClick = async (event) => {
    const button = event.target.closest('[data-copy]');
    if (!button) return;
    const code = button.closest('.code-block').querySelector('code').textContent;
    if (await copyText(code)) {
      button.textContent = '已复制';
      setTimeout(() => { button.textContent = '复制'; }, 1400);
    }
  };

  const messages = state.messages;
  const excluded = new Set(state.context?.excluded || []);
  const lastExcluded = messages.reduce((found, m, i) => (excluded.has(m.id) ? i : found), -1);
  const summaryIndex = state.conversation?.summary_upto
    ? messages.findIndex((m) => m.id === state.conversation.summary_upto) : -1;

  if (!messages.length && !state.loadingConversation) {
    return html`<div class="messages"><${Welcome} /></div>`;
  }

  return html`<div class="messages-wrap">
    <div class="messages" ref=${ref} onScroll=${onScroll} onClick=${onClick}>
      <div class="messages-inner">
        ${messages.map((message, index) => html`
          <div key=${message.id} class=${index <= lastExcluded || index <= summaryIndex ? 'dimmed' : ''}>
            ${message.role === 'user'
              ? html`<${UserMessage} message=${message} />`
              : html`<${AssistantMessage} message=${message} isLast=${index === messages.length - 1} />`}
            ${index === summaryIndex && html`<${Divider} kind="summary" count=${index + 1} />`}
            ${index === lastExcluded && index > summaryIndex && html`<${Divider} kind="excluded" count=${index - summaryIndex} />`}
          </div>`)}
      </div>
    </div>
    ${showJump && html`<button type="button" class="jump" aria-label="回到底部" onClick=${toBottom}><${Icon} name="arrowDown" /></button>`}
  </div>`;
}
