import { html, useEffect, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { bytes, compact, contextLabel, dateTime, number, percent, seconds } from '../format.js';
import {
  addLocalFolder, bulkAction, closeModal, confirmDialog, currentModelId, loadedModelId, loadModel, lockNow,
  openConversation, pickFolder, refreshConversations, refreshModels, refreshPrompts, refreshVault, removeConversation,
  removeLocalFolder, selectModel, setArchived, setUi, state, toast, trackDownloads, unloadModel, updateSettings,
  wipeEverything,
} from '../store.js';
import { askDelete, Icon, IconButton, Modal, Segmented, Spinner, Toggle } from './ui.js';

// ---------------------------------------------------------------- 模型管理
async function startDownload(repo) {
  try {
    await api.post('/api/downloads', { repo });
    const { tasks } = await api.get('/api/downloads');
    trackDownloads(tasks);
    toast('已开始下载', 'success');
  } catch (error) {
    toast(error.message, 'error', 7000);
  }
}

function Downloads() {
  const tasks = state.downloads;
  if (!tasks.length) return null;
  const cancel = (task) => api.post(`/api/downloads/${task.id}/cancel`).catch((e) => toast(e.message, 'error'));
  const dismiss = (task) => api.del(`/api/downloads/${task.id}`).catch((e) => toast(e.message, 'error'));
  const labels = { done: '已完成', failed: '失败', cancelled: '已取消' };
  return html`<div class="downloads">
    ${tasks.map((task) => {
      const share = percent(task.done, task.total);
      return html`<div class="download" key=${task.id}>
        <div class="download-head">
          <strong>${task.repo.split('/').pop()}</strong>
          <span class="num">
            ${task.status === 'running'
              ? `${bytes(task.done)} / ${bytes(task.total)} · ${bytes(task.speed)}/s`
              : labels[task.status]}
          </span>
          ${task.status === 'failed' && html`<button type="button" class="link-btn" onClick=${() => startDownload(task.repo)}>重试</button>`}
          ${task.status === 'running'
            ? html`<${IconButton} icon="x" tip="取消并删除已下载部分" onClick=${() => cancel(task)} />`
            : html`<${IconButton} icon="x" tip="移除记录" onClick=${() => dismiss(task)} />`}
        </div>
        ${task.status === 'running' && html`<div class="meter"><div style=${{ width: `${share}%` }}></div></div>`}
        ${task.error && html`<div class="download-error">${task.error}</div>`}
      </div>`;
    })}
  </div>`;
}

function FitLabel({ fit }) {
  if (fit === 'too_large') return html`<span class="badge danger">超出显存上限</span>`;
  if (fit === 'tight') return html`<span class="badge warn">偏紧，上下文要小</span>`;
  return html`<span class="badge good">适合这台 Mac</span>`;
}

function Installed() {
  const loaded = loadedModelId();
  if (!state.models.length) return html`<div class="empty">还没有模型。可以在“推荐”里下载，或在“手动添加”里选择已有的模型文件夹。</div>`;
  const remove = (model) => askDelete(
    '从磁盘删除这个模型？',
    `将释放 ${bytes(model.size_bytes)} 磁盘空间。之后需要重新下载才能使用。`,
    async () => {
      try {
        await api.post('/api/models/delete', { id: model.id });
        await refreshModels();
        toast('模型已删除', 'success');
      } catch (error) {
        toast(error.message, 'error');
      }
    },
  );
  return html`<div class="model-list">
    ${state.models.map((model) => html`<div class="model-card" key=${model.id}>
      <div class="grow">
        <div class="model-card-title">
          <span class="dot ${model.id === loaded ? 'on' : ''}"></span>
          <strong>${model.name}</strong>
          ${model.id === state.settings.default_model && html`<span class="badge">默认</span>`}
          ${model.fit !== 'ok' && html`<${FitLabel} fit=${model.fit} />`}
        </div>
        <div class="model-card-meta num">
          ${bytes(model.size_bytes)} · ${model.quantization} · ${model.model_type}${model.max_context ? ` · 最长 ${contextLabel(model.max_context)}` : ''}${model.reasoning ? ' · 支持思考' : ''}
        </div>
        <div class="model-card-path" title=${model.path}>${model.source === 'local' ? model.path : model.id}</div>
      </div>
      <div class="model-card-actions">
        ${model.id === loaded
          ? html`<button type="button" class="btn small" onClick=${unloadModel}>卸载</button>`
          : html`<button type="button" class="btn small" onClick=${() => loadModel(model.id)}>加载</button>`}
        <button type="button" class="btn small" onClick=${() => { selectModel(model.id); toast(`已切换到 ${model.name}`, 'success'); }}>使用</button>
        ${model.deletable && html`<${IconButton} icon="trash" tip="从磁盘删除" onClick=${() => remove(model)} />`}
      </div>
    </div>`)}
  </div>`;
}

function RemoteCard({ model }) {
  const downloading = state.downloads.some((t) => t.repo === model.id && t.status === 'running');
  const blocked = !model.disk_ok;
  return html`<div class="model-card">
    <div class="grow">
      <div class="model-card-title">
        <strong>${model.name}</strong>
        <${FitLabel} fit=${model.fit} />
      </div>
      <div class="model-card-meta num">${bytes(model.size_bytes, 2)} · ${model.model_type || '未知架构'}${model.downloads ? ` · ${compact(model.downloads)} 次下载` : ''}</div>
      ${model.note && html`<div class="model-card-note">${model.note}</div>`}
      ${blocked && html`<div class="model-card-warn"><${Icon} name="alert" size=${13} />磁盘空间不足</div>`}
    </div>
    <div class="model-card-actions">
      ${model.installed
        ? html`<span class="badge good">已安装</span>`
        : downloading
          ? html`<span class="muted small">下载中</span>`
          : html`<button type="button" class="btn small primary" disabled=${blocked} onClick=${() => startDownload(model.id)}>
              <${Icon} name="download" size=${14} />下载
            </button>`}
    </div>
  </div>`;
}

function Recommended() {
  const [items, setItems] = useState(null);
  useEffect(() => {
    api.get('/api/models/recommended').then((r) => setItems(r.models)).catch((e) => { setItems([]); toast(e.message, 'error'); });
  }, [state.models.length, state.disk?.free]);
  if (items === null) return html`<div class="empty"><${Spinner} /></div>`;
  return html`<div class="model-list">${items.map((m) => html`<${RemoteCard} key=${m.id} model=${m} />`)}</div>`;
}

function Search() {
  const [query, setQuery] = useState('');
  const [items, setItems] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const run = async () => {
    if (!query.trim()) return;
    setBusy(true);
    setError('');
    try {
      setItems((await api.get(`/api/models/search?q=${encodeURIComponent(query.trim())}`)).models);
    } catch (e) {
      setError(e.message);
      setItems(null);
    } finally {
      setBusy(false);
    }
  };
  const isRepo = /^[\w.-]+\/[\w.-]+$/.test(query.trim());
  return html`<div>
    <form class="search-row" onSubmit=${(e) => { e.preventDefault(); run(); }}>
      <input class="field" placeholder="搜索 Hugging Face 上的 MLX 模型，例如 qwen3" aria-label="搜索模型" value=${query}
        ref=${(el) => el?.focus()} onInput=${(e) => setQuery(e.target.value)} />
      <button type="submit" class="btn primary" disabled=${busy}>${busy ? html`<${Spinner} size=${14} />` : '搜索'}</button>
      ${isRepo && html`<button type="button" class="btn" onClick=${() => startDownload(query.trim())}>直接下载此仓库</button>`}
    </form>
    ${error && html`<div class="form-error">${error}</div>`}
    ${items && !items.length && html`<div class="empty">没有找到 mlx-lm 能运行的模型</div>`}
    ${items && html`<div class="model-list">${items.map((m) => html`<${RemoteCard} key=${m.id} model=${m} />`)}</div>`}
  </div>`;
}

function FoundModel({ model }) {
  const selected = model.id === currentModelId();
  return html`<div class="model-card">
    <div class="grow">
      <div class="model-card-title">
        <strong>${model.name}</strong>
        ${model.fit !== 'ok' && html`<${FitLabel} fit=${model.fit} />`}
      </div>
      <div class="model-card-meta num">${bytes(model.size_bytes)} · ${model.quantization} · ${model.model_type}${model.max_context ? ` · 最长 ${contextLabel(model.max_context)}` : ''}${model.reasoning ? ' · 支持思考' : ''}</div>
      <div class="model-card-path" title=${model.path}>${model.path}</div>
    </div>
    <div class="model-card-actions">
      ${selected
        ? html`<span class="badge good">当前使用</span>`
        : html`<button type="button" class="btn small primary" onClick=${() => { selectModel(model.id); toast(`已切换到 ${model.name}`, 'success'); }}>使用</button>`}
    </div>
  </div>`;
}

function LocalModels() {
  const [path, setPath] = useState('');
  const [busy, setBusy] = useState('');
  const [result, setResult] = useState(null);

  const add = async (value) => {
    setBusy('add');
    try {
      const data = await addLocalFolder(value);
      setResult(data);
      // 只有一个模型时直接用它；一个都没选时先选第一个，避免发消息时报“没有模型”
      if (data.models.length === 1 || (data.models.length > 1 && !currentModelId())) {
        await selectModel(data.models[0].id);
        toast(`已切换到 ${data.models[0].name}`, 'success');
      }
      if (data.models.length) setPath('');
    } catch (error) {
      toast(error.message, 'error', 6000);
    } finally {
      setBusy('');
    }
  };

  const choose = async () => {
    setBusy('pick');
    try {
      const picked = await pickFolder();
      if (picked) {
        setPath(picked);
        await add(picked);
      }
    } catch (error) {
      toast(error.message, 'error', 6000);
    } finally {
      setBusy('');
    }
  };

  const remove = (row) => confirmDialog({
    title: '从列表移除这个文件夹？',
    body: `里面的 ${row.count} 个模型不再显示，磁盘上的文件不会被删除。`,
    confirmLabel: '移除',
    onConfirm: () => removeLocalFolder(row.path),
  });

  const models = result?.models ?? [];
  const problems = result?.problems ?? [];

  return html`<div class="local">
    <p class="local-intro">选择模型所在的文件夹，或存放多个模型的上级文件夹。模型留在原处，不会被移动或复制。</p>
    <button type="button" class="btn primary large" disabled=${!!busy} onClick=${choose}>
      ${busy === 'pick' ? html`<${Spinner} size=${14} />` : html`<${Icon} name="file" />`}
      ${busy === 'pick' ? '请在弹出的窗口中选择…' : '选择文件夹…'}
    </button>
    ${busy === 'pick' && html`<p class="local-hint">没看到窗口的话，它可能被浏览器挡住了。</p>`}

    <form class="search-row local-path" onSubmit=${(e) => { e.preventDefault(); if (path.trim()) add(path); }}>
      <input class="field" placeholder="或粘贴路径，例如 ~/models/Qwen3-4B-4bit" aria-label="模型文件夹路径"
        value=${path} onInput=${(e) => setPath(e.target.value)} />
      <button type="submit" class="btn" disabled=${!path.trim() || !!busy}>${busy === 'add' ? html`<${Spinner} size=${14} />` : '添加'}</button>
    </form>

    ${models.length > 0 && html`<section>
      <h3 class="section-label">
        找到 ${models.length} 个模型${result.added === 0 ? '（之前已添加）' : ''}
      </h3>
      <div class="model-list">${models.map((m) => html`<${FoundModel} key=${m.id} model=${m} />`)}</div>
    </section>`}

    ${problems.length > 0 && html`<section role="alert">
      <h3 class="section-label">${models.length ? '以下文件夹被跳过' : '无法添加'}</h3>
      ${problems.map((p) => html`<div class="problem" key=${p.path}>
        <${Icon} name="alert" />
        <div class="grow"><div>${p.message}</div><div class="problem-path">${p.path}</div></div>
      </div>`)}
    </section>`}

    <section>
      <h3 class="section-label">已添加的文件夹</h3>
      ${state.localDirs.map((row) => html`<div class="folder-row" key=${row.path}>
        <div class="grow">
          <div class="folder-path" title=${row.path}>${row.path}</div>
          <div class="muted small num">
            ${row.exists ? `${row.count} 个模型` : '找不到这个文件夹，可能是外接硬盘没有连接'}${row.removable ? '' : ' · LM Studio（自动发现）'}
          </div>
        </div>
        ${row.removable && html`<${IconButton} icon="x" tip="从列表移除（不删除文件）" onClick=${() => remove(row)} />`}
      </div>`)}
      ${!state.localDirs.length && html`<p class="muted small">还没有添加过文件夹</p>`}
    </section>

    <p class="local-hint">
      文件夹里需要有 config.json、.safetensors 权重和分词器文件，最好是 MLX 版本（例如 mlx-community 上的模型）。不支持 GGUF。
    </p>
  </div>`;
}

function ModelsModal() {
  const [tab, setTab] = useState(state.modal.tab || (state.models.length ? 'installed' : 'recommended'));
  useEffect(() => { refreshModels(); }, []);
  const disk = state.disk;
  return html`<${Modal} title="模型" width=${780} class="tall">
    <div class="tabbar">
      <${Segmented} value=${tab} onChange=${setTab}
        options=${[['installed', `已安装 ${state.models.length}`], ['recommended', '推荐'], ['search', '搜索'], ['local', '手动添加']]} />
      <div class="grow"></div>
      ${disk && html`<span class="muted small num">磁盘剩余 ${bytes(disk.free)} · 显存上限 ${bytes(state.workingSetLimit)}</span>`}
    </div>
    <div class="modal-scroll pad">
      <${Downloads} />
      ${tab === 'installed' && html`<${Installed} />`}
      ${tab === 'recommended' && html`<${Recommended} />`}
      ${tab === 'search' && html`<${Search} />`}
      ${tab === 'local' && html`<${LocalModels} />`}
    </div>
  <//>`;
}

// ------------------------------------------------------------------ 记忆
const KIND_LABELS = { profile: '关于我', preference: '偏好', person: '人物', event: '近况', style: '聊天方式', insight: '反思' };
const MEMORY_TABS = [['active', '使用中'], ['pending', '待确认'], ['archived', '已归档'], ['expired', '已过期']];

function ImportanceDots({ value }) {
  return html`<span class="dots-row" role="img" aria-label=${`重要度 ${value}`} data-tip=${`重要度 ${value} / 5`}>
    ${[1, 2, 3, 4, 5].map((n) => html`<i key=${n} class=${n <= value ? 'on' : ''}></i>`)}
  </span>`;
}

function MemoryEditor({ initial, onSave, onCancel }) {
  const [content, setContent] = useState(initial.content);
  const [kind, setKind] = useState(initial.kind);
  const [importance, setImportance] = useState(initial.importance);
  return html`<div class="memory-edit">
    <textarea class="field" rows="2" aria-label="记忆内容" ref=${(el) => el?.focus()} value=${content}
      onInput=${(e) => setContent(e.target.value)}></textarea>
    <div class="row-end">
      <select class="field short" aria-label="类别" value=${kind} onChange=${(e) => setKind(e.target.value)}>
        ${Object.entries(KIND_LABELS).map(([key, label]) => html`<option key=${key} value=${key}>${label}</option>`)}
      </select>
      <select class="field short" aria-label="重要度" value=${importance} onChange=${(e) => setImportance(+e.target.value)}>
        ${[1, 2, 3, 4, 5].map((n) => html`<option key=${n} value=${n}>重要度 ${n}</option>`)}
      </select>
      <div class="grow"></div>
      <button type="button" class="btn small" onClick=${onCancel}>取消</button>
      <button type="button" class="btn small primary" disabled=${!content.trim()}
        onClick=${() => onSave({ content, kind, importance })}>保存</button>
    </div>
  </div>`;
}

function MemoryRow({ item, reload }) {
  const [editing, setEditing] = useState(false);
  const act = async (fn) => {
    try {
      await fn();
      await reload();
    } catch (error) {
      toast(error.message, 'error');
    }
  };
  const patch = (body) => act(() => api.patch(`/api/memory/${item.id}`, body));
  const bulk = (action) => act(() => api.post('/api/memory/bulk', { ids: [item.id], action }));
  if (editing) {
    return html`<${MemoryEditor} initial=${item} onCancel=${() => setEditing(false)}
      onSave=${(body) => { setEditing(false); patch(body); }} />`;
  }
  const expires = item.expires_at ? `到期 ${dateTime(item.expires_at).slice(5, 10)}` : '';
  return html`<div class="memory-row ${item.seen ? '' : 'fresh'}">
    <div class="grow">
      <div class="memory-text">${item.content}</div>
      <div class="memory-meta">
        <span class="badge">${KIND_LABELS[item.kind] || item.kind}</span>
        <${ImportanceDots} value=${item.importance} />
        ${item.pinned && html`<span>已钉住</span>`}
        ${item.origin === 'manual' ? html`<span>手动添加</span>` : item.sources.slice(0, 2).map((source) => html`
          <button type="button" class="link-btn" key=${source.conversation_id} title="打开来源对话"
            onClick=${() => { closeModal(); openConversation(source.conversation_id); }}>${source.title}</button>`)}
        ${item.use_count > 0 && html`<span class="num">用过 ${item.use_count} 次</span>`}
        ${expires && html`<span class="num">${expires}</span>`}
      </div>
    </div>
    <div class="memory-actions">
      ${item.status === 'pending' && html`<button type="button" class="btn small primary" onClick=${() => bulk('approve')}>记住</button>`}
      ${item.status === 'active' && html`<${IconButton} icon="pin" tip=${item.pinned ? '取消钉住' : '钉住：总是用到，模型也不能修改'} active=${item.pinned} onClick=${() => patch({ pinned: !item.pinned })} />`}
      <${IconButton} icon="edit" tip="编辑" onClick=${() => setEditing(true)} />
      ${item.status === 'active'
        ? html`<${IconButton} icon="archive" tip="归档：暂时不用" onClick=${() => bulk('archive')} />`
        : item.status !== 'pending' && html`<${IconButton} icon="restore" tip="恢复使用" onClick=${() => bulk('restore')} />`}
      <${IconButton} icon="trash" tip="彻底删除" onClick=${() => act(() => api.del(`/api/memory/${item.id}`))} />
    </div>
  </div>`;
}

function MemoryOptions() {
  const memory = state.settings.memory;
  return html`<div class="memory-options">
    <label class="inline"><${Toggle} checked=${memory.enabled} label="启用长期记忆" onChange=${(enabled) => updateSettings({ memory: { enabled } })} /><span>启用记忆</span></label>
    <label class="inline" data-tip="开启后，模型提炼的内容先进入“待确认”，你同意后才会用到"><${Toggle} checked=${memory.review} label="新记忆需要我确认" onChange=${(review) => updateSettings({ memory: { review } })} /><span>新记忆需要确认</span></label>
    <label class="inline" data-tip="对话停下来多久之后开始反思"><span>空闲</span><input class="field tiny num" type="number" min="10" aria-label="开始反思前的空闲秒数" value=${memory.idle_seconds} onChange=${(e) => updateSettings({ memory: { idle_seconds: +e.target.value } })} /><span>秒后反思</span></label>
    <label class="inline" data-tip="每轮最多放进上下文的记忆量，不超过上下文窗口的五分之一"><span>最多占</span><input class="field tiny num" type="number" min="50" step="50" aria-label="记忆占用的 token 上限" value=${memory.budget_tokens} onChange=${(e) => updateSettings({ memory: { budget_tokens: +e.target.value } })} /><span>tokens</span></label>
  </div>`;
}

function MemoryModal() {
  const [status, setStatus] = useState('active');
  const [kind, setKind] = useState('');
  const [query, setQuery] = useState('');
  const [data, setData] = useState(null);
  const [adding, setAdding] = useState(false);
  const [reflecting, setReflecting] = useState(false);
  const version = state.memoryStatus?.version;

  const reload = async () => {
    const params = new URLSearchParams({ status, kind, q: query });
    try {
      setData(await api.get(`/api/memory?${params}`));
    } catch (error) {
      toast(error.message, 'error');
    }
  };

  useEffect(() => { reload(); }, [status, kind, query, version]);
  useEffect(() => { api.post('/api/memory/seen').catch(() => {}); }, []);

  const add = async (body) => {
    setAdding(false);
    try {
      await api.post('/api/memory', body);
      await reload();
    } catch (error) {
      toast(error.message, 'error');
    }
  };
  const reflect = async () => {
    setReflecting(true);
    try {
      const r = await api.post('/api/memory/reflect');
      const changed = (r.added || 0) + (r.updated || 0) + (r.insights || 0) + (r.merged || 0) + (r.removed || 0);
      toast(changed ? `反思完成：新增 ${r.added || 0} 条，更新 ${r.updated || 0} 条，得出 ${r.insights || 0} 条新理解` : '反思完成，没有新的内容', 'success');
      await reload();
    } catch (error) {
      toast(error.message, 'error', 6000);
    } finally {
      setReflecting(false);
    }
  };
  const clearAll = () => confirmDialog({
    title: '清除全部记忆？',
    body: '所有记忆都会被彻底删除，模型将不再记得这些事。对话本身不受影响。',
    confirmLabel: '全部清除',
    danger: true,
    onConfirm: async () => {
      await api.post('/api/memory/clear');
      await reload();
    },
  });

  const counts = data?.counts;
  const items = data?.memories ?? [];
  const tabs = MEMORY_TABS.map(([key, label]) => [key, counts?.status[key] ? `${label} ${counts.status[key]}` : label]);

  return html`<${Modal} title="记忆" width=${820} class="tall">
    <div class="tabbar">
      <${Segmented} value=${status} onChange=${setStatus} options=${tabs} />
      <div class="grow"></div>
      <button type="button" class="btn small" disabled=${reflecting} onClick=${reflect}
        data-tip="现在就读一遍还没整理的对话，并综合已有记忆">
        ${reflecting ? html`<${Spinner} size=${12} />` : html`<${Icon} name="think" size=${14} />`}立即反思
      </button>
      <button type="button" class="btn small primary" onClick=${() => setAdding(!adding)}><${Icon} name="plus" size=${14} />添加</button>
    </div>
    <div class="memory-filter">
      <div class="kind-chips" role="radiogroup" aria-label="按类别筛选">
        <button type="button" class=${kind === '' ? 'on' : ''} onClick=${() => setKind('')}>全部</button>
        ${Object.entries(KIND_LABELS).map(([key, label]) => html`
          <button type="button" key=${key} class=${kind === key ? 'on' : ''} onClick=${() => setKind(key)}>
            ${label}${status === 'active' && counts?.kind[key] ? html` <span class="num">${counts.kind[key]}</span>` : ''}
          </button>`)}
      </div>
      <input class="field search" placeholder="搜索记忆" aria-label="搜索记忆" value=${query} onInput=${(e) => setQuery(e.target.value)} />
    </div>
    <div class="modal-scroll pad">
      ${adding && html`<${MemoryEditor} initial=${{ content: '', kind: 'profile', importance: 3 }} onCancel=${() => setAdding(false)} onSave=${add} />`}
      ${data === null && html`<div class="empty"><${Spinner} /></div>`}
      ${data && !items.length && !adding && html`<div class="empty memory-empty">
        ${query || kind ? '没有匹配的记忆' : status === 'active'
          ? '还没有记忆。多聊几句，对话停下来后我会慢慢记住你的习惯和近况；也可以直接告诉我“记住……”。'
          : '这里是空的'}
      </div>`}
      ${items.map((item) => html`<${MemoryRow} key=${item.id} item=${item} reload=${reload} />`)}
    </div>
    <footer class="memory-foot">
      <${MemoryOptions} />
      <button type="button" class="link-btn danger" onClick=${clearAll}>全部清除</button>
    </footer>
  <//>`;
}

// ------------------------------------------------------------------ 归档
function ArchiveModal() {
  const [items, setItems] = useState(null);
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(new Set());
  const load = async () => {
    setItems((await api.get('/api/conversations?archived=true')).conversations);
    setSelected(new Set());
  };
  useEffect(() => { load().catch((e) => { setItems([]); toast(e.message, 'error'); }); }, []);

  const shown = (items || []).filter((item) => item.title.toLowerCase().includes(query.toLowerCase()));
  const toggle = (id) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };
  const allSelected = shown.length > 0 && shown.every((i) => selected.has(i.id));
  const ids = [...selected];

  const bulk = async (action) => {
    await bulkAction(ids, action);
    await load();
  };
  const restore = async (id) => {
    await setArchived(id, false);
    await load();
  };
  const remove = (item) => askDelete('彻底删除？', `“${item.title}”和由它产生的记忆都会被清除，无法恢复。`, async () => {
    await removeConversation(item.id);
    await load();
  });

  return html`<${Modal} title="已归档的对话" width=${680} class="tall">
    <div class="tabbar">
      <input class="field" placeholder="筛选标题" aria-label="筛选已归档的对话" value=${query} onInput=${(e) => setQuery(e.target.value)} />
      ${shown.length > 0 && html`<button type="button" class="btn small" onClick=${() => setSelected(allSelected ? new Set() : new Set(shown.map((i) => i.id)))}>
        ${allSelected ? '取消全选' : '全选'}
      </button>`}
      ${ids.length > 0 && html`
        <button type="button" class="btn small" onClick=${() => bulk('unarchive')}>恢复 ${ids.length} 项</button>
        <button type="button" class="btn small danger" onClick=${() => askDelete(`永久删除 ${ids.length} 个对话？`, '删除后无法恢复。', () => bulk('delete'))}>删除</button>`}
    </div>
    <div class="modal-scroll">
      ${items === null && html`<div class="empty"><${Spinner} /></div>`}
      ${items && !shown.length && html`<div class="empty">${items.length ? '没有匹配的对话' : '归档是空的。在对话的“⋯”菜单里可以归档。'}</div>`}
      ${shown.map((item) => html`<div class="archive-row" key=${item.id}>
        <button type="button" class="check ${selected.has(item.id) ? 'on' : ''}" aria-label="选择" aria-pressed=${selected.has(item.id)} onClick=${() => toggle(item.id)}>
          ${selected.has(item.id) && html`<${Icon} name="check" size=${12} />`}
        </button>
        <button type="button" class="archive-main" onClick=${() => { openConversation(item.id); closeModal(); }}>
          <span class="archive-title">${item.title}</span>
          <span class="muted small num">${item.message_count} 条消息 · 归档于 ${dateTime(item.archived_at)}</span>
        </button>
        <${IconButton} icon="restore" tip="恢复到对话列表" onClick=${() => restore(item.id)} />
        <${IconButton} icon="trash" tip="永久删除" onClick=${() => remove(item)} />
      </div>`)}
    </div>
  <//>`;
}

// ---------------------------------------------------------------- 提示词库
function PromptsModal() {
  const [editing, setEditing] = useState(null);
  const save = async () => {
    try {
      await api.post('/api/prompts', editing);
      await refreshPrompts();
      setEditing(null);
    } catch (error) {
      toast(error.message, 'error');
    }
  };
  const remove = (prompt) => askDelete('删除这个提示词？', `“${prompt.name}”会被永久删除。`, async () => {
    await api.del(`/api/prompts/${prompt.id}`);
    await refreshPrompts();
    setEditing(null);
  });
  return html`<${Modal} title="提示词" width=${680} class="tall">
    <div class="tabbar">
      <span class="muted small">在输入框键入 / 可以快速调用；也能在“参数”里作为系统提示使用</span>
      <div class="grow"></div>
      <button type="button" class="btn small primary" onClick=${() => setEditing({ name: '', content: '' })}><${Icon} name="plus" size=${14} />新建</button>
    </div>
    <div class="modal-scroll pad">
      ${editing && html`<div class="prompt-edit">
        <input class="field" placeholder="名称" aria-label="名称" value=${editing.name} onInput=${(e) => setEditing({ ...editing, name: e.target.value })} />
        <textarea class="field" rows="6" placeholder="提示词内容" aria-label="提示词内容" value=${editing.content}
          onInput=${(e) => setEditing({ ...editing, content: e.target.value })}></textarea>
        <div class="row-end">
          ${editing.id && html`<button type="button" class="btn small danger" onClick=${() => remove(editing)}>删除</button>`}
          <div class="grow"></div>
          <button type="button" class="btn small" onClick=${() => setEditing(null)}>取消</button>
          <button type="button" class="btn small primary" disabled=${!editing.name.trim() || !editing.content.trim()} onClick=${save}>保存</button>
        </div>
      </div>`}
      ${state.prompts.map((prompt) => html`
        <button type="button" class="prompt-row" key=${prompt.id} onClick=${() => setEditing({ id: prompt.id, name: prompt.name, content: prompt.content })}>
          <strong>${prompt.name}</strong><span>${prompt.content}</span>
        </button>`)}
      ${!state.prompts.length && !editing && html`<div class="empty">还没有提示词</div>`}
    </div>
  <//>`;
}

// ------------------------------------------------------------------ 搜索
function SearchModal() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [active, setActive] = useState(0);
  const ticket = useRef(0);

  useEffect(() => {
    const trimmed = query.trim();
    if (!trimmed) {
      setResults([]);
      return undefined;
    }
    const timer = setTimeout(async () => {
      const current = ++ticket.current;
      try {
        const data = await api.get(`/api/search?q=${encodeURIComponent(trimmed)}`);
        if (current === ticket.current) {
          setResults(data.results);
          setActive(0);
        }
      } catch (error) {
        toast(error.message, 'error');
      }
    }, 150);
    return () => clearTimeout(timer);
  }, [query]);

  const items = query.trim()
    ? results
    : state.conversations.slice(0, 8).map((c) => ({ id: c.id, title: c.title, snippet: '', archived: false }));
  const open = (item) => {
    closeModal();
    openConversation(item.id);
  };
  const onKey = (event) => {
    if (event.isComposing) return;
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setActive((active + 1) % Math.max(items.length, 1));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setActive((active - 1 + items.length) % Math.max(items.length, 1));
    } else if (event.key === 'Enter' && items[active]) {
      open(items[active]);
    }
  };

  return html`<div class="modal-backdrop top" onPointerDown=${(e) => e.target === e.currentTarget && closeModal()}
    onKeyDown=${(e) => e.key === 'Escape' && closeModal()}>
    <div class="palette" role="dialog" aria-modal="true" aria-label="搜索对话">
      <div class="palette-input">
        <${Icon} name="search" />
        <input ref=${(el) => el?.focus()} placeholder="搜索标题和消息内容" aria-label="搜索" value=${query}
          onInput=${(e) => setQuery(e.target.value)} onKeyDown=${onKey} />
        <kbd>Esc</kbd>
      </div>
      <div class="palette-list">
        ${!query.trim() && items.length > 0 && html`<div class="group-label">最近</div>`}
        ${items.map((item, index) => html`
          <button type="button" key=${item.id} class="palette-item ${index === active ? 'on' : ''}"
            onClick=${() => open(item)} onPointerEnter=${() => setActive(index)}>
            <span class="palette-title">${item.title}${item.archived && html`<span class="badge">已归档</span>`}</span>
            ${item.snippet && html`<span class="palette-snippet">${item.snippet}</span>`}
          </button>`)}
        ${query.trim() && !items.length && html`<div class="empty">没有找到匹配的内容</div>`}
      </div>
    </div>
  </div>`;
}

// ------------------------------------------------------------------ 设置
function UsageChart({ days }) {
  const data = days.slice(-30);
  const totals = data.map((d) => d.completion_tokens + d.prompt_tokens + d.cached_tokens);
  const maximum = Math.max(...totals, 1);
  const step = niceStep(maximum);
  const top = Math.ceil(maximum / step) * step;
  const ticks = [top, top / 2, 0];
  return html`<figure class="usage-chart">
    <figcaption>每日处理的 tokens</figcaption>
    <div class="usage-plot">
      <div class="usage-axis num">${ticks.map((t) => html`<span key=${t}>${compact(t)}</span>`)}</div>
      <div class="usage-bars">
        <div class="usage-grid"><i></i><i></i><i></i></div>
        ${data.map((d, i) => html`<div class="usage-slot" key=${d.day}
          data-tip=${`${d.day}\n总计 ${number(totals[i])} tokens\n生成 ${number(d.completion_tokens)} · 计算 ${number(d.prompt_tokens)} · 缓存 ${number(d.cached_tokens)}\n${d.requests} 次请求`}>
          <div class="usage-bar" style=${{ height: `${(totals[i] / top) * 100}%` }}></div>
        </div>`)}
      </div>
    </div>
    <div class="usage-x num"><span>${data[0]?.day.slice(5)}</span>${data.length > 1 && html`<span>${data[data.length - 1].day.slice(5)}</span>`}</div>
  </figure>`;
}

function niceStep(maximum) {
  const rough = maximum / 2;
  const power = 10 ** Math.floor(Math.log10(rough));
  const fraction = rough / power;
  return (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * power;
}

function UsageSection() {
  const [usage, setUsage] = useState(null);
  const load = () => api.get('/api/usage').then(setUsage).catch((e) => toast(e.message, 'error'));
  useEffect(() => { load(); }, []);
  if (!usage) return html`<${Spinner} />`;
  const sum = (key) => usage.by_model.reduce((total, m) => total + m[key], 0);
  const enabled = state.settings.privacy.usage_stats;
  return html`<div>
    <${Field} label="记录用量统计" hint="默认关闭。开启后按天和模型累计 token 数，不包含对话内容">
      <${Toggle} checked=${enabled} label="记录用量统计" onChange=${(usage_stats) => updateSettings({ privacy: { usage_stats } })} />
    <//>
    ${usage.by_day.length > 0 && html`<${Field} label="清除已有统计">
      <button type="button" class="btn small" onClick=${() => api.post('/api/usage/clear').then(load)}>清除</button>
    <//>`}
    <div class="stat-row">
      <div class="tile"><div class="tile-label">对话</div><div class="tile-value">${number(usage.counts.active)}</div></div>
      <div class="tile"><div class="tile-label">已归档</div><div class="tile-value">${number(usage.counts.archived)}</div></div>
      <div class="tile"><div class="tile-label">消息</div><div class="tile-value">${number(usage.counts.messages)}</div></div>
      <div class="tile"><div class="tile-label">生成 tokens</div><div class="tile-value">${compact(sum('completion_tokens'))}</div></div>
    </div>
    ${usage.by_day.length > 0 ? html`<${UsageChart} days=${usage.by_day} />` : html`<div class="empty">${enabled ? '还没有用量数据' : '用量统计已关闭，不记录任何使用情况'}</div>`}
    ${usage.by_model.length > 0 && html`<table class="data-table">
      <thead><tr><th>模型</th><th>请求</th><th>生成</th><th>计算</th><th>缓存命中</th><th>总耗时</th></tr></thead>
      <tbody>
        ${usage.by_model.map((m) => html`<tr key=${m.model}>
          <td title=${m.model}>${m.model.split('/').pop()}</td>
          <td class="num">${number(m.requests)}</td>
          <td class="num">${number(m.completion_tokens)}</td>
          <td class="num">${number(m.prompt_tokens)}</td>
          <td class="num">${number(m.cached_tokens)}</td>
          <td class="num">${seconds(m.seconds)}</td>
        </tr>`)}
      </tbody>
    </table>`}
  </div>`;
}

const Field = ({ label, hint, children }) => html`<div class="set-field">
  <div class="set-label"><div>${label}</div>${hint && html`<div class="set-hint">${hint}</div>`}</div>
  <div class="set-control">${children}</div>
</div>`;

function DataSection() {
  const fileInput = useRef(null);
  const archiveAll = () => {
    const ids = state.conversations.map((c) => c.id);
    if (!ids.length) {
      toast('没有可归档的对话');
      return;
    }
    confirmDialog({
      title: `归档全部 ${ids.length} 个对话？`,
      body: '它们会从列表中移到归档，之后可以恢复。',
      confirmLabel: '全部归档',
      onConfirm: () => bulkAction(ids, 'archive'),
    });
  };
  const importFile = async (file) => {
    try {
      const { count } = await api.post('/api/import', file);
      await refreshConversations();
      toast(`已导入 ${count} 个对话`, 'success');
    } catch (error) {
      toast(error.message, 'error', 6000);
    }
  };
  return html`<div>
    <${Field} label="备份全部对话" hint="导出为 JSON，包含所有分支、归档和记忆。注意：导出的文件没有加密">
      <a class="btn small" href="/api/export" download><${Icon} name="download" size=${14} />导出</a>
    <//>
    <${Field} label="导入备份" hint="只接受本应用导出的 JSON，不会覆盖现有对话">
      <button type="button" class="btn small" onClick=${() => fileInput.current.click()}>选择文件</button>
      <input ref=${fileInput} type="file" accept=".json" hidden onChange=${(e) => { if (e.target.files[0]) importFile(e.target.files[0]); e.target.value = ''; }} />
    <//>
    <${Field} label="归档全部对话"><button type="button" class="btn small" onClick=${archiveAll}>全部归档</button><//>
    <${Field} label="数据目录"><code class="path">${state.status.data_dir}</code><//>
  </div>`;
}

function ApiSection() {
  const base = `${location.origin}/v1`;
  const model = state.settings.default_model || 'your-model';
  const curl = `curl ${base}/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"${model}","messages":[{"role":"user","content":"你好"}]}'`;
  return html`<div>
    <p class="set-intro">其他工具可以把这里当作 OpenAI 接口使用，共用已加载的模型和用量统计。</p>
    <${Field} label="Base URL"><code class="path">${base}</code><//>
    <${Field} label="API Key" hint="本地服务不校验，填任意值即可"><code class="path">any</code><//>
    <pre class="prompt-view">${curl}</pre>
  </div>`;
}

function PasswordForm({ hasPassword, onDone }) {
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [repeat, setRepeat] = useState('');
  const [busy, setBusy] = useState(false);
  const minimum = state.vault.min_password;
  const submit = async (event) => {
    event.preventDefault();
    if (next !== repeat) {
      toast('两次输入的密码不一致', 'error');
      return;
    }
    setBusy(true);
    try {
      await api.post('/api/vault/password', { current, password: next });
      await refreshVault();
      toast(hasPassword ? '密码已修改，数据已用新密钥重新加密' : '启动密码已开启', 'success');
      onDone();
    } catch (error) {
      toast(error.message, 'error', 6000);
    } finally {
      setBusy(false);
    }
  };
  return html`<form class="password-form" onSubmit=${submit}>
    ${hasPassword && html`<input class="field" type="password" autocomplete="off" placeholder="当前密码" aria-label="当前密码" value=${current} onInput=${(e) => setCurrent(e.target.value)} />`}
    <input class="field" type="password" autocomplete="off" placeholder=${`新密码，至少 ${minimum} 位`} aria-label="新密码" value=${next} onInput=${(e) => setNext(e.target.value)} />
    <input class="field" type="password" autocomplete="off" placeholder="再输入一次" aria-label="再次输入新密码" value=${repeat} onInput=${(e) => setRepeat(e.target.value)} />
    <div class="row-end">
      <div class="grow"></div>
      <button type="button" class="btn small" onClick=${onDone}>取消</button>
      <button type="submit" class="btn small primary" disabled=${busy || next.length < minimum}>${busy ? html`<${Spinner} size=${12} />` : '确定'}</button>
    </div>
  </form>`;
}

function RemovePassword({ onDone }) {
  const [current, setCurrent] = useState('');
  const submit = async (event) => {
    event.preventDefault();
    try {
      await api.post('/api/vault/password/remove', { current });
      await refreshVault();
      toast('已取消启动密码，改由系统钥匙串保管密钥', 'success');
      onDone();
    } catch (error) {
      toast(error.message, 'error', 6000);
    }
  };
  return html`<form class="password-form" onSubmit=${submit}>
    <input class="field" type="password" autocomplete="off" placeholder="当前密码" aria-label="当前密码" value=${current} onInput=${(e) => setCurrent(e.target.value)} />
    <div class="row-end">
      <div class="grow"></div>
      <button type="button" class="btn small" onClick=${onDone}>取消</button>
      <button type="submit" class="btn small danger" disabled=${!current}>取消启动密码</button>
    </div>
  </form>`;
}

function WipeField() {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState('');
  if (!open) {
    return html`<${Field} label="抹除全部数据" hint="销毁密钥并删除所有对话、记忆和设置，无法恢复">
      <button type="button" class="btn small danger" onClick=${() => setOpen(true)}>抹除…</button>
    <//>`;
  }
  return html`<div class="wipe-box" role="alert">
    <p>确定的话，请输入“抹除”。这一步无法撤销。</p>
    <div class="row-end">
      <input class="field" aria-label="输入抹除以确认" value=${text} onInput=${(e) => setText(e.target.value)} />
      <button type="button" class="btn danger" disabled=${text !== '抹除'} onClick=${wipeEverything}>抹除</button>
      <button type="button" class="btn" onClick=${() => { setOpen(false); setText(''); }}>取消</button>
    </div>
  </div>`;
}

function PrivacySection() {
  const [form, setForm] = useState('');
  const vault = state.vault;
  const hasPassword = vault.mode === 'password';
  const rotate = () => confirmDialog({
    title: '更换数据库密钥？',
    body: '会用一把新密钥重写整个数据库。之后磁盘上可能残留的旧密文，即使被恢复出来也无法解开。',
    confirmLabel: '更换',
    onConfirm: async () => {
      try {
        await api.post('/api/vault/rotate');
        toast('密钥已更换', 'success');
      } catch (error) {
        toast(error.message, 'error', 6000);
      }
    },
  });
  return html`<div>
    <div class="privacy-status">
      <div><${Icon} name="shield" /><span>数据库已加密（AES-256）</span><span class="status good"><${Icon} name="check" size=${13} />已开启</span></div>
      <div class="privacy-note">对话、记忆、设置和 API Key 都在加密数据库里。密钥${hasPassword ? '由启动密码派生，磁盘上不保存' : '保存在 macOS 钥匙串，启动时自动解锁'}。</div>
      ${vault.filevault === false && html`<div class="privacy-warn"><${Icon} name="alert" size=${14} /><span>这台 Mac 没有开启 FileVault。数据库本身是加密的，但你导出的文件和磁盘上的其他内容不受保护。建议在“系统设置 → 隐私与安全性”里开启。</span></div>`}
    </div>

    <${Field} label="启动密码" hint=${hasPassword ? '每次启动都要输入，忘记后数据无法找回' : '更安全：密钥不再保存在这台电脑上，每次启动输入'}>
      ${form === '' && html`<div class="inline">
        <button type="button" class="btn small" onClick=${() => setForm('password')}>${hasPassword ? '修改' : '设置'}</button>
        ${hasPassword && html`<button type="button" class="btn small" onClick=${() => setForm('remove')}>取消密码</button>`}
      </div>`}
    <//>
    ${form === 'password' && html`<${PasswordForm} hasPassword=${hasPassword} onDone=${() => setForm('')} />`}
    ${form === 'remove' && html`<${RemovePassword} onDone=${() => setForm('')} />`}

    ${hasPassword && html`<${Field} label="立即锁定" hint="关闭数据库，并清除内存中的对话缓存">
      <button type="button" class="btn small" onClick=${lockNow}><${Icon} name="lock" size=${14} />锁定</button>
    <//>`}
    ${hasPassword && html`<${Field} label="空闲后自动锁定" hint="0 表示不自动锁定">
      <div class="inline"><input class="field short num" type="number" min="0" aria-label="自动锁定分钟数" value=${state.settings.privacy.auto_lock_minutes} onChange=${(e) => updateSettings({ privacy: { auto_lock_minutes: +e.target.value } })} /><span>分钟</span></div>
    <//>`}
    ${!hasPassword && html`<${Field} label="更换密钥" hint="重写整个数据库，让磁盘上可能残留的旧密文彻底失效">
      <button type="button" class="btn small" onClick=${rotate}>更换</button>
    <//>`}
    <${WipeField} />

    <h3 class="section-label privacy-head">这个工具的隐私承诺</h3>
    <ul class="privacy-list">
      <li><strong>除对话本身外不留日志。</strong>不写日志文件；出错时终端只显示错误类型，不含内容；用量统计默认关闭。</li>
      <li><strong>删除即清除。</strong>删除对话会同时移除由它产生的记忆、显存里的缓存，并截断数据库日志，被删内容所在的数据页会被清零。</li>
      <li><strong>联网是可选的。</strong>只有开启“联网”的对话才会发出检索词，正文和记忆不会发送出去。</li>
      <li><strong>做不到的事。</strong>固态硬盘的物理擦除由系统控制，无法保证旧数据块立刻消失，因此这里只保证残留的都是密文。你主动导出的文件不受保护。</li>
    </ul>
  </div>`;
}

const SEARXNG_COMMAND = 'docker run -d --name searxng -p 127.0.0.1:8888:8080 -e SEARXNG_SEARCH_FORMATS=html,json searxng/searxng';

function ApiKeyField({ label, name, isSet, hint }) {
  const [value, setValue] = useState('');
  const save = (key) => updateSettings({ search: { [name]: key } });
  return html`<${Field} label=${label} hint=${hint}>
    <div class="inline grow-input">
      <input class="field" type="password" autocomplete="off" aria-label=${label} placeholder=${isSet ? '已保存，输入新值可替换' : '粘贴 API Key'}
        value=${value} onInput=${(e) => setValue(e.target.value)}
        onChange=${(e) => { if (e.target.value.trim()) { save(e.target.value); setValue(''); } }} />
      ${isSet && html`<button type="button" class="btn small" onClick=${() => save('')}>清除</button>`}
    </div>
  <//>`;
}

function SearchSection() {
  const search = state.settings.search;
  const save = (patch) => updateSettings({ search: patch });
  return html`<div>
    <p class="set-intro">在输入框点“联网”才会搜索。发送给搜索服务的只有模型写的一行检索词，你的记忆和附件不会发送出去。</p>
    <${Field} label="搜索服务">
      <${Segmented} value=${search.provider} onChange=${(provider) => save({ provider })}
        options=${[['wikipedia', '维基百科'], ['searxng', 'SearXNG'], ['brave', 'Brave'], ['tavily', 'Tavily']]} />
    <//>
    ${search.provider === 'wikipedia' && html`<p class="local-hint">免配置，但只能查百科知识，查不到新闻和普通网页。想搜全网，请选择下面任一服务。</p>`}
    ${search.provider === 'searxng' && html`
      <${Field} label="实例地址" hint="隐私最好：查询只经过你自己运行的服务">
        <input class="field" placeholder="http://127.0.0.1:8888" aria-label="SearXNG 实例地址" value=${search.searxng_url} onChange=${(e) => save({ searxng_url: e.target.value })} />
      <//>
      <p class="local-hint">这台 Mac 上有 Docker 的话，可以这样启动一个本地实例（需要下载约 200 MB 镜像）：</p>
      <pre class="prompt-view">${SEARXNG_COMMAND}</pre>`}
    ${search.provider === 'brave' && html`<${ApiKeyField} label="Brave API Key" name="brave_api_key" isSet=${search.brave_key_set} hint="在 brave.com/search/api 申请，密钥保存在加密数据库中" />`}
    ${search.provider === 'tavily' && html`<${ApiKeyField} label="Tavily API Key" name="tavily_api_key" isSet=${search.tavily_key_set} hint="在 tavily.com 申请，密钥保存在加密数据库中" />`}
    <${Field} label="搜索结果数">
      <input class="field short num" type="number" min="1" max="10" aria-label="搜索结果数" value=${search.max_results} onChange=${(e) => save({ max_results: +e.target.value })} />
    <//>
    <${Field} label="读取网页正文" hint="读取前几条结果的正文，回答更准确；0 表示只用搜索摘要">
      <div class="inline"><span>前</span><input class="field tiny num" type="number" min="0" max="5" aria-label="读取网页数" value=${search.fetch_pages} onChange=${(e) => save({ fetch_pages: +e.target.value })} /><span>条</span></div>
    <//>
  </div>`;
}

function SettingsModal() {
  const [section, setSection] = useState('general');
  const settings = state.settings;
  const ui = settings.ui;
  const status = state.status;
  const nav = [
    ['general', '通用'], ['privacy', '隐私与安全'], ['search', '联网搜索'], ['inference', '推理与显存'],
    ['sources', '模型来源'], ['usage', '用量统计'], ['data', '数据'], ['api', 'API'], ['about', '关于'],
  ];
  const [dirs, setDirs] = useState(settings.model_dirs.join('\n'));
  const mirror = settings.hf_endpoint.includes('hf-mirror') ? 'mirror' : settings.hf_endpoint.includes('huggingface.co') ? 'official' : 'custom';

  return html`<${Modal} title="设置" width=${820} class="tall">
    <div class="settings">
      <nav class="set-nav">
        ${nav.map(([key, label]) => html`<button type="button" key=${key} class=${section === key ? 'on' : ''} onClick=${() => setSection(key)}>${label}</button>`)}
      </nav>
      <div class="set-body">
        ${section === 'general' && html`<div>
          <${Field} label="主题">
            <${Segmented} value=${ui.theme} onChange=${(theme) => setUi({ theme })} options=${[['system', '跟随系统'], ['light', '浅色'], ['dark', '深色']]} />
          <//>
          <${Field} label="字号">
            <div class="inline"><input type="range" min="13" max="19" step="1" aria-label="字号" value=${ui.font_size} onInput=${(e) => setUi({ font_size: +e.target.value })} /><span class="num">${ui.font_size}px</span></div>
          <//>
          <${Field} label="发送方式">
            <${Segmented} value=${ui.send_key} onChange=${(send_key) => setUi({ send_key })} options=${[['enter', 'Enter'], ['cmd', '⌘Enter']]} />
          <//>
          <${Field} label="显示消息统计" hint="在每条消息下显示 token 数和速度">
            <${Toggle} checked=${ui.show_stats} label="显示消息统计" onChange=${(show_stats) => setUi({ show_stats })} />
          <//>
          <${Field} label="自动生成标题" hint="第一轮对话结束后，让模型起一个简短的标题">
            <${Toggle} checked=${settings.auto_title} label="自动生成标题" onChange=${(auto_title) => updateSettings({ auto_title })} />
          <//>
        </div>`}

        ${section === 'inference' && html`<div>
          <${Field} label="KV 缓存上限" hint="跨对话保留的缓存总量，越大切换对话时越快。0 表示不保留">
            <div class="inline"><input class="field short num" type="number" min="0" step="256" aria-label="KV 缓存上限" value=${settings.kv_cache_limit_mb} onChange=${(e) => updateSettings({ kv_cache_limit_mb: +e.target.value })} /><span>MB</span></div>
          <//>
          <${Field} label="空闲后自动卸载模型" hint="过了这么久没有使用就释放显存。0 表示不自动卸载">
            <div class="inline"><input class="field short num" type="number" min="0" step="5" aria-label="空闲卸载分钟数" value=${settings.idle_unload_minutes} onChange=${(e) => updateSettings({ idle_unload_minutes: +e.target.value })} /><span>分钟</span></div>
          <//>
          <${Field} label="信任模型仓库中的代码" hint="开启后，部分模型会执行仓库自带的 Python 代码。只在你信任来源时开启">
            <${Toggle} checked=${settings.trust_remote_code} label="信任模型仓库中的代码" onChange=${(trust_remote_code) => updateSettings({ trust_remote_code })} />
          <//>
        </div>`}

        ${section === 'sources' && html`<div>
          <${Field} label="下载源" hint="如果访问 Hugging Face 不稳定，可以切换到镜像">
            <${Segmented} value=${mirror} onChange=${(value) => {
              if (value === 'official') updateSettings({ hf_endpoint: 'https://huggingface.co' });
              if (value === 'mirror') updateSettings({ hf_endpoint: 'https://hf-mirror.com' });
            }} options=${[['official', '官方'], ['mirror', 'hf-mirror']]} />
          <//>
          <${Field} label="下载位置" hint="留空使用默认缓存。磁盘紧张时可以指向外接硬盘">
            <input class="field" placeholder=${status.download_cache} aria-label="下载位置" value=${settings.download_dir} onChange=${(e) => updateSettings({ download_dir: e.target.value }).then(refreshModels)} />
          <//>
          <${Field} label="其他模型目录" hint="每行一个目录，自动发现里面的 MLX 模型。LM Studio 的目录会自动包含">
            <textarea class="field" rows="3" aria-label="其他模型目录" value=${dirs} onInput=${(e) => setDirs(e.target.value)}
              onChange=${(e) => updateSettings({ model_dirs: e.target.value.split('\n') }).then(refreshModels)}></textarea>
          <//>
        </div>`}

        ${section === 'privacy' && html`<${PrivacySection} />`}
        ${section === 'search' && html`<${SearchSection} />`}
        ${section === 'usage' && html`<${UsageSection} />`}
        ${section === 'data' && html`<${DataSection} />`}
        ${section === 'api' && html`<${ApiSection} />`}

        ${section === 'about' && html`<div>
          <${Field} label="设备">${status.device}<//>
          <${Field} label="统一内存">${bytes(status.memory_size)}<//>
          <${Field} label="显存建议上限">${bytes(status.working_set_limit)}<//>
          <${Field} label="版本"><span class="num">MLX UI ${status.version} · mlx ${status.mlx} · mlx-lm ${status.mlx_lm} · Python ${status.python}</span><//>
        </div>`}
      </div>
    </div>
  <//>`;
}

export function ModalHost() {
  const modal = state.modal;
  if (!modal) return null;
  switch (modal.name) {
    case 'models': return html`<${ModelsModal} />`;
    case 'memory': return html`<${MemoryModal} />`;
    case 'archive': return html`<${ArchiveModal} />`;
    case 'prompts': return html`<${PromptsModal} />`;
    case 'search': return html`<${SearchModal} />`;
    case 'settings': return html`<${SettingsModal} />`;
    default: return null;
  }
}
