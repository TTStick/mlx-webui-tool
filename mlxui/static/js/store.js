import { api } from './api.js';
import { clearRenderCache } from './markdown.js';

const freshDraft = () => ({ text: '', attachments: [], model: null, settings: {} });
const freshLive = () => ({
  state: null, position: 0, progress: null, cached: 0, total: 0,
  ttft: null, tps: null, tokens: 0, tpsHistory: [], stats: null, search: null, count: 0,
});

export const state = {
  ready: false,
  offline: false,
  vault: null,
  memoryStatus: null,
  status: null,
  settings: null,
  models: [],
  localDirs: [],
  disk: null,
  workingSetLimit: 0,
  monitor: null,
  memoryHistory: [],
  conversations: [],
  currentId: null,
  conversation: null,
  messages: [],
  params: null,
  draft: freshDraft(),
  context: null,
  job: null,
  live: freshLive(),
  compacting: false,
  prompts: [],
  downloads: [],
  modal: null,
  confirm: null,
  toasts: [],
  selecting: false,
  selected: new Set(),
  loadingConversation: false,
};

// ------------------------------------------------------------ 渲染调度
const listeners = new Set();
let scheduled = false;

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function notify() {
  if (scheduled) return;
  scheduled = true;
  let timer = null;
  const flush = () => {
    if (!scheduled) return;
    scheduled = false;
    clearTimeout(timer);
    listeners.forEach((listener) => listener());
  };
  // 窗口被遮挡时浏览器可能暂停 rAF，用定时器兜底，保证状态一定会渲染
  timer = setTimeout(flush, document.hidden ? 200 : 120);
  if (!document.hidden) requestAnimationFrame(flush);
}

// ---------------------------------------------------------------- 提示
let toastSeq = 0;
export function toast(message, kind = 'info', duration = 3800) {
  const id = ++toastSeq;
  state.toasts = [...state.toasts, { id, message, kind }];
  notify();
  setTimeout(() => {
    state.toasts = state.toasts.filter((t) => t.id !== id);
    notify();
  }, duration);
}

const fail = (error) => toast(error.message, 'error', 6000);

export function confirmDialog(options) {
  state.confirm = options;
  notify();
}

export function openModal(name, extra = {}) {
  state.modal = { name, ...extra };
  notify();
}

export function closeModal() {
  state.modal = null;
  notify();
}

// ---------------------------------------------------------------- 取值
export const modelInfo = (id) => state.models.find((m) => m.id === id) || null;

export function currentModelId() {
  if (state.conversation) return state.conversation.model || state.settings?.default_model || null;
  return state.draft.model || state.settings?.default_model || null;
}

export function effectiveParams() {
  if (state.params) return state.params;
  return { ...(state.settings?.params || {}), ...state.draft.settings };
}

export function kvBytesPerToken(model) {
  if (!model) return null;
  return state.monitor?.engine?.kv_bytes_per_token?.[model.id] ?? model.kv_bytes_per_token ?? null;
}

export const loadedModelId = () => state.monitor?.engine?.model_id ?? null;

export const busyConversationIds = () => {
  const engine = state.monitor?.engine;
  const ids = new Set();
  if (engine?.current_job?.conversation_id) ids.add(engine.current_job.conversation_id);
  (engine?.queue || []).forEach((job) => job.conversation_id && ids.add(job.conversation_id));
  return ids;
};

// ------------------------------------------------------------ 启动与轮询
export async function boot() {
  try {
    state.vault = await api.get('/api/vault/status');
    if (state.vault.state === 'unlocked') await startApp();
  } catch (error) {
    state.offline = true;
    fail(error);
  }
  notify();
}

let pollingStarted = false;

async function startApp() {
  try {
    const [status, settings, models, prompts, conversations] = await Promise.all([
      api.get('/api/status'),
      api.get('/api/settings'),
      api.get('/api/models?refresh=true'),
      api.get('/api/prompts'),
      api.get('/api/conversations'),
    ]);
    state.status = status;
    state.settings = settings;
    // 窄窗口下两侧面板是浮层，默认收起，且不写回设置
    if (window.innerWidth < 760) settings.ui.sidebar_open = false;
    if (window.innerWidth < 1180) settings.ui.inspector_open = false;
    applyModels(models);
    state.prompts = prompts.prompts;
    state.conversations = conversations.conversations;
    if (!settings.default_model && state.models.length) {
      await setDefaultModel(state.models[0].id);
    }
    applyTheme();
    state.ready = true;
    await routeFromHash();
    if (!pollingStarted) {
      pollingStarted = true;
      pollMonitor();
      pollDownloads();
    }
    scheduleContext(0);
  } catch (error) {
    if (error.status !== 423) {
      state.offline = true;
      fail(error);
    }
  }
  notify();
}

function resetSensitiveState() {
  detach();
  state.ready = false;
  state.currentId = null;
  state.conversation = null;
  state.messages = [];
  state.params = null;
  state.draft = freshDraft();
  state.context = null;
  state.job = null;
  state.live = freshLive();
  state.compacting = false;
  state.conversations = [];
  state.prompts = [];
  state.modal = null;
  state.confirm = null;
  state.selected = new Set();
  state.selecting = false;
  state.monitor = null;
  state.memoryStatus = null;
  state.memoryHistory = [];
  clearRenderCache();
}

async function showLockScreen() {
  resetSensitiveState();
  state.vault = await api.get('/api/vault/status');
  notify();
}

window.addEventListener('mlxui-locked', () => {
  if (state.ready) showLockScreen();
});

export async function unlock(password) {
  state.vault = await api.post('/api/vault/unlock', { password });
  await startApp();
  notify();
}

export async function setupPassword(password) {
  state.vault = await api.post('/api/vault/setup', { password });
  await startApp();
  notify();
}

export async function lockNow() {
  try {
    await api.post('/api/vault/lock');
    await showLockScreen();
  } catch (error) {
    fail(error);
  }
}

export async function wipeEverything() {
  try {
    const status = await api.post('/api/vault/wipe', { confirm: '抹除' });
    resetSensitiveState();
    if (location.hash) history.replaceState(null, '', location.pathname);
    state.vault = status;
    if (status.state === 'unlocked') await startApp();
    notify();
  } catch (error) {
    fail(error);
  }
}

export async function retryVault() {
  state.vault = await api.post('/api/vault/retry');
  if (state.vault.state === 'unlocked') await startApp();
  notify();
}

export async function refreshVault() {
  state.vault = await api.get('/api/vault/status');
  notify();
}

function applyModels(data) {
  state.models = data.models;
  state.localDirs = data.local_dirs || [];
  // 新对话选中的模型被删除或移除后，不再保留失效的选择
  if (state.draft.model && !data.models.some((m) => m.id === state.draft.model)) state.draft.model = null;
  state.workingSetLimit = data.working_set_limit;
  state.disk = data.disk;
}

export async function refreshModels() {
  applyModels(await api.get('/api/models?refresh=true'));
  notify();
}

let previousBusy = '';

async function pollMonitor() {
  if (!state.ready) {
    setTimeout(pollMonitor, 1000);
    return;
  }
  try {
    const snapshot = await api.get('/api/monitor');
    state.monitor = snapshot;
    state.offline = false;
    const previous = state.memoryStatus;
    if (previous && snapshot.memory.unseen > previous.unseen && state.modal?.name !== 'memory') {
      const added = snapshot.memory.unseen - previous.unseen;
      toast(snapshot.memory.pending > previous.pending ? `有 ${added} 条新记忆等你确认` : `记住了 ${added} 件新的事`, 'success');
    }
    state.memoryStatus = snapshot.memory;
    state.memoryHistory = [
      ...state.memoryHistory.slice(-89),
      { t: snapshot.time, active: snapshot.mlx.active, kv: snapshot.engine.kv_bytes },
    ];
    // 后台对话跑完后要刷新列表（标题、时间可能变了）
    const busy = [...busyConversationIds()].sort().join(',');
    if (busy !== previousBusy) {
      previousBusy = busy;
      refreshConversations();
    }
  } catch (error) {
    if (error.status !== 423) state.offline = true;
  }
  notify();
  setTimeout(pollMonitor, document.hidden ? 5000 : state.job ? 1000 : 2000);
}

let downloadsWereRunning = new Set();

async function pollDownloads() {
  let anyRunning = false;
  if (!state.ready) {
    setTimeout(pollDownloads, 1000);
    return;
  }
  try {
    const { tasks } = await api.get('/api/downloads');
    state.downloads = tasks;
    const running = new Set(tasks.filter((t) => t.status === 'running').map((t) => t.id));
    anyRunning = running.size > 0;
    for (const task of tasks) {
      if (downloadsWereRunning.has(task.id) && task.status === 'done') {
        toast(`${task.repo.split('/').pop()} 下载完成`, 'success');
        await refreshModels();
        if (!state.settings.default_model && state.models.length) await setDefaultModel(task.repo);
        scheduleContext(0);
      } else if (downloadsWereRunning.has(task.id) && task.status === 'failed') {
        toast(`下载失败：${task.error || task.repo}`, 'error', 8000);
      }
    }
    downloadsWereRunning = running;
    notify();
  } catch {
    // 服务断开时由 monitor 轮询负责提示
  }
  setTimeout(pollDownloads, anyRunning ? 800 : 2500);
}

export function trackDownloads(tasks) {
  state.downloads = tasks;
  tasks.filter((t) => t.status === 'running').forEach((t) => downloadsWereRunning.add(t.id));
  notify();
}

export async function refreshConversations() {
  try {
    state.conversations = (await api.get('/api/conversations')).conversations;
    notify();
  } catch (error) {
    fail(error);
  }
}

// ---------------------------------------------------------------- 路由
const hashId = () => {
  const match = location.hash.match(/^#\/c\/([0-9a-f]+)$/);
  return match ? match[1] : null;
};

export async function routeFromHash() {
  if (!state.ready) return;
  const id = hashId();
  if (id === state.currentId) return;
  if (id) await loadConversation(id);
  else resetToNewChat();
}

window.addEventListener('hashchange', () => routeFromHash());

function setHash(id) {
  const target = id ? `#/c/${id}` : '#/';
  if (location.hash !== target) location.hash = target;
}

function resetToNewChat() {
  detach();
  flushParamPatch();
  state.currentId = null;
  state.conversation = null;
  state.messages = [];
  state.params = null;
  state.job = null;
  state.live = freshLive();
  state.compacting = false;
  scheduleContext(0);
  notify();
}

export function newChat() {
  setHash(null);
  resetToNewChat();
}

export function openConversation(id) {
  state.selecting = false;
  setHash(id);
}

async function loadConversation(id) {
  detach();
  flushParamPatch();
  state.currentId = id;
  state.loadingConversation = true;
  state.live = freshLive();
  state.job = null;
  state.compacting = false;
  state.draft.text = '';
  state.draft.attachments = [];
  notify();
  try {
    const detail = await api.get(`/api/conversations/${id}`);
    if (state.currentId !== id) return;
    applyDetail(detail);
    if (detail.job) {
      if (detail.job.kind === 'compact') attachCompact(detail.job.id, id);
      else attach(detail.job, id);
    }
  } catch (error) {
    if (error.status === 404) {
      toast('这个对话不存在了', 'error');
      newChat();
    } else {
      fail(error);
    }
  } finally {
    state.loadingConversation = false;
    scheduleContext(0);
    notify();
  }
}

function applyDetail(detail) {
  state.conversation = detail.conversation;
  state.messages = detail.messages;
  state.params = detail.params;
  const last = [...detail.messages].reverse().find((m) => m.role === 'assistant' && m.stats?.completion_tokens != null);
  if (!state.job) state.live.stats = last?.stats ?? null;
  notify();
}

async function reloadCurrent() {
  if (!state.currentId) return;
  const id = state.currentId;
  try {
    const detail = await api.get(`/api/conversations/${id}`);
    if (state.currentId === id) applyDetail(detail);
  } catch (error) {
    if (error.status !== 404) fail(error);
  }
}

// -------------------------------------------------------------- 上下文
let contextTimer = null;
let contextTicket = 0;

export function scheduleContext(delay = 250) {
  clearTimeout(contextTimer);
  contextTimer = setTimeout(refreshContext, delay);
}

async function refreshContext() {
  const ticket = ++contextTicket;
  try {
    const result = await api.post('/api/context/preview', {
      conversation_id: state.currentId,
      draft: state.draft.text,
      attachments: state.draft.attachments,
      model: currentModelId(),
      settings: state.conversation ? null : state.draft.settings,
    });
    if (ticket === contextTicket) {
      state.context = result;
      notify();
    }
  } catch {
    // 统计失败不打断用户输入
  }
}

export async function fetchPromptPreview() {
  return api.post('/api/context/preview', {
    conversation_id: state.currentId,
    draft: state.draft.text,
    attachments: state.draft.attachments,
    model: currentModelId(),
    settings: state.conversation ? null : state.draft.settings,
    include_prompt: true,
  });
}

// ---------------------------------------------------------------- 任务流
let source = null;

function detach() {
  if (source) source.close();
  source = null;
}

function liveMessage(id) {
  return state.messages.find((m) => m.id === id);
}

function attach(job, conversationId) {
  detach();
  const prefix = job.prefix || '';
  const target = liveMessage(job.message_id);
  if (target) {
    target.content = prefix;
    target.status = 'streaming';
  }
  state.job = job;
  state.live = freshLive();
  let text = '';
  let lastSeq = -1;

  const es = new EventSource(`/api/jobs/${job.id}/events`);
  source = es;
  es.onmessage = (raw) => {
    const event = JSON.parse(raw.data);
    if (event.seq <= lastSeq) return; // 自动重连时服务端会从头重放
    lastSeq = event.seq;
    if (state.currentId !== conversationId) return;
    const live = state.live;
    const message = liveMessage(job.message_id);
    switch (event.type) {
      case 'status':
        live.state = event.state;
        if (event.position != null) live.position = event.position;
        if (event.count != null) live.count = event.count;
        if (event.ttft != null) live.ttft = event.ttft;
        if (event.total != null) {
          live.total = event.total;
          live.cached = event.cached ?? 0;
        }
        break;
      case 'progress':
        live.progress = { done: event.done, total: event.total };
        break;
      case 'search':
        live.search = { ...live.search, ...event };
        break;
      case 'context':
        state.context = { ...event };
        break;
      case 'delta':
        text += event.text;
        if (message) {
          message.content = prefix + text;
          message.status = 'streaming';
        }
        break;
      case 'tick':
        live.tps = event.tps;
        live.tokens = event.tokens;
        if (event.tokens >= 3) live.tpsHistory = [...live.tpsHistory.slice(-119), event.tps];
        break;
      case 'done':
        if (message) {
          message.status = event.status;
          message.stats = event.stats;
        }
        live.stats = event.stats;
        live.state = null;
        state.job = null;
        reloadCurrent().then(() => scheduleContext(0));
        refreshConversations();
        break;
      case 'title':
        if (state.conversation) state.conversation.title = event.title;
        refreshConversations();
        break;
      case 'error':
        state.job = null;
        live.state = null;
        reloadCurrent().then(() => scheduleContext(0));
        break;
      case 'end':
        es.close();
        if (source === es) source = null;
        break;
      default:
        break;
    }
    notify();
  };
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED && state.currentId === conversationId) {
      state.job = null;
      state.live.state = null;
      reloadCurrent();
      notify();
    }
  };
  notify();
}

function attachCompact(jobId, conversationId) {
  detach();
  state.compacting = true;
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  source = es;
  let failed = null;
  es.onmessage = (raw) => {
    const event = JSON.parse(raw.data);
    if (event.type === 'error') failed = event.message;
    if (event.type !== 'end') return;
    es.close();
    if (source === es) source = null;
    if (state.currentId !== conversationId) return;
    state.compacting = false;
    if (failed) toast(`压缩失败：${failed}`, 'error', 6000);
    else toast('上下文已压缩', 'success');
    reloadCurrent().then(() => scheduleContext(0));
    notify();
  };
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) {
      state.compacting = false;
      notify();
    }
  };
  notify();
}

function startFrom(result) {
  applyDetail(result);
  scheduleContext(0);
  refreshConversations();
  const target = result.messages[result.messages.length - 1];
  attach({ id: result.job_id, message_id: target.id, kind: 'chat', prefix: result.job?.prefix || '' }, result.conversation.id);
}

// ---------------------------------------------------------------- 动作
export async function send({ force = false } = {}) {
  const draft = state.draft;
  const content = draft.text;
  const attachments = draft.attachments;
  if ((!content.trim() && !attachments.length) || state.job) return;
  const model = currentModelId();
  if (!model) {
    toast('请先下载或选择一个模型', 'error');
    openModal('models', { tab: 'recommended' });
    return;
  }
  draft.text = '';
  draft.attachments = [];
  notify();
  try {
    let id = state.currentId;
    if (!id) {
      const created = await api.post('/api/conversations', { model, settings: draft.settings });
      id = created.id;
      state.currentId = id;
      state.draft = freshDraft();
      setHash(id);
    }
    startFrom(await api.post(`/api/conversations/${id}/send`, { content, attachments, force }));
  } catch (error) {
    state.draft.text = content;
    state.draft.attachments = attachments;
    fail(error);
    notify();
  }
}

export async function stop() {
  if (!state.job) return;
  try {
    await api.post(`/api/jobs/${state.job.id}/stop`);
  } catch (error) {
    fail(error);
  }
}

const conversationPath = (action) => `/api/conversations/${state.currentId}/${action}`;

export async function regenerate(message, force = false) {
  if (state.job) return;
  try {
    startFrom(await api.post(conversationPath('regenerate'), { message_id: message.id, force }));
  } catch (error) {
    fail(error);
  }
}

export async function editMessage(message, content) {
  if (state.job) return;
  try {
    startFrom(await api.post(conversationPath('edit'), { message_id: message.id, content }));
  } catch (error) {
    fail(error);
  }
}

export async function continueMessage(message, force = false) {
  if (state.job) return;
  try {
    startFrom(await api.post(conversationPath('continue'), { message_id: message.id, force }));
  } catch (error) {
    fail(error);
  }
}

export async function switchBranch(message, delta) {
  if (state.job) return;
  const target = message.siblings[message.sibling_index + delta];
  if (!target) return;
  try {
    applyDetail(await api.post(conversationPath('switch'), { message_id: target }));
    scheduleContext(0);
  } catch (error) {
    fail(error);
  }
}

export async function deleteMessage(message) {
  try {
    applyDetail(await api.post(conversationPath('delete-message'), { message_id: message.id }));
    clearRenderCache();
    scheduleContext(0);
    refreshConversations();
  } catch (error) {
    fail(error);
  }
}

export async function compactContext() {
  if (state.job || state.compacting || !state.currentId) return;
  try {
    const { job_id: jobId } = await api.post(conversationPath('compact'), {});
    attachCompact(jobId, state.currentId);
  } catch (error) {
    fail(error);
  }
}

export async function clearSummary() {
  try {
    applyDetail(await api.post(conversationPath('clear-summary')));
    scheduleContext(0);
  } catch (error) {
    fail(error);
  }
}

// -------------------------------------------------------------- 模型选择
async function setDefaultModel(id) {
  state.settings = await api.patch('/api/settings', { default_model: id });
}

export async function selectModel(id) {
  try {
    if (state.conversation) {
      applyDetail(await api.patch(`/api/conversations/${state.currentId}`, { model: id }));
    } else {
      state.draft.model = id;
    }
    await setDefaultModel(id); // 新对话沿用最近一次的选择
    scheduleContext(0);
    notify();
  } catch (error) {
    fail(error);
  }
}

export async function pickFolder() {
  return (await api.post('/api/system/pick-folder')).path;
}

export async function addLocalFolder(path) {
  const result = await api.post('/api/models/local', { path });
  state.settings.model_dirs = result.model_dirs ?? state.settings.model_dirs;
  await refreshModels();
  scheduleContext(0);
  return result;
}

export async function removeLocalFolder(path) {
  try {
    const result = await api.post('/api/models/local/remove', { path });
    state.settings.model_dirs = result.model_dirs;
    state.settings.default_model = result.default_model;
    await refreshModels();
    scheduleContext(0);
    toast('已从列表移除，文件还在原处', 'success');
  } catch (error) {
    fail(error);
  }
}

export async function loadModel(id, force = false) {
  try {
    await api.post('/api/models/load', { id, force });
    toast('模型已加载', 'success');
  } catch (error) {
    if (error.code === 'model_too_large') {
      confirmDialog({
        title: '这个模型可能放不下',
        body: `${error.message} 仍然要加载吗？`,
        confirmLabel: '仍然加载',
        danger: true,
        onConfirm: () => loadModel(id, true),
      });
    } else {
      fail(error);
    }
  }
  notify();
}

export async function unloadModel() {
  try {
    await api.post('/api/models/unload');
    toast('模型已卸载，显存已释放', 'success');
  } catch (error) {
    fail(error);
  }
}

export async function clearKvCache() {
  try {
    await api.post('/api/models/clear-cache');
    toast('KV 缓存已清空', 'success');
  } catch (error) {
    fail(error);
  }
}

// -------------------------------------------------------------- 参数修改
let paramPatch = {};
let paramTimer = null;
let paramConversation = null;

function flushParamPatch() {
  clearTimeout(paramTimer);
  if (!Object.keys(paramPatch).length || !paramConversation) return;
  const patch = paramPatch;
  const id = paramConversation;
  paramPatch = {};
  api.patch(`/api/conversations/${id}`, { settings: patch }).catch(fail);
}

export function setParam(key, value) {
  if (state.conversation) {
    state.params = { ...state.params, [key]: value };
    if (paramConversation !== state.currentId) flushParamPatch();
    paramConversation = state.currentId;
    paramPatch[key] = value;
    clearTimeout(paramTimer);
    paramTimer = setTimeout(flushParamPatch, 350);
  } else {
    state.draft.settings[key] = value;
  }
  scheduleContext(200);
  notify();
}

export async function resetParams() {
  if (state.conversation) {
    flushParamPatch();
    const clear = Object.fromEntries(Object.keys(state.conversation.settings).map((k) => [k, null]));
    try {
      applyDetail(await api.patch(`/api/conversations/${state.currentId}`, { settings: clear }));
    } catch (error) {
      fail(error);
    }
  } else {
    state.draft.settings = {};
  }
  scheduleContext(0);
  notify();
}

export async function saveParamsAsDefault() {
  try {
    state.settings = await api.patch('/api/settings', { params: effectiveParams() });
    toast('已保存为新对话的默认参数', 'success');
  } catch (error) {
    fail(error);
  }
}

// -------------------------------------------------------------- 对话管理
export async function renameConversation(id, title) {
  try {
    const detail = await api.patch(`/api/conversations/${id}`, { title });
    if (state.currentId === id) state.conversation = detail.conversation;
    await refreshConversations();
  } catch (error) {
    fail(error);
  }
}

export async function setPinned(id, pinned) {
  try {
    await api.patch(`/api/conversations/${id}`, { pinned });
    await refreshConversations();
  } catch (error) {
    fail(error);
  }
}

export async function setArchived(id, archived) {
  try {
    await api.patch(`/api/conversations/${id}`, { archived });
    if (archived && state.currentId === id) newChat();
    await refreshConversations();
    toast(archived ? '已归档' : '已恢复到对话列表', 'success');
  } catch (error) {
    fail(error);
  }
}

export async function removeConversation(id) {
  try {
    const { memories_removed: removed } = await api.del(`/api/conversations/${id}`);
    if (state.currentId === id) newChat();
    clearRenderCache();
    await refreshConversations();
    toast(removed ? `已彻底删除，并清除了由它产生的 ${removed} 条记忆` : '已彻底删除', 'success');
  } catch (error) {
    fail(error);
  }
}

export async function bulkAction(ids, action) {
  try {
    const { count } = await api.post('/api/conversations/bulk', { ids, action });
    if ((action === 'archive' || action === 'delete') && ids.includes(state.currentId)) newChat();
    if (action === 'delete') clearRenderCache();
    await refreshConversations();
    const labels = { archive: '已归档', unarchive: '已恢复', delete: '已删除', pin: '已置顶', unpin: '已取消置顶' };
    toast(`${labels[action]} ${count} 个对话`, 'success');
  } catch (error) {
    fail(error);
  }
}

export function exportConversation(id, format) {
  const link = document.createElement('a');
  link.href = `/api/conversations/${id}/export?format=${format}`;
  link.click();
}

// ---------------------------------------------------------------- 附件
export async function addAttachments(files) {
  for (const file of files) {
    try {
      const attachment = await api.post(`/api/attachments?name=${encodeURIComponent(file.name)}`, file);
      state.draft.attachments = [...state.draft.attachments, attachment];
      if (attachment.truncated) toast(`${attachment.name} 过长，已截断`, 'info');
      scheduleContext(0);
      notify();
    } catch (error) {
      toast(`${file.name}：${error.message}`, 'error', 6000);
    }
  }
}

export function removeAttachment(id) {
  state.draft.attachments = state.draft.attachments.filter((a) => a.id !== id);
  scheduleContext(0);
  notify();
}

// ---------------------------------------------------------------- 设置
export function applyTheme() {
  const ui = state.settings?.ui;
  if (!ui) return;
  const root = document.documentElement;
  if (ui.theme === 'light' || ui.theme === 'dark') root.dataset.theme = ui.theme;
  else delete root.dataset.theme;
  try {
    localStorage.setItem('mlxui-theme', ui.theme);
  } catch {
    // 无痕模式下不可写，下次启动回到跟随系统
  }
  root.style.setProperty('--fs', `${ui.font_size}px`);
}

export async function updateSettings(patch) {
  try {
    state.settings = await api.patch('/api/settings', patch);
    applyTheme();
    notify();
  } catch (error) {
    fail(error);
  }
}

export function setUi(patch) {
  state.settings.ui = { ...state.settings.ui, ...patch };
  applyTheme();
  notify();
  const panelOnly = Object.keys(patch).every((key) => key === 'sidebar_open' || key === 'inspector_open');
  if (panelOnly && window.innerWidth < 1180) return; // 窄窗口的开合不影响桌面布局的记忆
  api.patch('/api/settings', { ui: patch }).catch(fail);
}

export async function refreshPrompts() {
  state.prompts = (await api.get('/api/prompts')).prompts;
  notify();
}
