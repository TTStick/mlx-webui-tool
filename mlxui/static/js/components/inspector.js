import { html, useRef, useState } from '../lib.js';
import { bytes, number, percent, rate, seconds } from '../format.js';
import {
  clearKvCache, compactContext, currentModelId, effectiveParams, fetchPromptPreview, kvBytesPerToken, modelInfo,
  resetParams, saveParamsAsDefault, setParam, setUi, state, toast, unloadModel,
} from '../store.js';
import { hideTip, Icon, IconButton, Modal, showTip, Spinner, Toggle } from './ui.js';

const CONTEXT_PARTS = [
  ['system', '系统提示', 'var(--s1)'],
  ['summary', '压缩摘要', 'var(--s2)'],
  ['attachments', '附件', 'var(--s3)'],
  ['history', '历史消息', 'var(--s4)'],
  ['current', '当前消息', 'var(--s5)'],
  ['template', '模板开销', 'var(--s6)'],
  ['memory', '长期记忆', 'var(--s7)'],
  ['search', '联网资料', 'var(--s8)'],
];

function SegmentBar({ parts, total, label, unit }) {
  const description = parts.map((p) => `${p.name} ${unit(p.value)}`).join('，');
  return html`<div class="segbar" role="img" aria-label=${`${label}：${description}`}>
    ${parts.filter((p) => p.value > 0).map((part) => {
      const share = percent(part.value, total);
      return html`<div key=${part.name} class="seg ${part.muted ? 'muted' : ''}"
        style=${{ width: `max(3px, ${share}%)`, background: part.color }}
        data-tip=${`${part.name}：${unit(part.value)}（${share.toFixed(1)}%）`}></div>`;
    })}
  </div>`;
}

function Legend({ parts, unit }) {
  return html`<dl class="legend">
    ${parts.map((part) => html`<div key=${part.name}>
      <dt><i style=${{ background: part.color }}></i>${part.name}</dt>
      <dd class="num">${unit(part.value)}</dd>
    </div>`)}
  </dl>`;
}

function Sparkline({ values, format, label }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null);
  const width = 288;
  const height = 44;
  if (values.length < 2) return html`<div class="spark-empty">生成时显示实时曲线</div>`;
  const maximum = Math.max(...values) * 1.1 || 1;
  const x = (i) => (i / (values.length - 1)) * (width - 6) + 1;
  const y = (v) => height - 4 - (v / maximum) * (height - 10);
  const line = values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join('');
  const area = `${line}L${x(values.length - 1).toFixed(1)} ${height}L${x(0).toFixed(1)} ${height}Z`;
  const last = values.length - 1;

  const onMove = (event) => {
    const box = ref.current.getBoundingClientRect();
    const ratio = Math.max(0, Math.min((event.clientX - box.left) / box.width, 1));
    const index = Math.round(ratio * last);
    setHover(index);
    showTip(`${label} ${format(values[index])}`, event.clientX, box.top);
  };
  const onLeave = () => {
    setHover(null);
    hideTip();
  };

  return html`<svg ref=${ref} class="spark" viewBox=${`0 0 ${width} ${height}`} preserveAspectRatio="none"
    role="img" aria-label=${`${label}曲线，最新 ${format(values[last])}`} onPointerMove=${onMove} onPointerLeave=${onLeave}>
    <path d=${area} class="spark-area" />
    <path d=${line} class="spark-line" vector-effect="non-scaling-stroke" />
    ${hover != null && html`<line x1=${x(hover)} x2=${x(hover)} y1="0" y2=${height} class="spark-cross" vector-effect="non-scaling-stroke" />`}
  </svg>`;
}

function Row({ label, value, tip }) {
  return html`<div class="kv" data-tip=${tip}><span>${label}</span><span class="num">${value}</span></div>`;
}

function PromptModal({ onClose }) {
  const [prompt, setPrompt] = useState(null);
  if (prompt === null) {
    fetchPromptPreview().then((r) => setPrompt(r.prompt || '')).catch((e) => { toast(e.message, 'error'); onClose(); });
  }
  return html`<${Modal} title="实际发送给模型的 Prompt" width=${760} onClose=${onClose}>
    <div class="modal-scroll pad">
      ${prompt === null ? html`<${Spinner} />` : prompt
        ? html`<pre class="prompt-view">${prompt}</pre>`
        : html`<p class="muted">当前没有可预览的内容。先选择模型并输入一条消息。</p>`}
    </div>
  <//>`;
}

function ContextSection() {
  const [viewing, setViewing] = useState(false);
  const context = state.context;
  const params = effectiveParams();
  const model = modelInfo(currentModelId());
  if (!context) return html`<section class="ins-section"><h3>上下文</h3><div class="muted">统计中…</div></section>`;

  const parts = CONTEXT_PARTS.map(([key, name, color]) => ({ name, color, value: context.breakdown[key] || 0 }));
  const barParts = [...parts, { name: '预留回复', color: 'var(--seg-reserved)', value: context.reserve, muted: true }];
  const share = percent(context.total, context.limit);
  const perToken = kvBytesPerToken(model);
  const token = (v) => `${number(v)} tokens`;
  const canCompact = state.currentId && state.messages.filter((m) => m.role === 'user').length >= 2 && !state.job;

  return html`<section class="ins-section">
    <h3>上下文${context.estimated && html`<span class="tag">估算</span>`}</h3>
    <div class="headline">
      <div class="headline-value num">${number(context.total)}<small> / ${number(context.limit)}</small></div>
      <div class="headline-share num">${share.toFixed(share < 10 ? 1 : 0)}%</div>
    </div>
    <${SegmentBar} parts=${barParts} total=${context.limit} label="上下文组成" unit=${token} />
    <${Legend} parts=${parts.filter((p) => p.value > 0)} unit=${number} />
    <div class="kv-group">
      <${Row} label="预留回复" value=${number(context.reserve)} tip="发送时为模型的回答保留的 token" />
      <${Row} label="本次最多生成" value=${number(context.max_reply)} />
      <${Row} label="进入上下文的消息" value=${`${context.included.length} 条`} />
      ${context.excluded.length > 0 && html`<${Row} label="已被截断" value=${`${context.excluded.length} 条`} tip="最早的消息放不进窗口，发送时跳过" />`}
      ${perToken && html`<${Row} label="窗口占满时的 KV 缓存" value=${`约 ${bytes(perToken * params.context_length)}`} tip=${`每 token 约 ${bytes(perToken)}`} />`}
    </div>
    <div class="ins-actions">
      <button type="button" class="btn small" disabled=${!canCompact || state.compacting} onClick=${compactContext}
        data-tip="让模型把较早的对话压缩成摘要，释放上下文空间">
        ${state.compacting ? html`<${Spinner} size=${12} />` : html`<${Icon} name="compress" size=${14} />`}压缩上下文
      </button>
      <button type="button" class="btn small" onClick=${() => setViewing(true)}><${Icon} name="eye" size=${14} />查看 Prompt</button>
    </div>
    ${viewing && html`<${PromptModal} onClose=${() => setViewing(false)} />`}
  </section>`;
}

const PRESSURE = {
  normal: { label: '正常', icon: 'check', cls: 'good' },
  warning: { label: '偏高', icon: 'alert', cls: 'warn' },
  critical: { label: '严重', icon: 'alert', cls: 'critical' },
};

function MemorySection() {
  const monitor = state.monitor;
  if (!monitor) return null;
  const { mlx, engine, system, gpu } = monitor;
  const other = Math.max(mlx.active - engine.weights_bytes - engine.kv_bytes, 0);
  const parts = [
    { name: '模型权重', color: 'var(--s1)', value: engine.weights_bytes },
    { name: 'KV 缓存', color: 'var(--s2)', value: engine.kv_bytes },
    { name: '其他 MLX 占用', color: 'var(--s3)', value: other },
  ];
  const pressure = PRESSURE[system.pressure] || PRESSURE.normal;
  const systemShare = percent(system.used, system.total);
  const model = engine.model_id ? modelInfo(engine.model_id) : null;

  return html`<section class="ins-section">
    <h3>显存</h3>
    <div class="headline">
      <div class="headline-value num">${bytes(mlx.active)}<small> / ${bytes(mlx.working_set_limit)}</small></div>
      <div class="headline-share num">${percent(mlx.active, mlx.working_set_limit).toFixed(0)}%</div>
    </div>
    <${SegmentBar} parts=${parts} total=${mlx.working_set_limit} label="显存组成" unit=${bytes} />
    <${Legend} parts=${parts} unit=${bytes} />
    <div class="kv-group">
      ${engine.model_id
        ? html`<${Row} label="已加载" value=${model?.name || engine.model_id.split('/').pop()} tip=${`加载耗时 ${seconds(engine.load_seconds)}`} />`
        : html`<${Row} label="已加载" value="无" />`}
      ${engine.state === 'reflecting' && html`<${Row} label="后台" value="正在反思…" tip="空闲时整理记忆，你发消息时会立刻让路" />`}
      <${Row} label="KV 缓存对话数" value=${engine.kv_entries.length} tip="保留每个对话的缓存，追问时不必重算" />
      <${Row} label="MLX 峰值" value=${bytes(mlx.peak)} />
      <${Row} label="MLX 缓冲池" value=${bytes(mlx.cache)} tip="已释放、可被 MLX 立刻复用的显存" />
    </div>
    <div class="ins-actions">
      <button type="button" class="btn small" disabled=${!engine.model_id || engine.state !== 'idle'} onClick=${unloadModel}>卸载模型</button>
      <button type="button" class="btn small" disabled=${!engine.kv_entries.length || engine.state !== 'idle'} onClick=${clearKvCache}>清空 KV 缓存</button>
    </div>

    <h3 class="sub">系统</h3>
    <div class="meter-row">
      <div class="kv"><span>统一内存</span><span class="num">${bytes(system.used)} / ${bytes(system.total)}</span></div>
      <div class="meter"><div style=${{ width: `${systemShare}%` }} class=${system.pressure !== 'normal' ? 'hot' : ''}></div></div>
    </div>
    <div class="kv-group">
      <div class="kv"><span>内存压力</span><span class="status ${pressure.cls}"><${Icon} name=${pressure.icon} size=${13} />${pressure.label}</span></div>
      <${Row} label="交换区" value=${bytes(system.swap_used)} tip="持续增长说明内存已不够用，速度会明显下降" />
      <${Row} label="GPU 利用率" value=${gpu.utilization != null ? `${gpu.utilization}%` : '—'} />
      <${Row} label="CPU 利用率" value=${`${system.cpu_percent.toFixed(0)}%`} />
    </div>
  </section>`;
}

function Tile({ label, value, tip }) {
  return html`<div class="tile" data-tip=${tip}><div class="tile-label">${label}</div><div class="tile-value">${value}</div></div>`;
}

function PerformanceSection() {
  const live = state.live;
  const stats = live.stats;
  const generating = !!state.job && live.tokens > 0;
  const speed = generating ? live.tps : stats?.generation_tps;
  const tokens = generating ? live.tokens : stats?.completion_tokens;
  const ttft = generating ? live.ttft : stats?.ttft;
  if (!stats && !generating) {
    return html`<section class="ins-section"><h3>本次生成</h3><div class="muted">发送消息后显示速度与耗时</div></section>`;
  }
  const hit = stats && stats.prompt_tokens ? percent(stats.cached_tokens, stats.prompt_tokens) : null;
  return html`<section class="ins-section">
    <h3>本次生成${generating && html`<span class="tag live">进行中</span>`}</h3>
    <div class="tiles">
      <${Tile} label="生成速度" value=${rate(speed)} />
      <${Tile} label="首字延迟" value=${seconds(ttft)} tip="从开始处理到输出第一个字的时间" />
      <${Tile} label="回复 tokens" value=${number(tokens)} />
      ${stats && !generating && html`<${Tile} label="缓存命中" value=${`${number(stats.cached_tokens)}（${hit.toFixed(0)}%）`} tip="上一轮算过的部分不用重算" />`}
      ${stats && !generating && html`<${Tile} label="Prompt 速度" value=${rate(stats.prompt_tps)} />`}
      ${stats && !generating && html`<${Tile} label="峰值显存" value=${bytes(stats.peak_memory)} />`}
    </div>
    <h3 class="sub">生成速度曲线</h3>
    <${Sparkline} values=${live.tpsHistory} format=${rate} label="生成速度" />
  </section>`;
}

function Slider({ label, name, min, max, step, tip }) {
  const value = effectiveParams()[name];
  return html`<label class="param" data-tip=${tip}>
    <div class="param-head">
      <span>${label}</span>
      <input type="number" class="num" value=${value} min=${min} max=${max} step=${step} aria-label=${label}
        onChange=${(e) => setParam(name, Math.min(Math.max(parseFloat(e.target.value) || 0, min), max))} />
    </div>
    <input type="range" min=${min} max=${max} step=${step} value=${value} aria-label=${`${label}滑块`}
      onInput=${(e) => setParam(name, parseFloat(e.target.value))} />
  </label>`;
}

function ParamsPanel() {
  const params = effectiveParams();
  const model = modelInfo(currentModelId());
  const maxContext = model?.max_context || 262144;
  return html`<div class="params">
    <section class="ins-section">
      <div class="h3-row">
        <h3>系统提示</h3>
        ${state.prompts.length > 0 && html`<select class="preset" aria-label="选择预设提示词" value=""
          onChange=${(e) => {
            const prompt = state.prompts.find((p) => p.id === e.target.value);
            if (prompt) setParam('system_prompt', prompt.content);
            e.target.value = '';
          }}>
          <option value="">使用预设…</option>
          ${state.prompts.map((p) => html`<option key=${p.id} value=${p.id}>${p.name}</option>`)}
        </select>`}
      </div>
      <textarea class="sys-prompt" rows="4" placeholder="给模型设定角色和规则（可留空）" value=${params.system_prompt}
        aria-label="系统提示" onInput=${(e) => setParam('system_prompt', e.target.value)}></textarea>
    </section>

    <section class="ins-section">
      <h3>长度</h3>
      <${Slider} label="上下文窗口" name="context_length" min=${256} max=${Math.min(maxContext, 262144)} step=${256} tip="Prompt 和回复加起来的 token 上限" />
      <${Slider} label="回复上限" name="max_tokens" min=${16} max=${Math.min(params.context_length, 32768)} step=${16} tip="单次回复最多生成多少 token" />
    </section>

    <section class="ins-section">
      <h3>采样</h3>
      <${Slider} label="Temperature" name="temperature" min=${0} max=${2} step=${0.05} tip="越高越有创造性，0 表示每次都选概率最高的词" />
      <${Slider} label="Top P" name="top_p" min=${0} max=${1} step=${0.01} tip="只在累计概率前 P 的候选里采样" />
      <${Slider} label="Top K" name="top_k" min=${0} max=${200} step=${1} tip="只在概率最高的 K 个词里采样，0 表示不限制" />
      <${Slider} label="Min P" name="min_p" min=${0} max=${1} step=${0.01} tip="过滤概率低于最高概率 P 倍的候选" />
      <${Slider} label="重复惩罚" name="repetition_penalty" min=${1} max=${1.5} step=${0.01} tip="大于 1 时降低重复用词，1 表示关闭" />
    </section>

    <section class="ins-section">
      <h3>其他</h3>
      ${model?.thinking_toggle && html`<div class="param-line"><span>思考模式</span>
        <${Toggle} checked=${!!params.thinking} label="思考模式" onChange=${(v) => setParam('thinking', v)} /></div>`}
      <div class="param-line" data-tip="关闭后，这段对话不读取也不写入长期记忆"><span>长期记忆</span>
        <${Toggle} checked=${!!params.memory} label="长期记忆" onChange=${(v) => setParam('memory', v)} /></div>
      <div class="param-line" data-tip="开启后，检索词会发送给你选择的搜索服务"><span>联网搜索</span>
        <${Toggle} checked=${!!params.web_search} label="联网搜索" onChange=${(v) => setParam('web_search', v)} /></div>
      <label class="param">
        <div class="param-head"><span>随机种子</span></div>
        <input class="field num" type="number" placeholder="留空表示随机" value=${params.seed ?? ''}
          onChange=${(e) => setParam('seed', e.target.value === '' ? null : parseInt(e.target.value, 10))} />
      </label>
      <label class="param">
        <div class="param-head"><span>停止词（每行一个）</span></div>
        <textarea class="field" rows="2" value=${(params.stop || []).join('\n')}
          onChange=${(e) => setParam('stop', e.target.value.split('\n').filter(Boolean))}></textarea>
      </label>
    </section>

    <div class="ins-actions pad">
      <button type="button" class="btn small" onClick=${resetParams}>恢复默认</button>
      <button type="button" class="btn small" onClick=${saveParamsAsDefault}>存为新对话默认</button>
    </div>
  </div>`;
}

export function Inspector() {
  const [tab, setTab] = useState('status');
  return html`<aside class="inspector" aria-label="上下文与显存">
    <div class="ins-tabs" role="tablist">
      <button type="button" role="tab" aria-selected=${tab === 'status'} class=${tab === 'status' ? 'on' : ''} onClick=${() => setTab('status')}>状态</button>
      <button type="button" role="tab" aria-selected=${tab === 'params'} class=${tab === 'params' ? 'on' : ''} onClick=${() => setTab('params')}>参数</button>
      <div class="grow"></div>
      <${IconButton} icon="x" tip="关闭" onClick=${() => setUi({ inspector_open: false })} />
    </div>
    <div class="ins-scroll">
      ${tab === 'status'
        ? html`<${ContextSection} /><${MemorySection} /><${PerformanceSection} />`
        : html`<${ParamsPanel} />`}
    </div>
  </aside>`;
}
