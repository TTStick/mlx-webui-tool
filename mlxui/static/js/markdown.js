import { Marked } from '/static/vendor/marked.js';
import DOMPurify from '/static/vendor/purify.js';
import hljs from '/static/vendor/hljs/highlight.js';
import katex from '/static/vendor/katex/katex.js';

const escapeHtml = (text) =>
  text.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function renderMath(source, displayMode) {
  try {
    return katex.renderToString(source, { displayMode, throwOnError: false, strict: false, output: 'htmlAndMathml' });
  } catch {
    return `<code>${escapeHtml(source)}</code>`;
  }
}

const marked = new Marked({
  gfm: true,
  breaks: true,
  renderer: {
    code({ text, lang }) {
      const language = (lang || '').trim().split(/\s+/)[0];
      const body = language && hljs.getLanguage(language)
        ? hljs.highlight(text, { language, ignoreIllegals: true }).value
        : escapeHtml(text);
      return `<div class="code-block"><div class="code-head"><span>${escapeHtml(language || 'text')}</span>`
        + `<button type="button" class="code-copy" data-copy>复制</button></div>`
        + `<pre><code class="hljs">${body}</code></pre></div>`;
    },
  },
  extensions: [
    {
      name: 'blockMath',
      level: 'block',
      start(src) {
        const match = src.match(/(?:^|\n) {0,3}(?:\$\$|\\\[)/);
        return match ? match.index : undefined;
      },
      tokenizer(src) {
        const match = /^ {0,3}(?:\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]) *(?:\n|$)/.exec(src);
        if (match) return { type: 'blockMath', raw: match[0], text: (match[1] ?? match[2]).trim() };
        return undefined;
      },
      renderer: (token) => `<div class="math-block">${renderMath(token.text, true)}</div>`,
    },
    {
      name: 'inlineMath',
      level: 'inline',
      start(src) {
        const position = src.search(/\$|\\\(|\\\[/);
        return position >= 0 ? position : undefined;
      },
      tokenizer(src) {
        const match = /^\\\(([\s\S]+?)\\\)/.exec(src)
          || /^\\\[([\s\S]+?)\\\]/.exec(src)
          || /^\$\$([^\n]+?)\$\$/.exec(src)
          // 单个 $ 容易是价格：要求紧贴内容，且结束符后面不是数字
          || /^\$(?![\s$])((?:\\.|[^$\n\\])+?)(?<![\s\\])\$(?![\d$])/.exec(src);
        if (match) return { type: 'inlineMath', raw: match[0], text: match[1].trim() };
        return undefined;
      },
      renderer: (token) => renderMath(token.text, false),
    },
  ],
});

DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    node.setAttribute('target', '_blank');
    node.setAttribute('rel', 'noopener noreferrer');
  }
  if (node.tagName === 'INPUT') node.setAttribute('disabled', '');
});

// 模型输出不可信：禁止外链资源，否则一张图片就能把内容带出本机
const SANITIZE = {
  ADD_ATTR: ['target'],
  FORBID_TAGS: ['img', 'style', 'form', 'iframe', 'video', 'audio', 'source', 'link', 'meta', 'base', 'object', 'embed'],
};

const cache = new Map();
const CACHE_LIMIT = 300;

export function clearRenderCache() {
  cache.clear();
}

export function renderMarkdown(text) {
  const hit = cache.get(text);
  if (hit !== undefined) return hit;
  let output;
  try {
    output = DOMPurify.sanitize(marked.parse(text), SANITIZE);
  } catch {
    output = `<p>${escapeHtml(text)}</p>`;
  }
  cache.set(text, output);
  if (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
  return output;
}
