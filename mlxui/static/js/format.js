export function bytes(value, digits = 1) {
  if (value == null || Number.isNaN(value)) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = Math.max(value, 0);
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  const places = unit <= 1 ? 0 : digits;
  return `${size.toFixed(places)} ${units[unit]}`;
}

export const number = (value) => (value == null ? '—' : Math.round(value).toLocaleString('en-US'));

export function compact(value) {
  if (value == null) return '—';
  if (value < 1000) return String(Math.round(value));
  if (value < 1e6) {
    const k = value / 1000;
    return `${k >= 100 || Number.isInteger(k) ? Math.round(k) : k.toFixed(1)}K`;
  }
  return `${(value / 1e6).toFixed(1)}M`;
}

export function contextLabel(tokens) {
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}K`;
  return compact(tokens);
}

export const percent = (value, total) => (total > 0 ? Math.min((value / total) * 100, 100) : 0);

export function seconds(value) {
  if (value == null) return '—';
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  if (value < 60) return `${value.toFixed(1)} s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

export const rate = (value) => (value == null ? '—' : `${value.toFixed(1)} t/s`);

export function dateTime(ts) {
  if (!ts) return '';
  const date = new Date(ts * 1000);
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function groupByDate(list) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
  const day = 86400;
  const groups = [
    { key: 'pinned', label: '已置顶', items: [] },
    { key: 'today', label: '今天', items: [] },
    { key: 'yesterday', label: '昨天', items: [] },
    { key: 'week', label: '最近 7 天', items: [] },
    { key: 'month', label: '最近 30 天', items: [] },
    { key: 'older', label: '更早', items: [] },
  ];
  const index = Object.fromEntries(groups.map((g) => [g.key, g]));
  for (const item of list) {
    const t = item.updated_at;
    if (item.pinned) index.pinned.items.push(item);
    else if (t >= today) index.today.items.push(item);
    else if (t >= today - day) index.yesterday.items.push(item);
    else if (t >= today - 7 * day) index.week.items.push(item);
    else if (t >= today - 30 * day) index.month.items.push(item);
    else index.older.items.push(item);
  }
  return groups.filter((g) => g.items.length);
}

/** 拆出推理内容。推理未结束时 open 为 true。 */
export function splitThinking(text) {
  const start = text.indexOf('<think>');
  if (start === -1) return { thinking: null, answer: text, open: false };
  const end = text.indexOf('</think>', start);
  if (end === -1) {
    return { thinking: text.slice(start + 7).trim(), answer: text.slice(0, start), open: true };
  }
  return {
    thinking: text.slice(start + 7, end).trim(),
    answer: (text.slice(0, start) + text.slice(end + 8)).replace(/^\s+/, ''),
    open: false,
  };
}
