import { html, useState } from '../lib.js';
import { retryVault, setupPassword, state, unlock, wipeEverything } from '../store.js';
import { Icon, Spinner } from './ui.js';

function WipeDanger({ label }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState('');
  if (!open) return html`<button type="button" class="link-btn" onClick=${() => setOpen(true)}>${label}</button>`;
  return html`<div class="wipe-box" role="alert">
    <p>所有对话和记忆都会被永久销毁，无法恢复。确定的话，请输入“抹除”。</p>
    <div class="row-end">
      <input class="field" aria-label="输入抹除以确认" value=${text} onInput=${(e) => setText(e.target.value)} />
      <button type="button" class="btn danger" disabled=${text !== '抹除'} onClick=${wipeEverything}>抹除</button>
      <button type="button" class="btn" onClick=${() => { setOpen(false); setText(''); }}>取消</button>
    </div>
  </div>`;
}

function Unlock() {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async (event) => {
    event.preventDefault();
    if (!password || busy) return;
    setBusy(true);
    setError('');
    try {
      await unlock(password);
    } catch (e) {
      setError(e.message);
      setPassword('');
    } finally {
      setBusy(false);
    }
  };
  return html`<form class="vault-card" onSubmit=${submit}>
    <span class="vault-icon"><${Icon} name="lock" size=${22} /></span>
    <h1>已锁定</h1>
    <p>你的对话和记忆已加密，输入启动密码继续。</p>
    <input class="field" type="password" autocomplete="off" aria-label="启动密码" placeholder="启动密码"
      ref=${(el) => el?.focus()} value=${password} onInput=${(e) => setPassword(e.target.value)} />
    ${error && html`<div class="form-error" role="alert">${error}</div>`}
    <button type="submit" class="btn primary large" disabled=${!password || busy}>
      ${busy ? html`<${Spinner} size=${14} />` : '解锁'}
    </button>
    <${WipeDanger} label="忘记密码？只能抹除全部数据后重新开始" />
  </form>`;
}

function Setup() {
  const [first, setFirst] = useState('');
  const [second, setSecond] = useState('');
  const [error, setError] = useState('');
  const minimum = state.vault.min_password;
  const submit = async (event) => {
    event.preventDefault();
    if (first !== second) {
      setError('两次输入的密码不一致');
      return;
    }
    try {
      await setupPassword(first);
    } catch (e) {
      setError(e.message);
    }
  };
  return html`<form class="vault-card" onSubmit=${submit}>
    <span class="vault-icon"><${Icon} name="shield" size=${22} /></span>
    <h1>设置启动密码</h1>
    <p>无法使用系统钥匙串保存密钥，请设置一个密码来加密你的数据。忘记密码后数据无法找回。</p>
    <input class="field" type="password" autocomplete="off" aria-label="密码" placeholder=${`至少 ${minimum} 位`}
      ref=${(el) => el?.focus()} value=${first} onInput=${(e) => setFirst(e.target.value)} />
    <input class="field" type="password" autocomplete="off" aria-label="再次输入密码" placeholder="再输入一次"
      value=${second} onInput=${(e) => setSecond(e.target.value)} />
    ${error && html`<div class="form-error" role="alert">${error}</div>`}
    <button type="submit" class="btn primary large" disabled=${first.length < minimum}>开始使用</button>
  </form>`;
}

function KeyMissing() {
  const [message, setMessage] = useState('');
  const again = async () => {
    setMessage('');
    await retryVault();
    if (state.vault.state === 'key_missing') setMessage('还是读不到密钥');
  };
  return html`<div class="vault-card">
    <span class="vault-icon"><${Icon} name="alert" size=${22} /></span>
    <h1>读不到解密密钥</h1>
    <p>可能是钥匙串还没解锁；也可能这份数据来自别的电脑，或者钥匙串被重置了。没有密钥就无法读取这些数据，先试试重试。</p>
    ${message && html`<div class="form-error" role="alert">${message}</div>`}
    <button type="button" class="btn primary large" onClick=${again}>重试</button>
    <${WipeDanger} label="确认找不回了？抹除这份数据并重新开始" />
  </div>`;
}

export function VaultScreen() {
  const current = state.vault.state;
  return html`<div class="vault-screen">
    ${current === 'locked' && html`<${Unlock} />`}
    ${current === 'setup' && html`<${Setup} />`}
    ${current === 'key_missing' && html`<${KeyMissing} />`}
  </div>`;
}
