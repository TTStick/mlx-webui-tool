async function request(method, url, body) {
  const options = { method, headers: {} };
  if (body instanceof Blob) {
    options.headers['Content-Type'] = 'application/octet-stream';
    options.body = body;
  } else if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error('无法连接本地服务，请确认它还在运行');
  }
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (response.status === 423) window.dispatchEvent(new Event('mlxui-locked'));
  if (!response.ok) {
    const error = new Error(typeof data?.detail === 'string' ? data.detail : `请求失败（${response.status}）`);
    error.status = response.status;
    error.code = data?.code;
    throw error;
  }
  return data;
}

export const api = {
  get: (url) => request('GET', url),
  post: (url, body) => request('POST', url, body ?? {}),
  patch: (url, body) => request('PATCH', url, body),
  del: (url) => request('DELETE', url),
};
