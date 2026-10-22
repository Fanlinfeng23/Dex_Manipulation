// Keep connection diagnostics available even when the WebXR / Three.js module fails.
(() => {
  const el = id => document.getElementById(id);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  fetch('/api/health', {cache:'no-store', signal:controller.signal})
    .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
    .then(() => { el('http-status').textContent = '电脑 HTTP 服务：已连接'; })
    .catch(error => { el('http-status').textContent = `电脑 HTTP 服务：无法连接（${error.message}）。检查服务和 USB 转发后刷新。`; })
    .finally(() => clearTimeout(timer));
  const loading = setTimeout(() => {
    el('script-status').textContent = '采集程序加载较慢：检查 USB 转发，刷新本页；请使用更新后的 Quest 自带浏览器。';
  }, 8000);
  import('/static/quest.js').then(() => {
    clearTimeout(loading); el('script-status').textContent = '采集程序：已加载';
  }).catch(error => {
    clearTimeout(loading); el('script-status').textContent = `采集程序加载失败：${error.message}`;
    el('xr-status').textContent = '无法加载 VR 采集程序';
    el('xr-detail').textContent = '请刷新本页并更新 Quest 自带浏览器；若仍失败，请记录上方错误。';
  });
})();
