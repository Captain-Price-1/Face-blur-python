import { getJob } from "./api.js";

export function subscribe(jobId, onEvent) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${location.host}/api/jobs/${jobId}/events`);
  let pollTimer = null;
  let closed = false;

  function startPolling() {
    pollTimer = setInterval(async () => {
      const state = await getJob(jobId);
      onEvent({ phase: state.status, progress: state.progress });
      if (state.status === "done" || state.status === "error") stop();
    }, 1000);
  }

  function stop() {
    closed = true;
    if (pollTimer) clearInterval(pollTimer);
    try { ws.close(); } catch {}
  }

  ws.onmessage = (e) => onEvent(JSON.parse(e.data));
  ws.onerror = () => { if (!closed) startPolling(); };
  ws.onclose = (e) => {
    if (e.code !== 1000 && !closed && !pollTimer) startPolling();
  };

  return { stop };
}
