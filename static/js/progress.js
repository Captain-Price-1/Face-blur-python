import { getJob, wsUrl } from "./api.js?v=6";

/**
 * Drive a job's UI updates.
 *
 * HTTP polling every second is the GUARANTEED baseline — it always advances
 * the UI while the server is reachable. The WebSocket is a best-effort
 * enhancement for smoother, lower-latency progress. If the WS never connects,
 * errors, or goes silent, polling still drives everything, so the page can
 * never get stuck. `onEvent` may fire from both transports — keep it idempotent.
 */
export function subscribe(jobId, onEvent) {
  let stopped = false;
  let pollTimer = null;
  let ws = null;

  function stop() {
    stopped = true;
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    try { if (ws) ws.close(); } catch {}
  }

  async function poll() {
    if (stopped) return;
    try {
      const state = await getJob(jobId);
      onEvent({ phase: state.status, progress: state.progress, message: state.error });
      if (state.status === "done" || state.status === "error") stop();
    } catch { /* transient — retry next tick */ }
  }

  pollTimer = setInterval(poll, 1000);
  poll();

  try {
    ws = new WebSocket(wsUrl(jobId));
    ws.onmessage = (e) => {
      if (stopped) return;
      let ev;
      try { ev = JSON.parse(e.data); } catch { return; }
      onEvent(ev);
      if (ev.phase === "done" || ev.phase === "error") stop();
    };
  } catch { /* WS unavailable — polling covers us */ }

  return { stop };
}
