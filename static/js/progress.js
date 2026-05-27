import { getJob } from "./api.js";

/**
 * Drive a job's UI updates.
 *
 * Reliability model: HTTP polling every second is the GUARANTEED baseline —
 * it always advances the UI as long as the server is reachable. The WebSocket
 * is a best-effort enhancement layered on top for smoother, lower-latency
 * progress. If the WebSocket never connects, errors, or connects but goes
 * silent (which happens with some browsers/proxies), polling still drives
 * everything, so the page can never get stuck on a spinner.
 *
 * `onEvent` may therefore be called from both transports — handlers must be
 * idempotent.
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
    } catch {
      /* transient network error — next tick will retry */
    }
  }

  // Guaranteed baseline.
  pollTimer = setInterval(poll, 1000);
  poll();

  // Best-effort enhancement.
  try {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/api/jobs/${jobId}/events`);
    ws.onmessage = (e) => {
      if (stopped) return;
      let ev;
      try { ev = JSON.parse(e.data); } catch { return; }
      onEvent(ev);
      if (ev.phase === "done" || ev.phase === "error") stop();
    };
  } catch {
    /* WebSocket unavailable — polling already covers us */
  }

  return { stop };
}
