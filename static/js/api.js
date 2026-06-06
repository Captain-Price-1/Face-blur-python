import { getApiBase } from "./config.js?v=6";

const base = () => getApiBase();

export async function uploadVideo(file, blurAll = false) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("blur_all", blurAll ? "true" : "false");
  const r = await fetch(`${base()}/api/jobs`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
  return r.json();
}

export async function createJobFromUrl(url, blurAll = false) {
  const r = await fetch(`${base()}/api/jobs/from-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, blur_all: blurAll }),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try { detail = (await r.json()).detail || detail; } catch {}
    throw new Error(`could not start from link: ${detail}`);
  }
  return r.json();
}

export async function getJob(jobId) {
  const r = await fetch(`${base()}/api/jobs/${jobId}`);
  return r.json();
}

export async function getPeople(jobId) {
  const r = await fetch(`${base()}/api/jobs/${jobId}/people`);
  const data = await r.json();
  // Absolute-ize thumb URLs so they load when served from a static host.
  for (const p of data.people || []) {
    if (p.thumb_url && p.thumb_url.startsWith("/")) p.thumb_url = base() + p.thumb_url;
  }
  return data;
}

export async function startRender(jobId, blurPersonIds, blurMode = "face") {
  const r = await fetch(`${base()}/api/jobs/${jobId}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blur_person_ids: blurPersonIds, blur_mode: blurMode }),
  });
  if (!r.ok) {
    let detail = `render failed: ${r.status}`;
    try { detail = (await r.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return r.json();
}

export function downloadUrl(jobId) {
  return `${base()}/api/jobs/${jobId}/download`;
}

export function wsUrl(jobId) {
  const b = base();
  if (b) {
    return b.replace(/^http/, "ws") + `/api/jobs/${jobId}/events`;
  }
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/api/jobs/${jobId}/events`;
}

export async function deleteJob(jobId) {
  await fetch(`${base()}/api/jobs/${jobId}`, { method: "DELETE" });
}
