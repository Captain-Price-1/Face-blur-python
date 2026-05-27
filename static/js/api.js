const BASE = "";

export async function uploadVideo(file, blurAll = false) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("blur_all", blurAll ? "true" : "false");
  const r = await fetch(`${BASE}/api/jobs`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
  return r.json();
}

export async function createJobFromUrl(url, blurAll = false) {
  const r = await fetch(`${BASE}/api/jobs/from-url`, {
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
  const r = await fetch(`${BASE}/api/jobs/${jobId}`);
  return r.json();
}

export async function getPeople(jobId) {
  const r = await fetch(`${BASE}/api/jobs/${jobId}/people`);
  return r.json();
}

export async function startRender(jobId, blurPersonIds) {
  const r = await fetch(`${BASE}/api/jobs/${jobId}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blur_person_ids: blurPersonIds }),
  });
  return r.json();
}

export function downloadUrl(jobId) {
  return `${BASE}/api/jobs/${jobId}/download`;
}

export async function deleteJob(jobId) {
  await fetch(`${BASE}/api/jobs/${jobId}`, { method: "DELETE" });
}
