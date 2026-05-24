const BASE = "";

export async function uploadVideo(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/api/jobs`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
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
