// Resolves which backend the frontend talks to.
//
// - Served by the local FastAPI app  -> same origin ("") works directly.
// - Served from GitHub Pages (a *.github.io static host) -> there is no backend
//   there, so default to a locally-running backend and let the user override it
//   (persisted in localStorage). A banner explains this.

const IS_STATIC_HOST = /github\.io$/.test(location.hostname) || location.protocol === "file:";

export function getApiBase() {
  const saved = localStorage.getItem("faceblur_api_base");
  if (saved !== null) return saved.replace(/\/$/, "");
  return IS_STATIC_HOST ? "http://localhost:8000" : "";
}

export function setApiBase(url) {
  localStorage.setItem("faceblur_api_base", url.replace(/\/$/, ""));
}

export function isStaticHost() {
  return IS_STATIC_HOST;
}

// Inject the "showcase" banner when running on a static host (no backend).
export function mountShowcaseBanner() {
  if (!IS_STATIC_HOST) return;
  const base = getApiBase();
  const el = document.createElement("div");
  el.className = "banner";
  el.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>
    <div><b>UI showcase.</b> Processing runs on a local backend.
    Run it (<code>uvicorn app.main:app</code>) then set the API URL
    <a href="#" id="set-api">here</a> — currently <code>${base}</code>.</div>`;
  document.body.prepend(el);
  el.querySelector("#set-api").addEventListener("click", (e) => {
    e.preventDefault();
    const v = prompt("Backend API base URL", base);
    if (v) { setApiBase(v); location.reload(); }
  });
}
