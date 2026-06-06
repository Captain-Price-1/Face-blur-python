// Shared UI helpers: theme toggle + toast. Imported by every page.
import { mountShowcaseBanner } from "./config.js?v=6";

const SUN = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>`;
const MOON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 109.8 9.8z"/></svg>`;

function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  const btn = document.querySelector(".theme-toggle");
  if (btn) btn.innerHTML = t === "light" ? MOON : SUN;
}

export function initShell() {
  const saved = localStorage.getItem("faceblur_theme") || "dark";
  applyTheme(saved);
  const btn = document.querySelector(".theme-toggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      localStorage.setItem("faceblur_theme", next);
      applyTheme(next);
    });
  }
  mountShowcaseBanner();
}

let toastTimer = null;
export function toast(msg, isError = false) {
  let el = document.querySelector(".toast");
  if (!el) {
    el = document.createElement("div");
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.toggle("err", isError);
  requestAnimationFrame(() => el.classList.add("show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 4200);
}
