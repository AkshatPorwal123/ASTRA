// Manages the operator's API key for this browser session. Only relevant
// when the backend has ASTRA_API_KEY set (see /api/config); if auth is
// disabled, every function here is a harmless no-op.
const STORAGE_KEY = "astra_api_key";

export function getStoredApiKey() {
  if (typeof window === "undefined") return "";
  try {
    return sessionStorage.getItem(STORAGE_KEY) || "";
  } catch {
    return "";   // sessionStorage can throw in some locked-down browser contexts
  }
}

export function setStoredApiKey(key) {
  if (typeof window === "undefined") return;
  try {
    sessionStorage.setItem(STORAGE_KEY, key);
  } catch {
    // best-effort — worst case the operator re-enters the key next reload
  }
}

export function authHeaders() {
  const key = getStoredApiKey();
  return key ? { "X-API-Key": key } : {};
}

export function withApiKeyParam(url) {
  const key = getStoredApiKey();
  if (!key) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}api_key=${encodeURIComponent(key)}`;
}
