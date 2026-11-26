// Cliente de la API del backend.

export function getClientId() {
  let id = null;
  try {
    id = localStorage.getItem("client_id");
    if (!id) {
      id = crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2);
      localStorage.setItem("client_id", id);
    }
  } catch {
    id = id || "anon-" + String(Math.random()).slice(2);
  }
  return id;
}

async function json(res) {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const b = await res.json();
      detail = b.detail || detail;
    } catch {}
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export const api = {
  documents: () => fetch("/api/documents").then(json),
  conversations: () => fetch(`/api/conversations?client_id=${getClientId()}`).then(json),
  messages: (id) => fetch(`/api/conversations/${id}/messages?client_id=${getClientId()}`).then(json),
  deleteConversation: (id) =>
    fetch(`/api/conversations/${id}?client_id=${getClientId()}`, { method: "DELETE" }).then(json),
  health: () => fetch("/api/health").then(json),
};

/**
 * Envía una pregunta y consume la respuesta en streaming (Server-Sent Events).
 * handlers: { onMeta, onStatus, onDelta(text, reset), onFinal(resp), onError(msg) }
 */
export async function chatStream(body, handlers, signal) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, client_id: getClientId() }),
    signal,
  });
  if (!res.ok || !res.body) {
    // Fallback sin streaming
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, client_id: getClientId() }),
      signal,
    });
    const data = await json(r);
    handlers.onFinal?.(data);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let terminado = false;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      const payload = JSON.parse(data);
      if (event === "meta") handlers.onMeta?.(payload);
      else if (event === "status") handlers.onStatus?.(payload.message);
      else if (event === "delta") handlers.onDelta?.(payload.text, payload.reset);
      else if (event === "final") {
        terminado = true;
        handlers.onFinal?.(payload);
      } else if (event === "error") {
        terminado = true;
        handlers.onError?.(payload.detail);
      }
    }
  }
  if (!terminado) handlers.onError?.("La conexión se cerró antes de terminar la respuesta.");
}

// ---------------- Admin ----------------
const TOKEN_KEY = "admin_token";
export const adminToken = {
  get: () => {
    try {
      return sessionStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set: (t) => {
    try {
      t ? sessionStorage.setItem(TOKEN_KEY, t) : sessionStorage.removeItem(TOKEN_KEY);
    } catch {}
  },
};

export async function adminLogin(username, password) {
  const r = await fetch("/api/admin/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const d = await json(r);
  adminToken.set(d.token);
  return d;
}

export async function adminGet(path, params = {}) {
  const qs = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== "" && v !== null && v !== undefined)
  ).toString();
  const r = await fetch(`/api/admin/${path}${qs ? "?" + qs : ""}`, {
    headers: { Authorization: `Bearer ${adminToken.get()}` },
  });
  return json(r);
}

export async function adminPost(path) {
  const r = await fetch(`/api/admin/${path}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${adminToken.get()}` },
  });
  return json(r);
}
