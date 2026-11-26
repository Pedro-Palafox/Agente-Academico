import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, chatStream } from "../api.js";
import Message from "../components/Message.jsx";

const SUGERENCIAS = [
  "¿Cuál es el objetivo principal del survey?",
  "¿Qué taxonomía propone el artículo?",
  "¿Cuáles son los desafíos abiertos?",
  "¿Qué muestra la Figura 1?",
];

export default function Chat() {
  const { convId } = useParams();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [docs, setDocs] = useState([]);
  const [docId, setDocId] = useState("");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadingConv, setLoadingConv] = useState(false);
  const [globalError, setGlobalError] = useState("");
  const [sidebar, setSidebar] = useState(false);
  const endRef = useRef(null);
  const abortRef = useRef(null);
  const convRef = useRef(convId || null);

  const cargarConversaciones = useCallback(() => {
    api.conversations().then(setConversations).catch(() => {});
  }, []);

  useEffect(() => {
    cargarConversaciones();
    api.documents().then(setDocs).catch(() => setGlobalError("No se pudo contactar con el backend."));
  }, [cargarConversaciones]);

  // Cargar historial cuando cambia la conversación en la URL
  useEffect(() => {
    if (convRef.current === convId && messages.length) return;
    convRef.current = convId || null;
    if (!convId) {
      setMessages([]);
      return;
    }
    setLoadingConv(true);
    api
      .messages(convId)
      .then((ms) => setMessages(ms.map((m) => ({ ...m }))))
      .catch(() => setGlobalError("No se pudo cargar la conversación."))
      .finally(() => setLoadingConv(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [convId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const actualizarUltimo = (fn) =>
    setMessages((ms) => {
      const copia = ms.slice();
      copia[copia.length - 1] = fn(copia[copia.length - 1]);
      return copia;
    });

  async function enviar(texto) {
    const pregunta = (texto ?? input).trim();
    if (!pregunta || busy) return;
    setInput("");
    setGlobalError("");
    setBusy(true);
    setMessages((ms) => [
      ...ms,
      { role: "user", content: pregunta },
      { role: "assistant", content: "", status: "Pensando…", streaming: true, pregunta },
    ]);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await chatStream(
        { message: pregunta, conversation_id: convRef.current, doc_id: docId || null },
        {
          onMeta: (m) => {
            if (!convRef.current) {
              convRef.current = m.conversation_id;
              navigate(`/c/${m.conversation_id}`, { replace: true });
            }
          },
          onStatus: (s) => actualizarUltimo((m) => ({ ...m, status: s })),
          onDelta: (t, reset) => actualizarUltimo((m) => ({ ...m, content: reset ? t : m.content + t })),
          onFinal: (r) =>
            actualizarUltimo((m) => ({
              ...m,
              content: r.answer,
              citations: r.citations,
              answerable: r.answerable,
              confidence: r.confidence,
              metrics: r.metrics,
              streaming: false,
              status: null,
            })),
          onError: (e) => actualizarUltimo((m) => ({ ...m, error: e, streaming: false, status: null })),
        },
        ctrl.signal
      );
    } catch (e) {
      if (e.name !== "AbortError")
        actualizarUltimo((m) => ({ ...m, error: e.message || "Error de red", streaming: false, status: null }));
      else actualizarUltimo((m) => ({ ...m, error: "Cancelado", streaming: false, status: null }));
    } finally {
      setBusy(false);
      abortRef.current = null;
      cargarConversaciones();
    }
  }

  function reintentar() {
    const last = messages[messages.length - 1];
    if (!last?.pregunta) return;
    setMessages((ms) => ms.slice(0, -2));
    enviar(last.pregunta);
  }

  function nueva() {
    convRef.current = null;
    setMessages([]);
    navigate("/");
    setSidebar(false);
  }

  async function borrar(id, e) {
    e.preventDefault();
    e.stopPropagation();
    await api.deleteConversation(id).catch(() => {});
    if (id === convRef.current) nueva();
    cargarConversaciones();
  }

  return (
    <div className="layout">
      <aside className={"sidebar" + (sidebar ? " open" : "")}>
        <div className="brand">
          📚 <span>Agente Académico</span>
        </div>
        <button className="btn primary block" onClick={nueva}>
          + Nueva conversación
        </button>
        {docs.length > 1 && (
          <label className="field">
            <span>Artículo</span>
            <select value={docId} onChange={(e) => setDocId(e.target.value)}>
              <option value="">Todos los artículos</option>
              {docs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.title}
                </option>
              ))}
            </select>
          </label>
        )}
        <div className="conv-list">
          <div className="conv-title">Historial</div>
          {conversations.length === 0 && <div className="muted small">Aún no hay conversaciones</div>}
          {conversations.map((c) => (
            <Link
              key={c.id}
              to={`/c/${c.id}`}
              className={"conv" + (c.id === convId ? " active" : "")}
              onClick={() => setSidebar(false)}
            >
              <span className="conv-name">{c.title}</span>
              <span className="conv-del" onClick={(e) => borrar(c.id, e)} title="Borrar">
                ×
              </span>
            </Link>
          ))}
        </div>
        <div className="sidebar-foot">
          {docs.map((d) => (
            <div key={d.id} className="doc-pill" title={d.title}>
              📄 {d.title}
              <small>
                {d.num_pages ?? "?"} págs · {d.num_chunks} chunks · {d.num_tables} tablas · {d.num_figures} figuras
              </small>
            </div>
          ))}
          <Link to="/admin" className="admin-link">
            Panel /admin →
          </Link>
        </div>
      </aside>

      <main className="chat">
        <header className="chat-header">
          <button className="btn ghost menu" onClick={() => setSidebar(!sidebar)} aria-label="Menú">
            ☰
          </button>
          <div>
            <div className="chat-h-title">Pregúntale al artículo</div>
            <div className="muted small">
              Respuestas fundamentadas con citas de sección y página · si no está en el artículo, te lo dice
            </div>
          </div>
        </header>

        {globalError && <div className="error-box global">⚠️ {globalError}</div>}
        {docs.length === 0 && !globalError && (
          <div className="warn-box">
            No hay artículos indexados todavía. Copia el PDF en <code>data/papers/</code> y ejecuta{" "}
            <code>docker compose run --rm ingest</code>.
          </div>
        )}

        <div className="messages">
          {loadingConv && <div className="muted center">Cargando conversación…</div>}
          {!loadingConv && messages.length === 0 && (
            <div className="empty">
              <h2>¿Qué quieres saber del artículo?</h2>
              <div className="suggestions">
                {SUGERENCIAS.map((s) => (
                  <button key={s} className="suggestion" onClick={() => enviar(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <Message
              key={m.id ?? i}
              m={m}
              onRetry={i === messages.length - 1 && m.error && !busy ? reintentar : null}
            />
          ))}
          <div ref={endRef} />
        </div>

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            enviar();
          }}
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                enviar();
              }
            }}
            placeholder="Escribe tu pregunta sobre el artículo…  (Enter para enviar, Shift+Enter salto de línea)"
            rows={2}
            disabled={busy}
          />
          {busy ? (
            <button type="button" className="btn" onClick={() => abortRef.current?.abort()}>
              Detener
            </button>
          ) : (
            <button type="submit" className="btn primary" disabled={!input.trim()}>
              Enviar
            </button>
          )}
        </form>
      </main>
    </div>
  );
}
