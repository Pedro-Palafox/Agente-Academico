import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, chatStream } from "../api.js";
import Message from "../components/Message.jsx";
import "./chat.css";

const SUGERENCIAS = [
  { t: "Objetivo", q: "¿Cuál es el objetivo principal del survey?" },
  { t: "Taxonomía", q: "¿Qué taxonomía propone el artículo?" },
  { t: "Desafíos", q: "¿Cuáles son los desafíos abiertos?" },
  { t: "Figuras", q: "¿Qué muestra la Figura 1?" },
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
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [online, setOnline] = useState(null); // null = comprobando
  const [globalError, setGlobalError] = useState("");
  const [sidebar, setSidebar] = useState(false);
  const endRef = useRef(null);
  const abortRef = useRef(null);
  const taRef = useRef(null);
  const convRef = useRef(convId || null);

  const cargarConversaciones = useCallback(() => {
    api.conversations().then(setConversations).catch(() => {});
  }, []);

  useEffect(() => {
    cargarConversaciones();
    api
      .documents()
      .then((d) => {
        setDocs(d);
        setOnline(true);
      })
      .catch(() => {
        setOnline(false);
        setGlobalError("No se pudo contactar con el backend.");
      })
      .finally(() => setLoadingDocs(false));
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

  // Textarea que crece con el contenido
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
  }, [input]);

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
      setOnline(true);
    } catch (e) {
      if (e.name !== "AbortError")
        actualizarUltimo((m) => ({ ...m, error: e.message || "Error de red", streaming: false, status: null }));
      else actualizarUltimo((m) => ({ ...m, error: "Cancelado", streaming: false, status: null }));
    } finally {
      setBusy(false);
      abortRef.current = null;
      cargarConversaciones();
      taRef.current?.focus();
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

  const docActual = docs.find((d) => d.id === docId);
  const convActual = conversations.find((c) => c.id === convId);
  const estado = online === null ? "checking" : online ? "on" : "off";
  const estadoTxt = { checking: "Conectando…", on: "En línea", off: "Sin conexión" }[estado];

  return (
    <div className="cx">
      {sidebar && <div className="cx-scrim" onClick={() => setSidebar(false)} />}

      {/* ---------- Barra lateral ---------- */}
      <aside className={"cx-side" + (sidebar ? " open" : "")}>
        <div className="cx-brand">
          <div className="cx-logo">AA</div>
          <div>
            <div className="cx-brand-name">Agente Académico</div>
            <div className="cx-brand-sub">RAG sobre artículos Q1</div>
          </div>
        </div>

        <button className="cx-new" onClick={nueva}>
          <span>＋</span> Nueva conversación
        </button>

        <div className="cx-section-label">Historial</div>
        <nav className="cx-convs">
          {conversations.length === 0 && <div className="cx-empty-small">Aún no hay conversaciones</div>}
          {conversations.map((c) => (
            <Link
              key={c.id}
              to={`/c/${c.id}`}
              className={"cx-conv" + (c.id === convId ? " active" : "")}
              onClick={() => setSidebar(false)}
              title={c.title}
            >
              <span className="cx-conv-dot" />
              <span className="cx-conv-name">{c.title}</span>
              <button className="cx-conv-del" onClick={(e) => borrar(c.id, e)} title="Borrar" aria-label="Borrar">
                ×
              </button>
            </Link>
          ))}
        </nav>

        <div className="cx-section-label">Artículos indexados</div>
        <div className="cx-docs">
          {loadingDocs && <div className="cx-skel" />}
          {!loadingDocs && docs.length === 0 && <div className="cx-empty-small">Ninguno todavía</div>}
          {docs.map((d) => (
            <button
              key={d.id}
              className={"cx-doc" + (docId === d.id ? " active" : "")}
              onClick={() => setDocId(docId === d.id ? "" : d.id)}
              title={d.title}
            >
              <div className="cx-doc-title">{d.title}</div>
              <div className="cx-doc-stats">
                <span>{d.num_pages ?? "?"} págs</span>
                <span>{d.num_chunks} chunks</span>
                <span>{d.num_tables} tablas</span>
                <span>{d.num_figures} figs</span>
              </div>
            </button>
          ))}
        </div>

        <Link to="/admin" className="cx-admin">
          Panel de administración <span>→</span>
        </Link>
      </aside>

      {/* ---------- Zona principal ---------- */}
      <main className="cx-main">
        <header className="cx-head">
          <button className="cx-icon-btn cx-menu" onClick={() => setSidebar(!sidebar)} aria-label="Menú">
            ☰
          </button>
          <div className="cx-head-text">
            <div className="cx-head-title">{convActual?.title || "Pregúntale al artículo"}</div>
            <div className="cx-head-sub">Respuestas con cita de sección y página · si no está en el artículo, te lo dice</div>
          </div>
          <div className="cx-head-right">
            {docs.length > 0 && (
              <select className="cx-select" value={docId} onChange={(e) => setDocId(e.target.value)} title="Artículo a consultar">
                <option value="">Todos los artículos</option>
                {docs.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.title}
                  </option>
                ))}
              </select>
            )}
            <span className={"cx-status " + estado}>
              <i />
              {estadoTxt}
            </span>
          </div>
        </header>

        {globalError && (
          <div className="cx-alert err">
            <span>⚠ {globalError}</span>
            <button onClick={() => setGlobalError("")} aria-label="Cerrar">
              ×
            </button>
          </div>
        )}
        {!loadingDocs && docs.length === 0 && online && (
          <div className="cx-alert warn">
            No hay artículos indexados todavía. Copia el PDF en <code>data/papers/</code> y ejecuta{" "}
            <code>docker compose run --rm ingest</code>.
          </div>
        )}

        <div className="cx-scroll">
          <div className="cx-thread">
            {loadingConv && (
              <div className="cx-loading">
                <div className="cx-skel w60" />
                <div className="cx-skel w80 right" />
                <div className="cx-skel w70" />
              </div>
            )}

            {!loadingConv && messages.length === 0 && (
              <div className="cx-hero">
                <div className="cx-hero-badge">Survey Q1 · RAG híbrido</div>
                <h1>
                  ¿Qué quieres saber <span>del artículo</span>?
                </h1>
                <p>
                  {docActual
                    ? `Consultando: ${docActual.title}`
                    : docs.length > 1
                    ? `Buscando en ${docs.length} artículos`
                    : "Haz una pregunta o elige una sugerencia"}
                </p>
                <div className="cx-sugs">
                  {SUGERENCIAS.map((s) => (
                    <button key={s.q} className="cx-sug" onClick={() => enviar(s.q)} disabled={busy}>
                      <span className="cx-sug-tag">{s.t}</span>
                      <span className="cx-sug-q">{s.q}</span>
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
        </div>

        <div className="cx-composer-wrap">
          <form
            className={"cx-composer" + (busy ? " busy" : "")}
            onSubmit={(e) => {
              e.preventDefault();
              enviar();
            }}
          >
            <textarea
              ref={taRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  enviar();
                }
              }}
              placeholder={busy ? "Generando respuesta…" : "Escribe tu pregunta sobre el artículo…"}
              rows={1}
              disabled={busy}
            />
            {busy ? (
              <button type="button" className="cx-send stop" onClick={() => abortRef.current?.abort()} title="Detener">
                ■
              </button>
            ) : (
              <button type="submit" className="cx-send" disabled={!input.trim()} title="Enviar">
                ↑
              </button>
            )}
          </form>
          <div className="cx-hint">
            <kbd>Enter</kbd> enviar · <kbd>Shift</kbd>+<kbd>Enter</kbd> nueva línea
          </div>
        </div>
      </main>
    </div>
  );
}
