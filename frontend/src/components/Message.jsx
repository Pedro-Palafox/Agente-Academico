import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function paginas(c) {
  if (c.page_start == null) return "";
  return c.page_end && c.page_end !== c.page_start ? `pp. ${c.page_start}–${c.page_end}` : `p. ${c.page_start}`;
}

const ICONO = { table: "▦", figure: "◩", text: "¶", reference: "❝" };

export function Citations({ citations }) {
  const [open, setOpen] = useState(null);
  if (!citations?.length) return null;
  return (
    <div className="citations">
      <div className="citations-title">Fuentes del artículo</div>
      <div className="citation-chips">
        {citations.map((c) => (
          <button
            key={c.chunk_id}
            className={"chip" + (open === c.chunk_id ? " active" : "")}
            onClick={() => setOpen(open === c.chunk_id ? null : c.chunk_id)}
            title={c.doc_title}
          >
            <span className="chip-n">{c.n}</span>
            <span className="chip-icon">{ICONO[c.chunk_type] || "¶"}</span>
            <span className="chip-sec">{c.section}</span>
            <span className="chip-page">{paginas(c)}</span>
          </button>
        ))}
      </div>
      {citations
        .filter((c) => c.chunk_id === open)
        .map((c) => (
          <div className="citation-detail" key={c.chunk_id}>
            <div className="citation-meta">
              <strong>[{c.n}]</strong> {c.doc_title} · {c.section} · {paginas(c)}
            </div>
            {c.image_url && <img src={c.image_url} alt={c.section} className="citation-img" />}
            <blockquote>{c.snippet}</blockquote>
          </div>
        ))}
    </div>
  );
}

export default function Message({ m, onRetry }) {
  if (m.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{m.content}</div>
      </div>
    );
  }
  return (
    <div className="msg assistant">
      <div className="avatar">📚</div>
      <div className="bubble">
        {m.status && !m.content && (
          <div className="status">
            <span className="dots">
              <i />
              <i />
              <i />
            </span>
            {m.status}
          </div>
        )}
        {m.content && (
          <div className="md">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
            {m.streaming && <span className="cursor">▍</span>}
          </div>
        )}
        {m.status && m.content && m.streaming && <div className="status small">{m.status}</div>}
        {m.error && (
          <div className="error-box">
            <span>⚠️ {m.error}</span>
            {onRetry && (
              <button className="btn small" onClick={onRetry}>
                Reintentar
              </button>
            )}
          </div>
        )}
        {m.answerable === false && <div className="badge warn">No está en el artículo · el agente no inventa</div>}
        <Citations citations={m.citations} />
        {m.metrics && (
          <div className="metrics">
            {m.metrics.cache_hit && <span className="badge ok">⚡ caché Redis</span>}
            {m.confidence && <span>confianza: {m.confidence}</span>}
            <span>{(m.metrics.latency_ms / 1000).toFixed(2)} s</span>
            {!m.metrics.cache_hit && (
              <span>
                {m.metrics.input_tokens} tok entrada · {m.metrics.output_tokens} tok salida
              </span>
            )}
            <span>{m.metrics.model}</span>
          </div>
        )}
      </div>
    </div>
  );
}
