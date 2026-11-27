import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { adminGet, adminLogin, adminPost, adminToken } from "../api.js";
import "./admin.css";

const fmt = (n, d = 0) => (n == null || Number.isNaN(Number(n)) ? "—" : Number(n).toLocaleString("es", { maximumFractionDigits: d }));
const seg = (ms) => (ms == null ? "—" : `${(Number(ms) / 1000).toFixed(2)} s`);
const pct = (x) => (x == null ? "—" : `${(Number(x) * 100).toFixed(1)} %`);

function Login({ onOk }) {
  const [u, setU] = useState("");
  const [p, setP] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <div className="login-wrap">
      <form
        className="card login"
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setErr("");
          try {
            await adminLogin(u, p);
            onOk();
          } catch (e2) {
            setErr(e2.message);
          } finally {
            setBusy(false);
          }
        }}
      >
        <h2>🔒 Panel de administración</h2>
        <p className="muted small">Acceso protegido. Usuario y contraseña definidos en el archivo .env.</p>
        <label className="field">
          <span>Usuario</span>
          <input value={u} onChange={(e) => setU(e.target.value)} autoFocus />
        </label>
        <label className="field">
          <span>Contraseña</span>
          <input type="password" value={p} onChange={(e) => setP(e.target.value)} />
        </label>
        {err && <div className="error-box">{err}</div>}
        <button className="btn primary block" disabled={busy}>
          {busy ? "Entrando…" : "Entrar"}
        </button>
        <Link to="/" className="muted small">
          ← Volver al chat
        </Link>
      </form>
    </div>
  );
}

function Tile({ label, value, sub }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

function ChartCard({ title, children, empty }) {
  return (
    <div className="card chart-card">
      <div className="card-title">{title}</div>
      {empty ? <div className="muted center pad">Sin datos para estos filtros</div> : children}
    </div>
  );
}

const axis = { stroke: "var(--text-secondary)", fontSize: 12, tickLine: false, axisLine: false };
const tooltipStyle = {
  contentStyle: { background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text)" },
  labelStyle: { color: "var(--text-secondary)" },
  cursor: { fill: "var(--hover)" },
};

function Dashboard({ onLogout }) {
  const [tab, setTab] = useState("resumen");
  const [f, setF] = useState({ desde: "", hasta: "", modelo: "", cache_hit: "", respondible: "", q: "" });
  const [data, setData] = useState({});
  const [runs, setRuns] = useState({ total: 0, items: [] });
  const [page, setPage] = useState(0);
  const [detalle, setDetalle] = useState(null);
  const [modelos, setModelos] = useState([]);
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const PAGE = 20;

  const manejar = useCallback(
    (e) => {
      if (e.status === 401) {
        adminToken.set(null);
        onLogout();
      } else setErr(e.message);
    },
    [onLogout]
  );

  const cargar = useCallback(async () => {
    setLoading(true);
    setErr("");
    try {
      const [summary, series, stages, models, cache] = await Promise.all([
        adminGet("summary", f),
        adminGet("timeseries", { ...f, bucket: "day" }),
        adminGet("stages", f),
        adminGet("models", f),
        adminGet("cache"),
      ]);
      setData({ summary, series, stages, models, cache });
    } catch (e) {
      manejar(e);
    } finally {
      setLoading(false);
    }
  }, [f, manejar]);

  const cargarRuns = useCallback(async () => {
    try {
      setRuns(await adminGet("runs", { ...f, limit: PAGE, offset: page * PAGE }));
    } catch (e) {
      manejar(e);
    }
  }, [f, page, manejar]);

  useEffect(() => {
    adminGet("filters").then((d) => setModelos(d.modelos)).catch(manejar);
  }, [manejar]);
  useEffect(() => {
    cargar();
  }, [cargar]);
  useEffect(() => {
    cargarRuns();
  }, [cargarRuns]);
  useEffect(() => {
    if (tab === "eval") adminGet("eval").then(setEvalData).catch(manejar);
  }, [tab, manejar]);

  const s = data.summary || {};
  const series = (data.series || []).map((r) => ({
    ...r,
    dia: new Date(r.t).toLocaleDateString("es", { day: "2-digit", month: "short" }),
    latencia_s: Number(r.latencia_ms) / 1000,
  }));
  const st = data.stages || {};
  const stages = [
    { etapa: "Embedding", ms: st.embedding },
    { etapa: "Búsqueda OpenSearch", ms: st.busqueda },
    { etapa: "Reranking", ms: st.rerank },
    { etapa: "Visión (Qwen-VL)", ms: st.vision },
    { etapa: "LLM (agente)", ms: st.llm },
  ].map((x) => ({ ...x, ms: Number(x.ms || 0) }));
  const models = (data.models || []).map((m) => ({ ...m, nombre: `${m.model.split("/").pop()} · ${m.purpose}` }));
  const set = (k) => (e) => {
    setPage(0);
    setF({ ...f, [k]: e.target.value });
  };

  return (
    <div className="admin">
      <header className="admin-header">
        <div className="brand">
          📊 <span>Panel /admin</span>
        </div>
        <nav className="tabs">
          {[
            ["resumen", "Resumen"],
            ["conversaciones", "Preguntas y respuestas"],
            ["eval", "Evaluación"],
          ].map(([k, l]) => (
            <button key={k} className={"tab" + (tab === k ? " active" : "")} onClick={() => setTab(k)}>
              {l}
            </button>
          ))}
        </nav>
        <div className="row gap">
          <Link to="/" className="btn ghost">
            Chat
          </Link>
          <button
            className="btn ghost"
            onClick={() => {
              adminToken.set(null);
              onLogout();
            }}
          >
            Salir
          </button>
        </div>
      </header>

      {tab !== "eval" && (
        <div className="filters">
          <label className="field">
            <span>Desde</span>
            <input type="date" value={f.desde} onChange={set("desde")} />
          </label>
          <label className="field">
            <span>Hasta</span>
            <input type="date" value={f.hasta} onChange={set("hasta")} />
          </label>
          <label className="field">
            <span>Modelo</span>
            <select value={f.modelo} onChange={set("modelo")}>
              <option value="">Todos</option>
              {modelos.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Caché</span>
            <select value={f.cache_hit} onChange={set("cache_hit")}>
              <option value="">Todas</option>
              <option value="true">Hit</option>
              <option value="false">Miss</option>
            </select>
          </label>
          <label className="field">
            <span>Respondible</span>
            <select value={f.respondible} onChange={set("respondible")}>
              <option value="">Todas</option>
              <option value="true">Sí</option>
              <option value="false">No (abstención)</option>
            </select>
          </label>
          <label className="field grow">
            <span>Buscar texto</span>
            <input value={f.q} onChange={set("q")} placeholder="pregunta o respuesta…" />
          </label>
          <button className="btn" onClick={() => { cargar(); cargarRuns(); }} disabled={loading}>
            {loading ? "Cargando…" : "Actualizar"}
          </button>
        </div>
      )}
      {err && <div className="error-box global">⚠️ {err}</div>}

      {tab === "resumen" && (
        <>
          <div className="tiles">
            <Tile label="Consultas" value={fmt(s.total_runs)} sub={`${fmt(s.conversaciones)} conversaciones · ${fmt(s.mensajes)} mensajes`} />
            <Tile label="Tokens de entrada" value={fmt(s.input_tokens)} sub={`${fmt(s.avg_input_tokens)} por consulta`} />
            <Tile label="Tokens de salida" value={fmt(s.output_tokens)} sub={`${fmt(s.avg_output_tokens)} por consulta`} />
            <Tile label="Tiempo promedio" value={seg(s.avg_latency_ms)} sub={`p50 ${seg(s.p50_latency_ms)} · p95 ${seg(s.p95_latency_ms)}`} />
            <Tile label="Tasa de caché (Redis)" value={pct(s.cache_hit_rate)} sub={`hit ${seg(s.avg_latency_cache_hit_ms)} vs miss ${seg(s.avg_latency_cache_miss_ms)}`} />
            <Tile label="Abstenciones" value={pct(s.tasa_no_respondible)} sub={`${fmt(s.errores)} errores · ${fmt(s.avg_tool_calls, 1)} tools/consulta`} />
          </div>

          <div className="grid2">
            <ChartCard title="Consultas por día" empty={!series.length}>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={series} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="dia" {...axis} />
                  <YAxis allowDecimals={false} {...axis} width={40} />
                  <Tooltip {...tooltipStyle} cursor={{ stroke: "var(--text-secondary)", strokeDasharray: "3 3" }} />
                  <Line type="monotone" dataKey="consultas" name="Consultas" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard title="Tiempo promedio de ejecución por día (s)" empty={!series.length}>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={series} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="dia" {...axis} />
                  <YAxis {...axis} width={40} />
                  <Tooltip {...tooltipStyle} formatter={(v) => `${Number(v).toFixed(2)} s`} cursor={{ stroke: "var(--text-secondary)", strokeDasharray: "3 3" }} />
                  <Line type="monotone" dataKey="latencia_s" name="Latencia" stroke="var(--series-2)" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard title="Latencia promedio por etapa (ms, sin caché)" empty={!st.total}>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={stages} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 0 }}>
                  <CartesianGrid stroke="var(--grid)" horizontal={false} />
                  <XAxis type="number" {...axis} />
                  <YAxis type="category" dataKey="etapa" {...axis} width={140} />
                  <Tooltip {...tooltipStyle} formatter={(v) => `${fmt(v, 1)} ms`} />
                  <Bar dataKey="ms" name="ms" fill="var(--series-1)" radius={[0, 4, 4, 0]} barSize={18} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard title="Uso por modelo (llamadas)" empty={!models.length}>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={models} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 0 }}>
                  <CartesianGrid stroke="var(--grid)" horizontal={false} />
                  <XAxis type="number" allowDecimals={false} {...axis} />
                  <YAxis type="category" dataKey="nombre" {...axis} width={190} />
                  <Tooltip {...tooltipStyle} />
                  <Bar dataKey="llamadas" name="Llamadas" fill="var(--series-3)" radius={[0, 4, 4, 0]} barSize={18} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          <div className="card">
            <div className="card-title">Detalle por modelo</div>
            <table className="table">
              <thead>
                <tr>
                  <th>Modelo</th>
                  <th>Servidor</th>
                  <th>Rol</th>
                  <th className="num">Llamadas</th>
                  <th className="num">Tokens entrada</th>
                  <th className="num">Tokens salida</th>
                  <th className="num">Latencia media</th>
                </tr>
              </thead>
              <tbody>
                {(data.models || []).map((m) => (
                  <tr key={m.model + m.purpose}>
                    <td>{m.model}</td>
                    <td>{m.provider}</td>
                    <td>{m.purpose}</td>
                    <td className="num">{fmt(m.llamadas)}</td>
                    <td className="num">{fmt(m.input_tokens)}</td>
                    <td className="num">{fmt(m.output_tokens)}</td>
                    <td className="num">{fmt(m.latencia_media_ms)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {data.cache && (
            <div className="card">
              <div className="row between">
                <div className="card-title">Efecto de Redis</div>
                <button className="btn small" onClick={() => adminPost("cache/flush").then(cargar).catch(manejar)}>
                  Vaciar caché de respuestas
                </button>
              </div>
              <div className="tiles">
                <Tile label="Latencia con caché" value={seg(data.cache.hit_ms)} sub={`${fmt(data.cache.hits)} consultas`} />
                <Tile label="Latencia sin caché" value={seg(data.cache.miss_ms)} sub={`${fmt(data.cache.misses)} consultas`} />
                <Tile label="Aceleración" value={data.cache.speedup ? `${fmt(data.cache.speedup)}×` : "—"} />
                <Tile
                  label="Redis"
                  value={data.cache.redis?.claves != null ? `${fmt(data.cache.redis.claves)} claves` : "—"}
                  sub={data.cache.redis?.memoria ? `memoria ${data.cache.redis.memoria} · emb. hits ${fmt(data.cache.redis.embeddings_hits)}` : data.cache.redis?.error}
                />
              </div>
            </div>
          )}
        </>
      )}

      {tab === "conversaciones" && (
        <div className="card">
          <div className="row between">
            <div className="card-title">Preguntas y respuestas ({fmt(runs.total)})</div>
            <div className="row gap">
              <button className="btn small" disabled={page === 0} onClick={() => setPage(page - 1)}>
                ←
              </button>
              <span className="small muted">
                Página {page + 1} de {Math.max(1, Math.ceil(runs.total / PAGE))}
              </span>
              <button className="btn small" disabled={(page + 1) * PAGE >= runs.total} onClick={() => setPage(page + 1)}>
                →
              </button>
            </div>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Fecha</th>
                  <th>Pregunta</th>
                  <th>Respuesta</th>
                  <th className="num">Tok. entrada</th>
                  <th className="num">Tok. salida</th>
                  <th className="num">Tiempo</th>
                  <th>Modelo</th>
                  <th>Estado</th>
                </tr>
              </thead>
              <tbody>
                {runs.items.map((r) => (
                  <tr key={r.run_id} className="clickable" onClick={() => adminGet(`runs/${r.run_id}`).then(setDetalle).catch(manejar)}>
                    <td className="nowrap">{new Date(r.created_at).toLocaleString("es")}</td>
                    <td className="clip">{r.question}</td>
                    <td className="clip">{r.answer || <em className="muted">{r.error}</em>}</td>
                    <td className="num">{fmt(r.input_tokens)}</td>
                    <td className="num">{fmt(r.output_tokens)}</td>
                    <td className="num">{seg(r.latency_total_ms)}</td>
                    <td className="small">{r.agent_model}</td>
                    <td>
                      {r.error ? (
                        <span className="badge err">✕ error</span>
                      ) : r.cache_hit ? (
                        <span className="badge ok">⚡ caché</span>
                      ) : r.answerable === false ? (
                        <span className="badge warn">abstención</span>
                      ) : (
                        <span className="badge">✓ ok</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === "eval" && (
        <div className="card">
          <div className="card-title">Última evaluación automática (LLM como juez)</div>
          {!evalData?.run ? (
            <p className="muted">
              Aún no hay evaluaciones. Ejecuta <code>docker compose exec backend python -m eval.run_eval</code>.
            </p>
          ) : (
            <>
              <p className="small muted">
                {evalData.run.label} · juez {evalData.run.judge_model} · agente {evalData.run.agent_model}
              </p>
              <div className="tiles">
                <Tile label="Preguntas" value={fmt(evalData.resumen.n)} />
                <Tile label="Faithfulness (1-5)" value={fmt(evalData.resumen.faithfulness, 2)} />
                <Tile label="Relevancia (1-5)" value={fmt(evalData.resumen.relevance, 2)} />
                <Tile label="Corrección (1-5)" value={fmt(evalData.resumen.correctness, 2)} />
                <Tile label="Abstención correcta" value={pct(evalData.resumen.abstencion_ok)} />
                <Tile label="Con citas" value={pct(evalData.resumen.citas_ok)} />
              </div>
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>id</th>
                      <th>Categoría</th>
                      <th>Pregunta</th>
                      <th>Respuesta</th>
                      <th className="num">F</th>
                      <th className="num">R</th>
                      <th className="num">C</th>
                      <th>Abst.</th>
                      <th>Juez</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evalData.items.map((i) => (
                      <tr key={i.id}>
                        <td>{i.question_id}</td>
                        <td>{i.category}</td>
                        <td className="clip">{i.question}</td>
                        <td className="clip">{i.answer}</td>
                        <td className="num">{i.faithfulness ?? "—"}</td>
                        <td className="num">{i.relevance ?? "—"}</td>
                        <td className="num">{i.correctness ?? "—"}</td>
                        <td>{i.abstention_ok ? "✓" : "✕"}</td>
                        <td className="clip small">{i.judge_reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      {detalle && (
        <div className="modal-bg" onClick={() => setDetalle(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="row between">
              <h3>Ejecución #{detalle.run.run_id}</h3>
              <button className="btn ghost" onClick={() => setDetalle(null)}>
                ✕
              </button>
            </div>
            <p>
              <strong>Pregunta:</strong> {detalle.run.question}
            </p>
            <p className="pre">
              <strong>Respuesta:</strong> {detalle.run.answer || detalle.run.error}
            </p>
            <div className="tiles small-tiles">
              <Tile label="Embedding" value={`${fmt(detalle.run.latency_embed_ms)} ms`} />
              <Tile label="Búsqueda" value={`${fmt(detalle.run.latency_search_ms)} ms`} />
              <Tile label="Rerank" value={`${fmt(detalle.run.latency_rerank_ms)} ms`} />
              <Tile label="Visión" value={`${fmt(detalle.run.latency_vision_ms)} ms`} />
              <Tile label="LLM" value={`${fmt(detalle.run.latency_llm_ms)} ms`} />
              <Tile label="Total" value={seg(detalle.run.latency_total_ms)} />
            </div>
            <h4>Llamadas a modelos</h4>
            <table className="table">
              <tbody>
                {detalle.model_calls.map((m, i) => (
                  <tr key={i}>
                    <td>{m.model}</td>
                    <td>{m.purpose}</td>
                    <td className="num">{fmt(m.input_tokens)} / {fmt(m.output_tokens)} tok</td>
                    <td className="num">{fmt(m.latency_ms)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h4>Fragmentos recuperados</h4>
            <table className="table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Consulta del agente</th>
                  <th>chunk</th>
                  <th className="num">score híbrido</th>
                  <th className="num">rerank</th>
                </tr>
              </thead>
              <tbody>
                {detalle.hits.map((h) => (
                  <tr key={h.rank}>
                    <td>{h.rank}</td>
                    <td className="clip">{h.query}</td>
                    <td>{h.chunk_id}</td>
                    <td className="num">{fmt(h.score, 3)}</td>
                    <td className="num">{fmt(h.rerank_score, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Admin() {
  const [logged, setLogged] = useState(!!adminToken.get());
  return (
    <div className="adm">
      {logged ? <Dashboard onLogout={() => setLogged(false)} /> : <Login onOk={() => setLogged(true)} />}
    </div>
  );
}
