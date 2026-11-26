"""Panel /admin: estadísticas leídas de PostgreSQL. Protegido con JWT (usuario/clave en .env)."""
from __future__ import annotations

import hmac
from datetime import date, datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .. import cache
from ..config import get_settings
from ..db import pool
from ..schemas import LoginIn

router = APIRouter(prefix="/api/admin", tags=["admin"])
bearer = HTTPBearer(auto_error=False)


# --------------------------------------------------------------------------- autenticación
@router.post("/login")
async def login(body: LoginIn):
    s = get_settings()
    ok = hmac.compare_digest(body.username, s.admin_user) and hmac.compare_digest(body.password, s.admin_password)
    if not ok:
        raise HTTPException(401, "Credenciales inválidas")
    exp = datetime.now(timezone.utc) + timedelta(minutes=s.jwt_expire_minutes)
    token = jwt.encode({"sub": body.username, "exp": exp}, s.jwt_secret, algorithm="HS256")
    return {"token": token, "expires_at": exp.isoformat()}


def requiere_admin(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> str:
    if cred is None:
        raise HTTPException(401, "Falta token")
    try:
        data = jwt.decode(cred.credentials, get_settings().jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as e:
        raise HTTPException(401, "Token inválido o expirado") from e
    return data["sub"]


# --------------------------------------------------------------------------- filtros
class Filtros:
    def __init__(
        self,
        desde: date | None = Query(None, description="YYYY-MM-DD"),
        hasta: date | None = Query(None, description="YYYY-MM-DD (inclusive)"),
        modelo: str | None = None,
        cache_hit: bool | None = None,
        respondible: bool | None = None,
        errores: bool | None = None,
        q: str | None = Query(None, description="texto en la pregunta o respuesta"),
    ):
        self.desde, self.hasta, self.modelo = desde, hasta, modelo
        self.cache_hit, self.respondible, self.errores, self.q = cache_hit, respondible, errores, q

    def sql(self, alias: str = "v") -> tuple[str, list]:
        cond, params = [], []

        def add(expr, val):
            params.append(val)
            cond.append(expr.replace("?", f"${len(params)}"))

        if self.desde:
            add(f"{alias}.created_at >= ?", datetime.combine(self.desde, datetime.min.time(), timezone.utc))
        if self.hasta:
            add(f"{alias}.created_at < ?", datetime.combine(self.hasta + timedelta(days=1), datetime.min.time(), timezone.utc))
        if self.modelo:
            add(f"{alias}.agent_model = ?", self.modelo)
        if self.cache_hit is not None:
            add(f"{alias}.cache_hit = ?", self.cache_hit)
        if self.respondible is not None:
            add(f"{alias}.answerable = ?", self.respondible)
        if self.errores is not None:
            cond.append(f"{alias}.error IS {'NOT ' if self.errores else ''}NULL")
        if self.q:
            params.append(f"%{self.q}%")
            n = len(params)
            cond.append(f"({alias}.question ILIKE ${n} OR {alias}.answer ILIKE ${n})")
        return ("WHERE " + " AND ".join(cond)) if cond else "", params


# --------------------------------------------------------------------------- endpoints
@router.get("/summary")
async def resumen(f: Filtros = Depends(), _=Depends(requiere_admin)):
    w, p = f.sql()
    r = await pool().fetchrow(
        f"""SELECT count(*)                                   AS total_runs,
                   count(DISTINCT conversation_id)            AS conversaciones,
                   coalesce(sum(input_tokens),0)              AS input_tokens,
                   coalesce(sum(output_tokens),0)             AS output_tokens,
                   coalesce(avg(input_tokens) FILTER (WHERE NOT cache_hit),0)  AS avg_input_tokens,
                   coalesce(avg(output_tokens) FILTER (WHERE NOT cache_hit),0) AS avg_output_tokens,
                   coalesce(avg(latency_total_ms),0)          AS avg_latency_ms,
                   coalesce(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_total_ms),0) AS p50_latency_ms,
                   coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_total_ms),0) AS p95_latency_ms,
                   coalesce(avg(latency_total_ms) FILTER (WHERE cache_hit),0)     AS avg_latency_cache_hit_ms,
                   coalesce(avg(latency_total_ms) FILTER (WHERE NOT cache_hit AND error IS NULL),0) AS avg_latency_cache_miss_ms,
                   coalesce(avg(cache_hit::int),0)            AS cache_hit_rate,
                   coalesce(avg((answerable = false)::int) FILTER (WHERE answerable IS NOT NULL),0) AS tasa_no_respondible,
                   count(*) FILTER (WHERE error IS NOT NULL)  AS errores,
                   coalesce(avg(tool_calls) FILTER (WHERE NOT cache_hit),0) AS avg_tool_calls
            FROM v_qa v {w}""",
        *p,
    )
    out = dict(r)
    out["mensajes"] = await pool().fetchval("SELECT count(*) FROM messages")
    out["documentos"] = await pool().fetchval("SELECT count(*) FROM documents")
    return out


@router.get("/timeseries")
async def serie(bucket: str = Query("day", pattern="^(hour|day)$"), f: Filtros = Depends(), _=Depends(requiere_admin)):
    w, p = f.sql()
    rows = await pool().fetch(
        f"""SELECT date_trunc('{bucket}', created_at) AS t, count(*) AS consultas,
                   avg(latency_total_ms) AS latencia_ms, sum(input_tokens) AS input_tokens,
                   sum(output_tokens) AS output_tokens, sum(cache_hit::int) AS cache_hits
            FROM v_qa v {w} GROUP BY 1 ORDER BY 1""",
        *p,
    )
    return [dict(r) for r in rows]


@router.get("/stages")
async def etapas(f: Filtros = Depends(), _=Depends(requiere_admin)):
    """Latencia promedio por etapa del pipeline (solo ejecuciones que no salieron de la caché)."""
    w, p = f.sql()
    w = (w + " AND " if w else "WHERE ") + "NOT v.cache_hit AND v.error IS NULL"
    r = await pool().fetchrow(
        f"""SELECT coalesce(avg(latency_embed_ms),0) AS embedding, coalesce(avg(latency_search_ms),0) AS busqueda,
                   coalesce(avg(latency_rerank_ms),0) AS rerank, coalesce(avg(latency_vision_ms),0) AS vision,
                   coalesce(avg(latency_llm_ms),0) AS llm, coalesce(avg(latency_total_ms),0) AS total
            FROM v_qa v {w}""",
        *p,
    )
    return dict(r)


@router.get("/models")
async def modelos(f: Filtros = Depends(), _=Depends(requiere_admin)):
    w, p = f.sql()
    rows = await pool().fetch(
        f"""SELECT mc.model, mc.provider, mc.purpose, count(*) AS llamadas,
                   sum(mc.input_tokens) AS input_tokens, sum(mc.output_tokens) AS output_tokens,
                   avg(mc.latency_ms) AS latencia_media_ms
            FROM model_calls mc JOIN v_qa v ON v.run_id = mc.run_id {w}
            GROUP BY 1,2,3 ORDER BY llamadas DESC""",
        *p,
    )
    return [dict(r) for r in rows]


@router.get("/runs")
async def ejecuciones(
    f: Filtros = Depends(), limit: int = Query(25, le=200), offset: int = 0, _=Depends(requiere_admin)
):
    w, p = f.sql()
    total = await pool().fetchval(f"SELECT count(*) FROM v_qa v {w}", *p)
    rows = await pool().fetch(
        f"SELECT * FROM v_qa v {w} ORDER BY created_at DESC LIMIT {int(limit)} OFFSET {int(offset)}", *p
    )
    return {"total": total, "items": [dict(r) for r in rows]}


@router.get("/runs/{run_id}")
async def detalle(run_id: int, _=Depends(requiere_admin)):
    p = pool()
    run = await p.fetchrow("SELECT * FROM v_qa WHERE run_id=$1", run_id)
    if not run:
        raise HTTPException(404, "No existe")
    hits = await p.fetch("SELECT query, chunk_id, rank, score, rerank_score FROM retrieval_hits WHERE run_id=$1 ORDER BY rank", run_id)
    calls = await p.fetch("SELECT model, provider, purpose, input_tokens, output_tokens, latency_ms FROM model_calls WHERE run_id=$1", run_id)
    cits = await p.fetch(
        "SELECT c.chunk_id, c.section, c.page, c.snippet FROM citations c JOIN agent_runs r ON r.assistant_message_id=c.message_id WHERE r.id=$1",
        run_id,
    )
    return {"run": dict(run), "hits": [dict(h) for h in hits], "model_calls": [dict(c) for c in calls],
            "citations": [dict(c) for c in cits]}


@router.get("/filters")
async def opciones(_=Depends(requiere_admin)):
    rows = await pool().fetch("SELECT DISTINCT agent_model FROM agent_runs ORDER BY 1")
    return {"modelos": [r["agent_model"] for r in rows]}


@router.get("/cache")
async def estado_cache(_=Depends(requiere_admin)):
    r = await pool().fetchrow(
        """SELECT coalesce(avg(latency_total_ms) FILTER (WHERE cache_hit),0) AS hit_ms,
                  coalesce(avg(latency_total_ms) FILTER (WHERE NOT cache_hit AND error IS NULL),0) AS miss_ms,
                  count(*) FILTER (WHERE cache_hit) AS hits, count(*) FILTER (WHERE NOT cache_hit) AS misses
           FROM agent_runs"""
    )
    d = dict(r)
    d["speedup"] = (d["miss_ms"] / d["hit_ms"]) if d["hit_ms"] else None
    d["redis"] = await cache.stats()
    return d


@router.post("/cache/flush")
async def vaciar_cache(_=Depends(requiere_admin)):
    return {"borradas": await cache.flush_respuestas()}


@router.get("/eval")
async def evaluacion(_=Depends(requiere_admin)):
    p = pool()
    run = await p.fetchrow("SELECT * FROM eval_runs ORDER BY id DESC LIMIT 1")
    if not run:
        return {"run": None, "items": [], "resumen": None}
    items = await p.fetch("SELECT * FROM eval_items WHERE eval_run_id=$1 ORDER BY id", run["id"])
    res = await p.fetchrow(
        """SELECT count(*) AS n, avg(faithfulness) AS faithfulness, avg(relevance) AS relevance,
                  avg(correctness) AS correctness, avg(citation_ok::int) AS citas_ok,
                  avg(abstention_ok::int) AS abstencion_ok, avg(latency_ms) AS latencia_ms
           FROM eval_items WHERE eval_run_id=$1""",
        run["id"],
    )
    return {"run": dict(run), "items": [dict(i) for i in items], "resumen": dict(res)}
