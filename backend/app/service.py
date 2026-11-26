"""Orquestación de un turno de chat: persistencia, caché, agente y métricas."""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID

from . import cache
from .agent import Deps, ejecutar, ejecutar_stream
from .config import get_settings
from .db import pool
from .schemas import ChatRequest, ChatResponse, CitaOut, MetricasOut
from .search import Fragmento, get_retriever

log = logging.getLogger(__name__)


class AgenteError(Exception):
    pass


def image_url(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        rel = p.relative_to(get_settings().figures_dir)
    except ValueError:
        return None
    return f"/api/figures/{rel.as_posix()}"


def a_cita(n: int, f: Fragmento, cita_textual: str = "") -> CitaOut:
    snippet = cita_textual.strip() if cita_textual and len(cita_textual) > 15 else f.text[:350]
    return CitaOut(
        n=n, chunk_id=f.chunk_id, doc_id=f.doc_id, doc_title=f.doc_title, section=f.section,
        page_start=f.page_start, page_end=f.page_end, chunk_type=f.chunk_type, snippet=snippet,
        image_url=image_url(f.image_path) if f.chunk_type in ("figure", "table") else None,
    )


async def documentos() -> list[dict]:
    rows = await pool().fetch("SELECT id, title, num_pages, num_chunks, num_tables, num_figures, ingested_at FROM documents ORDER BY ingested_at")
    return [dict(r) for r in rows]


async def _conversacion(req: ChatRequest) -> tuple[UUID, list[dict]]:
    p = pool()
    if req.conversation_id:
        row = await p.fetchrow("SELECT id FROM conversations WHERE id=$1 AND client_id=$2", req.conversation_id, req.client_id)
        if row:
            n = get_settings().max_history_messages
            hist = await p.fetch(
                "SELECT role, content FROM (SELECT id, role, content FROM messages WHERE conversation_id=$1 "
                "ORDER BY id DESC LIMIT $2) t ORDER BY id", req.conversation_id, n,
            )
            return req.conversation_id, [dict(h) for h in hist]
    titulo = req.message.strip().replace("\n", " ")[:70]
    cid = await p.fetchval("INSERT INTO conversations (client_id, title) VALUES ($1,$2) RETURNING id", req.client_id, titulo)
    return cid, []


async def procesar(
    req: ChatRequest,
    on_meta: Callable[[dict], Awaitable[None]] | None = None,
    on_status: Callable[[str], Awaitable[None]] | None = None,
    on_delta: Callable[[str, bool], Awaitable[None]] | None = None,
) -> ChatResponse:
    s = get_settings()
    t0 = time.perf_counter()
    p = pool()
    conv_id, historial = await _conversacion(req)
    user_msg_id = await p.fetchval(
        "INSERT INTO messages (conversation_id, role, content) VALUES ($1,'user',$2) RETURNING id", conv_id, req.message
    )
    if on_meta:
        await on_meta({"conversation_id": str(conv_id), "user_message_id": user_msg_id})

    modelo_agente = s.vllm_model if s.agent_backend == "vllm" else s.ollama_model
    usar_cache = not historial and not req.no_cache
    key = cache.clave_respuesta(req.message, req.doc_id, modelo_agente)

    # ---------------- 1) caché de respuestas (Redis) ----------------
    if usar_cache:
        hit = await cache.get_json(key)
        if hit:
            await cache.incr("cache_hit")
            if on_status:
                await on_status("Respuesta recuperada de la caché (Redis)")
            if on_delta:
                await on_delta(hit["answer"], True)
            total = (time.perf_counter() - t0) * 1000
            citas = [CitaOut(**c) for c in hit["citations"]]
            return await _persistir(
                conv_id, user_msg_id, hit["answer"], hit["answerable"], hit["confidence"], citas,
                modelo_agente, "cache", 0, 0, 0, 0, total, {}, True, on_delta is not None, [], [],
                contexts=hit.get("contexts") if req.debug else None,
            )
        await cache.incr("cache_miss")

    # ---------------- 2) agente ----------------
    docs = await documentos()
    deps = Deps(retriever=get_retriever(), doc_id=req.doc_id, documentos=docs, on_status=on_status)
    try:
        if on_delta:
            r = await ejecutar_stream(req.message, historial, deps, on_delta)
        else:
            r = await ejecutar(req.message, historial, deps)
    except Exception as e:
        total = (time.perf_counter() - t0) * 1000
        log.exception("Fallo del agente")
        run_id = await p.fetchval(
            """INSERT INTO agent_runs (user_message_id, agent_model, agent_provider, latency_total_ms,
                 latency_embed_ms, latency_search_ms, latency_rerank_ms, streamed, error)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING id""",
            user_msg_id, modelo_agente, s.agent_backend, total, deps.tiempos.embed_ms, deps.tiempos.search_ms,
            deps.tiempos.rerank_ms, on_delta is not None, f"{type(e).__name__}: {e}"[:2000],
        )
        log.info("run %s registrado con error", run_id)
        raise AgenteError(_mensaje_error(e)) from e

    out = r.salida
    citas = []
    for c in out.citas:
        f = deps.recuperados.get(c.chunk_id)
        if f and all(x.chunk_id != f.chunk_id for x in citas):
            citas.append(a_cita(len(citas) + 1, f, c.cita_textual))
    t = deps.tiempos
    stages = {
        "embedding": round(t.embed_ms, 1), "busqueda": round(t.search_ms, 1), "rerank": round(t.rerank_ms, 1),
        "vision": round(t.extra.get("vision_ms", 0), 1), "llm": round(r.llm_ms, 1),
    }
    llamadas = [{
        "model": r.modelo, "provider": r.proveedor, "purpose": "agente",
        "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "latency_ms": r.llm_ms,
    }] + deps.llamadas_vision
    if t.embed_ms:
        from .embeddings import get_embedder
        llamadas.append({"model": get_embedder().model_name, "provider": "huggingface", "purpose": "embedding",
                         "input_tokens": 0, "output_tokens": 0, "latency_ms": t.embed_ms})
    if t.rerank_ms:
        from .reranker import get_reranker
        llamadas.append({"model": get_reranker().model_name, "provider": "huggingface", "purpose": "rerank",
                         "input_tokens": 0, "output_tokens": 0, "latency_ms": t.rerank_ms})
    contexts = [h["frag"].para_llm() for h in deps.hits]
    total = (time.perf_counter() - t0) * 1000
    resp = await _persistir(
        conv_id, user_msg_id, out.respuesta, out.respondible, out.confianza, citas, r.modelo, r.proveedor,
        r.input_tokens, r.output_tokens, r.tool_calls, r.requests, total, stages, False, on_delta is not None,
        llamadas, deps.hits, contexts=contexts if req.debug else None,
    )
    if usar_cache:
        await cache.set_json(key, {
            "answer": out.respuesta, "answerable": out.respondible, "confidence": out.confianza,
            "citations": [c.model_dump() for c in citas], "contexts": contexts,
        })
    return resp


def _mensaje_error(e: Exception) -> str:
    n = type(e).__name__
    txt = str(e)
    if "Connect" in n or "connection" in txt.lower():
        return "No pude conectar con el servidor de modelos (Ollama/vLLM). ¿Está arriba y con el modelo descargado?"
    if "Timeout" in n or "timed out" in txt.lower():
        return "El modelo tardó demasiado en responder. Intenta de nuevo o usa un modelo más pequeño."
    if "UnexpectedModelBehavior" in n:
        return "El modelo no produjo una respuesta válida con citas tras varios intentos. Reformula la pregunta."
    if "NotFound" in n or "index_not_found" in txt:
        return "El índice de OpenSearch no existe. Ejecuta la ingesta del artículo (docker compose run --rm ingest)."
    return f"Error interno del agente ({n})."


async def _persistir(
    conv_id, user_msg_id, answer, answerable, confidence, citas: list[CitaOut], modelo, proveedor,
    tin, tout, tool_calls, requests, total_ms, stages, cache_hit, streamed, llamadas, hits, contexts=None,
) -> ChatResponse:
    p = pool()
    async with p.acquire() as con, con.transaction():
        msg_id = await con.fetchval(
            "INSERT INTO messages (conversation_id, role, content, answerable, confidence) "
            "VALUES ($1,'assistant',$2,$3,$4) RETURNING id", conv_id, answer, answerable, confidence,
        )
        run_id = await con.fetchval(
            """INSERT INTO agent_runs (user_message_id, assistant_message_id, agent_model, agent_provider,
                 input_tokens, output_tokens, llm_requests, tool_calls, latency_total_ms, latency_embed_ms,
                 latency_search_ms, latency_rerank_ms, latency_llm_ms, latency_vision_ms, cache_hit, streamed)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16) RETURNING id""",
            user_msg_id, msg_id, modelo, proveedor, tin, tout, requests, tool_calls, total_ms,
            stages.get("embedding", 0), stages.get("busqueda", 0), stages.get("rerank", 0),
            stages.get("llm", 0), stages.get("vision", 0), cache_hit, streamed,
        )
        doc_ids = {r["id"] for r in await con.fetch("SELECT id FROM documents")}
        if citas:
            await con.executemany(
                "INSERT INTO citations (message_id, chunk_id, document_id, section, page, snippet) VALUES ($1,$2,$3,$4,$5,$6)",
                [(msg_id, c.chunk_id, c.doc_id if c.doc_id in doc_ids else None, c.section, c.page_start, c.snippet) for c in citas],
            )
        if llamadas:
            await con.executemany(
                "INSERT INTO model_calls (run_id, model, provider, purpose, input_tokens, output_tokens, latency_ms) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7)",
                [(run_id, m["model"], m["provider"], m["purpose"], m["input_tokens"], m["output_tokens"], m["latency_ms"]) for m in llamadas],
            )
        if hits:
            await con.executemany(
                "INSERT INTO retrieval_hits (run_id, query, chunk_id, document_id, rank, score, rerank_score) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7)",
                [(run_id, h["query"], h["frag"].chunk_id, h["frag"].doc_id if h["frag"].doc_id in doc_ids else None,
                  h["rank"], h["frag"].score, h["frag"].rerank_score) for h in hits],
            )
        await con.execute("UPDATE conversations SET updated_at=now() WHERE id=$1", conv_id)
    return ChatResponse(
        conversation_id=conv_id, message_id=msg_id, answer=answer, answerable=answerable, confidence=confidence,
        citations=citas, contexts=contexts,
        metrics=MetricasOut(model=modelo, provider=proveedor, input_tokens=tin, output_tokens=tout,
                            tool_calls=tool_calls, latency_ms=round(total_ms, 1), stages=stages, cache_hit=cache_hit),
    )
