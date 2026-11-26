"""Endpoints públicos: chat (normal y streaming SSE), conversaciones, documentos."""
from __future__ import annotations

import asyncio
import json
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..db import pool
from ..schemas import ChatRequest, ChatResponse, CitaOut, ConversationOut, MessageOut
from ..service import AgenteError, documentos, procesar

router = APIRouter(prefix="/api", tags=["chat"])


@router.get("/documents")
async def listar_documentos():
    return await documentos()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        return await procesar(req)
    except AgenteError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


def _sse(evento: str, data) -> str:
    return f"event: {evento}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events: meta -> status* -> delta* -> final | error."""
    q: asyncio.Queue[str | None] = asyncio.Queue()

    async def on_meta(d):
        await q.put(_sse("meta", d))

    async def on_status(msg):
        await q.put(_sse("status", {"message": msg}))

    async def on_delta(texto, reset):
        await q.put(_sse("delta", {"text": texto, "reset": reset}))

    async def trabajo():
        try:
            r = await procesar(req, on_meta=on_meta, on_status=on_status, on_delta=on_delta)
            await q.put(_sse("final", r.model_dump(mode="json")))
        except AgenteError as e:
            await q.put(_sse("error", {"detail": str(e)}))
        except Exception as e:  # noqa: BLE001
            await q.put(_sse("error", {"detail": f"Error inesperado: {type(e).__name__}"}))
        finally:
            await q.put(None)

    async def generador():
        tarea = asyncio.create_task(trabajo())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # evita cortes de ngrok/nginx en respuestas largas
                    continue
                if item is None:
                    break
                yield item
        finally:
            if not tarea.done():
                tarea.cancel()

    return StreamingResponse(
        generador(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations", response_model=list[ConversationOut])
async def conversaciones(client_id: str = Query(..., min_length=1)):
    rows = await pool().fetch(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE client_id=$1 ORDER BY updated_at DESC LIMIT 50",
        client_id,
    )
    return [dict(r) for r in rows]


@router.get("/conversations/{conv_id}/messages", response_model=list[MessageOut])
async def mensajes(conv_id: UUID, client_id: str = Query(...)):
    p = pool()
    ok = await p.fetchval("SELECT 1 FROM conversations WHERE id=$1 AND client_id=$2", conv_id, client_id)
    if not ok:
        raise HTTPException(404, "Conversación no encontrada")
    rows = await p.fetch(
        "SELECT id, role, content, answerable, confidence, created_at FROM messages WHERE conversation_id=$1 ORDER BY id",
        conv_id,
    )
    cits = await p.fetch(
        """SELECT c.message_id, c.chunk_id, c.document_id, coalesce(d.title,'') AS title, c.section, c.page, c.snippet
           FROM citations c JOIN messages m ON m.id=c.message_id LEFT JOIN documents d ON d.id=c.document_id
           WHERE m.conversation_id=$1 ORDER BY c.id""",
        conv_id,
    )
    por_msg: dict[int, list[CitaOut]] = {}
    for c in cits:
        lst = por_msg.setdefault(c["message_id"], [])
        tipo = "table" if "-T" in c["chunk_id"] else "figure" if "-F" in c["chunk_id"] else "text"
        lst.append(CitaOut(n=len(lst) + 1, chunk_id=c["chunk_id"], doc_id=c["document_id"] or "", doc_title=c["title"],
                           section=c["section"] or "", page_start=c["page"], page_end=c["page"], chunk_type=tipo,
                           snippet=c["snippet"] or ""))
    return [MessageOut(**dict(r), citations=por_msg.get(r["id"], [])) for r in rows]


@router.delete("/conversations/{conv_id}")
async def borrar(conv_id: UUID, client_id: str = Query(...)):
    n = await pool().execute("DELETE FROM conversations WHERE id=$1 AND client_id=$2", conv_id, client_id)
    return {"deleted": n.endswith("1")}
