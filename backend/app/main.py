"""API FastAPI del Agente Académico."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import cache
from .config import get_settings
from .db import close_pool, init_pool, pool
from .routers import admin, chat
from .search import get_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


async def _precargar_modelos():
    """Carga embeddings y reranker en segundo plano para que la primera pregunta no espere."""
    try:
        from .embeddings import get_embedder
        from .reranker import get_reranker

        await anyio.to_thread.run_sync(get_embedder)
        if get_settings().rerank_enabled:
            await anyio.to_thread.run_sync(get_reranker)
    except Exception:
        log.exception("No se pudieron precargar los modelos de Hugging Face")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    Path(get_settings().figures_dir).mkdir(parents=True, exist_ok=True)
    tarea = asyncio.create_task(_precargar_modelos())
    yield
    tarea.cancel()
    await close_pool()


s = get_settings()
app = FastAPI(title="Agente Académico", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=[o.strip() for o in s.cors_origins.split(",")], allow_methods=["*"], allow_headers=["*"]
)
app.include_router(chat.router)
app.include_router(admin.router)
Path(s.figures_dir).mkdir(parents=True, exist_ok=True)
app.mount("/api/figures", StaticFiles(directory=s.figures_dir), name="figures")


@app.get("/api/health")
async def health():
    """Healthcheck usado por docker compose. Reporta el estado de cada dependencia."""
    estado = {"api": "ok"}
    try:
        await pool().fetchval("SELECT 1")
        estado["postgres"] = "ok"
    except Exception as e:
        estado["postgres"] = f"error: {e}"
    try:
        await cache.client().ping()
        estado["redis"] = "ok"
    except Exception as e:
        estado["redis"] = f"error: {e}"
    try:
        r = get_retriever()
        existe = await anyio.to_thread.run_sync(lambda: r.client.indices.exists(index=s.opensearch_index))
        estado["opensearch"] = "ok" if existe else "sin índice (ejecuta la ingesta)"
    except Exception as e:
        estado["opensearch"] = f"error: {e}"
    return estado
