"""Caché en Redis.

Dos usos justificados y medibles:
1. Caché de respuestas completas para preguntas sin historial (primer turno). Una pregunta
   repetida (muy común en una demo o en clase) se responde en milisegundos en lugar de
   segundos, sin gastar GPU. Se mide en /admin (latencia cache_hit vs. miss) y con
   `python -m eval.bench_cache`.
2. Caché de embeddings de consultas: evita recalcular el embedding de la misma consulta.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata

import redis.asyncio as aioredis

from .config import get_settings

log = logging.getLogger(__name__)
_client: aioredis.Redis | None = None


def client() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _client


def normalizar(texto: str) -> str:
    """Normaliza la pregunta para que variaciones triviales compartan entrada de caché."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _key(prefijo: str, *partes: str) -> str:
    h = hashlib.sha256("||".join(partes).encode()).hexdigest()[:32]
    return f"{prefijo}:{h}"


def clave_respuesta(pregunta: str, doc_id: str | None, modelo: str) -> str:
    return _key("resp", normalizar(pregunta), doc_id or "*", modelo)


async def get_json(key: str):
    if not get_settings().cache_enabled:
        return None
    try:
        raw = await client().get(key)
        return json.loads(raw) if raw else None
    except Exception as e:  # Redis caído: la app sigue funcionando sin caché
        log.warning("Redis GET falló: %s", e)
        return None


async def set_json(key: str, value, ttl: int | None = None) -> None:
    if not get_settings().cache_enabled:
        return
    try:
        await client().set(key, json.dumps(value, ensure_ascii=False), ex=ttl or get_settings().cache_ttl_seconds)
    except Exception as e:
        log.warning("Redis SET falló: %s", e)


async def incr(nombre: str) -> None:
    try:
        await client().incr(f"stats:{nombre}")
    except Exception:
        pass


async def stats() -> dict:
    try:
        c = client()
        hits = int(await c.get("stats:cache_hit") or 0)
        misses = int(await c.get("stats:cache_miss") or 0)
        emb_hits = int(await c.get("stats:embed_hit") or 0)
        emb_miss = int(await c.get("stats:embed_miss") or 0)
        info = await c.info("memory")
        keys = await c.dbsize()
        return {
            "respuestas_hits": hits,
            "respuestas_misses": misses,
            "embeddings_hits": emb_hits,
            "embeddings_misses": emb_miss,
            "claves": keys,
            "memoria": info.get("used_memory_human"),
        }
    except Exception as e:
        return {"error": str(e)}


async def flush_respuestas() -> int:
    c = client()
    n = 0
    async for k in c.scan_iter("resp:*"):
        await c.delete(k)
        n += 1
    return n


def clave_embedding(texto: str, modelo: str) -> str:
    return _key("emb", modelo, texto)
