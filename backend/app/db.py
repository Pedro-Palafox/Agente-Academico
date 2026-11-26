"""Acceso a PostgreSQL con asyncpg (SQL explícito, esquema en sql/schema.sql)."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import asyncpg

from .config import get_settings

log = logging.getLogger(__name__)
_pool: asyncpg.Pool | None = None


async def init_pool(retries: int = 30) -> asyncpg.Pool:
    """Crea el pool y garantiza que el esquema exista (idempotente)."""
    global _pool
    s = get_settings()
    last_err: Exception | None = None
    for intento in range(retries):
        try:
            _pool = await asyncpg.create_pool(s.postgres_dsn, min_size=1, max_size=10)
            break
        except Exception as e:  # PostgreSQL todavía arrancando
            last_err = e
            log.warning("PostgreSQL no disponible (intento %s): %s", intento + 1, e)
            await asyncio.sleep(2)
    if _pool is None:
        raise RuntimeError(f"No se pudo conectar a PostgreSQL: {last_err}")
    schema = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"
    if schema.exists():
        async with _pool.acquire() as con:
            await con.execute(schema.read_text(encoding="utf-8"))
    return _pool


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Pool de PostgreSQL no inicializado")
    return _pool


async def close_pool() -> None:
    if _pool is not None:
        await _pool.close()
