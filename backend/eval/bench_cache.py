"""Mide el efecto de Redis en la latencia.

    docker compose exec backend python -m eval.bench_cache [--n 5]

1. Vacía la caché de respuestas.
2. Pregunta N preguntas (miss: pasa por el agente completo) y luego las MISMAS N (hit: Redis).
3. Mide también el embedding de consultas con y sin caché.
Salida: eval/resultados/cache.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

import httpx

from app import cache
from app.search import Retriever, Tiempos

AQUI = Path(__file__).parent


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--api", default="http://localhost:8000")
    a = ap.parse_args()
    preguntas = [p["pregunta"] for p in json.loads((AQUI / "preguntas.json").read_text(encoding="utf-8"))["preguntas"]
                 if p["respondible"] and not p.get("contexto_previo")][: a.n]
    await cache.flush_respuestas()
    cid = f"bench-{uuid.uuid4().hex[:6]}"
    miss, hit = [], []
    async with httpx.AsyncClient(timeout=600) as cli:
        for fase, lista in (("miss", miss), ("hit", hit)):
            for q in preguntas:
                t0 = time.perf_counter()
                r = await cli.post(f"{a.api}/api/chat", json={"message": q, "client_id": cid})
                lista.append((time.perf_counter() - t0) * 1000)
                ok = r.status_code == 200 and r.json()["metrics"]["cache_hit"] == (fase == "hit")
                print(f"{fase:4s} {lista[-1]:9.1f} ms {'OK' if ok else '??'}  {q[:60]}")

    # Embeddings de consulta
    ret = Retriever()
    emb_miss, emb_hit = [], []
    for q in preguntas:
        k = f"bench embedding {uuid.uuid4().hex} {q}"
        t = Tiempos(); ret._embedding(k, t); emb_miss.append(t.embed_ms)
        t = Tiempos(); ret._embedding(k, t); emb_hit.append(t.embed_ms)

    def fmt(v):
        return f"{statistics.mean(v):.1f}"

    md = ["# Efecto de Redis en la latencia\n", "| caso | sin caché (ms) | con caché (ms) | aceleración |", "|---|---|---|---|",
          f"| respuesta completa | {fmt(miss)} | {fmt(hit)} | {statistics.mean(miss) / statistics.mean(hit):.0f}× |",
          f"| embedding de consulta | {fmt(emb_miss)} | {fmt(emb_hit)} | {statistics.mean(emb_miss) / max(statistics.mean(emb_hit), 0.01):.0f}× |"]
    out = AQUI / "resultados"
    out.mkdir(exist_ok=True)
    (out / "cache.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    asyncio.run(main())
