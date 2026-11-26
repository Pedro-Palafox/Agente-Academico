"""Evidencia de que la búsqueda híbrida (y el reranking) mejoran la recuperación.

    docker compose exec backend python -m eval.eval_retrieval

Compara bm25 | knn | hybrid | hybrid_rerank sobre:
  a) eval/qrels.jsonl (generado con eval.generar_qrels; relevante = el chunk de origen)
  b) preguntas de eval/preguntas.json que tengan "paginas" (relevante = chunk en esas páginas)
Métricas: Recall@{1,3,5,10}, MRR@10, nDCG@10 y latencia media.
También barre los pesos léxico/vectorial de la búsqueda híbrida para justificar el valor elegido
(HYBRID_WEIGHT_LEXICAL / HYBRID_WEIGHT_VECTOR en .env).
Salida: eval/resultados/retrieval.md y retrieval.json
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from app.config import get_settings
from app.search import MODOS, Retriever, asegurar_pipeline

AQUI = Path(__file__).parent
KS = (1, 3, 5, 10)


def cargar_consultas() -> list[dict]:
    qs = []
    qrels = AQUI / "qrels.jsonl"
    if qrels.exists():
        for line in qrels.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                qs.append({"query": d["query"], "gold_chunks": {d["gold_chunk"]}, "gold_pages": set(), "origen": "sintética"})
    pj = json.loads((AQUI / "preguntas.json").read_text(encoding="utf-8"))["preguntas"]
    for p in pj:
        if p["respondible"] and p.get("paginas"):
            qs.append({"query": p["pregunta"], "gold_chunks": set(), "gold_pages": set(p["paginas"]), "origen": "manual"})
    return qs


def relevante(frag, q) -> bool:
    if frag.chunk_id in q["gold_chunks"]:
        return True
    if q["gold_pages"] and frag.page_start is not None:
        return any(p in q["gold_pages"] for p in range(frag.page_start, (frag.page_end or frag.page_start) + 1))
    return False


def metricas(rels: list[bool]) -> dict:
    m = {f"R@{k}": float(any(rels[:k])) for k in KS}
    first = next((i for i, r in enumerate(rels[:10]) if r), None)
    m["MRR@10"] = 1 / (first + 1) if first is not None else 0.0
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rels[:10]) if r)
    n_rel = max(1, min(10, sum(rels[:10]) or 1))
    idcg = sum(1 / math.log2(i + 2) for i in range(n_rel))
    m["nDCG@10"] = dcg / idcg
    return m


def evaluar(r: Retriever, qs, modo: str, pipeline: str | None = None) -> dict:
    filas, lat = [], []
    for q in qs:
        t0 = time.perf_counter()
        frags = r.buscar(q["query"], modo=modo, k=10, tipos=["text", "table", "figure"], pipeline=pipeline)
        lat.append((time.perf_counter() - t0) * 1000)
        filas.append(metricas([relevante(f, q) for f in frags]))
    agg = {k: round(statistics.mean(f[k] for f in filas), 3) for k in filas[0]}
    agg["latencia_ms"] = round(statistics.mean(lat), 1)
    return agg


def main() -> None:
    s = get_settings()
    qs = cargar_consultas()
    if not qs:
        print("No hay consultas. Ejecuta primero: python -m eval.generar_qrels  (o agrega 'paginas' a preguntas.json)")
        return
    r = Retriever()
    r.buscar("warmup", modo="hybrid_rerank")  # carga modelos antes de medir latencias
    print(f"{len(qs)} consultas ({sum(q['origen'] == 'sintética' for q in qs)} sintéticas)")

    resultados = {}
    for modo in MODOS:
        resultados[modo] = evaluar(r, qs, modo)
        print(modo, resultados[modo])

    barrido = {}
    for w in (0.1, 0.3, 0.5, 0.7, 0.9):
        nombre = f"pipeline-w{int(w * 100)}"
        asegurar_pipeline(r.client, nombre, (w, round(1 - w, 2)))
        barrido[w] = evaluar(r, qs, "hybrid", pipeline=nombre)
        print(f"peso léxico={w}", barrido[w])

    out = AQUI / "resultados"
    out.mkdir(exist_ok=True)
    (out / "retrieval.json").write_text(json.dumps({"modos": resultados, "barrido_pesos": barrido}, indent=2), encoding="utf-8")
    cols = [f"R@{k}" for k in KS] + ["MRR@10", "nDCG@10", "latencia_ms"]
    md = ["# Evaluación de recuperación\n", f"Consultas: {len(qs)} · índice `{s.opensearch_index}` · "
          f"candidatos para rerank: {s.retrieval_candidates}\n",
          "## Comparación de modos\n", "| modo | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for modo, m in resultados.items():
        md.append(f"| {modo} | " + " | ".join(str(m[c]) for c in cols) + " |")
    md += ["\n## Barrido de pesos de la búsqueda híbrida (léxico / vectorial)\n",
           "| peso léxico | peso vectorial | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for w, m in barrido.items():
        md.append(f"| {w} | {round(1 - w, 2)} | " + " | ".join(str(m[c]) for c in cols) + " |")
    mejor = max(barrido, key=lambda w: barrido[w]["MRR@10"])
    md.append(f"\n**Mejor peso léxico por MRR@10: {mejor}** (configurado actualmente: {s.hybrid_weight_lexical}). "
              "Si difiere, actualicen HYBRID_WEIGHT_LEXICAL/HYBRID_WEIGHT_VECTOR en .env y reinicien el backend.")
    (out / "retrieval.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
