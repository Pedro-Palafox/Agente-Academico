"""Genera un conjunto de evaluación de RECUPERACIÓN (qrels) sin etiquetado manual.

    docker compose exec backend python -m eval.generar_qrels --n 40

Toma N fragmentos de texto al azar (semilla fija) del índice y le pide a un LLM una pregunta
específica que solo ese fragmento responde. El fragmento de origen es el "gold".
Salida: eval/qrels.jsonl  -> la consume eval/eval_retrieval.py
Revisen a mano unas cuantas preguntas: es buena práctica y la pueden mostrar en la presentación.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from openai import OpenAI

from app.config import get_settings
from app.search import os_client

PROMPT = ("Write ONE specific question in English that can be answered using ONLY the following passage from a "
          "scientific survey. Do not copy sentences verbatim; paraphrase like a student would. "
          "Return only the question.\n\nPASSAGE:\n{texto}")


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=["ollama", "vllm"], default="ollama")
    a = ap.parse_args()

    cli = os_client()
    res = cli.search(index=s.opensearch_index, body={
        "size": 2000, "_source": ["chunk_id", "doc_id", "raw_text", "page_start", "section"],
        "query": {"term": {"chunk_type": "text"}},
    })
    chunks = [h["_source"] for h in res["hits"]["hits"] if len(h["_source"].get("raw_text") or "") > 400]
    random.Random(a.seed).shuffle(chunks)
    chunks = chunks[: a.n]
    if a.backend == "ollama":
        llm, modelo = OpenAI(base_url=s.ollama_base_url, api_key="ollama"), s.ollama_model
    else:
        llm, modelo = OpenAI(base_url=s.vllm_base_url, api_key=s.vllm_api_key), s.vllm_model

    out = Path(__file__).parent / "qrels.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, 1):
            r = llm.chat.completions.create(model=modelo, temperature=0.3, max_tokens=4000,
                                            messages=[{"role": "user", "content": PROMPT.format(texto=c["raw_text"][:2500])}])
            q = (r.choices[0].message.content or "").strip().split("\n")[-1].strip()
            f.write(json.dumps({"query": q, "gold_chunk": c["chunk_id"], "gold_pages": [c["page_start"]],
                                "section": c["section"]}, ensure_ascii=False) + "\n")
            print(f"{i:02d}/{len(chunks)} [{c['section'][:30]}] {q}")
    print(f"Guardado {out}")


if __name__ == "__main__":
    main()
