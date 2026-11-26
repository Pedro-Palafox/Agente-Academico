"""Evaluación automática de respuestas con LLM como juez.

    docker compose exec backend python -m eval.run_eval [--label mi-run] [--api http://localhost:8000]

Para cada pregunta de eval/preguntas.json:
  1. Llama a la API real (/api/chat, sin caché, debug=true para obtener los contextos).
  2. Un juez (Qwen2.5-VL en vLLM por defecto, distinto del generador para evitar autoevaluación)
     puntúa con salida tipada (Pydantic AI):
       - faithfulness (1-5): ¿todo lo afirmado está respaldado por los fragmentos recuperados?
       - relevance    (1-5): ¿responde a lo que se preguntó?
       - correctness  (1-5): coincidencia con la respuesta esperada (si se escribió)
  3. Métricas deterministas: abstención correcta (respondible esperado vs. obtenido) y citas presentes.
Resultados: eval/resultados/eval_<label>.json + .md y tablas eval_runs/eval_items (visibles en /admin).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path

import asyncpg
import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import get_settings

AQUI = Path(__file__).parent


class Juicio(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="1=inventa/contradice el contexto, 5=todo respaldado por el contexto")
    relevance: int = Field(ge=1, le=5, description="1=no responde la pregunta, 5=responde exactamente lo preguntado")
    correctness: int | None = Field(default=None, ge=1, le=5, description="vs. respuesta esperada; null si no hay esperada")
    justificacion: str = Field(description="1-2 frases en español")


PROMPT_JUEZ = """Eres un evaluador estricto de sistemas RAG. Evalúa la RESPUESTA de un asistente a una PREGUNTA
sobre un artículo científico usando SOLO los FRAGMENTOS recuperados como evidencia.
- faithfulness: penaliza cualquier dato que no esté en los fragmentos. Si el asistente dijo que no puede
  responder y efectivamente los fragmentos no contienen la respuesta, faithfulness=5.
- relevance: ¿la respuesta atiende la pregunta?
- correctness: compara con la RESPUESTA ESPERADA si existe; si está vacía devuelve null."""


def juez() -> Agent[None, Juicio]:
    s = get_settings()
    import os

    backend = os.getenv("JUDGE_BACKEND", "vllm")
    if backend == "ollama":
        m = OpenAIChatModel(s.ollama_model, provider=OllamaProvider(base_url=s.ollama_base_url))
        nombre = s.ollama_model
    else:
        m = OpenAIChatModel(s.vllm_model, provider=OpenAIProvider(base_url=s.vllm_base_url, api_key=s.vllm_api_key))
        nombre = s.vllm_model
    a = Agent(m, output_type=Juicio, instructions=PROMPT_JUEZ, retries=2, model_settings={"temperature": 0})
    a.nombre = nombre  # type: ignore[attr-defined]
    return a


async def preguntar(cli: httpx.AsyncClient, api: str, pregunta: str, client_id: str, conv=None) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = await cli.post(f"{api}/api/chat", json={
        "message": pregunta, "client_id": client_id, "conversation_id": conv, "no_cache": True, "debug": True,
    })
    ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return {"error": r.text}, ms
    return r.json(), ms


async def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=datetime.now().strftime("%Y%m%d-%H%M"))
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--preguntas", default=str(AQUI / "preguntas.json"))
    a = ap.parse_args()

    preguntas = json.loads(Path(a.preguntas).read_text(encoding="utf-8"))["preguntas"]
    J = juez()
    client_id = f"eval-{uuid.uuid4().hex[:8]}"
    filas = []
    async with httpx.AsyncClient(timeout=600) as cli:
        for q in preguntas:
            conv = None
            if q.get("contexto_previo"):  # pregunta multi-turno
                prev, _ = await preguntar(cli, a.api, q["contexto_previo"], client_id)
                conv = prev.get("conversation_id")
            resp, ms = await preguntar(cli, a.api, q["pregunta"], client_id, conv)
            if "error" in resp:
                print(f"[{q['id']}] ERROR {resp['error'][:200]}")
                filas.append({**q, "answer": None, "error": resp["error"], "latency_ms": ms})
                continue
            contextos = "\n\n".join(
                f"[{c['chunk_id']} | {c['seccion']} | p.{c['pagina']}]\n{c['texto']}" for c in (resp.get("contexts") or [])
            ) or "(sin fragmentos)"
            prompt = (f"PREGUNTA: {q['pregunta']}\n\nRESPUESTA ESPERADA: {q.get('respuesta_esperada') or '(vacía)'}\n\n"
                      f"FRAGMENTOS:\n{contextos[:12000]}\n\nRESPUESTA DEL ASISTENTE:\n{resp['answer']}")
            try:
                j = (await J.run(prompt)).output
            except Exception as e:  # el juez falló: se registra sin puntaje
                j = Juicio(faithfulness=1, relevance=1, correctness=None, justificacion=f"juez falló: {e}")
            answered = bool(resp["answerable"])
            fila = {
                **q, "answer": resp["answer"], "answered": answered, "faithfulness": j.faithfulness,
                "relevance": j.relevance, "correctness": j.correctness if q.get("respuesta_esperada") else None,
                "citation_ok": (len(resp["citations"]) > 0) if answered else True,
                "abstention_ok": answered == q["respondible"], "latency_ms": ms, "judge_reason": j.justificacion,
                "citas": [f"{c['section']} p.{c['page_start']}" for c in resp["citations"]],
                "input_tokens": resp["metrics"]["input_tokens"], "output_tokens": resp["metrics"]["output_tokens"],
            }
            filas.append(fila)
            print(f"[{q['id']}] F={j.faithfulness} R={j.relevance} C={fila['correctness']} "
                  f"abst={'OK' if fila['abstention_ok'] else 'X'} {ms/1000:.1f}s")

    ok = [f for f in filas if f.get("answer") is not None]

    def media(k):
        v = [f[k] for f in ok if f.get(k) is not None]
        return round(statistics.mean(v), 2) if v else None

    resumen = {
        "n": len(filas), "errores": len(filas) - len(ok),
        "faithfulness": media("faithfulness"), "relevance": media("relevance"), "correctness": media("correctness"),
        "abstencion_correcta": round(sum(f["abstention_ok"] for f in ok) / max(len(ok), 1), 3),
        "con_citas": round(sum(f["citation_ok"] for f in ok) / max(len(ok), 1), 3),
        "latencia_media_s": round(statistics.mean([f["latency_ms"] for f in ok]) / 1000, 2) if ok else None,
    }
    out = AQUI / "resultados"
    out.mkdir(exist_ok=True)
    (out / f"eval_{a.label}.json").write_text(json.dumps({"resumen": resumen, "filas": filas}, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# Evaluación `{a.label}`\n", f"Juez: `{J.nombre}` · Agente: `{s.ollama_model if s.agent_backend == 'ollama' else s.vllm_model}`\n",
          "| Métrica | Valor |", "|---|---|"] + [f"| {k} | {v} |" for k, v in resumen.items()]
    md += ["\n| id | categoría | resp.? | F | R | C | abst. | citas | latencia s | justificación |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in filas:
        md.append(f"| {f['id']} | {f['categoria']} | {f.get('answered')} | {f.get('faithfulness')} | {f.get('relevance')} | "
                  f"{f.get('correctness')} | {'✅' if f.get('abstention_ok') else '❌'} | {', '.join(f.get('citas', []))} | "
                  f"{f['latency_ms']/1000:.1f} | {(f.get('judge_reason') or f.get('error') or '')[:120].replace('|', '/')} |")
    (out / f"eval_{a.label}.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    con = await asyncpg.connect(s.postgres_dsn)
    try:
        rid = await con.fetchval(
            "INSERT INTO eval_runs (label, judge_model, agent_model) VALUES ($1,$2,$3) RETURNING id",
            a.label, J.nombre, s.ollama_model if s.agent_backend == "ollama" else s.vllm_model,
        )
        await con.executemany(
            """INSERT INTO eval_items (eval_run_id, question_id, question, category, expected_answerable, expected_answer,
                 answer, answered, faithfulness, relevance, correctness, citation_ok, abstention_ok, latency_ms, judge_reason)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)""",
            [(rid, f["id"], f["pregunta"], f["categoria"], f["respondible"], f.get("respuesta_esperada") or None,
              f.get("answer"), f.get("answered"), f.get("faithfulness"), f.get("relevance"), f.get("correctness"),
              f.get("citation_ok"), f.get("abstention_ok"), f["latency_ms"], f.get("judge_reason") or f.get("error"))
             for f in filas],
        )
    finally:
        await con.close()
    print(f"Guardado en {out} y en PostgreSQL (eval_run {rid}).")


if __name__ == "__main__":
    asyncio.run(main())
