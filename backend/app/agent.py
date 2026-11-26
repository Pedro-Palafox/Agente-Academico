"""Agente con Pydantic AI.

Roles de los servidores de modelos:
  * Ollama  -> Nemotron-3-Nano: "cerebro" del agente. Decide qué buscar, llama herramientas
              (tool calling) y redacta la respuesta tipada con citas.
  * vLLM    -> Qwen2.5-VL-7B-Instruct: modelo de visión. Se invoca desde la herramienta
              `analizar_figura_o_tabla` con la imagen recortada por Docling, describe figuras
              durante la ingesta (opcional) y actúa como juez en la evaluación automática
              (un modelo distinto al generador reduce el sesgo de autoevaluación).

Herramientas:
  * buscar_en_articulo     : recuperación híbrida + reranking en OpenSearch
  * indice_del_articulo    : secciones del artículo con su página
  * analizar_figura_o_tabla: Qwen2.5-VL sobre la imagen de una figura/tabla
Salida tipada: RespuestaAgente (respondible, respuesta, citas, confianza), validada para que
solo cite fragmentos realmente recuperados (si no, ModelRetry).
"""
from __future__ import annotations

import base64
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anyio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from .config import get_settings
from .search import Fragmento, Retriever, Tiempos

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- salidas tipadas
class Cita(BaseModel):
    chunk_id: str = Field(description="chunk_id EXACTO de un fragmento devuelto por buscar_en_articulo")
    cita_textual: str = Field(
        default="", description="Frase corta copiada del fragmento que respalda la afirmación", max_length=500
    )


class RespuestaAgente(BaseModel):
    respondible: bool = Field(
        description="true solo si los fragmentos del artículo permiten responder; false si la información no está en el artículo"
    )
    respuesta: str = Field(
        description="Respuesta en español, en Markdown, con marcadores [1], [2]... que remiten al orden de 'citas'. "
        "Si respondible=false, explica brevemente que el artículo no contiene esa información."
    )
    citas: list[Cita] = Field(default_factory=list, description="Fragmentos que respaldan la respuesta, en orden")
    confianza: Literal["alta", "media", "baja"] = "media"


NO_RESPONDIBLE = (
    "No encuentro esa información en el artículo, así que no puedo responderla sin inventar. "
    "Puedo ayudarte con preguntas sobre su contenido: métodos, taxonomía, resultados, tablas o figuras."
)


# --------------------------------------------------------------------------- dependencias
@dataclass
class Deps:
    retriever: Retriever
    doc_id: str | None = None
    documentos: list[dict] = field(default_factory=list)
    tiempos: Tiempos = field(default_factory=Tiempos)
    recuperados: dict[str, Fragmento] = field(default_factory=dict)
    hits: list[dict] = field(default_factory=list)
    llamadas_vision: list[dict] = field(default_factory=list)
    on_status: Callable[[str], Awaitable[None]] | None = None

    async def estado(self, msg: str) -> None:
        if self.on_status:
            try:
                await self.on_status(msg)
            except Exception:
                pass


INSTRUCCIONES = """Eres un asistente académico que responde preguntas EXCLUSIVAMENTE con base en el/los
artículo(s) científicos indexados (surveys). Reglas:

1. Para cualquier pregunta sobre el contenido, llama SIEMPRE a `buscar_en_articulo` antes de responder.
   Escribe la consulta de búsqueda en INGLÉS (el idioma del artículo) con términos técnicos concretos.
   Si el primer resultado no basta, reformula y busca otra vez (máximo 3 búsquedas).
2. Para preguntas sobre la estructura del artículo usa `indice_del_articulo`.
3. Si la pregunta es sobre lo que MUESTRA una figura o tabla (valores, ejes, diagrama), busca con
   tipo="figura" o tipo="tabla" y luego llama `analizar_figura_o_tabla` con su chunk_id.
4. Responde en español, de forma precisa y concisa, usando solo lo que dicen los fragmentos.
   Pon marcadores [1], [2]... y en `citas` incluye el chunk_id de cada fragmento usado, en ese orden.
5. Si los fragmentos no contienen la respuesta, o la pregunta no tiene relación con el artículo
   (actualidad, opiniones personales, otros temas), devuelve respondible=false, citas vacías y explica
   que el artículo no contiene esa información. NUNCA inventes datos, cifras, autores ni páginas.
6. Usa el historial para resolver referencias como "eso", "la segunda técnica", etc.
"""


def _modelo():
    s = get_settings()
    settings = {"temperature": s.llm_temperature, "timeout": s.llm_timeout_seconds}
    if s.agent_backend == "vllm":
        return OpenAIChatModel(
            s.vllm_model, provider=OpenAIProvider(base_url=s.vllm_base_url, api_key=s.vllm_api_key), settings=settings
        ), s.vllm_model, "vllm"
    return OpenAIChatModel(
        s.ollama_model, provider=OllamaProvider(base_url=s.ollama_base_url), settings=settings
    ), s.ollama_model, "ollama"


def construir_agente(model=None) -> Agent[Deps, RespuestaAgente]:
    agent = Agent(
        model or _modelo()[0],
        deps_type=Deps,
        output_type=RespuestaAgente,
        instructions=INSTRUCCIONES,
        retries=2,
        name="agente_academico",
    )

    @agent.instructions
    def documentos_disponibles(ctx: RunContext[Deps]) -> str:
        if not ctx.deps.documentos:
            return ""
        lst = "\n".join(f"- {d['id']}: {d['title']}" for d in ctx.deps.documentos)
        filtro = f"\nLa búsqueda está restringida al documento '{ctx.deps.doc_id}'." if ctx.deps.doc_id else ""
        return f"Artículos indexados:\n{lst}{filtro}"

    @agent.tool
    async def buscar_en_articulo(
        ctx: RunContext[Deps],
        consulta: str,
        tipo: Literal["todo", "texto", "tabla", "figura", "referencias"] = "todo",
    ) -> list[dict] | str:
        """Busca fragmentos relevantes del artículo (búsqueda híbrida léxica+vectorial con reranking).

        Args:
            consulta: consulta de búsqueda en inglés, con términos técnicos concretos.
            tipo: restringe el tipo de fragmento: todo, texto, tabla, figura o referencias.
        """
        tipos = {
            "todo": ["text", "table", "figure"],
            "texto": ["text"],
            "tabla": ["table"],
            "figura": ["figure"],
            "referencias": ["reference"],
        }[tipo]
        await ctx.deps.estado(f"Buscando en el artículo: «{consulta}»")
        try:
            frags = await anyio.to_thread.run_sync(
                lambda: ctx.deps.retriever.buscar(consulta, doc_id=ctx.deps.doc_id, tipos=tipos, tiempos=ctx.deps.tiempos)
            )
        except Exception as e:
            log.exception("Error en la recuperación")
            raise ModelRetry(f"La búsqueda falló ({e}). Intenta con otra consulta más corta.") from e
        if not frags:
            return "Sin resultados. Reformula la consulta o concluye que el artículo no trata ese tema."
        base = len(ctx.deps.hits)
        for i, f in enumerate(frags, 1):
            ctx.deps.recuperados[f.chunk_id] = f
            ctx.deps.hits.append({"query": consulta, "rank": base + i, "frag": f})
        return [f.para_llm() for f in frags]

    @agent.tool
    async def indice_del_articulo(ctx: RunContext[Deps]) -> list[dict]:
        """Devuelve la lista de secciones del artículo con la página donde empiezan."""
        await ctx.deps.estado("Consultando el índice del artículo")
        return await anyio.to_thread.run_sync(lambda: ctx.deps.retriever.secciones(ctx.deps.doc_id))

    @agent.tool
    async def analizar_figura_o_tabla(ctx: RunContext[Deps], chunk_id: str, pregunta: str) -> str:
        """Analiza visualmente una figura o tabla del artículo con Qwen2.5-VL.

        Args:
            chunk_id: chunk_id de una figura o tabla devuelto por buscar_en_articulo.
            pregunta: qué quieres saber de la imagen.
        """
        frag = ctx.deps.recuperados.get(chunk_id)
        if frag is None:
            mapa = await anyio.to_thread.run_sync(lambda: ctx.deps.retriever.por_id([chunk_id]))
            frag = mapa.get(chunk_id)
        if frag is None or not frag.image_path or not Path(frag.image_path).exists():
            raise ModelRetry("Ese chunk_id no es una figura/tabla con imagen. Busca primero con tipo='figura' o 'tabla'.")
        ctx.deps.recuperados[chunk_id] = frag
        await ctx.deps.estado(f"Analizando imagen de {frag.caption or chunk_id} con Qwen2.5-VL")
        return await analizar_imagen(frag, pregunta, ctx.deps)

    @agent.output_validator
    async def validar(ctx: RunContext[Deps], out: RespuestaAgente) -> RespuestaAgente:
        if ctx.partial_output:
            return out
        if not out.respondible:
            out.citas = []
            if not out.respuesta.strip():
                out.respuesta = NO_RESPONDIBLE
            return out
        if not ctx.deps.recuperados:
            raise ModelRetry("Antes de responder debes llamar a buscar_en_articulo para fundamentar la respuesta.")
        validas = [c for c in out.citas if c.chunk_id in ctx.deps.recuperados]
        if not validas:
            ids = ", ".join(list(ctx.deps.recuperados)[:10])
            raise ModelRetry(
                f"Las citas deben usar chunk_id devueltos por la herramienta ({ids}). "
                "Si ninguno respalda la respuesta, devuelve respondible=false."
            )
        out.citas = validas
        return out

    return agent


async def analizar_imagen(frag: Fragmento, pregunta: str, deps: Deps) -> str:
    """Llamada directa a vLLM (API OpenAI) con la imagen en base64."""
    from openai import AsyncOpenAI

    s = get_settings()
    t0 = time.perf_counter()
    cli = AsyncOpenAI(base_url=s.vllm_base_url, api_key=s.vllm_api_key, timeout=s.llm_timeout_seconds)
    b64 = base64.b64encode(Path(frag.image_path).read_bytes()).decode()
    try:
        r = await cli.chat.completions.create(
            model=s.vllm_model,
            temperature=0,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": f"This is {frag.caption or 'a figure/table'} from a scientific survey "
                                             f"(page {frag.page_start}). Answer factually based only on the image: {pregunta}"},
                ],
            }],
        )
    except Exception as e:
        raise ModelRetry(f"El modelo de visión no está disponible ({e}). Responde con el pie y el texto del fragmento.") from e
    ms = (time.perf_counter() - t0) * 1000
    deps.tiempos.extra["vision_ms"] = deps.tiempos.extra.get("vision_ms", 0) + ms
    u = r.usage
    deps.llamadas_vision.append({
        "model": s.vllm_model, "provider": "vllm", "purpose": "vision",
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0, "latency_ms": ms,
    })
    return r.choices[0].message.content or ""


def historial_a_mensajes(historial: list[dict]) -> list[ModelMessage]:
    """Convierte el historial guardado en PostgreSQL a mensajes de Pydantic AI."""
    msgs: list[ModelMessage] = []
    for m in historial:
        if m["role"] == "user":
            msgs.append(ModelRequest(parts=[UserPromptPart(content=m["content"])]))
        else:
            msgs.append(ModelResponse(parts=[TextPart(content=m["content"])]))
    return msgs


@dataclass
class ResultadoAgente:
    salida: RespuestaAgente
    modelo: str
    proveedor: str
    input_tokens: int
    output_tokens: int
    requests: int
    tool_calls: int
    llm_ms: float
    total_ms: float
    deps: Deps


_agente: Agent | None = None


def get_agente() -> Agent[Deps, RespuestaAgente]:
    global _agente
    if _agente is None:
        _agente = construir_agente()
    return _agente


def _usage(u) -> tuple[int, int, int, int]:
    return (
        getattr(u, "input_tokens", 0) or 0,
        getattr(u, "output_tokens", 0) or 0,
        getattr(u, "requests", 0) or 0,
        getattr(u, "tool_calls", 0) or 0,
    )


def _resultado(salida, usage, deps: Deps, t0: float) -> ResultadoAgente:
    _, modelo, proveedor = _modelo()
    total = (time.perf_counter() - t0) * 1000
    t = deps.tiempos
    llm_ms = max(0.0, total - t.embed_ms - t.search_ms - t.rerank_ms - t.extra.get("vision_ms", 0))
    i, o, r, tc = _usage(usage)
    return ResultadoAgente(salida, modelo, proveedor, i, o, r, tc, llm_ms, total, deps)


async def ejecutar(pregunta: str, historial: list[dict], deps: Deps) -> ResultadoAgente:
    t0 = time.perf_counter()
    res = await get_agente().run(pregunta, deps=deps, message_history=historial_a_mensajes(historial))
    return _resultado(res.output, res.usage(), deps, t0)


async def ejecutar_stream(
    pregunta: str,
    historial: list[dict],
    deps: Deps,
    on_delta: Callable[[str, bool], Awaitable[None]],
) -> ResultadoAgente:
    """Igual que `ejecutar` pero emite el texto de `respuesta` a medida que el modelo lo genera.

    Se usa `agent.run(..., event_stream_handler=...)` (y no `run_stream`) para conservar los
    reintentos del validador de salida. Los argumentos JSON de la herramienta de salida llegan
    por fragmentos y se parsean de forma parcial con pydantic_core. `on_delta(texto, reset)`:
    reset=True indica que empezó un nuevo intento y el cliente debe borrar lo mostrado.
    """
    from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, ToolCallPart, ToolCallPartDelta
    from pydantic_core import from_json

    t0 = time.perf_counter()

    async def handler(_ctx, events):
        buffers: dict[int, str] = {}
        enviado: dict[int, str] = {}
        async for ev in events:
            if isinstance(ev, PartStartEvent) and isinstance(ev.part, ToolCallPart) and ev.part.tool_name == "final_result":
                args = ev.part.args
                buffers[ev.index] = args if isinstance(args, str) else ""
                enviado[ev.index] = ""
                await on_delta("", True)
            elif isinstance(ev, PartDeltaEvent) and isinstance(ev.delta, ToolCallPartDelta) and ev.index in buffers:
                if isinstance(ev.delta.args_delta, str):
                    buffers[ev.index] += ev.delta.args_delta
                else:
                    continue
            else:
                continue
            try:
                parcial = from_json(buffers[ev.index] or "{}", allow_partial=True)
            except ValueError:
                continue
            texto = parcial.get("respuesta") or "" if isinstance(parcial, dict) else ""
            prev = enviado[ev.index]
            if isinstance(texto, str) and texto.startswith(prev) and len(texto) > len(prev):
                await on_delta(texto[len(prev):], False)
                enviado[ev.index] = texto

    res = await get_agente().run(
        pregunta, deps=deps, message_history=historial_a_mensajes(historial), event_stream_handler=handler
    )
    return _resultado(res.output, res.usage(), deps, t0)
