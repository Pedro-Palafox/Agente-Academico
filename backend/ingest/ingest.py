"""Ingesta de artículos con Docling -> chunks jerárquicos -> embeddings HF -> OpenSearch.

Uso (dentro del contenedor, lo ejecuta automáticamente el servicio `ingest` del compose):
    python -m ingest.ingest                 # ingiere todos los PDF de /data/papers (idempotente)
    python -m ingest.ingest --force         # recrea el índice y reingiere todo
    python -m ingest.ingest --pdf /data/papers/mi_survey.pdf

Qué conserva:
  * Estructura: Docling detecta títulos, secciones, listas, tablas, figuras con pie y referencias.
  * Chunking: HybridChunker de Docling (respeta la jerarquía: nunca mezcla secciones y
    fusiona párrafos hermanos pequeños) con límite de tokens del tokenizer del modelo.
  * Metadatos por chunk: documento, ruta de encabezados, sección, página inicial/final, tipo.
  * Tablas: chunk propio con la tabla en Markdown + pie de tabla (tipo "table").
  * Figuras: imagen PNG guardada + pie de figura (+ descripción opcional generada con
    Qwen2.5-VL en vLLM) como chunk propio (tipo "figure").
  * Referencias bibliográficas: chunks de tipo "reference".
  * Exporta el documento a Markdown y JSON en /data/processed como evidencia de la conversión.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import sys
import time
from pathlib import Path

from opensearchpy import helpers

from app.config import get_settings
from app.embeddings import get_embedder
from app.search import asegurar_pipeline, crear_indice, os_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ingest] %(message)s")
log = logging.getLogger("ingest")

REF_RE = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography|referencias|bibliograf[ií]a)\s*$", re.I)


def slug(nombre: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", Path(nombre).stem.lower()).strip("-")
    return s[:60] or "articulo"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- Docling
def convertir(pdf: Path):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False  # los surveys Q1 son PDF digitales; activar si el PDF es escaneado
    opts.do_table_structure = True
    opts.table_structure_options.do_cell_matching = True
    opts.generate_picture_images = True
    opts.generate_page_images = True  # necesario para recortar tablas como imagen
    opts.images_scale = 2.0
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    t0 = time.perf_counter()
    res = conv.convert(str(pdf))
    log.info("Docling convirtió %s en %.1fs", pdf.name, time.perf_counter() - t0)
    return res.document


def crear_chunker():
    s = get_settings()
    from docling.chunking import HybridChunker
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(s.chunk_tokenizer, token=s.hf_token)
    try:  # API actual de docling-core
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

        tok = HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=s.chunk_max_tokens)
        return HybridChunker(tokenizer=tok, merge_peers=True)
    except ImportError:  # versiones antiguas
        return HybridChunker(tokenizer=hf_tok, max_tokens=s.chunk_max_tokens, merge_peers=True)


def titulo_documento(doc, fallback: str) -> str:
    from docling_core.types.doc import DocItemLabel

    for t in doc.texts:
        if t.label == DocItemLabel.TITLE and t.text.strip():
            return t.text.strip()
    return fallback


def paginas(items) -> list[int]:
    return sorted({p.page_no for it in items for p in (getattr(it, "prov", None) or [])})


def describir_imagen_vlm(png: bytes, pie: str) -> str | None:
    """Descripción opcional de figuras con Qwen2.5-VL (vLLM) para hacerlas buscables."""
    s = get_settings()
    try:
        from openai import OpenAI

        cli = OpenAI(base_url=s.vllm_base_url, api_key=s.vllm_api_key, timeout=120)
        b64 = base64.b64encode(png).decode()
        r = cli.chat.completions.create(
            model=s.vllm_model,
            temperature=0,
            max_tokens=350,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Describe in English, factually and concisely, what this figure from a "
                                             f"scientific survey shows (axes, components, flow, key values). Caption: {pie}"},
                ],
            }],
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        log.warning("No se pudo describir la figura con vLLM (%s). Se indexa solo el pie.", e)
        return None


def construir_chunks(doc, doc_id: str, titulo: str, describir_figuras: bool) -> tuple[list[dict], dict]:
    s = get_settings()
    from docling_core.types.doc import DocItemLabel

    chunker = crear_chunker()
    figs_dir = Path(s.figures_dir) / doc_id
    figs_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict] = []
    orden = 0
    en_referencias = False

    # 1) Texto: chunking jerárquico
    for ch in chunker.chunk(dl_doc=doc):
        items = ch.meta.doc_items
        labels = {it.label for it in items}
        # Las tablas/figuras se indexan aparte con su propio chunk enriquecido
        if labels and labels <= {DocItemLabel.TABLE, DocItemLabel.PICTURE, DocItemLabel.CAPTION}:
            continue
        heads = [h for h in (ch.meta.headings or []) if h]
        seccion = heads[-1] if heads else "Front matter"
        if any(REF_RE.match(h) for h in heads):
            en_referencias = True
        tipo = "reference" if (en_referencias or DocItemLabel.REFERENCE in labels) else "text"
        pags = paginas(items) or [None]
        texto = chunker.contextualize(chunk=ch)  # antepone la ruta de encabezados
        orden += 1
        chunks.append({
            "chunk_id": f"{doc_id}-{orden:04d}", "doc_id": doc_id, "doc_title": titulo,
            "text": texto, "raw_text": ch.text, "section": seccion, "headings": heads,
            "page_start": pags[0], "page_end": pags[-1], "chunk_type": tipo, "order": orden,
        })

    # Sección a la que pertenece un elemento = último chunk de texto que empieza en su página o antes
    def seccion_para(pagina: int | None) -> str:
        sec = "Front matter"
        for c in chunks:
            if c["chunk_type"] == "text" and c["page_start"] and pagina and c["page_start"] <= pagina:
                sec = c["section"]
        return sec

    # 2) Tablas
    n_tab = 0
    for i, tab in enumerate(doc.tables, 1):
        md = tab.export_to_markdown(doc=doc)
        pie = tab.caption_text(doc) or f"Table {i}"
        pag = tab.prov[0].page_no if tab.prov else None
        img_path = None
        try:
            img = tab.get_image(doc)
            if img is not None:
                p = figs_dir / f"tabla_{i:02d}.png"
                img.save(p)
                img_path = str(p)
        except Exception:
            pass
        n_tab += 1
        orden += 1
        sec = seccion_para(pag)
        chunks.append({
            "chunk_id": f"{doc_id}-T{i:02d}", "doc_id": doc_id, "doc_title": titulo,
            "text": f"{sec}\n{pie}\n\n{md}", "raw_text": md, "section": sec, "headings": [sec],
            "page_start": pag, "page_end": pag, "chunk_type": "table", "caption": pie,
            "image_path": img_path, "order": orden,
        })

    # 3) Figuras
    n_fig = 0
    for i, pic in enumerate(doc.pictures, 1):
        pie = pic.caption_text(doc) or ""
        pag = pic.prov[0].page_no if pic.prov else None
        img = pic.get_image(doc)
        if img is None or (img.width < 80 and img.height < 80):  # logos e iconos
            continue
        p = figs_dir / f"figura_{i:02d}.png"
        img.save(p)
        desc = None
        if describir_figuras:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            desc = describir_imagen_vlm(buf.getvalue(), pie)
        n_fig += 1
        orden += 1
        sec = seccion_para(pag)
        texto = f"{sec}\nFigure {i}: {pie or '(sin pie)'}" + (f"\n\nDescription: {desc}" if desc else "")
        chunks.append({
            "chunk_id": f"{doc_id}-F{i:02d}", "doc_id": doc_id, "doc_title": titulo,
            "text": texto, "raw_text": pie, "section": sec, "headings": [sec],
            "page_start": pag, "page_end": pag, "chunk_type": "figure", "caption": pie or f"Figure {i}",
            "image_path": str(p), "order": orden,
        })
    stats = {"tables": n_tab, "figures": n_fig, "pages": doc.num_pages() if callable(getattr(doc, "num_pages", None)) else None}
    return chunks, stats


# ----------------------------------------------------------------------------- Postgres
async def registrar_documento(doc_id, titulo, pdf: Path, digest, stats, n_chunks) -> None:
    import asyncpg

    s = get_settings()
    con = await asyncpg.connect(s.postgres_dsn)
    try:
        schema = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"
        await con.execute(schema.read_text(encoding="utf-8"))
        await con.execute(
            """INSERT INTO documents (id, title, filename, sha256, num_pages, num_chunks, num_tables, num_figures)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
               ON CONFLICT (id) DO UPDATE SET title=EXCLUDED.title, sha256=EXCLUDED.sha256,
                 num_pages=EXCLUDED.num_pages, num_chunks=EXCLUDED.num_chunks,
                 num_tables=EXCLUDED.num_tables, num_figures=EXCLUDED.num_figures, ingested_at=now()""",
            doc_id, titulo, pdf.name, digest, stats.get("pages"), n_chunks, stats["tables"], stats["figures"],
        )
    finally:
        await con.close()


def ya_indexado(client, doc_id: str) -> bool:
    s = get_settings()
    if not client.indices.exists(index=s.opensearch_index):
        return False
    r = client.count(index=s.opensearch_index, body={"query": {"term": {"doc_id": doc_id}}})
    return r["count"] > 0


def ingerir(pdf: Path, client, describir_figuras: bool, force: bool) -> None:
    s = get_settings()
    doc_id = slug(pdf.name)
    if not force and ya_indexado(client, doc_id):
        log.info("'%s' ya está indexado. Usa --force para reingerir.", doc_id)
        return
    doc = convertir(pdf)
    out = Path(s.papers_dir).parent / "processed"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{doc_id}.md").write_text(doc.export_to_markdown(), encoding="utf-8")
    (out / f"{doc_id}.json").write_text(json.dumps(doc.export_to_dict(), ensure_ascii=False), encoding="utf-8")

    titulo = titulo_documento(doc, pdf.stem)
    chunks, stats = construir_chunks(doc, doc_id, titulo, describir_figuras)
    log.info("%s: %d chunks (%d tablas, %d figuras)", doc_id, len(chunks), stats["tables"], stats["figures"])

    emb = get_embedder()
    vectores = emb.embed_documents([c["text"] for c in chunks])
    client.delete_by_query(index=s.opensearch_index, body={"query": {"term": {"doc_id": doc_id}}}, refresh=True)
    acciones = [
        {"_index": s.opensearch_index, "_id": c["chunk_id"], "_source": {**c, "embedding": v}}
        for c, v in zip(chunks, vectores)
    ]
    helpers.bulk(client, acciones, refresh=True)
    (out / f"{doc_id}.chunks.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in c.items()}, ensure_ascii=False) for c in chunks), encoding="utf-8"
    )
    asyncio.run(registrar_documento(doc_id, titulo, pdf, sha256(pdf), stats, len(chunks)))
    log.info("OK: '%s' indexado en OpenSearch (%s)", titulo, s.opensearch_index)


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description="Ingesta con Docling")
    ap.add_argument("--pdf", type=Path, help="PDF concreto (por defecto todos los de PAPERS_DIR)")
    ap.add_argument("--force", action="store_true", help="recrear el índice y reingerir")
    ap.add_argument("--describir-figuras", action="store_true",
                    help="describir figuras con Qwen2.5-VL (requiere vLLM arriba)")
    a = ap.parse_args()
    describir = a.describir_figuras or str(__import__("os").getenv("INGEST_DESCRIBE_FIGURES", "false")).lower() == "true"

    pdfs = [a.pdf] if a.pdf else sorted(Path(s.papers_dir).glob("*.pdf"))
    if not pdfs:
        log.warning("=" * 70)
        log.warning("No hay PDFs en %s. Copia tu survey Q1 ahí y ejecuta:", s.papers_dir)
        log.warning("   docker compose run --rm ingest")
        log.warning("=" * 70)
        return 0

    client = os_client()
    emb = get_embedder()  # define la dimensión del índice
    crear_indice(client, emb.dim, recrear=a.force)
    asegurar_pipeline(client)  # por si cambiaron los pesos en .env
    for pdf in pdfs:
        try:
            ingerir(pdf, client, describir, a.force)
        except Exception:
            log.exception("Falló la ingesta de %s", pdf)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
