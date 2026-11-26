"""Embeddings con modelos de Hugging Face (sentence-transformers).

Modelo principal: nvidia/Nemotron-3-Embed-1B-BF16 (configurable con EMBED_MODEL).
Si no se puede cargar (sin GPU/RAM, sin acceso al repo, versión de transformers, ...)
se usa EMBED_FALLBACK_MODEL (BAAI/bge-m3, multilingüe, 1024 dims).

La dimensión del índice k-NN se obtiene del propio modelo, así el índice de OpenSearch
siempre coincide con el modelo realmente cargado.
"""
from __future__ import annotations

import logging
import threading
import time

from .config import get_settings

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        s = get_settings()
        self.query_prefix = s.embed_query_prefix
        self.doc_prefix = s.embed_doc_prefix
        self.batch_size = s.embed_batch_size
        t0 = time.perf_counter()
        try:
            self.model = SentenceTransformer(
                s.embed_model, device=s.embed_device, trust_remote_code=True, token=s.hf_token
            )
            self.model_name = s.embed_model
        except Exception as e:
            log.warning("No se pudo cargar %s (%s). Usando fallback %s", s.embed_model, e, s.embed_fallback_model)
            self.model = SentenceTransformer(s.embed_fallback_model, device=s.embed_device, token=s.hf_token)
            self.model_name = s.embed_fallback_model
            # bge-m3 no usa prefijos
            self.query_prefix = self.doc_prefix = ""
        self.dim = len(self.model.encode(["dimension"], normalize_embeddings=True)[0])
        log.info("Embeddings: %s (dim=%s) cargado en %.1fs", self.model_name, self.dim, time.perf_counter() - t0)

    def embed_queries(self, textos: list[str]) -> list[list[float]]:
        v = self.model.encode([self.query_prefix + t for t in textos], normalize_embeddings=True,
                              batch_size=self.batch_size)
        return [x.tolist() for x in v]

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        v = self.model.encode([self.doc_prefix + t for t in textos], normalize_embeddings=True,
                              batch_size=self.batch_size, show_progress_bar=True)
        return [x.tolist() for x in v]


_embedder: Embedder | None = None
_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = Embedder()
    return _embedder
