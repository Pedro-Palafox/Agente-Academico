"""OpenSearch: índice k-NN, pipeline de búsqueda híbrida y recuperador.

Modos de recuperación (todos se comparan en eval/eval_retrieval.py):
  - bm25           : solo léxica (match sobre el texto contextualizado)
  - knn            : solo vectorial (HNSW, similitud coseno)
  - hybrid         : consulta `hybrid` + normalization-processor (min-max + media ponderada)
  - hybrid_rerank  : hybrid + cross-encoder (llama-nemotron-rerank-1b-v2)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import redis
from opensearchpy import OpenSearch

from .cache import clave_embedding
from .config import get_settings

log = logging.getLogger(__name__)

MODOS = ("bm25", "knn", "hybrid", "hybrid_rerank")


def os_client() -> OpenSearch:
    return OpenSearch(get_settings().opensearch_url, timeout=60, max_retries=3, retry_on_timeout=True)


def index_body(dim: int) -> dict:
    """Mapping del índice. `dim` sale del modelo de embeddings cargado."""
    return {
        "settings": {
            "index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0},
            "analysis": {
                "analyzer": {
                    # El artículo está en inglés: stemming en inglés + minúsculas + ascii folding
                    "articulo": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding", "english_stop", "english_stemmer"],
                    }
                },
                "filter": {
                    "english_stop": {"type": "stop", "stopwords": "_english_"},
                    "english_stemmer": {"type": "stemmer", "language": "english"},
                },
            },
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "doc_title": {"type": "text"},
                "text": {"type": "text", "analyzer": "articulo"},  # texto contextualizado (con encabezados)
                "raw_text": {"type": "text", "index": False},
                "section": {"type": "text", "analyzer": "articulo", "fields": {"kw": {"type": "keyword"}}},
                "headings": {"type": "keyword"},
                "page_start": {"type": "integer"},
                "page_end": {"type": "integer"},
                "chunk_type": {"type": "keyword"},  # text | table | figure
                "caption": {"type": "text", "analyzer": "articulo"},
                "image_path": {"type": "keyword"},
                "order": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",  # embeddings normalizados -> coseno
                        "engine": "lucene",
                        "parameters": {"m": 16, "ef_construction": 128},
                    },
                },
            }
        },
    }


def asegurar_pipeline(client: OpenSearch, nombre: str | None = None, pesos: tuple[float, float] | None = None) -> None:
    s = get_settings()
    w_lex, w_vec = pesos or (s.hybrid_weight_lexical, s.hybrid_weight_vector)
    client.search_pipeline.put(
        id=nombre or s.opensearch_pipeline,
        body={
            "description": "Búsqueda híbrida: min-max + media aritmética ponderada (léxica, vectorial)",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [w_lex, w_vec]},
                        },
                    }
                }
            ],
        },
    )


def crear_indice(client: OpenSearch, dim: int, recrear: bool = False) -> None:
    s = get_settings()
    if client.indices.exists(index=s.opensearch_index):
        if not recrear:
            return
        client.indices.delete(index=s.opensearch_index)
    client.indices.create(index=s.opensearch_index, body=index_body(dim))
    asegurar_pipeline(client)


@dataclass
class Fragmento:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    page_start: int | None
    page_end: int | None
    chunk_type: str
    text: str
    caption: str | None = None
    image_path: str | None = None
    score: float | None = None
    rerank_score: float | None = None

    def para_llm(self, max_chars: int = 1800) -> dict:
        pag = self.page_start if self.page_start == self.page_end else f"{self.page_start}-{self.page_end}"
        return {
            "chunk_id": self.chunk_id,
            "articulo": self.doc_title,
            "seccion": self.section,
            "pagina": pag,
            "tipo": self.chunk_type,
            "texto": self.text[:max_chars],
        }


@dataclass
class Tiempos:
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    embed_cache_hit: bool = False
    extra: dict = field(default_factory=dict)


class Retriever:
    def __init__(self) -> None:
        self.s = get_settings()
        self.client = os_client()
        self._redis = redis.Redis.from_url(self.s.redis_url, decode_responses=True)

    # ---------- embeddings con caché en Redis ----------
    def _embedding(self, consulta: str, t: Tiempos) -> list[float]:
        from .embeddings import get_embedder

        emb = get_embedder()
        key = clave_embedding(consulta, emb.model_name)
        t0 = time.perf_counter()
        try:
            raw = self._redis.get(key) if self.s.cache_enabled else None
        except Exception:
            raw = None
        if raw:
            t.embed_cache_hit = True
            self._safe_incr("embed_hit")
            v = json.loads(raw)
        else:
            v = emb.embed_queries([consulta])[0]
            self._safe_incr("embed_miss")
            try:
                if self.s.cache_enabled:
                    self._redis.set(key, json.dumps(v), ex=self.s.embed_cache_ttl_seconds)
            except Exception:
                pass
        t.embed_ms += (time.perf_counter() - t0) * 1000
        return v

    def _safe_incr(self, nombre: str) -> None:
        try:
            self._redis.incr(f"stats:{nombre}")
        except Exception:
            pass

    # ---------- construcción de consultas ----------
    @staticmethod
    def _filtros(doc_id: str | None, tipos: list[str] | None) -> list[dict]:
        f = []
        if doc_id:
            f.append({"term": {"doc_id": doc_id}})
        if tipos:
            f.append({"terms": {"chunk_type": tipos}})
        return f

    def _q_lexica(self, consulta: str, filtros: list[dict]) -> dict:
        return {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": consulta,
                        "fields": ["text", "section^2", "caption^2"],
                    }
                }],
                "filter": filtros,
            }
        }

    def _q_knn(self, vector: list[float], k: int, filtros: list[dict]) -> dict:
        knn: dict = {"vector": vector, "k": k}
        if filtros:
            knn["filter"] = {"bool": {"filter": filtros}}
        return {"knn": {"embedding": knn}}

    # ---------- búsqueda ----------
    def buscar(
        self,
        consulta: str,
        modo: str = "hybrid_rerank",
        k: int | None = None,
        doc_id: str | None = None,
        tipos: list[str] | None = None,
        tiempos: Tiempos | None = None,
        pipeline: str | None = None,
    ) -> list[Fragmento]:
        if modo not in MODOS:
            raise ValueError(f"modo inválido: {modo}")
        if modo == "hybrid_rerank" and not self.s.rerank_enabled:
            modo = "hybrid"
        t = tiempos or Tiempos()
        k = k or self.s.retrieval_top_k
        n_cand = max(self.s.retrieval_candidates, k) if modo == "hybrid_rerank" else k
        filtros = self._filtros(doc_id, tipos)
        params = {}
        excl = {"_source": {"excludes": ["embedding"]}}

        if modo == "bm25":
            body = {"size": n_cand, "query": self._q_lexica(consulta, filtros), **excl}
        elif modo == "knn":
            v = self._embedding(consulta, t)
            body = {"size": n_cand, "query": self._q_knn(v, n_cand, filtros), **excl}
        else:
            v = self._embedding(consulta, t)
            body = {
                "size": n_cand,
                "query": {"hybrid": {"queries": [self._q_lexica(consulta, filtros), self._q_knn(v, n_cand, filtros)]}},
                **excl,
            }
            params = {"search_pipeline": pipeline or self.s.opensearch_pipeline}

        t0 = time.perf_counter()
        res = self.client.search(index=self.s.opensearch_index, body=body, params=params)
        t.search_ms += (time.perf_counter() - t0) * 1000
        frags = [self._a_fragmento(h) for h in res["hits"]["hits"]]

        if modo == "hybrid_rerank" and frags:
            from .reranker import get_reranker

            t0 = time.perf_counter()
            scores = get_reranker().score(consulta, [f.text for f in frags])
            for f, sc in zip(frags, scores):
                f.rerank_score = sc
            frags.sort(key=lambda f: f.rerank_score or 0.0, reverse=True)
            t.rerank_ms += (time.perf_counter() - t0) * 1000
        return frags[:k]

    @staticmethod
    def _a_fragmento(hit: dict) -> Fragmento:
        s = hit["_source"]
        return Fragmento(
            chunk_id=s["chunk_id"], doc_id=s["doc_id"], doc_title=s.get("doc_title", ""),
            section=s.get("section") or "(sin sección)", page_start=s.get("page_start"),
            page_end=s.get("page_end"), chunk_type=s.get("chunk_type", "text"), text=s.get("text", ""),
            caption=s.get("caption"), image_path=s.get("image_path"), score=hit.get("_score"),
        )

    def secciones(self, doc_id: str | None = None) -> list[dict]:
        """Índice del artículo (secciones con su primera página), útil para preguntas de estructura."""
        filtros = self._filtros(doc_id, None)
        body = {
            "size": 0,
            "query": {"bool": {"filter": filtros}} if filtros else {"match_all": {}},
            "aggs": {
                "docs": {
                    "terms": {"field": "doc_id", "size": 20},
                    "aggs": {
                        "secs": {
                            "terms": {"field": "section.kw", "size": 200, "order": {"primera": "asc"}},
                            "aggs": {"primera": {"min": {"field": "order"}}, "pag": {"min": {"field": "page_start"}}},
                        }
                    },
                }
            },
        }
        res = self.client.search(index=self.s.opensearch_index, body=body)
        out = []
        for d in res["aggregations"]["docs"]["buckets"]:
            for b in d["secs"]["buckets"]:
                out.append({"doc_id": d["key"], "seccion": b["key"], "pagina": int(b["pag"]["value"] or 0)})
        return out

    def por_id(self, chunk_ids: list[str]) -> dict[str, Fragmento]:
        if not chunk_ids:
            return {}
        res = self.client.search(
            index=self.s.opensearch_index,
            body={"size": len(chunk_ids), "query": {"terms": {"chunk_id": chunk_ids}}, "_source": {"excludes": ["embedding"]}},
        )
        return {h["_source"]["chunk_id"]: self._a_fragmento(h) for h in res["hits"]["hits"]}


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
