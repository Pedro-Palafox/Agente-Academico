"""Reranking con un cross-encoder de Hugging Face.

Principal: nvidia/llama-nemotron-rerank-1b-v2 (formato "question: ... passage: ...").
Fallback:  BAAI/bge-reranker-v2-m3 (CrossEncoder de sentence-transformers).
El efecto del reranking se mide en eval/eval_retrieval.py (modo hybrid_rerank).
"""
from __future__ import annotations

import logging
import threading

from .config import get_settings

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self) -> None:
        s = get_settings()
        self.device = s.embed_device
        self._mode = "nemotron"
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(s.rerank_model, trust_remote_code=True, token=s.hf_token)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForSequenceClassification.from_pretrained(
                s.rerank_model, trust_remote_code=True, token=s.hf_token,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            ).to(self.device).eval()
            self.model_name = s.rerank_model
        except Exception as e:
            log.warning("No se pudo cargar %s (%s). Usando fallback %s", s.rerank_model, e, s.rerank_fallback_model)
            from sentence_transformers import CrossEncoder

            self.ce = CrossEncoder(s.rerank_fallback_model, device=self.device)
            self.model_name = s.rerank_fallback_model
            self._mode = "crossencoder"
        log.info("Reranker cargado: %s", self.model_name)

    def score(self, consulta: str, pasajes: list[str]) -> list[float]:
        if not pasajes:
            return []
        if self._mode == "crossencoder":
            return [float(x) for x in self.ce.predict([(consulta, p) for p in pasajes])]
        import torch

        textos = [f"question:{consulta} \n \n passage:{p}" for p in pasajes]
        with torch.no_grad():
            enc = self.tok(textos, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
        return logits.view(-1).float().cpu().tolist()


_reranker: Reranker | None = None
_lock = threading.Lock()


def get_reranker() -> Reranker:
    global _reranker
    with _lock:
        if _reranker is None:
            _reranker = Reranker()
    return _reranker
