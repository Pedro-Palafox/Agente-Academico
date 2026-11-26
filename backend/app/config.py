"""Configuración central leída desde variables de entorno (.env).

Todas las credenciales y parámetros viven en el archivo .env (nunca en el repositorio).
Ver .env.example para la lista completa y su explicación.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- PostgreSQL ---
    postgres_user: str = "agente"
    postgres_password: str = "cambia-esto"
    postgres_db: str = "agente"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # --- Redis ---
    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = 60 * 60 * 24  # 24 h para respuestas
    embed_cache_ttl_seconds: int = 60 * 60 * 24 * 7  # 7 días para embeddings de consultas
    cache_enabled: bool = True

    # --- OpenSearch ---
    opensearch_url: str = "http://opensearch:9200"
    opensearch_index: str = "articulos_chunks"
    opensearch_pipeline: str = "pipeline-hibrido"
    # Peso de cada sub-consulta en la búsqueda híbrida (léxica, vectorial). Ver eval/eval_retrieval.py
    hybrid_weight_lexical: float = 0.3
    hybrid_weight_vector: float = 0.7

    # --- Recuperación ---
    retrieval_candidates: int = 20  # candidatos que trae OpenSearch antes del reranking
    retrieval_top_k: int = 6  # fragmentos que finalmente ve el LLM
    rerank_enabled: bool = True

    # --- Hugging Face (embeddings y reranker, se ejecutan dentro del backend) ---
    hf_token: str | None = None
    embed_model: str = "nvidia/Nemotron-3-Embed-1B-BF16"
    embed_fallback_model: str = "BAAI/bge-m3"
    embed_query_prefix: str = "query: "
    embed_doc_prefix: str = "passage: "
    embed_device: str = "cpu"  # "cuda" si tienes VRAM libre
    embed_batch_size: int = 8
    rerank_model: str = "nvidia/llama-nemotron-rerank-1b-v2"
    rerank_fallback_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Modelos generativos ---
    # Ollama: Nemotron-3-Nano = cerebro del agente (razonamiento + tool calling)
    ollama_base_url: str = "http://ollama:11434/v1"
    ollama_model: str = "nemotron-3-nano"
    # vLLM: Qwen2.5-VL = visión (figuras/tablas) y juez de evaluación
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vllm_api_key: str = "sin-clave"
    # Qué servidor ejecuta el agente principal: "ollama" (por defecto) o "vllm"
    agent_backend: str = "ollama"
    llm_timeout_seconds: float = 180.0
    llm_temperature: float = 0.1

    # --- Docling / ingesta ---
    papers_dir: str = "/data/papers"
    figures_dir: str = "/data/figures"
    chunk_max_tokens: int = 512
    chunk_tokenizer: str = "BAAI/bge-m3"

    # --- Admin ---
    admin_user: str = "admin"
    admin_password: str = "cambia-esto"
    jwt_secret: str = "cambia-esto-por-un-secreto-largo"
    jwt_expire_minutes: int = 480

    # --- Varios ---
    max_history_messages: int = 10  # turnos previos enviados al agente
    cors_origins: str = "*"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
