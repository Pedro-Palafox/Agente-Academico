# Esquema de PostgreSQL

Definido en [`backend/sql/schema.sql`](../backend/sql/schema.sql). Está en tercera forma normal: cada
hecho se guarda una sola vez y las métricas se relacionan con el mensaje que las produjo.

```mermaid
erDiagram
    conversations ||--o{ messages : contiene
    messages ||--o| agent_runs : "pregunta (user_message_id)"
    messages ||--o| agent_runs : "respuesta (assistant_message_id)"
    agent_runs ||--o{ model_calls : "uso por modelo"
    agent_runs ||--o{ retrieval_hits : "fragmentos recuperados"
    messages ||--o{ citations : "citas de la respuesta"
    documents ||--o{ retrieval_hits : ""
    documents ||--o{ citations : ""
    eval_runs ||--o{ eval_items : ""

    documents {
        text id PK
        text title
        text filename
        text sha256
        int num_pages
        int num_chunks
        int num_tables
        int num_figures
        timestamptz ingested_at
    }
    conversations {
        uuid id PK
        text client_id
        text title
        timestamptz created_at
        timestamptz updated_at
    }
    messages {
        bigint id PK
        uuid conversation_id FK
        text role
        text content
        bool answerable
        text confidence
        timestamptz created_at
    }
    agent_runs {
        bigint id PK
        bigint user_message_id FK
        bigint assistant_message_id FK
        text agent_model
        text agent_provider
        int input_tokens
        int output_tokens
        int tool_calls
        float latency_total_ms
        float latency_embed_ms
        float latency_search_ms
        float latency_rerank_ms
        float latency_llm_ms
        float latency_vision_ms
        bool cache_hit
        text error
    }
    model_calls {
        bigint id PK
        bigint run_id FK
        text model
        text provider
        text purpose
        int input_tokens
        int output_tokens
        float latency_ms
    }
    retrieval_hits {
        bigint id PK
        bigint run_id FK
        text query
        text chunk_id
        text document_id FK
        int rank
        float score
        float rerank_score
    }
    citations {
        bigint id PK
        bigint message_id FK
        text chunk_id
        text document_id FK
        text section
        int page
        text snippet
    }
    eval_runs {
        bigint id PK
        text label
        text judge_model
        text agent_model
        timestamptz created_at
    }
    eval_items {
        bigint id PK
        bigint eval_run_id FK
        text question
        bool expected_answerable
        int faithfulness
        int relevance
        int correctness
        bool abstention_ok
    }
```

| Tabla | Para qué sirve |
|---|---|
| `documents` | Artículos ingeridos con Docling (hash para no reingerir, conteo de tablas y figuras). |
| `conversations` | Una conversación por cliente anónimo (id guardado en el navegador). |
| `messages` | Turnos de usuario y asistente. El asistente guarda si pudo responder y su confianza. |
| `agent_runs` | Una fila por respuesta: tokens de entrada y salida, latencia total y por etapa, caché y errores. |
| `model_calls` | Detalle por modelo (agente en Ollama, visión en vLLM, embeddings y reranker de HF). Alimenta "uso por modelo". |
| `retrieval_hits` | Qué fragmentos recuperó el agente, con qué consulta y con qué puntaje (híbrido y rerank). |
| `citations` | Citas (sección y página) que aparecen en la respuesta final. |
| `eval_runs` / `eval_items` | Resultados de la evaluación automática con LLM como juez. |
| `v_qa` (vista) | Une pregunta, respuesta y métricas. Es la base del panel /admin. |

Consultas útiles:

```sql
-- Tiempo promedio y tokens por modelo
SELECT agent_model, count(*), avg(latency_total_ms), sum(input_tokens), sum(output_tokens)
FROM agent_runs GROUP BY 1;

-- Efecto de Redis
SELECT cache_hit, avg(latency_total_ms) FROM agent_runs WHERE error IS NULL GROUP BY 1;
```
