-- =====================================================================
-- Esquema normalizado (3FN) del Agente Académico
-- Se ejecuta automáticamente la primera vez que arranca el contenedor
-- de PostgreSQL (montado en /docker-entrypoint-initdb.d).
--
--   documents 1──* retrieval_hits / citations
--   conversations 1──* messages
--   messages (assistant) 1──1 agent_runs 1──* model_calls
--                                       1──* retrieval_hits
--   messages (assistant) 1──* citations
--   eval_runs 1──* eval_items
-- Ver docs/esquema_bd.md para el diagrama y la descripción de cada tabla.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Artículos ingeridos con Docling
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,             -- slug del archivo (p. ej. "survey-rag-2024")
    title         TEXT NOT NULL,
    filename      TEXT NOT NULL,
    sha256        TEXT NOT NULL UNIQUE,
    num_pages     INTEGER,
    num_chunks    INTEGER NOT NULL DEFAULT 0,
    num_tables    INTEGER NOT NULL DEFAULT 0,
    num_figures   INTEGER NOT NULL DEFAULT 0,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Una conversación pertenece a un cliente anónimo (id generado en el navegador)
CREATE TABLE IF NOT EXISTS conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id   TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT 'Nueva conversación',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conversations_client ON conversations (client_id, updated_at DESC);

-- Mensajes de usuario y del asistente
CREATE TABLE IF NOT EXISTS messages (
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    answerable       BOOLEAN,                  -- solo asistente: ¿se pudo responder con el artículo?
    confidence       TEXT,                     -- solo asistente: alta | media | baja
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id, id);

-- Una ejecución del agente por cada respuesta del asistente (métricas agregadas)
CREATE TABLE IF NOT EXISTS agent_runs (
    id                    BIGSERIAL PRIMARY KEY,
    user_message_id       BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    assistant_message_id  BIGINT REFERENCES messages(id) ON DELETE CASCADE,
    agent_model           TEXT NOT NULL,
    agent_provider        TEXT NOT NULL,       -- ollama | vllm | cache
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    llm_requests          INTEGER NOT NULL DEFAULT 0,
    tool_calls            INTEGER NOT NULL DEFAULT 0,
    latency_total_ms      DOUBLE PRECISION NOT NULL,
    latency_embed_ms      DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_search_ms     DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_rerank_ms     DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_llm_ms        DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_vision_ms     DOUBLE PRECISION NOT NULL DEFAULT 0,
    cache_hit             BOOLEAN NOT NULL DEFAULT FALSE,
    streamed              BOOLEAN NOT NULL DEFAULT FALSE,
    error                 TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON agent_runs (created_at DESC);

-- Detalle por modelo: permite "uso por modelo" (agente en Ollama, visión en vLLM, ...)
CREATE TABLE IF NOT EXISTS model_calls (
    id             BIGSERIAL PRIMARY KEY,
    run_id         BIGINT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    model          TEXT NOT NULL,
    provider       TEXT NOT NULL,
    purpose        TEXT NOT NULL,              -- agente | vision | embedding | rerank
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    latency_ms     DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_model_calls_run ON model_calls (run_id);

-- Fragmentos recuperados en cada ejecución (trazabilidad del RAG)
CREATE TABLE IF NOT EXISTS retrieval_hits (
    id          BIGSERIAL PRIMARY KEY,
    run_id      BIGINT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    query       TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
    rank        INTEGER NOT NULL,
    score       DOUBLE PRECISION,
    rerank_score DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_hits_run ON retrieval_hits (run_id);

-- Citas que aparecen en la respuesta final
CREATE TABLE IF NOT EXISTS citations (
    id           BIGSERIAL PRIMARY KEY,
    message_id   BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    chunk_id     TEXT NOT NULL,
    document_id  TEXT REFERENCES documents(id) ON DELETE SET NULL,
    section      TEXT,
    page         INTEGER,
    snippet      TEXT
);
CREATE INDEX IF NOT EXISTS idx_citations_msg ON citations (message_id);

-- Evaluación automática (LLM como juez)
CREATE TABLE IF NOT EXISTS eval_runs (
    id           BIGSERIAL PRIMARY KEY,
    label        TEXT NOT NULL,
    judge_model  TEXT NOT NULL,
    agent_model  TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_items (
    id                 BIGSERIAL PRIMARY KEY,
    eval_run_id        BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    question_id        TEXT NOT NULL,
    question           TEXT NOT NULL,
    category           TEXT NOT NULL,
    expected_answerable BOOLEAN NOT NULL,
    expected_answer    TEXT,
    answer             TEXT,
    answered           BOOLEAN,
    faithfulness       INTEGER,   -- 1..5
    relevance          INTEGER,   -- 1..5
    correctness        INTEGER,   -- 1..5 (NULL si no hay respuesta esperada)
    citation_ok        BOOLEAN,
    abstention_ok      BOOLEAN,
    latency_ms         DOUBLE PRECISION,
    judge_reason       TEXT
);

-- Vista de conveniencia para el panel /admin
CREATE OR REPLACE VIEW v_qa AS
SELECT r.id            AS run_id,
       c.id            AS conversation_id,
       um.content      AS question,
       am.content      AS answer,
       am.answerable,
       r.agent_model, r.agent_provider,
       r.input_tokens, r.output_tokens, r.tool_calls,
       r.latency_total_ms, r.latency_embed_ms, r.latency_search_ms,
       r.latency_rerank_ms, r.latency_llm_ms, r.latency_vision_ms,
       r.cache_hit, r.error, r.created_at
FROM agent_runs r
JOIN messages um ON um.id = r.user_message_id
JOIN conversations c ON c.id = um.conversation_id
LEFT JOIN messages am ON am.id = r.assistant_message_id;
