# Guion sugerido para la demo (10 min)

Ambos integrantes deben poder explicar cualquier parte. Repartan los bloques y ensayen.

1. **Arquitectura (1 min):** diagrama del README; rol de cada componente del stack.
2. **Ingesta (1,5 min):** `data/processed/<doc>.md` (secciones, tablas en Markdown), un par de líneas de `.chunks.jsonl` (sección y página por chunk) y las figuras extraídas.
3. **Chat (3 min), desde la URL pública de ngrok:**
   - Pregunta factual: mostrar las citas con sección y página y abrir el fragmento.
   - Pregunta de seguimiento ("¿y la segunda?"): el contexto entre turnos se conserva.
   - Pregunta sobre una figura: el agente usa Qwen2.5-VL (se ve en /admin, uso por modelo).
   - Pregunta fuera del artículo ("¿quién ganó el mundial?"): el agente se abstiene.
   - Repetir la primera pregunta: respuesta instantánea desde Redis (insignia "caché Redis").
4. **/admin (2 min):** login, indicadores, latencia por etapa, uso por modelo, filtros y detalle de una ejecución con sus fragmentos.
5. **Evidencia (2 min):** `eval/resultados/retrieval.md` (híbrida y rerank frente a BM25 y kNN), `eval_final.md` (juez) y `cache.md`.
6. **Preguntas del profesor (resto).**

Antes de la demo: `docker compose ps` (todo healthy), una pregunta de calentamiento para cargar los modelos en GPU y la URL de ngrok abierta.
