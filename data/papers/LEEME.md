# Coloca aquí el PDF del artículo

1. Descarga el PDF de tu survey Q1 (versión publicada en la revista si tu universidad tiene acceso;
   si no, la versión de arXiv del mismo artículo).
2. Guárdalo en esta carpeta con un nombre corto, por ejemplo `survey-alucinaciones.pdf`
   (el nombre se convierte en el `doc_id`).
3. Ejecuta `docker compose run --rm ingest` (o simplemente `docker compose up -d`: el servicio
   `ingest` procesa automáticamente los PDF nuevos).

Puedes poner varios PDF: el sistema soporta múltiples artículos y el chat permite filtrar por artículo.
Los PDF están en `.gitignore` por derechos de autor.
