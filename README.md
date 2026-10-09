# Semantic Search

Aplicación web para guardar textos, calcular embeddings con Ollama y comparar consultas contra un corpus usando similitud semántica.

## ¿Qué hace?

- Guarda textos en un archivo local.
- Genera embeddings con modelos compatibles de Ollama.
- Compara una consulta contra todos los documentos guardados.
- Ordena resultados por similitud.
- Permite elegir entre varias métricas: coseno, euclidiana y manhattan.
- Muestra una explicación de por qué un texto quedó mejor rankeado.

## Requisitos

- Python 3.13+
- Ollama instalado y corriendo localmente.
- Un modelo de embeddings descargado, por ejemplo:
  - `nomic-embed-text`

## Instalación

1. Entra a la carpeta del proyecto:

```bash
cd "/Users/Enrique_Davila/Desktop/portfolio/langchain_course/semantic search"
```

2. Crea o activa el entorno virtual:

```bash
python3 -m venv semantic_search
source semantic_search/bin/activate
```

3. Instala dependencias:

```bash
pip install -r requirements.txt
```

4. Descarga el modelo de embeddings de Ollama:

```bash
ollama pull nomic-embed-text
```

5. Asegúrate de que Ollama esté corriendo:

```bash
ollama serve
```

## Ejecución

```bash
python app.py
```

La aplicación queda disponible en:

- http://127.0.0.1:5001

> Si el puerto 5000 está ocupado por macOS (por ejemplo, AirPlay), la app usa 5001.

## Estructura

- `app.py`: lógica principal de la aplicación.
- `textos.json`: textos guardados por la app.
- `embeddings.json`: caché de embeddings calculados.
- `requirements.txt`: dependencias del proyecto.

## Notas

- El archivo de caché `embeddings.json` se reutiliza para evitar recalcular embeddings ya generados.
- Los ajustes de tuning se guardan en sesión y pueden cambiar la métrica, el prefijo, el lote y la explicación.
