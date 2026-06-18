# Document Analyzer

API REST para análisis de documentos mediante LLM. Recibe un archivo (PDF, DOCX o texto plano), extrae su contenido y devuelve un análisis estructurado en JSON: resumen, palabras clave, categoría y 10 pares pregunta/respuesta.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Comprobación de disponibilidad del servicio |
| `POST` | `/analyze` | Analiza el documento y devuelve el análisis en JSON |

### `POST /analyze`

**Autenticación** — cabecera HTTP obligatoria

| Cabecera | Descripción |
|----------|-------------|
| `X-API-Key` | Clave configurada en `APP_API_KEY`. Requerida en todas las peticiones a `/analyze`. |

**Request** — `multipart/form-data`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `file` | archivo | PDF, DOCX, TXT, MD o CSV |

**Response** `200 OK`

```json
{
  "filename": "documento.pdf",
  "model": "Reasoning",
  "characters_analyzed": 12500,
  "truncated": false,
  "summary": "...",
  "keywords": ["keyword1", "keyword2"],
  "category": "...",
  "questions_answers": [
    { "question": "...", "answer": "..." }
  ]
}
```

**Errores**

| Código | Causa |
|--------|-------|
| `401` | Cabecera `X-API-Key` ausente o incorrecta |
| `413` | El archivo supera el límite de 20 MB |
| `415` | Formato de archivo no soportado |
| `422` | No se pudo extraer texto del archivo (p. ej. PDF escaneado sin capa de texto) |
| `502` | Fallo en la llamada al LLM o respuesta incompleta o no válida |

## Instalación y uso local

### Requisitos

- Python 3.10+

### Pasos

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd doc-analyzer

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tu API key y configuración

# 5. Arrancar el servidor
uvicorn main:app --reload
```

La API quedará disponible en `http://localhost:8000`. La documentación interactiva (Swagger UI) en `http://localhost:8000/docs`.

## Variables de entorno

Copia `.env.example` a `.env` y rellena los valores:

| Variable | Por defecto | Descripción |
|----------|-------------|-------------|
| `BASE_PATH` | *(vacío)* | Prefijo de ruta inyectado por el proxy inverso (ChurroStack) |
| `LLM_BASE_URL` | `https://apps.fi-group.com/api/openai` | URL base de la API compatible con OpenAI |
| `LLM_API_KEY` | *(requerido)* | Clave de autenticación de la API del LLM |
| `LLM_MODEL` | `Reasoning` | Modelo a utilizar (`Reasoning` \| `Performance`) |
| `MAX_CHARS` | `60000` | Límite de caracteres enviados al LLM |
| `APP_API_KEY` | *(requerido)* | Clave que deben enviar los clientes en la cabecera `X-API-Key` |

## Formatos soportados

| Extensión | Método de extracción |
|-----------|----------------------|
| `.pdf` | Texto nativo con `pypdf` (no soporta PDFs escaneados sin OCR) |
| `.docx` | Párrafos con `python-docx` |
| `.txt`, `.md`, `.csv`, otros | Decodificación UTF-8 |

## Dependencias principales

| Paquete | Uso |
|---------|-----|
| `fastapi` | Framework web |
| `uvicorn` | Servidor ASGI |
| `pydantic` | Validación y serialización de modelos de respuesta |
| `openai` | Cliente para la API compatible con OpenAI |
| `pypdf` | Extracción de texto de PDFs |
| `python-docx` | Extracción de texto de DOCX |
| `python-dotenv` | Carga de variables de entorno desde `.env` |

## Autor

**Juan Ceballos** — Senior .NET Analyst
[juan.ceballos@epsa.com](mailto:juan.ceballos@epsa.com)
