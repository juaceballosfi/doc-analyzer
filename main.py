"""
Document Analyzer · API de análisis de documentos con LLM
=====================================================
Expone un único endpoint POST /analyze que recibe un documento (PDF, DOCX o
texto plano), extrae su contenido, lo envía a un LLM compatible con la API de
OpenAI y devuelve un análisis estructurado en JSON.

Endpoints públicos
------------------
GET  /health   → comprobación de disponibilidad del servicio
POST /analyze  → análisis completo del documento subido

Variables de entorno (ver .env.example)
----------------------------------------
BASE_PATH   : prefijo de ruta inyectado por el proxy inverso (ChurroStack)
LLM_BASE_URL: URL base de la API compatible con OpenAI
LLM_API_KEY : clave de autenticación de la API
LLM_MODEL   : nombre del modelo a utilizar ("Reasoning" | "Performance")
MAX_CHARS   : límite de caracteres enviados al LLM (por defecto 60 000)
"""

import logging
import os
import time
import uuid
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Formato estructurado para que los logs sean fáciles de filtrar en producción.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("doc-analyzer")

# ---------------------------------------------------------------------------
# Configuración — todas las variables se inyectan como env vars en producción.
# En desarrollo local se pueden definir en un archivo .env cargado con
# `python-dotenv` o exportadas manualmente en la terminal.
# ---------------------------------------------------------------------------
BASE_PATH = os.environ.get("BASE_PATH", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://apps.fi-group.com/api/openai")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "Reasoning")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "60000"))
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}

# Falla en arranque para evitar errores silenciosos en la primera petición.
if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY no está configurada. Revisa el archivo .env.")

logger.info(
    "Configuracion cargada | base_url=%s | modelo=%s | max_chars=%d | api_key_presente=%s",
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_CHARS,
    bool(LLM_API_KEY),
)

# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------
# root_path es necesario para que /docs funcione correctamente detrás del
# proxy inverso de ChurroStack, que elimina el segmento {BASE_PATH}/fastapi
# antes de reenviar la petición a este servicio.
app = FastAPI(
    title="Document Analyzer",
    description="Analiza documentos PDF, DOCX y texto plano mediante un LLM y devuelve un resumen estructurado.",
    version="1.0.0",
    root_path=f"{BASE_PATH}/fastapi" if BASE_PATH else "",
)

# Cliente OpenAI apuntando al endpoint corporativo en lugar de api.openai.com.
client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

# ---------------------------------------------------------------------------
# Prompt del sistema
# ---------------------------------------------------------------------------
# Se exige respuesta en JSON puro para evitar post-procesamiento frágil.
# El idioma de la respuesta se adapta automáticamente al del documento.
SYSTEM_PROMPT = """You are a document analyst. Analyze the user's document and respond
ONLY with a valid JSON object, without additional text or code blocks.
The structure must be EXACTLY:
{
  "summary": "<brief and clear summary of the document>",
  "keywords": ["<keyword>"],
  "category": "<a single general category of the document>",
  "questions_answers": [
    {"question": "<question about the document>", "answer": "<answer based on the document>"}
  ]
}
Include between 5 and 10 keywords, and EXACTLY 10 question/answer pairs.
Answer in the same language as the document."""


# ---------------------------------------------------------------------------
# Modelos de respuesta (Pydantic)
# ---------------------------------------------------------------------------

class QA(BaseModel):
    question: str
    answer: str


class AnalysisResponse(BaseModel):
    filename: str
    model: str
    characters_analyzed: int
    truncated: bool
    summary: str
    keywords: list[str]
    category: str
    questions_answers: list[QA]


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def extract_text(filename: str, data: bytes) -> str:
    """Extrae el contenido textual de un archivo según su extensión.

    Formatos soportados
    -------------------
    .pdf  : extracción de texto nativo con pypdf (no admite PDFs escaneados).
    .docx : extracción de párrafos con python-docx.
    otros : decodificación UTF-8 con tolerancia a errores (txt, md, csv, etc.).

    Parameters
    ----------
    filename : str
        Nombre original del archivo, usado únicamente para detectar la extensión.
    data : bytes
        Contenido binario del archivo tal como fue recibido en la petición.

    Returns
    -------
    str
        Texto extraído y limpio de espacios redundantes. Cadena vacía si no se
        pudo extraer contenido (p. ej. PDF escaneado sin capa de texto).
    """
    name = (filename or "").lower()

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()

    if name.endswith(".docx"):
        import docx  # python-docx; importación diferida para no cargar si no se usa

        document = docx.Document(BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs).strip()

    # Texto plano: .txt, .md, .csv u otros formatos sin extensión reconocida.
    return data.decode("utf-8", errors="ignore").strip()


def parse_json(text: str) -> dict:
    """Extrae y parsea el ULTIMO objeto JSON balanceado de la respuesta del LLM.

    No usa una regex "greedy" (``\\{.*\\}``) porque esa estrategia falla en
    cuanto el texto contiene más de una llave de apertura/cierre fuera del
    JSON real (p. ej. el modelo recuerda el formato esperado antes de dar la
    respuesta). En su lugar, se recorre el texto contando aperturas y cierres
    de llave para localizar TODOS los objetos JSON completos de nivel
    superior, y se queda con el ULTIMO: si el modelo menciona un ejemplo de
    formato antes de responder, ese ejemplo siempre aparece antes que la
    respuesta real, nunca después.

    Parameters
    ----------
    text : str
        Texto completo devuelto por el LLM.

    Returns
    -------
    dict
        Diccionario Python con los campos del análisis.

    Raises
    ------
    ValueError
        Si no se encuentra ningún objeto JSON, o si está mal cerrado.
    json.JSONDecodeError
        Si el último objeto localizado no es sintácticamente válido.
    """
    import json

    cleaned = text.strip()
    candidates: list[str] = []
    depth = 0
    start = -1

    for index, char in enumerate(cleaned):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(cleaned[start : index + 1])

    if not candidates:
        raise ValueError("La respuesta del modelo no contenia JSON.")

    return json.loads(candidates[-1])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Comprobación de salud", tags=["Infraestructura"])
def health():
    """Devuelve ``{"status": "ok"}`` si el servicio está en funcionamiento.

    Usado por el orquestador (ChurroStack / load balancer) para verificar
    que la instancia está lista para recibir tráfico.
    """
    logger.info("Chequeo de salud solicitado")
    return {"status": "ok"}


@app.post("/analyze", summary="Analizar documento", tags=["Análisis"], response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    """Analiza el documento subido y devuelve un resumen estructurado en JSON.

    Flujo de procesamiento
    ----------------------
    1. Valida la extensión del archivo.
    2. Lee el archivo en memoria y valida su tamaño.
    3. Extrae el texto según el formato del archivo (.pdf, .docx, texto plano).
    4. Trunca el texto a ``MAX_CHARS`` para respetar la ventana de contexto del LLM.
    5. Llama al LLM con un prompt de sistema que exige respuesta en JSON.
    6. Parsea el JSON devuelto por el modelo y lo valida contra ``AnalysisResponse``
       (campos faltantes o mal tipados se tratan como fallo del LLM, no como error interno).
    7. Retorna el análisis validado, enriquecido con metadatos de la petición.

    Parameters
    ----------
    file : UploadFile
        Archivo a analizar. Formatos admitidos: PDF, DOCX, TXT, MD, CSV.

    Returns
    -------
    AnalysisResponse
        Objeto JSON con los campos:

        - ``filename``            : nombre original del archivo.
        - ``model``               : modelo LLM utilizado.
        - ``characters_analyzed`` : caracteres efectivamente enviados al LLM.
        - ``truncated``           : ``true`` si el documento fue recortado.
        - ``summary``             : resumen ejecutivo del documento.
        - ``keywords``            : lista de palabras clave (5–10 elementos).
        - ``category``            : categoría general del documento.
        - ``questions_answers``   : lista de 10 pares pregunta/respuesta.

    Raises
    ------
    - HTTPException 415: Si el formato del archivo no está soportado.
    - HTTPException 413: Si el archivo supera el límite de tamaño permitido.
    - HTTPException 422: Si no se pudo extraer texto del archivo (p. ej. PDF escaneado).
    - HTTPException 502: Si la llamada al LLM falló o el modelo no devolvió JSON válido.
    """
    request_id = uuid.uuid4().hex[:8]
    start_time = time.perf_counter()
    logger.info("[%s] Peticion recibida | archivo=%s", request_id, file.filename)

    # --- 1. Validación de formato ---
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Formato '{ext}' no soportado. Usa: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    # --- 2. Lectura del archivo ---
    data = await file.read()
    logger.info("[%s] Archivo leido en memoria | bytes=%d", request_id, len(data))

    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo supera el límite de {MAX_FILE_BYTES // (1024 * 1024)} MB.",
        )

    # --- 3. Extracción de texto ---
    try:
        text = extract_text(file.filename, data)
    except Exception as exc:
        logger.warning("[%s] Error al extraer texto | error=%s", request_id, exc)
        raise HTTPException(status_code=422, detail="No se pudo procesar el archivo. Puede estar dañado.")

    if not text:
        logger.warning(
            "[%s] No se pudo extraer texto del archivo (posible PDF escaneado)",
            request_id,
        )
        raise HTTPException(
            status_code=422,
            detail="No se pudo extraer texto. Si es un PDF escaneado necesitarias OCR o el modelo Multimodal.",
        )

    # --- 4. Truncado ---
    content = text[:MAX_CHARS]
    was_truncated = len(text) > MAX_CHARS
    logger.info(
        "[%s] Texto extraido | caracteres_totales=%d | caracteres_enviados=%d | truncado=%s",
        request_id,
        len(text),
        len(content),
        was_truncated,
    )

    # --- 5. Llamada al LLM ---
    logger.info("[%s] Llamando al LLM | modelo=%s", request_id, LLM_MODEL)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,  # respuestas deterministas para facilitar el parseo del JSON
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"DOCUMENTO:\n{content}"},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] Fallo la llamada al LLM | error=%s", request_id, exc)
        raise HTTPException(status_code=502, detail=f"Error llamando al LLM: {exc}")

    raw = response.choices[0].message.content or ""
    logger.info("[%s] Respuesta del LLM recibida | caracteres=%d", request_id, len(raw))

    # --- 6. Parseo y validacion del JSON ---
    # parse_json y AnalysisResponse(...) se hacen en el MISMO try: un JSON
    # sintacticamente valido pero con un campo faltante (p. ej. el LLM omite
    # "category", o entrega 8 preguntas en vez de 10) NO falla en parse_json,
    # sino al validar contra el modelo Pydantic. Si esa validacion quedara
    # fuera de este bloque, el error escaparia al manejo de errores: FastAPI
    # devolveria un 500 generico y nunca pasaria por logger.error ni por el
    # 502 que disenamos para fallos del LLM.
    try:
        result = parse_json(raw)
        validated = AnalysisResponse(
            filename=file.filename,
            model=LLM_MODEL,
            characters_analyzed=len(content),
            truncated=was_truncated,
            **result,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[%s] JSON invalido o incompleto | error=%s | respuesta_cruda=%s",
            request_id,
            exc,
            raw[:300],
        )
        raise HTTPException(
            status_code=502,
            detail="El modelo no devolvio una respuesta valida o completa. Reintenta o prueba el modelo 'Reasoning'.",
        )

    elapsed = time.perf_counter() - start_time
    logger.info(
        "[%s] Analisis completado con exito | duracion_seg=%.2f | categoria=%s",
        request_id,
        elapsed,
        validated.category,
    )

    # --- 7. Respuesta final ---
    return validated