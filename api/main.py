"""
Aplicación FastAPI de la Parte 2 (Computación Evolutiva).

Expone el optimizador como servicio REST en el puerto **8001**: la API de la
Parte 1 (tesis) ocupa el 8000 (docs/ARCHITECTURE.md, "¿Por que 8001 y no 8000?").

El dataset y el `Problema` se construyen una sola vez en el arranque y quedan en
`app.state`, porque normalizar y preparar el problema es caro y no depende de la
petición. El `Problema` compartido es de solo lectura: los endpoints que necesitan
pesos distintos construyen uno nuevo (ver `api/dependencias.py`).

Todos los montos que devuelve la API van en **miles de soles (kS/)**, la escala de
la columna C del dataset (sum(C) ≈ 209.7 kS/, B ≈ 73.4 kS/).

Referencia: README.md, arquitectura del proyecto — docs/API_SPEC.md
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.dependencias import VERSION_API
from api.routers import analysis, health, optimize, pareto
from config.params import Params
from core.problem import Problema

logger = logging.getLogger(__name__)

#: Orígenes autorizados por CORS: dashboard React y API de la Parte 1.
ORIGENES_CORS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

#: Título de la aplicación, visible en /docs.
TITULO: str = "CE API - Optimizacion de Obras Publicas"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Carga el dataset y construye el problema al arrancar; libera al apagar.

    La carga sigue la cascada de `data/loader.cargar_obras`: CSV si se indica, si
    no la API de la Parte 1, y si tampoco está disponible el generador sintético
    calibrado. Un fallo de la Parte 1 no impide arrancar: se degrada a datos
    simulados, como indica docs/API_SPEC.md (503 no bloqueante).

    Si la carga falla por completo, la aplicación arranca igualmente con el estado
    vacío: /health responde "degraded" y los demás endpoints devuelven 503, que es
    más diagnosticable que un proceso que no levanta.

    Args:
        app: Aplicación FastAPI.

    Yields:
        None, mientras la aplicación está en servicio.
    """
    params = Params.get()
    app.state.params = params
    app.state.df = None
    app.state.problema = None

    logger.info("Arrancando %s v%s | puerto configurado=%d", TITULO, VERSION_API, params.API_PORT)
    try:
        from data.loader import cargar_obras

        df = cargar_obras(
            base_url=params.PARTE1_API_URL,
            n=int(params.N_OBRAS),
            seed=int(params.SEED),
        )
        problema = Problema(df, params)
        app.state.df = df
        app.state.problema = problema
        logger.info(
            "Estado listo | n_obras=%d | B=%.4f kS/ | K=%d | m_r=%s",
            len(df),
            problema.B,
            problema.K,
            problema.m_r,
        )
    except Exception as exc:  # noqa: BLE001 — el arranque no debe tumbar el proceso
        logger.exception(
            "Fallo la inicializacion del dataset (%s): la API arranca degradada",
            type(exc).__name__,
        )

    yield

    app.state.df = None
    app.state.problema = None
    logger.info("Apagando %s", TITULO)


app = FastAPI(
    title=TITULO,
    version=VERSION_API,
    description=(
        "Optimizacion evolutiva de inversiones publicas para control preventivo "
        "(Parte 2: Computacion Evolutiva). Toma Ri = P(Extrem. Riesgosa) de la "
        "Parte 1 y resuelve el MCKP con AG binario (Tema 3) y NSGA-II (Tema 10). "
        "Montos en miles de soles (kS/)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENES_CORS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(optimize.router)
app.include_router(pareto.router)
app.include_router(analysis.router)


@app.exception_handler(StarletteHTTPException)
async def manejar_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """
    Devuelve los errores HTTP con el cuerpo uniforme de docs/API_SPEC.md.

    Se registra sobre `starlette.exceptions.HTTPException`, no sobre la de
    FastAPI: esta última la hereda, así que la base cubre ambas, incluidos los 404
    que emite el propio router de Starlette y que de otro modo responderían con el
    `{"detail": "Not Found"}` por defecto.

    Args:
        request: Petición que provocó el error.
        exc: Excepción HTTP lanzada por un endpoint, una dependencia o el router.

    Returns:
        Respuesta JSON con las claves `error` y `detail`.
    """
    logger.warning("%s %s -> %d | %s", request.method, request.url.path, exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail), "detail": f"{request.method} {request.url.path}"},
    )


@app.exception_handler(RequestValidationError)
async def manejar_validacion(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Normaliza los errores de validación de Pydantic al cuerpo de la API.

    Responde 422, el código que docs/API_SPEC.md reserva para los errores de
    validación, pero con las claves `error` y `detail` en lugar del formato por
    defecto de FastAPI.

    Args:
        request: Petición que provocó el error.
        exc: Error de validación del cuerpo o de los parámetros de consulta.

    Returns:
        Respuesta JSON con las claves `error` y `detail`.
    """
    campos = [
        f"{'.'.join(str(p) for p in error.get('loc', []))}: {error.get('msg', '')}"
        for error in exc.errors()
    ]
    logger.warning(
        "%s %s -> 422 | validacion: %s", request.method, request.url.path, "; ".join(campos)
    )
    return JSONResponse(
        status_code=422,
        content={"error": "Parametros de peticion invalidos", "detail": "; ".join(campos)},
    )


@app.exception_handler(Exception)
async def manejar_excepcion_generica(request: Request, exc: Exception) -> JSONResponse:
    """
    Captura cualquier error no previsto, lo registra con traza y responde 500.

    Args:
        request: Petición que provocó el error.
        exc: Excepción no controlada.

    Returns:
        Respuesta JSON con las claves `error` y `detail`.
    """
    logger.exception("Error no controlado en %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": f"{type(exc).__name__} en {request.url.path}"},
    )


@app.get("/", tags=["health"], summary="Informacion de la API")
def raiz() -> dict:
    """
    Punto de entrada informativo con el inventario de endpoints.

    Returns:
        Diccionario con el nombre, la versión y las rutas disponibles.
    """
    return {
        "nombre": TITULO,
        "version": VERSION_API,
        "parte": "2 de 2 (Computacion Evolutiva)",
        "unidad_monetaria": "miles S/ (kS/)",
        "endpoints": [
            "GET  /health",
            "POST /optimize",
            "GET  /pareto",
            "GET  /analysis/runs",
            "GET  /analysis/schemas",
            "GET  /analysis/sensitivity",
            "GET  /docs",
        ],
    }
