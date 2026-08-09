"""
Router de estado del servicio.

Referencia: docs/API_SPEC.md — GET /health
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Request

from api.dependencias import VERSION_API
from data.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Estado de la API")
def health(request: Request) -> HealthResponse:
    """
    Informa si la API está lista para atender optimizaciones.

    Devuelve `status="degraded"` en lugar de fallar cuando el dataset no se pudo
    cargar en el arranque: así un orquestador puede distinguir "proceso vivo pero
    sin datos" de "proceso caído".

    Args:
        request: Petición HTTP en curso.

    Returns:
        Estado, versión, número de obras cargadas y marca de tiempo ISO 8601.
    """
    df = getattr(request.app.state, "df", None)
    problema = getattr(request.app.state, "problema", None)
    n_obras = 0 if df is None else int(len(df))
    listo = n_obras > 0 and problema is not None

    return HealthResponse(
        status="ok" if listo else "degraded",
        version=VERSION_API,
        n_obras=n_obras,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
