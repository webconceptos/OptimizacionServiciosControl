"""
Dependencias compartidas de los routers.

Aísla el acceso al estado de la aplicación (`app.state`) para que los routers no
tengan que importar `api.main`, lo que crearía un import circular.

Referencia: docs/API_SPEC.md — docs/ARCHITECTURE.md
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from fastapi import HTTPException, Request, status

from config.params import Params
from core.problem import Problema

logger = logging.getLogger(__name__)

#: Versión de la API, expuesta en /health y en el título de OpenAPI.
VERSION_API: str = "1.0.0"


def obtener_params(request: Request) -> Params:
    """
    Devuelve los parámetros del proyecto asociados a la aplicación.

    Args:
        request: Petición HTTP en curso.

    Returns:
        Instancia de `Params` guardada en `app.state`, o el Singleton si no está.
    """
    return getattr(request.app.state, "params", None) or Params.get()


def obtener_df(request: Request) -> pd.DataFrame:
    """
    Devuelve el dataset de obras cargado al arrancar.

    Args:
        request: Petición HTTP en curso.

    Returns:
        DataFrame de obras.

    Raises:
        HTTPException: 503 si el dataset no se pudo cargar en el arranque.
    """
    df: Optional[pd.DataFrame] = getattr(request.app.state, "df", None)
    if df is None or len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El dataset de obras no esta disponible: fallo la carga en el arranque. "
                "Revisar el log del lifespan"
            ),
        )
    return df


def obtener_problema(request: Request) -> Problema:
    """
    Devuelve el problema MCKP compartido por todas las peticiones.

    Es de solo lectura: quien necesite pesos distintos debe construir un
    `Problema` nuevo con `construir_problema`, nunca mutar este.

    Args:
        request: Petición HTTP en curso.

    Returns:
        Instancia de `Problema`.

    Raises:
        HTTPException: 503 si no se pudo construir en el arranque.
    """
    problema: Optional[Problema] = getattr(request.app.state, "problema", None)
    if problema is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El problema MCKP no esta disponible: fallo la inicializacion en el "
                "arranque. Revisar el log del lifespan"
            ),
        )
    return problema


def construir_problema(
    request: Request, pesos: Optional[dict] = None
) -> Problema:
    """
    Devuelve el problema compartido, o uno nuevo si la petición trae pesos propios.

    Crear una instancia nueva es imprescindible: mutar los pesos del `Problema` de
    `app.state` afectaría a todas las peticiones concurrentes.

    Args:
        request: Petición HTTP en curso.
        pesos: Diccionario {R, M, P, E, G} con pesos personalizados, o None.

    Returns:
        El `Problema` compartido si `pesos` es None; en caso contrario, uno nuevo
        con esos pesos aplicados.

    Raises:
        HTTPException: 503 si el estado no está disponible; 400 si los pesos son
            inválidos.
    """
    if not pesos:
        return obtener_problema(request)

    df = obtener_df(request)
    params = obtener_params(request)
    problema = Problema(df, params)
    try:
        problema.w = pesos
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pesos invalidos: {exc}",
        ) from exc
    logger.info("Problema construido con pesos personalizados: %s", pesos)
    return problema
