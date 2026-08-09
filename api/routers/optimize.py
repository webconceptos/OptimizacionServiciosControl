"""
Router de optimización.

Ejecuta el AG (Tema 3) o NSGA-II (Tema 10) sobre el dataset cargado y devuelve el
portafolio resultante. Los montos van en miles de soles (kS/), la escala de la
columna C del dataset.

Los endpoints se declaran como funciones síncronas (`def`, no `async def`) a
propósito: FastAPI las ejecuta en un hilo del threadpool, de modo que una
optimización de varios segundos no bloquea el bucle de eventos ni las peticiones
a /health.

Referencia: docs/API_SPEC.md — POST /optimize
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from fastapi import APIRouter, HTTPException, Request, status

from algorithms.benchmarks import greedy
from algorithms.genetic import AG
from algorithms.nsga2 import NSGA2
from analysis.statistics import analisis_wilcoxon, multiples_corridas
from api.dependencias import construir_problema, obtener_params
from core.problem import Problema
from core.solution import Solucion
from data.schemas import (
    CorridaItem,
    EstadisticasResponse,
    OptimizeRequest,
    SolucionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["optimize"])


def solucion_a_response(
    solucion: Solucion, tiempo_seg: float
) -> SolucionResponse:
    """
    Convierte una `Solucion` del dominio en la respuesta de la API.

    Args:
        solucion: Solución a serializar.
        tiempo_seg: Tiempo de cómputo de la corrida que la produjo.

    Returns:
        `SolucionResponse` con el costo total en kS/.
    """
    return SolucionResponse(
        fitness=round(solucion.fitness, 6),
        n_seleccionadas=solucion.n_seleccionadas,
        costo_total=round(solucion.costo, 4),
        factible=solucion.factible,
        tiempo_seg=round(float(tiempo_seg), 4),
        obras_seleccionadas=solucion.codigos,
        distribucion_territorial={
            str(region): int(n) for region, n in solucion.distribucion_territorial.items()
        },
    )


@router.post(
    "/optimize",
    response_model=None,
    summary="Ejecuta el AG o NSGA-II",
)
def optimize(
    peticion: OptimizeRequest, request: Request
) -> Union[SolucionResponse, EstadisticasResponse]:
    """
    Ejecuta el algoritmo solicitado y devuelve el portafolio o sus estadísticas.

    Tres comportamientos, según docs/API_SPEC.md:

      * `algoritmo="AG"` con `n_runs=1`: una corrida, devuelve `SolucionResponse`.
      * `algoritmo="AG"` con `n_runs>1`: N corridas independientes con semillas
        `seed..seed+N-1`, devuelve `EstadisticasResponse` con la prueba de
        Wilcoxon contra el Greedy y la mejor solución encontrada.
      * `algoritmo="NSGA2"`: devuelve la primera solución del frente de Pareto, que
        es la de menor costo f2 (el frente viene ordenado por costo ascendente).

    Si la petición trae `pesos`, se construye un `Problema` nuevo con ellos: nunca
    se mutan los pesos del problema compartido, que atiende a todas las peticiones.

    Args:
        peticion: Cuerpo de la petición.
        request: Petición HTTP en curso.

    Returns:
        `SolucionResponse` o `EstadisticasResponse`.

    Raises:
        HTTPException: 503 si el estado no está disponible, 400 si los pesos son
            inválidos, 500 si el optimizador falla.

    Referencia: Tema 3 y Tema 10 del curso CE UNI 2026.
    """
    problema = construir_problema(request, peticion.pesos)
    params = obtener_params(request)

    logger.info(
        "POST /optimize | algoritmo=%s n_runs=%d seed=%d pesos=%s",
        peticion.algoritmo,
        peticion.n_runs,
        peticion.seed,
        "personalizados" if peticion.pesos else "por defecto",
    )

    if peticion.algoritmo == "NSGA2":
        return _optimizar_nsga2(problema, params, peticion.seed)
    if peticion.n_runs > 1:
        return _optimizar_ag_multiple(problema, params, peticion)
    return _optimizar_ag_simple(problema, params, peticion.seed)


def _optimizar_ag_simple(problema: Problema, params, seed: int) -> SolucionResponse:
    """
    Ejecuta una sola corrida del AG.

    Args:
        problema: Problema MCKP.
        params: Parámetros del proyecto.
        seed: Semilla de la corrida.

    Returns:
        La solución encontrada.
    """
    algoritmo = AG(problema, params, seed=seed)
    solucion = algoritmo.ejecutar()
    return solucion_a_response(solucion, algoritmo.tiempo_seg)


def _optimizar_ag_multiple(
    problema: Problema, params, peticion: OptimizeRequest
) -> EstadisticasResponse:
    """
    Ejecuta N corridas del AG y las contrasta con el Greedy.

    Args:
        problema: Problema MCKP.
        params: Parámetros del proyecto.
        peticion: Cuerpo de la petición, con n_runs y seed.

    Returns:
        Estadísticas de las corridas más la mejor solución.
    """
    df_runs = multiples_corridas(
        problema, params, n_runs=peticion.n_runs, seed=peticion.seed
    )
    solucion_greedy = greedy(problema)
    resumen = analisis_wilcoxon(df_runs, solucion_greedy.fitness)

    fila_mejor = df_runs.loc[df_runs["fitness"].idxmax()]
    algoritmo_mejor = AG(problema, params, seed=int(fila_mejor["seed"]))
    mejor = algoritmo_mejor.ejecutar()

    return EstadisticasResponse(
        media=resumen["media"],
        std=resumen["std"],
        mediana=resumen["mediana"],
        minimo=resumen["minimo"],
        maximo=resumen["maximo"],
        greedy=resumen["greedy"],
        mejora_media_pct=resumen["mejora_media_pct"],
        wilcoxon_stat=resumen["wilcoxon_stat"],
        wilcoxon_p=resumen["wilcoxon_p"],
        significativo=resumen["significativo"],
        n_runs_factibles=resumen["n_runs_factibles"],
        tiempo_total_seg=resumen["tiempo_total_seg"],
        corridas=[CorridaItem(**fila) for fila in df_runs.to_dict("records")],
        mejor_solucion=solucion_a_response(mejor, algoritmo_mejor.tiempo_seg),
    )


def _optimizar_nsga2(problema: Problema, params, seed: int) -> SolucionResponse:
    """
    Ejecuta NSGA-II y devuelve la primera solución del frente de Pareto.

    Args:
        problema: Problema MCKP.
        params: Parámetros del proyecto.
        seed: Semilla de la corrida.

    Returns:
        La solución de menor costo del frente.

    Raises:
        HTTPException: 500 si el frente sale vacío, es decir, si no se encontró
            ninguna solución factible.
    """
    algoritmo = NSGA2(problema, params, seed=seed)
    frente = algoritmo.ejecutar()
    if not frente:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "NSGA-II no encontro ninguna solucion factible: revisar si K, B y m_r "
                "son compatibles, o aumentar el numero de generaciones"
            ),
        )
    return solucion_a_response(frente[0], algoritmo.tiempo_seg)
