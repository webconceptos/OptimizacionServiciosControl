"""
Router del frente de Pareto.

Ejecuta NSGA-II (Tema 10) y devuelve el frente de soluciones no dominadas
factibles. Los costos van en miles de soles (kS/).

Referencia: docs/API_SPEC.md — GET /pareto
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request

from algorithms.nsga2 import NSGA2
from api.dependencias import obtener_params, obtener_problema
from data.schemas import ParetoPoint, ParetoResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pareto"])


@router.get(
    "/pareto",
    response_model=ParetoResponse,
    summary="Frente de Pareto de NSGA-II",
)
def pareto(
    request: Request,
    pop: int = Query(100, ge=10, le=500, description="Tamano de poblacion NSGA-II"),
    gen: int = Query(200, ge=10, le=1000, description="Numero de generaciones"),
) -> ParetoResponse:
    """
    Calcula el frente de Pareto del compromiso riesgo cubierto vs costo.

    Objetivos: f1 = Σ R̃ᵢ·xᵢ a maximizar y f2 = Σ Cᵢ·xᵢ (kS/) a minimizar. El
    frente viene ordenado por f2 ascendente y solo contiene soluciones factibles y
    no dominadas entre sí.

    Args:
        request: Petición HTTP en curso.
        pop: Tamaño de población de NSGA-II.
        gen: Número de generaciones.

    Returns:
        `ParetoResponse` con el número de soluciones, el tiempo de cómputo y la
        lista de puntos del frente.

    Referencia: Tema 10 del curso CE UNI 2026.
    """
    problema = obtener_problema(request)
    params = obtener_params(request)

    logger.info("GET /pareto | pop=%d gen=%d", pop, gen)

    algoritmo = NSGA2(problema, params)
    algoritmo.pop_size = pop
    algoritmo.n_gen = gen
    algoritmo.ejecutar()

    puntos = [ParetoPoint(**punto) for punto in algoritmo.pareto_a_dicts()]
    if not puntos:
        logger.warning(
            "GET /pareto devolvio un frente vacio: no se encontro ninguna solucion "
            "factible con pop=%d gen=%d",
            pop,
            gen,
        )
    return ParetoResponse(
        n_soluciones=len(puntos),
        tiempo_seg=round(algoritmo.tiempo_seg, 4),
        soluciones=puntos,
    )
