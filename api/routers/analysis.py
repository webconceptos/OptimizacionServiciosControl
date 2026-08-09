"""
Router de análisis.

Tres endpoints de solo lectura sobre el mismo dataset: estadística de múltiples
corridas (Wilcoxon), Teoría de Esquemas (Tema 4) y sensibilidad del peso w₁.

Los tres son costosos: ejecutan uno o varios AG completos. Se declaran como
funciones síncronas para que FastAPI los lleve al threadpool y no bloqueen el
bucle de eventos.

Referencia: docs/API_SPEC.md — GET /analysis/*
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request

from algorithms.benchmarks import greedy
from analysis.schema_theory import analizar_building_blocks, resumen_building_blocks
from analysis.sensitivity import (
    GEN_SENSIBILIDAD,
    POP_SENSIBILIDAD,
    resumen_sensibilidad,
    sensibilidad_w1,
)
from analysis.statistics import analisis_wilcoxon, multiples_corridas
from api.dependencias import obtener_df, obtener_params, obtener_problema
from data.schemas import (
    BuildingBlock,
    CorridaItem,
    EstadisticasResponse,
    SchemasResponse,
    SensitivityResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get(
    "/runs",
    response_model=EstadisticasResponse,
    summary="Estadisticas de multiples corridas del AG",
)
def analysis_runs(
    request: Request,
    n_runs: int = Query(10, ge=2, le=20, description="Corridas independientes"),
    seed: int = Query(42, description="Semilla base; la corrida i usa seed+i"),
) -> EstadisticasResponse:
    """
    Ejecuta N corridas del AG y las contrasta con el Greedy por Wilcoxon.

    Args:
        request: Petición HTTP en curso.
        n_runs: Número de corridas independientes, entre 2 y 20.
        seed: Semilla base de las corridas.

    Returns:
        `EstadisticasResponse` con media, dispersión, el p-valor de Wilcoxon y el
        detalle por corrida.

    Referencia: docs/ARCHITECTURE.md — Wilcoxon signed-rank sobre t-test.
    """
    problema = obtener_problema(request)
    params = obtener_params(request)
    logger.info("GET /analysis/runs | n_runs=%d seed=%d", n_runs, seed)

    df_runs = multiples_corridas(problema, params, n_runs=n_runs, seed=seed)
    resumen = analisis_wilcoxon(df_runs, greedy(problema).fitness)

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
    )


@router.get(
    "/schemas",
    response_model=SchemasResponse,
    summary="Building blocks (Teoria de Esquemas)",
)
def analysis_schemas(
    request: Request,
    top_k: int = Query(10, ge=5, le=20, description="Obras de mayor ratio b/C"),
) -> SchemasResponse:
    """
    Identifica los building blocks y su factor de crecimiento esperado.

    El crecimiento es K_G·K_S, con K_S la cota inferior de Goldberg: omite el
    término (1 − Pr(H,t)) de la ec. 23 de la Clase 03, así que subestima la
    supervivencia real. Los esquemas con crecimiento > 1 son amplificados por
    selección.

    Args:
        request: Petición HTTP en curso.
        top_k: Número de obras de mayor ratio con las que se arman los esquemas.

    Returns:
        `SchemasResponse` con los esquemas ordenados por crecimiento descendente.

    Referencia: Tema 4 del curso CE UNI 2026.
    """
    problema = obtener_problema(request)
    logger.info("GET /analysis/schemas | top_k=%d", top_k)

    df_bbs = analizar_building_blocks(problema, top_k=top_k)
    resumen = resumen_building_blocks(df_bbs)

    columnas = ["rank", "o_H", "delta_H", "f_H", "fitness_relativo", "crecimiento", "descripcion"]
    bloques = [
        BuildingBlock(**fila) for fila in df_bbs[columnas].to_dict("records")
    ]
    return SchemasResponse(
        building_blocks=bloques,
        umbral_favorecido=float(resumen["umbral_favorecido"]),
        interpretacion=str(resumen["interpretacion"]),
    )


@router.get(
    "/sensitivity",
    response_model=SensitivityResponse,
    summary="Sensibilidad del peso w1 del riesgo",
)
def analysis_sensitivity(
    request: Request,
    pop: int = Query(POP_SENSIBILIDAD, ge=10, le=300, description="Poblacion del AG"),
    gen: int = Query(GEN_SENSIBILIDAD, ge=10, le=600, description="Generaciones del AG"),
) -> SensitivityResponse:
    """
    Barre w₁ y devuelve el óptimo alcanzado con cada valor.

    ADVERTENCIA: `fitness_optimo` NO es comparable entre valores de w₁, porque cada
    w₁ define una función objetivo distinta. Su crecimiento monótono es un
    artefacto de escala y no señala un w₁ óptimo: la elección de w₁ es normativa.
    La robustez de la decisión se mide con la similitud del portafolio, que el
    módulo `analysis/sensitivity.py` calcula y la figura 6 grafica.

    Args:
        request: Petición HTTP en curso.
        pop: Población del AG en cada punto del barrido.
        gen: Generaciones del AG en cada punto del barrido.

    Returns:
        `SensitivityResponse` con el rango de w₁, los fitness y el valor usado en
        el trabajo.

    Referencia: README.md, formulacion del MCKP (pesos de la funcion objetivo).
    """
    df = obtener_df(request)
    params = obtener_params(request)
    logger.info("GET /analysis/sensitivity | pop=%d gen=%d", pop, gen)

    df_sens = sensibilidad_w1(df, params, pop_size=pop, n_gen=gen)
    resumen = resumen_sensibilidad(df_sens)

    return SensitivityResponse(
        w1_rango=resumen["w1_rango"],
        fitness_optimo=resumen["fitness_optimo"],
        w1_usado=resumen["w1_usado"],
        fitness_en_w1_usado=resumen["fitness_en_w1_usado"],
    )
