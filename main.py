"""
Punto de entrada por línea de comandos.

Orquesta todo lo construido en `data/`, `core/`, `algorithms/`, `analysis/`,
`visualization/` y `api/`. No implementa lógica propia: solo compone.

Modos disponibles:

    full      dataset -> AG (N corridas) + Wilcoxon -> NSGA-II -> benchmarks ->
              esquemas -> sensibilidad -> 9 figuras -> metricas.json y
              metricas_ext.json -> resumen en consola
    ag        una corrida del AG con la semilla indicada
    nsga2     solo NSGA-II, con el frente de Pareto
    analysis  solo el analisis estadistico de N corridas + Wilcoxon
    api       levanta la API REST con uvicorn en el puerto configurado

Todos los montos se reportan en **miles de soles (kS/)**, la escala de la columna
C del dataset: sum(C) ≈ 209.7 kS/ y B = 35 % de eso ≈ 73.4 kS/.

Ejemplos:
    python main.py --mode full --output ./outputs
    python main.py --mode full --con-sensibilidad-parametros
    python main.py --mode ag --seed 42
    python main.py --mode analysis --n-runs 10
    python main.py --mode api --port 8001

El barrido de parámetros del AG (pop_size, pc, pm) NO corre por defecto en modo
full: son 39 corridas extra que casi duplican la duración (~67 s frente a ~130 s)
y su resultado no altera las métricas del informe. Se activa con
`--con-sensibilidad-parametros`, que además genera la figura 10 y el CSV
`sensibilidad_parametros.csv`.

Referencia: README.md, comandos de uso
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from algorithms.benchmarks import aleatorio, greedy
from algorithms.genetic import AG
from algorithms.nsga2 import NSGA2
from analysis.schema_theory import analizar_building_blocks, resumen_building_blocks
from analysis.sensitivity import (
    N_GEN_PARAMETROS,
    resumen_sensibilidad,
    resumen_sensibilidad_parametros,
    sensibilidad_parametros,
    sensibilidad_w1,
)
from analysis.statistics import analisis_wilcoxon, multiples_corridas, tabla_comparacion
from config.params import Params
from core.problem import Problema
from core.solution import Solucion
from data.generator import generar_obras
from visualization import plots

logger = logging.getLogger("main")

#: Formato de log estructurado del proyecto.
FORMATO_LOG: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

#: Nombres de las figuras del informe.
FIGURAS: Dict[str, str] = {
    "convergencia": "fig1_convergencia.png",
    "comparacion": "fig2_comparacion.png",
    "territorial": "fig3_territorial.png",
    "top_obras": "fig4_top_obras.png",
    "scatter": "fig5_scatter.png",
    "sensibilidad": "fig6_sensibilidad.png",
    "multiples_corridas": "fig7_multiples_corridas.png",
    "pareto": "fig8_pareto_nsga2.png",
    "esquemas": "fig9_esquemas.png",
    "parametros": "fig10_sensibilidad_parametros.png",
}

#: Semillas del barrido de parámetros del AG en modo full.
SEEDS_PARAMETROS: tuple[int, ...] = (42, 43, 44)

#: Nombre del CSV del dataset recalibrado (v2) que se guarda en modo full.
CSV_DATASET: str = "obras_326_recalibradas.csv"

#: Población y generaciones de NSGA-II en modo full (frente amplio sin encarecer).
POP_NSGA2: int = 100
GEN_NSGA2: int = 200

#: Intentos del benchmark aleatorio.
N_INTENTOS_ALEATORIO: int = 10_000

#: Obras de mayor ratio con las que se arman los esquemas.
TOP_K_ESQUEMAS: int = 10

#: Ancho de las líneas del resumen en consola.
ANCHO: int = 78


def configurar_logging(nivel: int = logging.INFO) -> None:
    """
    Configura el logging estructurado del proceso.

    Fuerza `errors="replace"` en la salida estándar: la consola de Windows usa
    cp1252 y de otro modo los caracteres no representables la rompen o la
    ensucian con escapes.

    Args:
        nivel: Nivel mínimo a registrar.
    """
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    logging.basicConfig(level=nivel, format=FORMATO_LOG, stream=sys.stdout, force=True)


def construir_parser(params: Params) -> argparse.ArgumentParser:
    """
    Construye el parser de argumentos con los valores por defecto de `params`.

    Args:
        params: Parámetros del proyecto, de donde salen los defaults de semilla y
            puerto (nada hard-codeado).

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Optimizacion evolutiva de inversiones publicas para control preventivo "
            "(Parte 2: Computacion Evolutiva). Montos en miles de soles (kS/)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "ag", "nsga2", "analysis", "api"],
        default="full",
        help="Modo de ejecucion",
    )
    parser.add_argument(
        "--output", default="./outputs", help="Carpeta de salida de figuras y metricas"
    )
    parser.add_argument(
        "--n-runs", type=int, default=int(params.N_RUNS), help="Corridas independientes del AG"
    )
    parser.add_argument("--seed", type=int, default=int(params.SEED), help="Semilla base")
    parser.add_argument("--port", type=int, default=int(params.API_PORT), help="Puerto de la API")
    parser.add_argument(
        "--n-obras", type=int, default=int(params.N_OBRAS), help="Obras del dataset sintetico"
    )
    parser.add_argument(
        "--con-sensibilidad-parametros",
        action="store_true",
        help=(
            "En modo full, anade el barrido de parametros del AG (pop_size, pc, pm) "
            "y la figura 10. Cuesta 39 corridas extra y casi duplica la duracion "
            "(~67 s -> ~130 s), por eso esta desactivado por defecto"
        ),
    )
    return parser


def _contador_pasos(total: int) -> Callable[[str], None]:
    """
    Crea un logger de pasos numerados `[i/total]`.

    El total depende de si el barrido de parámetros del AG está activado, así que
    la numeración se calcula en tiempo de ejecución en lugar de estar escrita en
    cada mensaje.

    Args:
        total: Número total de pasos previstos.

    Returns:
        Función que registra el siguiente paso con su numeración.
    """
    estado = {"i": 0}

    def paso(mensaje: str) -> None:
        estado["i"] += 1
        logger.info("[%d/%d] %s", estado["i"], total, mensaje)

    return paso


def preparar_datos(params: Params, n_obras: int, seed: int) -> tuple[pd.DataFrame, Problema]:
    """
    Genera el dataset recalibrado y construye el problema MCKP.

    Args:
        params: Parámetros del proyecto.
        n_obras: Número de obras a generar.
        seed: Semilla del generador.

    Returns:
        Tupla (dataset, problema).
    """
    df = generar_obras(n=n_obras, seed=seed)
    problema = Problema(df, params)
    logger.info(
        "Problema listo | n=%d | B=%.4f kS/ (%.0f%% de %.4f) | K=%d | m_r=%s",
        problema.n,
        problema.B,
        100.0 * params.PRESUPUESTO_PCT,
        float(problema.C.sum()),
        problema.K,
        problema.m_r,
    )
    return df, problema


def _f1(solucion: Solucion, problema: Problema) -> float:
    """
    Primer objetivo de NSGA-II para una solución: riesgo cubierto.

    Args:
        solucion: Solución a evaluar.
        problema: Problema MCKP, del que se toma R_n.

    Returns:
        f1 = Σ R̃ᵢ·xᵢ.
    """
    r_norm = problema.df["R_n"].to_numpy(dtype=float)
    return float(np.asarray(solucion.x, dtype=float) @ r_norm)


def modo_ag(problema: Problema, params: Params, seed: int) -> tuple[Solucion, AG]:
    """
    Ejecuta una corrida del AG.

    Args:
        problema: Problema MCKP.
        params: Parámetros del proyecto.
        seed: Semilla de la corrida.

    Returns:
        Tupla (solución, instancia del AG con su historial).
    """
    algoritmo = AG(problema, params, seed=seed)
    solucion = algoritmo.ejecutar()
    return solucion, algoritmo


def modo_nsga2(problema: Problema, params: Params, seed: int) -> tuple[List[Solucion], NSGA2]:
    """
    Ejecuta NSGA-II y devuelve el frente de Pareto factible.

    Args:
        problema: Problema MCKP.
        params: Parámetros del proyecto.
        seed: Semilla de la corrida.

    Returns:
        Tupla (frente, instancia de NSGA2).
    """
    algoritmo = NSGA2(problema, params, seed=seed)
    algoritmo.pop_size = POP_NSGA2
    algoritmo.n_gen = GEN_NSGA2
    frente = algoritmo.ejecutar()
    return frente, algoritmo


def modo_analysis(
    problema: Problema, params: Params, n_runs: int, seed: int
) -> tuple[pd.DataFrame, Dict[str, Any], Solucion]:
    """
    Ejecuta N corridas del AG y las contrasta con el Greedy por Wilcoxon.

    Args:
        problema: Problema MCKP.
        params: Parámetros del proyecto.
        n_runs: Número de corridas independientes.
        seed: Semilla base.

    Returns:
        Tupla (DataFrame de corridas, resumen estadístico, solución Greedy).
    """
    df_runs = multiples_corridas(problema, params, n_runs=n_runs, seed=seed)
    solucion_greedy = greedy(problema)
    resumen = analisis_wilcoxon(df_runs, solucion_greedy.fitness)
    return df_runs, resumen, solucion_greedy


def modo_full(
    params: Params,
    salida: Path,
    n_obras: int,
    n_runs: int,
    seed: int,
    con_sensibilidad_parametros: bool = False,
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo y escribe figuras y métricas.

    Args:
        params: Parámetros del proyecto.
        salida: Carpeta de salida.
        n_obras: Obras del dataset.
        n_runs: Corridas independientes del AG.
        seed: Semilla base.
        con_sensibilidad_parametros: Si es True, añade el barrido de parámetros del
            AG (pop_size, pc, pm) y la figura 10. Está desactivado por defecto
            porque ejecuta 39 corridas extra y casi duplica la duración del modo
            full (de ~67 s a ~130 s), sin que su resultado cambie las métricas del
            informe.

    Returns:
        Diccionario de métricas principales (el contenido de metricas.json).
    """
    inicio = time.perf_counter()
    salida.mkdir(parents=True, exist_ok=True)
    total_pasos = 10 if con_sensibilidad_parametros else 9
    paso = _contador_pasos(total_pasos)

    # --- 1. Dataset -------------------------------------------------------
    paso("Dataset recalibrado (v2) con las proporciones reales de la Parte 1")
    df, problema = preparar_datos(params, n_obras, seed)
    ruta_csv = salida / CSV_DATASET
    df.to_csv(ruta_csv, index=False)
    logger.info("      CSV guardado | %s", ruta_csv)

    # --- 2. AG: N corridas + Wilcoxon -------------------------------------
    paso(f"AG: {n_runs} corridas independientes + prueba de Wilcoxon")
    df_runs, estadisticas, solucion_greedy = modo_analysis(problema, params, n_runs, seed)

    semilla_mejor = int(df_runs.loc[df_runs["fitness"].idxmax(), "seed"])
    solucion_ag, algoritmo_ag = modo_ag(problema, params, semilla_mejor)
    logger.info("      Mejor corrida: seed=%d fitness=%.6f", semilla_mejor, solucion_ag.fitness)

    # --- 3. NSGA-II --------------------------------------------------------
    paso(f"NSGA-II biobjetivo (pop={POP_NSGA2}, gen={GEN_NSGA2})")
    frente, algoritmo_nsga2 = modo_nsga2(problema, params, seed)

    # --- 4. Benchmarks -----------------------------------------------------
    paso(f"Benchmarks: Greedy (ya calculado) y Aleatorio ({N_INTENTOS_ALEATORIO} intentos)")
    solucion_aleatoria = aleatorio(problema, n_intentos=N_INTENTOS_ALEATORIO)

    # --- 5. Teoria de esquemas --------------------------------------------
    paso(f"Teoria de Esquemas: building blocks (top_k={TOP_K_ESQUEMAS})")
    df_bbs = analizar_building_blocks(problema, top_k=TOP_K_ESQUEMAS)
    resumen_bbs = resumen_building_blocks(df_bbs)

    # --- 6. Sensibilidad de w1 --------------------------------------------
    paso("Sensibilidad del peso w1 del riesgo")
    df_sens = sensibilidad_w1(df, params, seed=seed)
    resumen_sens = resumen_sensibilidad(df_sens)

    # --- 7. Sensibilidad de los parametros del AG (opcional) ---------------
    df_params: Optional[pd.DataFrame] = None
    resumen_params: Optional[Dict[str, Any]] = None
    if con_sensibilidad_parametros:
        paso(
            f"Sensibilidad de los parametros del AG (pop, pc, pm) con "
            f"{len(SEEDS_PARAMETROS)} semillas"
        )
        df_params = sensibilidad_parametros(problema, params, seeds=SEEDS_PARAMETROS)
        resumen_params = resumen_sensibilidad_parametros(df_params)
    else:
        logger.info(
            "      Barrido de parametros del AG omitido "
            "(activar con --con-sensibilidad-parametros)"
        )

    # --- 8. Figuras --------------------------------------------------------
    n_figuras = len(FIGURAS) if con_sensibilidad_parametros else len(FIGURAS) - 1
    paso(f"Generando las {n_figuras} figuras en {salida}")
    comparacion = tabla_comparacion(
        {
            "AG (propuesto)": solucion_ag,
            "Greedy": solucion_greedy,
            "Aleatorio": solucion_aleatoria,
        }
    )
    plots.plot_convergencia(algoritmo_ag.historial, salida / FIGURAS["convergencia"])
    plots.plot_comparacion(comparacion, salida / FIGURAS["comparacion"])
    plots.plot_territorial(solucion_ag, salida / FIGURAS["territorial"])
    plots.plot_top_obras(solucion_ag, salida / FIGURAS["top_obras"], top_n=20)
    plots.plot_scatter_riesgo_costo(solucion_ag, salida / FIGURAS["scatter"])
    plots.plot_sensibilidad(df_sens, salida / FIGURAS["sensibilidad"])
    plots.plot_multiples_corridas(df_runs, estadisticas, salida / FIGURAS["multiples_corridas"])
    plots.plot_pareto(
        algoritmo_nsga2.pareto_a_dicts(),
        salida / FIGURAS["pareto"],
        punto_ag=(_f1(solucion_ag, problema), solucion_ag.costo),
        punto_greedy=(_f1(solucion_greedy, problema), solucion_greedy.costo),
    )
    plots.plot_esquemas(
        df_bbs, salida / FIGURAS["esquemas"], n=problema.n, pc=params.PC, pm=params.PM
    )
    if df_params is not None:
        plots.plot_sensibilidad_parametros(
            df_params,
            salida / FIGURAS["parametros"],
            n_gen=N_GEN_PARAMETROS,
            n_seeds=len(SEEDS_PARAMETROS),
        )

    # --- 9. Metricas -------------------------------------------------------
    paso("Guardando metricas")
    duracion = time.perf_counter() - inicio
    metricas = _armar_metricas(
        params,
        problema,
        df,
        solucion_ag,
        solucion_greedy,
        solucion_aleatoria,
        estadisticas,
        frente,
        algoritmo_nsga2,
        semilla_mejor,
        duracion,
    )
    metricas_ext = _armar_metricas_ext(
        problema,
        df_runs,
        estadisticas,
        frente,
        algoritmo_nsga2,
        df_bbs,
        resumen_bbs,
        df_sens,
        resumen_sens,
        df_params,
        resumen_params,
    )
    _guardar_json(salida / "metricas.json", metricas)
    _guardar_json(salida / "metricas_ext.json", metricas_ext)
    df_runs.to_csv(salida / "corridas.csv", index=False)
    df_bbs.drop(columns=["posiciones", "obras"]).to_csv(salida / "building_blocks.csv", index=False)
    df_sens.to_csv(salida / "sensibilidad_w1.csv", index=False)
    if df_params is not None:
        df_params.to_csv(salida / "sensibilidad_parametros.csv", index=False)

    # --- 10. Resumen -------------------------------------------------------
    paso("Resumen")
    _imprimir_resumen(
        problema,
        comparacion,
        estadisticas,
        frente,
        algoritmo_nsga2,
        resumen_bbs,
        resumen_sens,
        resumen_params,
        solucion_ag,
        salida,
        duracion,
    )
    return metricas


def _armar_metricas(
    params: Params,
    problema: Problema,
    df: pd.DataFrame,
    solucion_ag: Solucion,
    solucion_greedy: Solucion,
    solucion_aleatoria: Solucion,
    estadisticas: Dict[str, Any],
    frente: List[Solucion],
    algoritmo_nsga2: NSGA2,
    semilla_mejor: int,
    duracion: float,
) -> Dict[str, Any]:
    """
    Compone el contenido de `metricas.json` (resultados principales).

    Args:
        params: Parámetros del proyecto.
        problema: Problema MCKP.
        df: Dataset usado.
        solucion_ag: Mejor solución del AG.
        solucion_greedy: Solución del Greedy.
        solucion_aleatoria: Mejor solución del benchmark aleatorio.
        estadisticas: Resumen de Wilcoxon.
        frente: Frente de Pareto.
        algoritmo_nsga2: Instancia de NSGA-II ya ejecutada.
        semilla_mejor: Semilla de la mejor corrida del AG.
        duracion: Duración total del pipeline, en segundos.

    Returns:
        Diccionario serializable a JSON.
    """
    return {
        "version_dataset": "v2 (recalibrado con data/calibracion_parte1.json)",
        "unidad_monetaria": "miles S/ (kS/)",
        "n_obras": problema.n,
        "presupuesto_B_kS": round(problema.B, 6),
        "costo_total_universo_kS": round(float(problema.C.sum()), 6),
        "presupuesto_pct": float(params.PRESUPUESTO_PCT),
        "capacidad_K": problema.K,
        "min_por_region": problema.m_r,
        "alpha": problema.alpha,
        "pesos": {a: float(w) for a, w in zip(["R", "M", "P", "E", "G"], problema.w)},
        "parametros_ag": {
            "pop_size": int(params.POP_SIZE),
            "n_gen": int(params.N_GEN),
            "pc": float(params.PC),
            "pm": float(params.PM),
            "k_torneo": int(params.K_TORNEO),
        },
        "seed_base": int(params.SEED),
        "seed_mejor_corrida": semilla_mejor,
        "ag": solucion_ag.to_dict(incluir_obras=True),
        "greedy": solucion_greedy.to_dict(incluir_obras=False),
        "aleatorio": solucion_aleatoria.to_dict(incluir_obras=False),
        "mejora_ag_vs_greedy_pct": round(
            100.0 * (solucion_ag.fitness - solucion_greedy.fitness) / abs(solucion_greedy.fitness), 4
        ),
        "mejora_ag_vs_aleatorio_pct": round(
            100.0
            * (solucion_ag.fitness - solucion_aleatoria.fitness)
            / abs(solucion_aleatoria.fitness),
            4,
        ),
        "estadisticas_corridas": {
            clave: estadisticas[clave]
            for clave in (
                "n_runs",
                "media",
                "std",
                "mediana",
                "minimo",
                "maximo",
                "greedy",
                "mejora_media_pct",
                "wilcoxon_stat",
                "wilcoxon_p",
                "significativo",
                "n_runs_factibles",
                "tiempo_total_seg",
            )
        },
        "nsga2": {
            "n_soluciones_pareto": len(frente),
            "f1_max": round(max((algoritmo_nsga2.f1(s) for s in frente), default=float("nan")), 6),
            "f2_min_kS": round(min((s.costo for s in frente), default=float("nan")), 6),
            "tiempo_seg": round(algoritmo_nsga2.tiempo_seg, 4),
        },
        "distribucion_riesgo_dataset": df["clase_riesgo"].value_counts().to_dict(),
        "tiempo_total_seg": round(duracion, 2),
    }


def _armar_metricas_ext(
    problema: Problema,
    df_runs: pd.DataFrame,
    estadisticas: Dict[str, Any],
    frente: List[Solucion],
    algoritmo_nsga2: NSGA2,
    df_bbs: pd.DataFrame,
    resumen_bbs: Dict[str, Any],
    df_sens: pd.DataFrame,
    resumen_sens: Dict[str, Any],
    df_params: Optional[pd.DataFrame] = None,
    resumen_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compone el contenido de `metricas_ext.json` (Wilcoxon, Pareto, esquemas).

    Args:
        problema: Problema MCKP.
        df_runs: Corridas del AG.
        estadisticas: Resumen de Wilcoxon completo.
        frente: Frente de Pareto.
        algoritmo_nsga2: Instancia de NSGA-II ya ejecutada.
        df_bbs: Building blocks.
        resumen_bbs: Resumen de los esquemas.
        df_sens: Barrido de sensibilidad.
        resumen_sens: Resumen del barrido.

    Returns:
        Diccionario serializable a JSON.
    """
    # El barrido de parametros es opcional: si no se ejecuto, la clave queda como
    # None para que el consumidor sepa que no se midio, en vez de faltar.
    parametros: Optional[Dict[str, Any]] = None
    if df_params is not None and resumen_params is not None:
        parametros = {
            "resumen": resumen_params,
            "barrido": df_params.to_dict("records"),
        }

    return {
        "wilcoxon": estadisticas,
        "corridas": df_runs.to_dict("records"),
        "pareto": {
            "n_soluciones": len(frente),
            "tiempo_seg": round(algoritmo_nsga2.tiempo_seg, 4),
            "metadatos": algoritmo_nsga2.metadatos(),
            "soluciones": algoritmo_nsga2.pareto_a_dicts(),
        },
        "esquemas": {
            "resumen": resumen_bbs,
            "top": df_bbs.head(20).drop(columns=["posiciones"]).to_dict("records"),
        },
        "sensibilidad_w1": {
            "resumen": resumen_sens,
            "barrido": df_sens.to_dict("records"),
        },
        "sensibilidad_parametros": parametros,
        "restricciones": {
            "R1_presupuesto_kS": round(problema.B, 6),
            "R2_capacidad": problema.K,
            "R3_min_por_region": problema.m_r,
            "nota_R3": (
                "R3 es un supuesto institucional de politica de control de la CGR, no un "
                "hecho derivado de los datos reales de la Parte 1 (65.03% de las obras "
                "reales son MULTIDEPARTAMENTALES)"
            ),
        },
    }


def _guardar_json(ruta: Path, contenido: Dict[str, Any]) -> None:
    """
    Escribe un diccionario como JSON legible en cualquier codificación.

    Usa escapes ASCII para poder leerlo con `open()` sin declarar encoding en
    Windows, y convierte los tipos de numpy que `json` no sabe serializar.

    Args:
        ruta: Ruta del archivo de salida.
        contenido: Diccionario a serializar.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, indent=2, default=_json_seguro), encoding="utf-8")
    logger.info("      %s", ruta)


def _json_seguro(valor: Any) -> Any:
    """
    Convierte tipos de numpy y pandas a tipos nativos de Python.

    Args:
        valor: Valor que `json` no pudo serializar.

    Returns:
        Equivalente nativo.

    Raises:
        TypeError: Si el tipo no es convertible.
    """
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, (pd.Timestamp,)):
        return valor.isoformat()
    raise TypeError(f"Tipo no serializable a JSON: {type(valor).__name__}")


def _imprimir_resumen(
    problema: Problema,
    comparacion: pd.DataFrame,
    estadisticas: Dict[str, Any],
    frente: List[Solucion],
    algoritmo_nsga2: NSGA2,
    resumen_bbs: Dict[str, Any],
    resumen_sens: Dict[str, Any],
    resumen_params: Optional[Dict[str, Any]],
    solucion_ag: Solucion,
    salida: Path,
    duracion: float,
) -> None:
    """
    Imprime el resumen final en consola, en ASCII.

    Args:
        problema: Problema MCKP.
        comparacion: Tabla comparativa de métodos.
        estadisticas: Resumen de Wilcoxon.
        frente: Frente de Pareto.
        algoritmo_nsga2: Instancia de NSGA-II ya ejecutada.
        resumen_bbs: Resumen de los esquemas.
        resumen_sens: Resumen de la sensibilidad de w1.
        resumen_params: Resumen de la sensibilidad de los parámetros del AG, o None
            si el barrido no se ejecutó.
        solucion_ag: Mejor solución del AG.
        salida: Carpeta de salida.
        duracion: Duración total, en segundos.
    """
    linea = "=" * ANCHO
    print("\n" + linea)
    print(" RESUMEN - Optimizacion Evolutiva de Inversiones Publicas (Parte 2)")
    print(linea)
    print(
        f" Problema MCKP | n={problema.n} obras | B={problema.B:.4f} kS/ | "
        f"K={problema.K} | m_r={list(problema.m_r.values())}"
    )
    print(f" Dataset v2 recalibrado con las proporciones reales de la Parte 1")

    print("\n TABLA COMPARATIVA")
    print(f" {'Metodo':<18}{'Fitness Z':>12}{'Obras':>7}{'Costo kS/':>11}{'Factible':>10}{'vs Greedy':>11}")
    print(" " + "-" * (ANCHO - 2))
    for fila in comparacion.itertuples():
        mejora = "" if pd.isna(fila.mejora_vs_greedy_pct) else f"{fila.mejora_vs_greedy_pct:+.2f}%"
        print(
            f" {fila.metodo:<18}{fila.fitness:>12.6f}{fila.n_seleccionadas:>7}"
            f"{fila.costo:>11.4f}{'si' if fila.factible else 'NO':>10}{mejora:>11}"
        )

    print(f"\n AG - {estadisticas['n_runs']} CORRIDAS INDEPENDIENTES")
    print(f"   media +- std   : {estadisticas['media']:.6f} +- {estadisticas['std']:.6f}")
    print(f"   mediana        : {estadisticas['mediana']:.6f}")
    print(f"   rango          : [{estadisticas['minimo']:.6f}, {estadisticas['maximo']:.6f}]")
    print(f"   Greedy         : {estadisticas['greedy']:.6f}")
    print(f"   mejora media   : {estadisticas['mejora_media_pct']:+.2f}%")
    print(
        f"   Wilcoxon       : stat={estadisticas['wilcoxon_stat']:.4f} "
        f"p={estadisticas['wilcoxon_p']:.6f} -> "
        f"{'SIGNIFICATIVO' if estadisticas['significativo'] else 'no significativo'} (p<0.05)"
    )
    print(
        f"   factibilidad   : {estadisticas['n_runs_factibles']}/{estadisticas['n_runs']} corridas | "
        f"corridas mejores que Greedy: {estadisticas['n_corridas_mejores']}/{estadisticas['n_runs']}"
    )

    print("\n NSGA-II (frente de Pareto)")
    if frente:
        print(f"   soluciones     : {len(frente)} no dominadas y factibles")
        print(f"   f1 (riesgo)    : max {max(algoritmo_nsga2.f1(s) for s in frente):.4f}")
        print(f"   f2 (costo)     : min {min(s.costo for s in frente):.4f} kS/")
        print(
            f"   obras por sol. : {min(s.n_seleccionadas for s in frente)} a "
            f"{max(s.n_seleccionadas for s in frente)}"
        )
    else:
        print("   frente vacio: no se hallaron soluciones factibles")
    print(f"   tiempo         : {algoritmo_nsga2.tiempo_seg:.2f} s")

    print("\n DISTRIBUCION TERRITORIAL (mejor solucion del AG)")
    for region, conteo in sorted(solucion_ag.distribucion_territorial.items()):
        minimo = problema.m_r.get(region, 0)
        print(f"   Macroregion {region}: {conteo:>2} obras (min {minimo}) {'ok' if conteo >= minimo else 'FALLA'}")

    print("\n TEORIA DE ESQUEMAS (Tema 4)")
    print(
        f"   {resumen_bbs['n_favorecidos']}/{resumen_bbs['n_esquemas']} esquemas con "
        f"crecimiento > {resumen_bbs['umbral_favorecido']:.1f} | maximo {resumen_bbs['crecimiento_max']:.4f}"
    )
    for orden, datos in sorted(resumen_bbs["por_orden"].items()):
        print(
            f"   o(H)={orden}: {datos['n_favorecidos']:>3}/{datos['n_esquemas']:<3} favorecidos | "
            f"supervivencia media {datos['supervivencia_media']:.4f}"
        )

    print("\n SENSIBILIDAD DE w1")
    print(f"   w1 usado       : {resumen_sens['w1_usado']} (fitness {resumen_sens['fitness_en_w1_usado']:.4f})")
    print(
        f"   similitud del portafolio (Jaccard) vs w1 usado: min {resumen_sens['similitud_min']:.3f}, "
        f"media {resumen_sens['similitud_media']:.3f}"
    )
    print(f"   riesgo medio de las obras elegidas: {resumen_sens['r_medio_rango']}")
    print("   NOTA: el fitness no es comparable entre valores de w1 (escalas distintas)")

    if resumen_params is None:
        print("\n SENSIBILIDAD DE LOS PARAMETROS DEL AG")
        print("   omitida (activar con --con-sensibilidad-parametros)")
    else:
        print(
            f"\n SENSIBILIDAD DE LOS PARAMETROS DEL AG (n_gen fijo={resumen_params['n_gen_fijo']})"
        )
        print(
            f" {'Parametro':<12}{'Usado':>8}{'Fitness':>11}{'Mejor':>8}{'Fitness':>11}"
            f"{'Ganancia':>10}{'Costo':>9}"
        )
        print(" " + "-" * (ANCHO - 2))
        for nombre, datos in resumen_params["por_parametro"].items():
            if datos.get("valor_usado") is None:
                continue
            marca = " =" if datos["el_usado_es_el_mejor"] else ""
            print(
                f" {nombre:<12}{datos['valor_usado']:>8g}{datos['fitness_usado']:>11.4f}"
                f"{datos['mejor_valor']:>8g}{datos['mejor_fitness']:>11.4f}"
                f"{datos['delta_fitness_vs_usado_pct']:>+9.2f}%"
                f"{datos['factor_tiempo_vs_usado']:>8.2f}x{marca}"
            )
        print(" " + "-" * (ANCHO - 2))
        print("   'Costo' = tiempo del mejor valor frente al usado; '=' indica que coinciden")

    print(f"\n Salidas en {salida.resolve()} | tiempo total {duracion:.1f} s")
    print(linea + "\n")


def modo_api(params: Params, puerto: int) -> None:
    """
    Levanta la API REST con uvicorn.

    Args:
        params: Parámetros del proyecto.
        puerto: Puerto de escucha.

    Raises:
        SystemExit: Si uvicorn no está instalado.
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise SystemExit(
            "uvicorn no esta instalado: pip install -r requirements.txt"
        ) from exc

    from api.main import app

    logger.info("Levantando la API en http://0.0.0.0:%d (docs en /docs)", puerto)
    uvicorn.run(app, host="0.0.0.0", port=puerto, log_config=None)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Punto de entrada del CLI.

    Args:
        argv: Argumentos de línea de comandos; None usa `sys.argv`.

    Returns:
        0 si la ejecución terminó bien.
    """
    configurar_logging()
    params = Params.get()
    args = construir_parser(params).parse_args(argv)
    salida = Path(args.output)

    logger.info(
        "Modo=%s | output=%s | n_runs=%d | seed=%d | port=%d",
        args.mode,
        salida,
        args.n_runs,
        args.seed,
        args.port,
    )

    if args.mode == "api":
        modo_api(params, args.port)
        return 0

    if args.mode == "full":
        modo_full(
            params,
            salida,
            args.n_obras,
            args.n_runs,
            args.seed,
            con_sensibilidad_parametros=args.con_sensibilidad_parametros,
        )
        return 0

    _, problema = preparar_datos(params, args.n_obras, args.seed)

    if args.mode == "ag":
        solucion, _ = modo_ag(problema, params, args.seed)
        print(
            f"\nAG | fitness={solucion.fitness:.6f} | obras={solucion.n_seleccionadas} | "
            f"costo={solucion.costo:.4f} kS/ | factible={'si' if solucion.factible else 'NO'}"
        )
        print(f"     territorial: {solucion.distribucion_territorial}")
    elif args.mode == "nsga2":
        frente, algoritmo = modo_nsga2(problema, params, args.seed)
        print(f"\nNSGA-II | {len(frente)} soluciones Pareto factibles en {algoritmo.tiempo_seg:.2f} s")
        if frente:
            print(f"        f1 max={max(algoritmo.f1(s) for s in frente):.4f} | f2 min={min(s.costo for s in frente):.4f} kS/")
    elif args.mode == "analysis":
        df_runs, estadisticas, solucion_greedy = modo_analysis(
            problema, params, args.n_runs, args.seed
        )
        print(f"\nAG | {args.n_runs} corridas: {estadisticas['media']:.6f} +- {estadisticas['std']:.6f}")
        print(f"   Greedy: {solucion_greedy.fitness:.6f} | mejora media {estadisticas['mejora_media_pct']:+.2f}%")
        print(
            f"   Wilcoxon p={estadisticas['wilcoxon_p']:.6f} -> "
            f"{'SIGNIFICATIVO' if estadisticas['significativo'] else 'no significativo'}"
        )
        print(f"   factibles: {estadisticas['n_runs_factibles']}/{args.n_runs}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
