"""
Análisis de sensibilidad del peso del riesgo w₁.

w₁ = 0.40 es la decisión más discutible de la formulación: es el peso que se le
da a Rᵢ = P(Extrem. Riesgosa), el output del Random Forest de la Parte 1, frente
a los criterios presupuestales. Este módulo barre w₁ y mide cómo cambia el óptimo
que alcanza el AG, para saber si la conclusión del trabajo depende de ese valor o
es robusta a él.

Redistribución: al fijar w₁, el resto (1 − w₁) se reparte entre M, P, E y G
**manteniendo sus proporciones originales**. Con los pesos del
proyecto (M=0.25, P=0.15, E=0.10, G=0.10, que suman 0.60), el nuevo peso de M es
0.25/0.60·(1 − w₁), y así con los demás. De ese modo los cuatro criterios
presupuestales conservan su importancia relativa entre sí y la única variable del
experimento es w₁, no la mezcla completa.

ADVERTENCIA DE INTERPRETACIÓN — `fitness_optimo` NO es comparable entre valores
de w₁. Cada w₁ define una función objetivo Z distinta, así que sus óptimos están
en escalas distintas: medido con n=326, el fitness crece de forma monótona de
15.92 (w₁=0.10) a 37.09 (w₁=0.75) simplemente porque R̃ toma valores más altos que
M̃, P̃, Ẽ y G̃ en las obras que el AG selecciona. Leer ese crecimiento como "w₁=0.75
es mejor" es un error: no hay ningún w₁ óptimo que este barrido pueda revelar,
porque la elección de w₁ es normativa (cuánto importa el riesgo frente al monto),
no empírica.

Lo que sí es informativo es cómo cambia la **decisión**, y para eso están las
columnas `similitud_vs_usado` (índice de Jaccard del portafolio frente al de
w₁=0.40) y `r_medio` (riesgo promedio de las obras elegidas). Si la similitud se
mantiene alta en un rango amplio de w₁, la recomendación del trabajo es robusta a
ese parámetro; si cae rápido, la conclusión depende del peso elegido y hay que
declararlo como limitación.

Este módulo **solo calcula**: no dibuja nada. La figura correspondiente vive en
`visualization/plots.py`.

Referencia: README.md, formulacion del MCKP (pesos de la funcion objetivo)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from algorithms.genetic import AG
from config.params import Params
from core.problem import Problema

logger = logging.getLogger(__name__)

#: Rango de w₁ por defecto: de 0.10 a 0.75 en pasos de 0.05.
W1_RANGE_DEFECTO: np.ndarray = np.arange(0.10, 0.76, 0.05)

#: Población del AG en el barrido: reducida frente a las 150 de la corrida
#: principal, porque se ejecuta un AG completo por cada valor de w₁.
POP_SENSIBILIDAD: int = 80

#: Generaciones del AG en el barrido, por el mismo motivo.
GEN_SENSIBILIDAD: int = 150

#: Atributos que absorben la redistribución de (1 − w₁).
ATRIBUTOS_RESTO: List[str] = ["M", "P", "E", "G"]

#: Tolerancia para marcar la fila que corresponde al w₁ usado en el trabajo.
TOLERANCIA_W1: float = 1e-9

#: Valores a barrer de cada parámetro del AG, uno a la vez.
GRID_PARAMETROS: Dict[str, Sequence[float]] = {
    "pop_size": (50, 100, 150, 200, 300),
    "pc": (0.60, 0.70, 0.85, 0.95),
    "pm": (0.005, 0.015, 0.03, 0.05),
}

#: Atributo de `Params` del que sale el valor por defecto de cada parámetro.
ATRIBUTO_PARAMS: Dict[str, str] = {
    "pop_size": "POP_SIZE",
    "pc": "PC",
    "pm": "PM",
}

#: Generaciones fijas en todo el barrido de parámetros. Se mantiene constante a
#: propósito: si n_gen variara, los tiempos no serían comparables entre
#: configuraciones y el barrido mediría dos cosas a la vez.
N_GEN_PARAMETROS: int = 200

#: Semillas por defecto del barrido de parámetros.
SEEDS_PARAMETROS: Sequence[int] = (42, 43, 44)

#: Columnas del DataFrame de sensibilidad de parámetros, en orden.
COLUMNAS_PARAMETROS: List[str] = [
    "parametro",
    "valor",
    "fitness_medio",
    "fitness_std",
    "tiempo_medio_seg",
    "factible",
    "es_valor_usado",
]

#: Columnas del DataFrame de sensibilidad, en orden.
COLUMNAS_SENSIBILIDAD: List[str] = [
    "w1",
    "w_M",
    "w_P",
    "w_E",
    "w_G",
    "fitness_optimo",
    "n_seleccionadas",
    "costo",
    "r_medio",
    "similitud_vs_usado",
    "factible",
    "es_valor_usado",
]


def _jaccard(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Índice de Jaccard entre dos portafolios binarios.

    Args:
        x1: Primer cromosoma binario.
        x2: Segundo cromosoma binario, de la misma longitud.

    Returns:
        |x1 ∩ x2| / |x1 ∪ x2| en [0, 1]; 1.0 si ambos están vacíos.
    """
    a = np.asarray(x1, dtype=bool)
    b = np.asarray(x2, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / union


def redistribuir_pesos(w1: float, pesos_originales: Dict[str, float]) -> Dict[str, float]:
    """
    Reparte (1 − w₁) entre M, P, E y G conservando sus proporciones originales.

    Args:
        w1: Nuevo peso del riesgo Rᵢ, en [0, 1].
        pesos_originales: Pesos de referencia por atributo (claves R, M, P, E, G).

    Returns:
        Diccionario {atributo: peso} cuyos valores suman 1.0.

    Raises:
        ValueError: Si w1 cae fuera de [0, 1], o si los pesos originales de
            M, P, E y G suman 0 (no habría con qué prorratear).

    Referencia: README.md, formulacion del MCKP.
    """
    if not 0.0 <= w1 <= 1.0:
        raise ValueError(f"w1 debe estar en [0, 1], se recibio {w1}")

    resto_original = np.array(
        [float(pesos_originales[a]) for a in ATRIBUTOS_RESTO], dtype=float
    )
    total_resto = float(resto_original.sum())
    if total_resto <= 0.0:
        raise ValueError(
            "Los pesos originales de M, P, E y G suman 0: no se puede redistribuir "
            f"(1 - w1) proporcionalmente. Pesos recibidos: {pesos_originales}"
        )

    proporciones = resto_original / total_resto
    nuevos = proporciones * (1.0 - float(w1))
    pesos = {"R": float(w1)}
    pesos.update({a: float(v) for a, v in zip(ATRIBUTOS_RESTO, nuevos)})
    return pesos


def sensibilidad_w1(
    df: pd.DataFrame,
    params: Optional[Params] = None,
    w1_range: Optional[Sequence[float]] = None,
    pop_size: int = POP_SENSIBILIDAD,
    n_gen: int = GEN_SENSIBILIDAD,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Barre w₁ y registra el óptimo que alcanza el AG con cada valor.

    Para cada w₁ del rango:
      1. Se redistribuye (1 − w₁) entre M, P, E y G proporcionalmente.
      2. Se crea un `Problema` nuevo sobre el mismo dataset y se le fijan esos
         pesos, lo que recalcula el beneficio unitario y el ratio b/C.
      3. Se ejecuta el AG con población y generaciones reducidas (80 / 150) y se
         registra el fitness de la mejor solución.

    Todas las corridas usan la **misma semilla**, para que las diferencias entre
    valores de w₁ no queden confundidas con la varianza estocástica del AG.

    Los pesos originales de referencia se leen de `params` en el momento de la
    llamada, no se hard-codean: si el `.env` cambia, el prorrateo cambia con él.

    Args:
        df: Dataset de obras sin normalizar (`data/generator.py` o `data/loader.py`).
        params: Parámetros del proyecto; None usa `Params.get()`.
        w1_range: Valores de w₁ a evaluar; None usa `np.arange(0.10, 0.76, 0.05)`.
        pop_size: Población del AG en cada corrida del barrido.
        n_gen: Generaciones del AG en cada corrida del barrido.
        seed: Semilla común a todas las corridas; None usa `params.SEED`.

    Returns:
        DataFrame con una fila por valor de w₁ y las columnas de
        `COLUMNAS_SENSIBILIDAD`: w1, los cuatro pesos redistribuidos (w_M, w_P,
        w_E, w_G), fitness_optimo, n_seleccionadas, costo, factible y
        `es_valor_usado`, que marca con True la fila cuyo w₁ coincide con el peso
        del riesgo configurado en el trabajo (0.40 por defecto).

    Raises:
        ValueError: Si el rango de w₁ está vacío, o si pop_size/n_gen no son
            positivos.

    Referencia: README.md, formulacion del MCKP (pesos de la funcion objetivo).
    """
    parametros = params if params is not None else Params.get()
    valores = np.asarray(
        W1_RANGE_DEFECTO if w1_range is None else w1_range, dtype=float
    ).ravel()

    if valores.size == 0:
        raise ValueError("w1_range esta vacio: no hay valores de w1 que evaluar")
    if pop_size < 1 or n_gen < 0:
        raise ValueError(
            f"pop_size debe ser >= 1 y n_gen >= 0, se recibio ({pop_size}, {n_gen})"
        )

    pesos_originales = dict(parametros.W)
    w1_usado = float(pesos_originales["R"])
    semilla = int(parametros.SEED if seed is None else seed)

    logger.info(
        "Sensibilidad de w1 | %d valores en [%.2f, %.2f] | AG pop=%d gen=%d seed=%d | "
        "w1 del trabajo=%.2f",
        valores.size,
        float(valores.min()),
        float(valores.max()),
        pop_size,
        n_gen,
        semilla,
        w1_usado,
    )

    filas: List[Dict[str, Any]] = []
    cromosomas: List[np.ndarray] = []
    for w1 in valores:
        pesos = redistribuir_pesos(float(w1), pesos_originales)

        problema = Problema(df, parametros)
        problema.w = pesos

        # Los overrides van en la instancia del AG, no en el Singleton params,
        # para no alterar la configuracion global del proceso.
        algoritmo = AG(problema, parametros, seed=semilla)
        algoritmo.pop_size = int(pop_size)
        algoritmo.n_gen = int(n_gen)
        solucion = algoritmo.evolucionar()

        cromosomas.append(solucion.x)
        filas.append(
            {
                "w1": float(w1),
                "w_M": pesos["M"],
                "w_P": pesos["P"],
                "w_E": pesos["E"],
                "w_G": pesos["G"],
                "fitness_optimo": float(solucion.fitness),
                "n_seleccionadas": int(solucion.n_seleccionadas),
                "costo": float(solucion.costo),
                "r_medio": float(solucion.r_medio),
                "similitud_vs_usado": float("nan"),
                "factible": bool(solucion.factible),
                "es_valor_usado": bool(abs(float(w1) - w1_usado) <= TOLERANCIA_W1),
            }
        )
        logger.info(
            "  w1=%.2f | pesos M/P/E/G=%.4f/%.4f/%.4f/%.4f | fitness=%.6f | factible=%s",
            w1,
            pesos["M"],
            pesos["P"],
            pesos["E"],
            pesos["G"],
            solucion.fitness,
            solucion.factible,
        )

    resultado = pd.DataFrame(filas)

    # Estabilidad del portafolio frente al w1 usado en el trabajo: es lo que
    # responde si la DECISION depende de w1 (ver la nota sobre fitness_optimo).
    marcadas = np.flatnonzero(resultado["es_valor_usado"].to_numpy())
    if marcadas.size:
        referencia = cromosomas[int(marcadas[0])]
        resultado["similitud_vs_usado"] = [_jaccard(x, referencia) for x in cromosomas]
    else:
        logger.warning(
            "El w1 del trabajo (%.2f) no esta en el rango evaluado: ninguna fila queda "
            "marcada como es_valor_usado y similitud_vs_usado queda en NaN",
            w1_usado,
        )

    resultado = resultado[COLUMNAS_SENSIBILIDAD]

    mejor = resultado.loc[resultado["fitness_optimo"].idxmax()]
    logger.info(
        "Sensibilidad completada | fitness en [%.6f, %.6f] | maximo en w1=%.2f | "
        "factibles=%d/%d",
        resultado["fitness_optimo"].min(),
        resultado["fitness_optimo"].max(),
        float(mejor["w1"]),
        int(resultado["factible"].sum()),
        len(resultado),
    )
    return resultado


def resumen_sensibilidad(df_sens: pd.DataFrame) -> Dict[str, Any]:
    """
    Resume el barrido para `GET /analysis/sensitivity` y `metricas_ext.json`.

    Args:
        df_sens: DataFrame devuelto por `sensibilidad_w1`.

    Returns:
        Diccionario con las claves de `SensitivityResponse` (w1_rango,
        fitness_optimo, w1_usado, fitness_en_w1_usado) más el w₁ que maximiza el
        fitness y el rango de variación observado.

    Raises:
        ValueError: Si el DataFrame está vacío.

    Referencia: docs/API_SPEC.md — GET /analysis/sensitivity.
    """
    if df_sens is None or len(df_sens) == 0:
        raise ValueError("df_sens esta vacio: no hay barrido que resumir")

    usados = df_sens.loc[df_sens["es_valor_usado"]]
    fila_maxima = df_sens.loc[df_sens["fitness_optimo"].idxmax()]
    fitness = df_sens["fitness_optimo"]

    return {
        "w1_rango": [round(float(v), 4) for v in df_sens["w1"]],
        "fitness_optimo": [round(float(v), 6) for v in fitness],
        # Redondeado: np.arange produce 0.40000000000000013 y ese ruido de coma
        # flotante se filtraba tal cual a la respuesta de la API.
        "w1_usado": round(float(usados["w1"].iloc[0]), 4) if len(usados) else None,
        "fitness_en_w1_usado": float(usados["fitness_optimo"].iloc[0])
        if len(usados)
        else None,
        "w1_optimo": float(fila_maxima["w1"]),
        "fitness_maximo": float(fila_maxima["fitness_optimo"]),
        "variacion_absoluta": float(fitness.max() - fitness.min()),
        "variacion_relativa_pct": (
            100.0 * (fitness.max() - fitness.min()) / abs(fitness.mean())
            if fitness.mean() != 0
            else float("nan")
        ),
        "n_valores": int(len(df_sens)),
        "similitud_min": float(df_sens["similitud_vs_usado"].min()),
        "similitud_media": float(df_sens["similitud_vs_usado"].mean()),
        "r_medio_rango": [
            round(float(df_sens["r_medio"].min()), 4),
            round(float(df_sens["r_medio"].max()), 4),
        ],
        "advertencia": (
            "fitness_optimo NO es comparable entre valores de w1: cada w1 define una "
            "funcion objetivo distinta, asi que su crecimiento monotono no indica un w1 "
            "optimo. La robustez de la decision se lee en similitud_vs_usado (Jaccard "
            "del portafolio frente a w1=0.40) y en r_medio"
        ),
    }


# =========================================================================
# Sensibilidad de los parámetros del algoritmo
# =========================================================================


def sensibilidad_parametros(
    problema: Problema,
    params: Optional[Params] = None,
    seeds: Sequence[int] = SEEDS_PARAMETROS,
    grid: Optional[Dict[str, Sequence[float]]] = None,
    n_gen: int = N_GEN_PARAMETROS,
) -> pd.DataFrame:
    """
    Barre los parámetros del AG de uno en uno y mide calidad y costo.

    A diferencia de `sensibilidad_w1`, aquí **la función objetivo no cambia**: se
    varía la configuración del algoritmo, no el problema. Por eso los fitness SÍ
    son comparables entre filas y tiene sentido hablar de un mejor valor.

    Barrido *one-factor-at-a-time*: al variar un parámetro, los otros dos quedan en
    su valor por defecto (`params.POP_SIZE`, `params.PC`, `params.PM`). No explora
    interacciones entre parámetros; para eso haría falta un diseño factorial, que
    multiplicaría las corridas.

    `n_gen` se mantiene FIJO en todo el barrido. Es lo que hace comparables los
    tiempos: si variara, no se sabría si una configuración es más lenta por su
    población o por haber corrido más generaciones. Como consecuencia, el costo
    total de cada configuración crece de forma aproximadamente lineal con
    `pop_size` (evaluaciones = pop_size × n_gen).

    Cada valor se evalúa con todas las semillas de `seeds` y se reporta la media y
    la desviación, porque una sola corrida de un algoritmo estocástico no permite
    comparar configuraciones.

    Los overrides se aplican en la INSTANCIA del `AG`, nunca en el Singleton
    `Params`: el barrido no altera la configuración global del proceso.

    Args:
        problema: Problema MCKP sobre el que se evalúan las configuraciones.
        params: Parámetros del proyecto; None usa `Params.get()`. De aquí salen los
            valores por defecto de los parámetros que no se están barriendo.
        seeds: Semillas con las que se repite cada configuración.
        grid: Valores a barrer por parámetro; None usa `GRID_PARAMETROS`.
        n_gen: Generaciones, iguales para todas las configuraciones.

    Returns:
        DataFrame con una fila por (parámetro, valor) y las columnas de
        `COLUMNAS_PARAMETROS`: parametro, valor, fitness_medio, fitness_std,
        tiempo_medio_seg, factible (True si TODAS las semillas dieron solución
        factible) y es_valor_usado (marca el valor configurado en el trabajo).

    Raises:
        ValueError: Si `seeds` está vacío, si n_gen < 1, o si el grid contiene un
            parámetro que el AG no expone.

    Referencia: Tema 3 del curso CE UNI 2026 (parametros del AG canonico).
    """
    parametros = params if params is not None else Params.get()
    rejilla = GRID_PARAMETROS if grid is None else grid
    semillas = list(seeds)

    if not semillas:
        raise ValueError("seeds esta vacio: hacen falta semillas para promediar")
    if n_gen < 1:
        raise ValueError(f"n_gen debe ser >= 1, se recibio {n_gen}")
    desconocidos = set(rejilla) - set(ATRIBUTO_PARAMS)
    if desconocidos:
        raise ValueError(
            f"Parametros no soportados: {sorted(desconocidos)}. "
            f"Admitidos: {sorted(ATRIBUTO_PARAMS)}"
        )

    total_corridas = sum(len(valores) for valores in rejilla.values()) * len(semillas)
    logger.info(
        "Sensibilidad de parametros del AG | %d configuraciones x %d semillas = "
        "%d corridas | n_gen fijo=%d",
        sum(len(v) for v in rejilla.values()),
        len(semillas),
        total_corridas,
        n_gen,
    )

    filas: List[Dict[str, Any]] = []
    for parametro, valores in rejilla.items():
        usado = getattr(parametros, ATRIBUTO_PARAMS[parametro])
        for valor in valores:
            fitness: List[float] = []
            tiempos: List[float] = []
            factibles: List[bool] = []

            for semilla in semillas:
                algoritmo = AG(problema, parametros, seed=int(semilla))
                algoritmo.n_gen = int(n_gen)
                _aplicar_override(algoritmo, parametro, valor)
                solucion = algoritmo.ejecutar()  # cronometra con time.perf_counter

                fitness.append(float(solucion.fitness))
                tiempos.append(float(algoritmo.tiempo_seg))
                factibles.append(bool(solucion.factible))

            valores_fitness = np.asarray(fitness, dtype=float)
            filas.append(
                {
                    "parametro": parametro,
                    "valor": float(valor),
                    "fitness_medio": float(valores_fitness.mean()),
                    "fitness_std": float(valores_fitness.std(ddof=1))
                    if len(valores_fitness) > 1
                    else 0.0,
                    "tiempo_medio_seg": float(np.mean(tiempos)),
                    "factible": bool(all(factibles)),
                    "es_valor_usado": bool(abs(float(valor) - float(usado)) <= TOLERANCIA_W1),
                }
            )
            logger.info(
                "  %-8s = %-6s | fitness=%.6f +- %.6f | %.2f s/corrida | factibles=%d/%d",
                parametro,
                valor,
                filas[-1]["fitness_medio"],
                filas[-1]["fitness_std"],
                filas[-1]["tiempo_medio_seg"],
                sum(factibles),
                len(factibles),
            )

    resultado = pd.DataFrame(filas)[COLUMNAS_PARAMETROS]

    for parametro in rejilla:
        if not resultado.loc[resultado["parametro"] == parametro, "es_valor_usado"].any():
            logger.warning(
                "El valor de '%s' usado en el trabajo (%s) no esta en el grid barrido",
                parametro,
                getattr(parametros, ATRIBUTO_PARAMS[parametro]),
            )

    logger.info(
        "Sensibilidad de parametros completada | %d filas | tiempo total de corridas=%.1f s",
        len(resultado),
        float(resultado["tiempo_medio_seg"].sum() * len(semillas)),
    )
    return resultado


def _aplicar_override(algoritmo: AG, parametro: str, valor: float) -> None:
    """
    Fija un parámetro en la instancia del AG, sin tocar el Singleton `Params`.

    Args:
        algoritmo: Instancia del AG a configurar.
        parametro: Nombre del parámetro (`pop_size`, `pc` o `pm`).
        valor: Valor a fijar.

    Raises:
        ValueError: Si el parámetro no es uno de los soportados.
    """
    if parametro == "pop_size":
        algoritmo.pop_size = int(valor)
    elif parametro == "pc":
        algoritmo.pc = float(valor)
    elif parametro == "pm":
        algoritmo.pm = float(valor)
    else:
        raise ValueError(f"Parametro no soportado: {parametro}")


def resumen_sensibilidad_parametros(df_params: pd.DataFrame) -> Dict[str, Any]:
    """
    Resume el barrido de parámetros para `metricas_ext.json` y el informe.

    Por cada parámetro compara el mejor valor encontrado contra el que usa el
    trabajo, en fitness y en tiempo. La lectura relevante no es solo qué valor
    maximiza el fitness, sino a qué costo: subir la población mejora el fitness de
    forma marginal y encarece la corrida de forma aproximadamente lineal.

    Args:
        df_params: DataFrame devuelto por `sensibilidad_parametros`.

    Returns:
        Diccionario con el detalle por parámetro (mejor valor, valor usado, deltas
        de fitness y tiempo, factor de tiempo) y una nota metodológica.

    Raises:
        ValueError: Si el DataFrame está vacío o le faltan columnas.
    """
    if df_params is None or len(df_params) == 0:
        raise ValueError("df_params esta vacio: no hay barrido que resumir")
    faltantes = [c for c in COLUMNAS_PARAMETROS if c not in df_params.columns]
    if faltantes:
        raise ValueError(f"df_params no tiene las columnas requeridas: {faltantes}")

    por_parametro: Dict[str, Any] = {}
    for parametro, grupo in df_params.groupby("parametro"):
        mejor = grupo.loc[grupo["fitness_medio"].idxmax()]
        usadas = grupo.loc[grupo["es_valor_usado"]]
        usado = usadas.iloc[0] if len(usadas) else None

        detalle: Dict[str, Any] = {
            "valores_evaluados": [float(v) for v in grupo["valor"]],
            "mejor_valor": float(mejor["valor"]),
            "mejor_fitness": float(mejor["fitness_medio"]),
            "mejor_fitness_std": float(mejor["fitness_std"]),
            "mejor_tiempo_seg": float(mejor["tiempo_medio_seg"]),
            "rango_fitness": float(grupo["fitness_medio"].max() - grupo["fitness_medio"].min()),
            "todas_factibles": bool(grupo["factible"].all()),
        }
        if usado is not None:
            detalle.update(
                {
                    "valor_usado": float(usado["valor"]),
                    "fitness_usado": float(usado["fitness_medio"]),
                    "fitness_usado_std": float(usado["fitness_std"]),
                    "tiempo_usado_seg": float(usado["tiempo_medio_seg"]),
                    "delta_fitness_vs_usado": float(
                        mejor["fitness_medio"] - usado["fitness_medio"]
                    ),
                    "delta_fitness_vs_usado_pct": (
                        100.0
                        * (mejor["fitness_medio"] - usado["fitness_medio"])
                        / abs(usado["fitness_medio"])
                        if usado["fitness_medio"] != 0
                        else float("nan")
                    ),
                    "factor_tiempo_vs_usado": (
                        float(mejor["tiempo_medio_seg"] / usado["tiempo_medio_seg"])
                        if usado["tiempo_medio_seg"] > 0
                        else float("nan")
                    ),
                    "el_usado_es_el_mejor": bool(
                        float(mejor["valor"]) == float(usado["valor"])
                    ),
                }
            )
        else:
            detalle["valor_usado"] = None
        por_parametro[str(parametro)] = detalle

    return {
        "n_configuraciones": int(len(df_params)),
        "n_gen_fijo": N_GEN_PARAMETROS,
        "por_parametro": por_parametro,
        "nota_metodologica": (
            "Barrido one-factor-at-a-time: al variar un parametro los otros quedan en "
            "su valor por defecto, asi que no se exploran interacciones. n_gen es fijo "
            "para que los tiempos sean comparables; en consecuencia el costo crece de "
            "forma aproximadamente lineal con pop_size (evaluaciones = pop_size * n_gen). "
            "A diferencia del barrido de w1, aqui la funcion objetivo no cambia, asi que "
            "los fitness SI son comparables entre filas."
        ),
    }
