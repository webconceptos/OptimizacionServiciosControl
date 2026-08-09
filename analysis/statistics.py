"""
Análisis estadístico de múltiples corridas del AG.

Un AG es estocástico: una sola corrida no dice nada sobre su desempeño. Este
módulo ejecuta N corridas independientes con semillas distintas y contrasta su
distribución de fitness contra el Greedy determinístico.

Por qué Wilcoxon signed-rank y no t-test (docs/ARCHITECTURE.md):

* No asume normalidad. Los fitness de un AG no tienen por qué ser normales:
  suelen estar truncados por arriba (cerca del óptimo) y con cola a la izquierda
  (corridas que caen en óptimos locales).
* Es robusto ante outliers, es decir, ante corridas que convergen mal.
* Es el estándar de facto en la comparación de algoritmos evolutivos
  (benchmarks IEEE CEC).

El contraste es pareado contra una constante: el Greedy es determinístico, así
que da el mismo valor siempre y la hipótesis nula queda limpia,
H₀ = "el AG no es mejor que el Greedy".

Referencia: README.md, resultados de referencia
            docs/ARCHITECTURE.md — "¿Por que Wilcoxon signed-rank y no t-test?"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from algorithms.genetic import AG
from config.params import Params
from core.problem import Problema
from core.solution import Solucion

logger = logging.getLogger(__name__)

#: Nivel de significancia por defecto del contraste de hipótesis.
ALPHA_SIGNIFICANCIA: float = 0.05

#: Columnas del DataFrame de corridas, en orden.
COLUMNAS_CORRIDAS: List[str] = [
    "corrida",
    "seed",
    "fitness",
    "n_seleccionadas",
    "costo",
    "r_medio",
    "factible",
    "tiempo_seg",
]

#: Columnas de la tabla comparativa de métodos, en orden.
COLUMNAS_COMPARACION: List[str] = [
    "metodo",
    "fitness",
    "n_seleccionadas",
    "costo",
    "factible",
    "mejora_vs_greedy_pct",
]

#: Nombre con el que se identifica al Greedy en la tabla comparativa.
ETIQUETA_GREEDY: str = "Greedy"


def multiples_corridas(
    problema: Problema,
    params: Optional[Params] = None,
    n_runs: int = 10,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Ejecuta N corridas independientes del AG y tabula sus resultados.

    La corrida i usa la semilla `seed_base + i`, de modo que las corridas son
    reproducibles individualmente y no comparten flujo aleatorio: cada `AG` crea
    su propio `np.random.Generator` (docs/TECHNICAL_DEBT.md, DT-008).

    Args:
        problema: Problema MCKP a resolver. Se reutiliza entre corridas: es
            inmutable respecto a la evaluación, así que no introduce acoplamiento.
        params: Parámetros del proyecto; None usa `Params.get()`. De aquí salen
            POP_SIZE, N_GEN y la semilla base.
        n_runs: Número de corridas independientes.
        seed: Semilla base; None usa `params.SEED`. Permite que la API atienda
            peticiones con semilla propia sin mutar el Singleton compartido.

    Returns:
        DataFrame con una fila por corrida y las columnas de `COLUMNAS_CORRIDAS`:
        corrida (base 1), seed, fitness, n_seleccionadas, costo, r_medio,
        factible y tiempo_seg.

    Raises:
        ValueError: Si n_runs < 1.

    Referencia: README.md, resultados de referencia.
    """
    if n_runs < 1:
        raise ValueError(f"n_runs debe ser >= 1, se recibio {n_runs}")

    parametros = params if params is not None else Params.get()
    semilla_base = int(parametros.SEED if seed is None else seed)

    logger.info(
        "Multiples corridas del AG | n_runs=%d | pop=%d gen=%d | semillas %d..%d",
        n_runs,
        int(parametros.POP_SIZE),
        int(parametros.N_GEN),
        semilla_base,
        semilla_base + n_runs - 1,
    )

    filas: List[Dict[str, Any]] = []
    for i in range(n_runs):
        semilla = semilla_base + i
        algoritmo = AG(problema, parametros, seed=semilla)
        solucion = algoritmo.ejecutar()
        filas.append(
            {
                "corrida": i + 1,
                "seed": semilla,
                "fitness": float(solucion.fitness),
                "n_seleccionadas": int(solucion.n_seleccionadas),
                "costo": float(solucion.costo),
                "r_medio": float(solucion.r_medio),
                "factible": bool(solucion.factible),
                "tiempo_seg": float(algoritmo.tiempo_seg),
            }
        )
        logger.info(
            "  Corrida %d/%d | seed=%d | fitness=%.6f | factible=%s | %.2f s",
            i + 1,
            n_runs,
            semilla,
            solucion.fitness,
            solucion.factible,
            algoritmo.tiempo_seg,
        )

    df = pd.DataFrame(filas)[COLUMNAS_CORRIDAS]
    logger.info(
        "Corridas completadas | media=%.6f std=%.6f | factibles=%d/%d | total %.2f s",
        df["fitness"].mean(),
        df["fitness"].std(ddof=1) if len(df) > 1 else 0.0,
        int(df["factible"].sum()),
        len(df),
        df["tiempo_seg"].sum(),
    )
    return df


def analisis_wilcoxon(
    df_runs: pd.DataFrame,
    f_greedy: float,
    alpha: float = ALPHA_SIGNIFICANCIA,
) -> Dict[str, Any]:
    """
    Contrasta la distribución de fitness del AG contra el Greedy.

    Aplica `scipy.stats.wilcoxon` sobre las diferencias `fitness_i − f_greedy`.
    El test es bilateral (el valor por defecto de scipy): con 10 corridas todas
    mejores que el Greedy, el p exacto es 2/2¹⁰ = 0.001953, que es el valor de
    referencia del proyecto.

    Casos degenerados que se manejan sin lanzar excepción:

    * **Todas las diferencias nulas** (el AG empata exactamente con el Greedy en
      cada corrida): no hay signos que rankear. Verificado con scipy 1.16: no
      lanza excepción, devuelve `p = NaN` con un RuntimeWarning de división
      inválida, que propagaría NaN a `metricas_ext.json` sin avisar. Se detecta
      antes y se devuelve `wilcoxon_p = 1.0` con `significativo = False`, que es
      la lectura correcta: sin diferencias no hay evidencia de diferencia.
    * **Una sola corrida**: el signed-rank no es aplicable con n=1; se omite el
      test y se devuelve `wilcoxon_p = NaN` con `significativo = False`.
    * **Cualquier otro fallo de scipy**: se captura, se registra y se informa en
      la clave `test_omitido`.

    Args:
        df_runs: DataFrame de `multiples_corridas`, con la columna `fitness`.
        f_greedy: Fitness del Greedy, el valor determinístico de comparación.
        alpha: Nivel de significancia; por defecto 0.05.

    Returns:
        Diccionario con: media, std, mediana, minimo, maximo, greedy,
        mejora_media_pct, wilcoxon_stat, wilcoxon_p, significativo, y además
        n_runs, n_runs_factibles, tiempo_total_seg, wilcoxon_p_unilateral,
        n_corridas_mejores y test_omitido (None si el test se aplicó).

    Raises:
        ValueError: Si `df_runs` está vacío o no tiene la columna `fitness`.

    Referencia: docs/ARCHITECTURE.md — eleccion de Wilcoxon sobre t-test.
    """
    if df_runs is None or len(df_runs) == 0:
        raise ValueError("df_runs esta vacio: no hay corridas que analizar")
    if "fitness" not in df_runs.columns:
        raise ValueError("df_runs no tiene la columna 'fitness'")

    fitness = df_runs["fitness"].to_numpy(dtype=float)
    n = int(fitness.size)
    greedy = float(f_greedy)

    media = float(fitness.mean())
    desviacion = float(fitness.std(ddof=1)) if n > 1 else 0.0
    mejora_pct = 100.0 * (media - greedy) / abs(greedy) if greedy != 0 else float("nan")

    diferencias = fitness - greedy
    estadistico: float = float("nan")
    p_valor: float = float("nan")
    p_unilateral: float = float("nan")
    test_omitido: Optional[str] = None

    if n < 2:
        test_omitido = (
            "El test de Wilcoxon signed-rank necesita al menos 2 corridas; "
            f"se recibio n={n}"
        )
    elif np.allclose(diferencias, 0.0):
        estadistico, p_valor, p_unilateral = 0.0, 1.0, 1.0
        test_omitido = (
            "Todas las diferencias respecto al Greedy son nulas: el signed-rank no "
            "puede rankear ceros, se reporta p=1.0 (no hay evidencia de diferencia)"
        )
    else:
        try:
            resultado = stats.wilcoxon(diferencias)
            estadistico = float(resultado.statistic)
            p_valor = float(resultado.pvalue)
            p_unilateral = float(stats.wilcoxon(diferencias, alternative="greater").pvalue)
        except (ValueError, ZeroDivisionError) as exc:
            test_omitido = f"scipy.stats.wilcoxon fallo ({type(exc).__name__}: {exc})"
            logger.warning("Test de Wilcoxon omitido: %s", test_omitido)

    significativo = bool(np.isfinite(p_valor) and p_valor < alpha and test_omitido is None)

    resumen: Dict[str, Any] = {
        "media": media,
        "std": desviacion,
        "mediana": float(np.median(fitness)),
        "minimo": float(fitness.min()),
        "maximo": float(fitness.max()),
        "greedy": greedy,
        "mejora_media_pct": mejora_pct,
        "wilcoxon_stat": estadistico,
        "wilcoxon_p": p_valor,
        "significativo": significativo,
        "n_runs": n,
        "n_runs_factibles": int(df_runs["factible"].sum())
        if "factible" in df_runs.columns
        else None,
        "tiempo_total_seg": float(df_runs["tiempo_seg"].sum())
        if "tiempo_seg" in df_runs.columns
        else None,
        "wilcoxon_p_unilateral": p_unilateral,
        "n_corridas_mejores": int((diferencias > 0).sum()),
        "alpha": float(alpha),
        "test_omitido": test_omitido,
    }

    logger.info(
        "Wilcoxon | n=%d | media=%.6f (std=%.6f) vs greedy=%.6f | mejora=%+.2f%% | "
        "stat=%.4f p=%.6f | significativo=%s | mejores=%d/%d",
        n,
        media,
        desviacion,
        greedy,
        mejora_pct,
        estadistico,
        p_valor,
        significativo,
        resumen["n_corridas_mejores"],
        n,
    )
    if test_omitido:
        logger.warning("  Nota: %s", test_omitido)
    return resumen


def tabla_comparacion(
    resultados: Mapping[str, Union[Solucion, Mapping[str, Any]]]
) -> pd.DataFrame:
    """
    Construye la tabla comparativa de métodos para el informe y la figura 2.

    Args:
        resultados: Diccionario {nombre del método: resultado}, donde el resultado
            puede ser una `Solucion` o un mapeo con las claves `fitness`,
            `n_seleccionadas` (o `n_sel`), `costo` y `factible`. Se admite el
            mapeo para poder tabular resultados ya serializados, por ejemplo
            leídos de `metricas.json`.

    Returns:
        DataFrame con las columnas de `COLUMNAS_COMPARACION`, en el mismo orden en
        que llegaron los métodos. La columna `mejora_vs_greedy_pct` se calcula
        respecto a la fila cuyo método se llame "Greedy"; si no hay ninguna, queda
        en NaN.

    Raises:
        ValueError: Si `resultados` está vacío o a un método le falta el fitness.

    Referencia: README.md, metodos de comparacion.
    """
    if not resultados:
        raise ValueError("resultados esta vacio: no hay metodos que comparar")

    filas: List[Dict[str, Any]] = []
    for nombre, resultado in resultados.items():
        filas.append({"metodo": nombre, **_extraer_metricas(nombre, resultado)})

    df = pd.DataFrame(filas)

    referencia = df.loc[df["metodo"] == ETIQUETA_GREEDY, "fitness"]
    if len(referencia) and float(referencia.iloc[0]) != 0.0:
        base = float(referencia.iloc[0])
        df["mejora_vs_greedy_pct"] = 100.0 * (df["fitness"] - base) / abs(base)
    else:
        df["mejora_vs_greedy_pct"] = np.nan

    df = df[COLUMNAS_COMPARACION]
    logger.info("Tabla comparativa | metodos: %s", ", ".join(df["metodo"].tolist()))
    return df


def _extraer_metricas(
    nombre: str, resultado: Union[Solucion, Mapping[str, Any]]
) -> Dict[str, Any]:
    """
    Normaliza las métricas de un método, venga como `Solucion` o como mapeo.

    Args:
        nombre: Nombre del método, usado en el mensaje de error.
        resultado: `Solucion` o mapeo con las métricas.

    Returns:
        Diccionario con fitness, n_seleccionadas, costo y factible.

    Raises:
        ValueError: Si el mapeo no trae el fitness.
        TypeError: Si el tipo del resultado no es soportado.
    """
    if isinstance(resultado, Solucion):
        return {
            "fitness": float(resultado.fitness),
            "n_seleccionadas": int(resultado.n_seleccionadas),
            "costo": float(resultado.costo),
            "factible": bool(resultado.factible),
        }

    if isinstance(resultado, Mapping):
        if "fitness" not in resultado:
            raise ValueError(f"El metodo '{nombre}' no trae la clave 'fitness'")
        n_sel = resultado.get("n_seleccionadas", resultado.get("n_sel"))
        costo = resultado.get("costo", resultado.get("costo_total"))
        return {
            "fitness": float(resultado["fitness"]),
            "n_seleccionadas": int(n_sel) if n_sel is not None else None,
            "costo": float(costo) if costo is not None else None,
            "factible": bool(resultado["factible"]) if "factible" in resultado else None,
        }

    raise TypeError(
        f"El metodo '{nombre}' tiene un resultado de tipo {type(resultado).__name__}; "
        "se esperaba Solucion o un mapeo con las metricas"
    )
