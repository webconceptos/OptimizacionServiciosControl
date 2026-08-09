"""
Fixtures compartidos de la suite de tests.

Dos escalas de problema, por una razón medida:

* `problema_small` (10 obras) sirve para probar aritmética: fitness, violaciones,
  normalización. **No puede ser factible por construcción**: R3 exige
  MIN_POR_REGION × 5 = 25 obras y solo hay 10, con macroregiones de 1 a 4 obras.
  Cualquier test de factibilidad sobre él fallaría por el planteamiento, no por un
  bug.
* `problema_mediano` (120 obras) es la escala mínima donde existen soluciones
  factibles y el AG las alcanza rápido (medido: factible en 10 generaciones con
  población 20, en 0.01 s). Es el fixture de los tests de algoritmos.
* `problema_grande` (326 obras) reproduce la escala del caso de estudio; se usa
  solo en el test de tiempo de NSGA-II.

`params_test` es una instancia **independiente** de `Params`: se construye con
`Params()` en lugar de `Params.get()`, así que no toca el Singleton del proceso
y ningún test puede contaminar a otro a través de la configuración global.

Referencia: docs/TECHNICAL_DEBT.md DT-004 (suite de tests) y DT-008 (semillas)
            los criterios de cobertura del proyecto
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from algorithms.benchmarks import greedy
from config.params import Params
from core.problem import Problema
from core.solution import Solucion
from data.generator import generar_obras

#: Obras del dataset pequeno (aritmetica; infactible por construccion).
N_SMALL: int = 10

#: Obras del dataset mediano (escala minima con soluciones factibles).
N_MEDIANO: int = 120

#: Obras del dataset grande (escala del caso de estudio).
N_GRANDE: int = 326

#: Semilla de los datasets de prueba.
SEED_TEST: int = 0


@pytest.fixture(autouse=True)
def silenciar_logs() -> Iterator[None]:
    """
    Baja el nivel de log a WARNING durante los tests.

    Los módulos registran mucho a nivel INFO (cada corrida, cada figura), lo que
    haría ilegible la salida de pytest. Se restaura al terminar.

    Yields:
        None, mientras el test corre.
    """
    raiz = logging.getLogger()
    nivel_previo = raiz.level
    raiz.setLevel(logging.WARNING)
    yield
    raiz.setLevel(nivel_previo)


@pytest.fixture(scope="session")
def df_small() -> pd.DataFrame:
    """
    Dataset de 10 obras, para tests de aritmética.

    Returns:
        DataFrame sin normalizar con 10 obras.
    """
    return generar_obras(n=N_SMALL, seed=SEED_TEST)


@pytest.fixture(scope="session")
def df_mediano() -> pd.DataFrame:
    """
    Dataset de 120 obras, la escala mínima con soluciones factibles.

    Returns:
        DataFrame sin normalizar con 120 obras.
    """
    return generar_obras(n=N_MEDIANO, seed=SEED_TEST)


@pytest.fixture(scope="session")
def df_grande() -> pd.DataFrame:
    """
    Dataset de 326 obras, la escala del caso de estudio.

    Returns:
        DataFrame sin normalizar con 326 obras.
    """
    return generar_obras(n=N_GRANDE, seed=SEED_TEST)


@pytest.fixture
def params_test() -> Params:
    """
    Parámetros reducidos para que los tests corran rápido.

    Se instancia con `Params()`, no con `Params.get()`: así el Singleton del
    proceso queda intacto y los tests no se contaminan entre sí.

    Returns:
        Instancia independiente con POP_SIZE=20, N_GEN=10 y N_RUNS=3.
    """
    params = Params()
    params.POP_SIZE = 20
    params.N_GEN = 10
    params.N_RUNS = 3
    return params


@pytest.fixture
def problema_small(df_small: pd.DataFrame, params_test: Params) -> Problema:
    """
    Problema de 10 obras. INFACTIBLE por construcción (ver el docstring del módulo).

    Args:
        df_small: Dataset de 10 obras.
        params_test: Parámetros de prueba.

    Returns:
        Instancia de `Problema`.
    """
    return Problema(df_small, params_test)


@pytest.fixture
def problema_mediano(df_mediano: pd.DataFrame, params_test: Params) -> Problema:
    """
    Problema de 120 obras, con soluciones factibles alcanzables.

    Args:
        df_mediano: Dataset de 120 obras.
        params_test: Parámetros de prueba.

    Returns:
        Instancia de `Problema`.
    """
    return Problema(df_mediano, params_test)


@pytest.fixture
def problema_grande(df_grande: pd.DataFrame, params_test: Params) -> Problema:
    """
    Problema de 326 obras, la escala del caso de estudio.

    Args:
        df_grande: Dataset de 326 obras.
        params_test: Parámetros de prueba.

    Returns:
        Instancia de `Problema`.
    """
    return Problema(df_grande, params_test)


@pytest.fixture
def solucion_factible(problema_mediano: Problema) -> Solucion:
    """
    Una solución factible construida de forma determinística con el Greedy.

    Args:
        problema_mediano: Problema de 120 obras.

    Returns:
        `Solucion` que cumple R1, R2 y R3.
    """
    solucion = greedy(problema_mediano)
    assert solucion.factible, (
        "El fixture solucion_factible depende de que el Greedy encuentre una "
        "solucion factible con 120 obras; si esto falla, revisar B, K y m_r"
    )
    return solucion


@pytest.fixture
def cromosoma_vacio(problema_mediano: Problema) -> np.ndarray:
    """
    Cromosoma sin ninguna obra seleccionada.

    Args:
        problema_mediano: Problema de 120 obras.

    Returns:
        Vector de ceros de dtype np.int8.
    """
    return np.zeros(problema_mediano.n, dtype=np.int8)
