"""
Operadores evolutivos como funciones puras.

Los operadores genéticos son operaciones matemáticas que no dependen del estado
de ningún objeto: aquí son funciones puras, no métodos de clase. Eso permite
testearlos unitariamente y que AG (Tema 3) y NSGA-II (Tema 10) compartan
exactamente los mismos operadores de variación.

Dos reglas que se cumplen en todo el módulo:

* **Sin semillas globales.** Toda aleatoriedad entra por un
  `np.random.Generator` explícito. Nunca se usa `np.random.seed()` ni
  `random.seed()`, para que dos corridas con semillas distintas sean
  reproducibles de forma independiente (docs/TECHNICAL_DEBT.md, DT-008).
* **Sin efectos secundarios.** Ninguna función modifica sus argumentos in-place:
  todas devuelven arrays nuevos. El costo de copiar un cromosoma de 326 int8 es
  despreciable frente a la evaluación del fitness.

Referencia: Tema 3 y Tema 5 del curso CE UNI 2026
            docs/ARCHITECTURE.md — "3. Operadores como funciones puras"
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from core.problem import Problema

logger = logging.getLogger(__name__)

#: Rango del jitter multiplicativo que diversifica la población inicial.
JITTER_INICIAL: Tuple[float, float] = (0.4, 1.6)

#: Factor de densidad de la inicializacion heuristica del prototipo.
#: Escala la probabilidad de activar cada bit; ver `inicializar_heuristico`.
FACTOR_DENSIDAD: float = 0.45


def inicializar_heuristico(
    problema: Problema, pop_size: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Genera la población inicial sesgada por el ratio beneficio/costo.

    Cada bit i se activa con probabilidad proporcional al ratio normalizado
    rᵢ = (bᵢ/Cᵢ) / Σ(b/C), perturbado por un jitter multiplicativo U(0.4, 1.6)
    distinto en cada individuo y escalado por `FACTOR_DENSIDAD`. Así las obras
    con mejor relación beneficio/costo tienen más chance de entrar, y los
    individuos no salen todos iguales.

    Frente a una inicialización uniforme (que activaría ~50 % de los bits y
    violaría R2 masivamente con n=326 y K=50), esta produce individuos dispersos.
    ADVERTENCIA MEDIDA: como rᵢ suma 1 sobre las n obras, el número esperado de
    bits activos es Σ(rᵢ·FACTOR_DENSIDAD) ≈ 0.45, es decir, individuos casi
    vacíos. El AG llega a la zona factible añadiendo bits por mutación bajo
    presión selectiva, no partiendo de soluciones ya buenas. Es el comportamiento
    del prototipo que produjo las métricas de referencia del proyecto.

    Args:
        problema: Problema MCKP; aporta `ratio()` y la longitud del cromosoma.
        pop_size: Número de individuos a generar.
        rng: Generador aleatorio explícito (np.random.Generator).

    Returns:
        Matriz binaria de forma (pop_size, problema.n) y dtype np.int8.

    Raises:
        ValueError: Si pop_size < 1.

    Referencia: Tema 3 del curso CE UNI 2026 (inicialización de la población).
    """
    if pop_size < 1:
        raise ValueError(f"pop_size debe ser >= 1, se recibio {pop_size}")

    n = problema.n
    ratio = np.asarray(problema.ratio(), dtype=float)
    total = float(ratio.sum())
    if total <= 0.0:
        logger.warning("Los ratios beneficio/costo suman 0: se inicializa con ceros")
        return np.zeros((pop_size, n), dtype=np.int8)
    r = ratio / total

    jitter = rng.uniform(JITTER_INICIAL[0], JITTER_INICIAL[1], size=(pop_size, n))
    probabilidades = np.clip(r * jitter, 0.0, 1.0) * FACTOR_DENSIDAD
    pop = (rng.random((pop_size, n)) < probabilidades).astype(np.int8)

    logger.debug(
        "Poblacion inicial | pop_size=%d n=%d | bits activos: media=%.2f max=%d",
        pop_size,
        n,
        float(pop.sum(axis=1).mean()),
        int(pop.sum(axis=1).max()),
    )
    return pop


def torneo(
    pop: np.ndarray, fit: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Selecciona un padre por torneo determinístico de tamaño k.

    Se toman k individuos al azar SIN reemplazo y gana el de mayor fitness. k
    controla la presión selectiva: k=1 equivale a selección aleatoria y valores
    altos aceleran la convergencia a costa de diversidad (k=3 en este trabajo).

    Args:
        pop: Población de forma (pop_size, n).
        fit: Vector de fitness de longitud pop_size, alineado con `pop`.
        k: Tamaño del torneo; se recorta a pop_size si lo excede.
        rng: Generador aleatorio explícito.

    Returns:
        Copia del cromosoma ganador, de forma (n,).

    Raises:
        ValueError: Si la población está vacía, si k < 1, o si `fit` no está
            alineado con `pop`.

    Referencia: Tema 3 del curso CE UNI 2026 (seleccion por torneo, presion selectiva).
    """
    pop_size = len(pop)
    if pop_size == 0:
        raise ValueError("La poblacion esta vacia")
    if len(fit) != pop_size:
        raise ValueError(
            f"fit tiene {len(fit)} elementos y la poblacion {pop_size}: deben coincidir"
        )
    if k < 1:
        raise ValueError(f"k debe ser >= 1, se recibio {k}")

    k_efectivo = min(k, pop_size)
    candidatos = rng.choice(pop_size, size=k_efectivo, replace=False)
    ganador = candidatos[int(np.argmax(np.asarray(fit)[candidatos]))]
    return np.array(pop[ganador], dtype=np.int8, copy=True)


def cruce_spx(
    p1: np.ndarray, p2: np.ndarray, pc: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cruce de un punto (Single Point Crossover) entre dos padres.

    Con probabilidad pc se sortea un punto de corte en [1, n-1] y se
    intercambian los segmentos; en caso contrario los hijos son copias de los
    padres. El corte nunca cae en 0 ni en n, casos en que no habría intercambio.

    El SPX explota el linkage entre genes contiguos: cuanto más separados están
    los bits de un esquema, más probable es que el corte lo destruya, lo que se
    formaliza en el factor (1 − pc·δ(H)/(n−1)) del Teorema de Esquemas (Tema 4).

    Args:
        p1: Primer padre, de forma (n,).
        p2: Segundo padre, de la misma forma que p1.
        pc: Probabilidad de cruce en [0, 1].
        rng: Generador aleatorio explícito.

    Returns:
        Tupla (hijo1, hijo2), arrays nuevos de dtype np.int8. Los padres no se
        modifican.

    Raises:
        ValueError: Si los padres no tienen la misma forma 1-D, o si n < 2.

    Referencia: Tema 3 del curso CE UNI 2026 (cruce de un punto).
    """
    a = np.asarray(p1, dtype=np.int8)
    b = np.asarray(p2, dtype=np.int8)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(
            f"Los padres deben ser vectores de la misma longitud: {a.shape} vs {b.shape}"
        )
    n = a.size
    if n < 2:
        raise ValueError(f"El cruce de un punto requiere n >= 2, se recibio n={n}")

    if rng.random() >= pc:
        return a.copy(), b.copy()

    corte = int(rng.integers(1, n))  # punto de corte en [1, n-1]
    h1 = np.concatenate([a[:corte], b[corte:]])
    h2 = np.concatenate([b[:corte], a[corte:]])
    return h1, h2


def mutacion_bitflip(x: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
    """
    Mutación por inversión de bit, aplicada gen a gen.

    Cada bit se invierte de forma independiente con probabilidad pm, de modo que
    se esperan pm·n inversiones por cromosoma (con pm=0.015 y n=326, ~4.9 bits).
    Es la fuente de diversidad del AG: garantiza que ningún alelo se pierda de
    forma irreversible y, en este problema, es el mecanismo que permite AÑADIR
    obras al portafolio, ya que la reparación greedy solo puede quitarlas.

    Args:
        x: Cromosoma binario de forma (n,).
        pm: Probabilidad de mutación por bit, en [0, 1].
        rng: Generador aleatorio explícito.

    Returns:
        Cromosoma mutado nuevo, de dtype np.int8. El original no se modifica.

    Referencia: Tema 3 del curso CE UNI 2026 (mutacion bit-flip, diversidad).
    """
    xv = np.asarray(x, dtype=np.int8)
    mascara = rng.random(xv.size) < pm
    mutado = xv.copy()
    mutado[mascara] = 1 - mutado[mascara]
    return mutado


def reparar_greedy(x: np.ndarray, problema: Problema) -> np.ndarray:
    """
    Repara las violaciones de R2 y R1 quitando las obras de peor ratio b/C.

    Procedimiento, en este orden:
      1. **R2 (capacidad)**: si hay más de K obras, se conservan las K de mayor
         ratio beneficio/costo y se desactivan las demás.
      2. **R1 (presupuesto)**: mientras el costo exceda B, se van desactivando
         las obras activas de menor ratio.

    Reparar en vez de penalizar mantiene la búsqueda dentro de la región factible
    de R1 y R2 sin desperdiciar evaluaciones en soluciones inviables. La versión
    del prototipo quitaba una obra por iteración recalculando el argmin; aquí se
    ordena una sola vez por ratio, que es equivalente (el ratio no depende de x)
    y evita un bucle O(k·n).

    R3 (cobertura territorial) NO se repara: exigiría añadir obras, lo que puede
    reintroducir violaciones de R1 y R2 y sesgaría la búsqueda. Su cumplimiento
    se induce por la penalización de `Problema.fitness()`.

    Args:
        x: Cromosoma binario de forma (problema.n,).
        problema: Problema MCKP; aporta ratio(), C, B y K.

    Returns:
        Cromosoma reparado nuevo, de dtype np.int8, que cumple R1 y R2. El
        original no se modifica.

    Raises:
        ValueError: Si la longitud del cromosoma no coincide con problema.n.

    Referencia: Tema 5 del curso CE UNI 2026 (operador de reparacion).
    """
    xv = np.asarray(x, dtype=np.int8)
    if xv.shape != (problema.n,):
        raise ValueError(
            f"El cromosoma debe tener forma ({problema.n},), se recibio {xv.shape}"
        )

    reparado = xv.copy()
    ratio = np.asarray(problema.ratio(), dtype=float)
    activos = np.flatnonzero(reparado == 1)
    if activos.size == 0:
        return reparado

    # Activos ordenados de PEOR a MEJOR ratio: son los primeros candidatos a salir.
    orden_ascendente = activos[np.argsort(ratio[activos], kind="stable")]

    # --- R2: exceso de obras ------------------------------------------------
    exceso = activos.size - problema.K
    if exceso > 0:
        reparado[orden_ascendente[:exceso]] = 0
        orden_ascendente = orden_ascendente[exceso:]

    # --- R1: exceso de presupuesto ------------------------------------------
    costo = float(problema.C[orden_ascendente].sum())
    if costo > problema.B:
        costos_ordenados = problema.C[orden_ascendente]
        # Costo restante tras quitar los j peores: total - suma acumulada.
        restante = costo - np.cumsum(costos_ordenados)
        # Se quita el mínimo número de obras que deja el costo dentro de B.
        dentro = np.flatnonzero(restante <= problema.B)
        a_quitar = int(dentro[0]) + 1 if dentro.size else orden_ascendente.size
        reparado[orden_ascendente[:a_quitar]] = 0

    return reparado


def evaluar_poblacion(pop: np.ndarray, problema: Problema) -> np.ndarray:
    """
    Evalúa el fitness de todos los individuos de una población.

    Args:
        pop: Población de forma (pop_size, n).
        problema: Problema MCKP sobre el que se evalúa.

    Returns:
        Vector de fitness de longitud pop_size, alineado con `pop`.

    Referencia: Tema 2 del curso CE UNI 2026 (ciclo evolutivo, evaluacion).
    """
    return np.array([problema.fitness(individuo) for individuo in pop], dtype=float)
