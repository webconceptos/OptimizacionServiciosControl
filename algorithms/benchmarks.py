"""
Métodos de referencia para contrastar el AG.

Dos líneas base con roles distintos en la comparación del informe:

* `greedy`: heurística **determinística** informada por el ratio beneficio/costo.
  Es el rival serio del AG y el término de comparación de la prueba de Wilcoxon,
  precisamente porque no tiene varianza: da el mismo resultado siempre, así que
  la hipótesis nula "el AG no supera al Greedy" queda limpia
  (docs/ARCHITECTURE.md, "¿Por qué comparar contra Greedy?").
* `aleatorio`: búsqueda aleatoria con reparación naive. Es el piso de referencia
  que cuantifica cuánto del resultado viene de la estructura del problema y
  cuánto del proceso evolutivo.

Ninguna de las dos es un algoritmo evolutivo, así que no heredan de
`AlgoritmoEvolutivo`: son funciones que devuelven una `Solucion`.

Referencia: README.md, metodos de comparacion (Greedy y Aleatorio)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

from core.operators import reparar_greedy
from core.problem import Problema
from core.solution import Solucion

logger = logging.getLogger(__name__)

#: Semilla por defecto del benchmark aleatorio (independiente de la del AG).
SEED_ALEATORIO: int = 99

#: Intentos por defecto de la búsqueda aleatoria.
N_INTENTOS_DEFECTO: int = 10_000

#: Estrategias de reparación admitidas por `aleatorio`.
Reparacion = Literal["aleatoria", "greedy"]


def greedy(problema: Problema) -> Solucion:
    """
    Heurística determinística: cobertura territorial primero, luego ratio b/C.

    Procedimiento en dos fases:

      1. **Cobertura territorial (R3)**: para cada macroregión r, se activan las
         mᵣ obras de mayor ratio bᵢ/Cᵢ de esa región, siempre que no se rompa el
         presupuesto B ni la capacidad K.
      2. **Relleno (R1, R2)**: se recorren todas las obras restantes en orden
         descendente de ratio y se activan las que caben en el presupuesto, hasta
         alcanzar K obras.

    El orden importa: si se rellenara primero por ratio, las obras más eficientes
    se concentrarían en pocas regiones y R3 quedaría sin cubrir. Atender R3
    primero garantiza factibilidad territorial a costa de algo de eficiencia.

    Es determinística: no consume aleatoriedad y con el mismo dataset devuelve
    siempre el mismo portafolio.

    Args:
        problema: Problema MCKP a resolver.

    Returns:
        La `Solucion` construida. Puede ser infactible si el propio problema no
        admite solución (por ejemplo, si cubrir Σmᵣ obras ya excede B); en ese
        caso se registra una advertencia.

    Referencia: README.md, metodos de comparacion (Greedy).
    """
    ratio = np.asarray(problema.ratio(), dtype=float)
    orden = np.argsort(ratio)[::-1]  # ratio descendente

    x = np.zeros(problema.n, dtype=np.int8)
    costo = 0.0
    seleccionadas = 0

    # --- Fase 1: minimo territorial por macroregion (R3) --------------------
    for region, minimo in problema.m_r.items():
        cubiertas = 0
        for indice in orden:
            if cubiertas >= minimo or seleccionadas >= problema.K:
                break
            if problema.mr[indice] != region or x[indice] == 1:
                continue
            if costo + problema.C[indice] > problema.B:
                continue
            x[indice] = 1
            costo += float(problema.C[indice])
            seleccionadas += 1
            cubiertas += 1
        if cubiertas < minimo:
            logger.warning(
                "Greedy no pudo cubrir el minimo de la macroregion %d: %d de %d obras",
                region,
                cubiertas,
                minimo,
            )

    # --- Fase 2: relleno por ratio dentro de B y K (R1, R2) -----------------
    for indice in orden:
        if seleccionadas >= problema.K:
            break
        if x[indice] == 1 or costo + problema.C[indice] > problema.B:
            continue
        x[indice] = 1
        costo += float(problema.C[indice])
        seleccionadas += 1

    solucion = Solucion(x, problema)
    if not solucion.factible:
        logger.warning(
            "Greedy devolvio una solucion infactible (violacion=%.4f): revisar si el "
            "problema admite solucion con K=%d, B=%.4f y m_r=%s",
            solucion.violacion,
            problema.K,
            problema.B,
            problema.m_r,
        )
    logger.info(
        "Greedy | fitness=%.6f | obras=%d | costo=%.4f | factible=%s",
        solucion.fitness,
        solucion.n_seleccionadas,
        solucion.costo,
        solucion.factible,
    )
    return solucion


def _reparar_aleatorio(
    x: np.ndarray, problema: Problema, rng: np.random.Generator
) -> np.ndarray:
    """
    Reparación naive: desactiva obras al azar hasta cumplir R2 y luego R1.

    Es la reparación deliberadamente **desinformada** del benchmark aleatorio:
    quita obras sin mirar su ratio beneficio/costo. Equivale a la del prototipo
    (quitar una obra activa al azar en un bucle), pero vectorizada: extraer k
    elementos al azar sin reemplazo es equivalente en distribución a quitar uno
    al azar k veces, y permutar los activos y quitarlos en ese orden hasta entrar
    en presupuesto equivale al bucle de R1.

    Args:
        x: Cromosoma binario de forma (problema.n,).
        problema: Problema MCKP; aporta C, B y K.
        rng: Generador aleatorio explícito.

    Returns:
        Cromosoma reparado nuevo que cumple R1 y R2. R3 no se repara.
    """
    reparado = np.asarray(x, dtype=np.int8).copy()
    activos = np.flatnonzero(reparado == 1)
    if activos.size == 0:
        return reparado

    # --- R2: quitar al azar el exceso de obras ------------------------------
    exceso = activos.size - problema.K
    if exceso > 0:
        fuera = rng.choice(activos, size=exceso, replace=False)
        reparado[fuera] = 0
        activos = np.flatnonzero(reparado == 1)

    # --- R1: quitar al azar hasta entrar en el presupuesto -------------------
    costo = float(problema.C[activos].sum())
    if costo > problema.B:
        permutados = rng.permutation(activos)
        restante = costo - np.cumsum(problema.C[permutados])
        dentro = np.flatnonzero(restante <= problema.B)
        a_quitar = int(dentro[0]) + 1 if dentro.size else permutados.size
        reparado[permutados[:a_quitar]] = 0

    return reparado


def aleatorio(
    problema: Problema,
    n_intentos: int = N_INTENTOS_DEFECTO,
    seed: int = SEED_ALEATORIO,
    reparacion: Reparacion = "aleatoria",
) -> Solucion:
    """
    Búsqueda aleatoria: mejor de `n_intentos` cromosomas reparados.

    Cada intento sortea un cromosoma uniforme en {0,1}ⁿ (≈50 % de bits activos),
    lo repara para cumplir R1 y R2, y se queda con el de mejor fitness. La
    reparación no toca R3, cuya violación queda penalizada por el fitness.

    Args:
        problema: Problema MCKP a resolver.
        n_intentos: Número de cromosomas a sortear.
        seed: Semilla del generador; por defecto 99, distinta de la del AG para
            que las dos búsquedas no compartan flujo aleatorio.
        reparacion: `"aleatoria"` (por defecto) usa la reparación desinformada,
            que es la que define este benchmark como línea base honesta;
            `"greedy"` usa `reparar_greedy` de `core/operators.py`, que quita las
            obras de peor ratio. La segunda opción existe para medir cuánta parte
            de la ventaja del AG proviene del operador de reparación y cuánta del
            proceso evolutivo, pero NO debe usarse como el benchmark aleatorio del
            informe: inyecta la heurística greedy en lo que debería ser azar.

    Returns:
        La mejor `Solucion` encontrada.

    Raises:
        ValueError: Si n_intentos < 1.

    Referencia: README.md, metodos de comparacion (Aleatorio).
    """
    if n_intentos < 1:
        raise ValueError(f"n_intentos debe ser >= 1, se recibio {n_intentos}")

    rng = np.random.default_rng(seed)
    mejor_x: Optional[np.ndarray] = None
    mejor_fitness = -np.inf
    n_factibles = 0

    for _ in range(n_intentos):
        candidato = (rng.random(problema.n) < 0.5).astype(np.int8)
        if reparacion == "greedy":
            candidato = reparar_greedy(candidato, problema)
        else:
            candidato = _reparar_aleatorio(candidato, problema, rng)

        fitness = problema.fitness(candidato)
        if problema.es_factible(candidato):
            n_factibles += 1
        if fitness > mejor_fitness:
            mejor_fitness = fitness
            mejor_x = candidato

    solucion = Solucion(mejor_x, problema)  # type: ignore[arg-type]
    logger.info(
        "Aleatorio | intentos=%d (reparacion=%s) | fitness=%.6f | obras=%d | "
        "costo=%.4f | factible=%s | factibles=%d/%d",
        n_intentos,
        reparacion,
        solucion.fitness,
        solucion.n_seleccionadas,
        solucion.costo,
        solucion.factible,
        n_factibles,
        n_intentos,
    )
    return solucion
