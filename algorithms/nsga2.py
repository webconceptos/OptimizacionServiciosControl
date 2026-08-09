"""
NSGA-II (variante elitista de NSGA) - Tema 10 CE UNI 2026.

Optimización multiobjetivo del MCKP sin agregar los criterios en un único
escalar: en lugar de un portafolio óptimo, produce el **frente de Pareto** del
compromiso entre riesgo cubierto y costo de supervisión, para que el decisor
institucional elija el punto de operación.

Linaje del algoritmo
--------------------
El curso presenta el NSGA de Srinivas & Deb (1994), de primera generación, cuya
diversidad se mantiene con **fitness compartido** y un radio de nicho
`sigma_share` fijado a mano. NSGA-II (Deb, Pratap, Agarwal & Meyarivan, 2002) es
su **variante elitista**, y cambia tres cosas:

* **Elitismo (μ+λ)**: padres e hijos compiten juntos (población combinada 2N), de
  donde se seleccionan los N mejores. En NSGA no había elitismo.
* **Diversidad sin parámetros**: el `crowding distance` sustituye al fitness
  compartido y cumple el mismo rol de repartir la población a lo largo del
  frente, pero sin necesidad de calibrar `sigma_share`.
* **Ordenamiento no dominado rápido**: `_fast_nds` en O(M·N²) en vez de O(M·N³).

Objetivos (docs/ARCHITECTURE.md, "¿Por qué solo 2 objetivos?")
--------------------------------------------------------------
    f1 = Σᵢ R̃ᵢ·xᵢ    riesgo cubierto      -> MAXIMIZAR  (w1 = -1)
    f2 = Σᵢ Cᵢ·xᵢ    costo de supervisión -> MINIMIZAR  (w2 = +1)

Son los dos criterios más directamente conflictivos y los más relevantes para el
decisor. Con 3 o más objetivos el crowding distance pierde eficacia y el frente
deja de ser visualizable.

Dominancia (ec. 5 de la Clase 10 — MOGA, Túpac)
-----------------------------------------------
    x1 domina a x2  (x1 ≺ x2)  sii
        ∀i:  wᵢ·fᵢ(x1) ≤ wᵢ·fᵢ(x2)      y
        ∃j:  wⱼ·fⱼ(x1) < wⱼ·fⱼ(x2)

con wᵢ = +1 si el objetivo i se minimiza y wᵢ = −1 si se maximiza. Multiplicar
por wᵢ lleva ambos objetivos a un espacio de minimización común, de modo que la
misma comparación sirve para f1 (maximizar) y f2 (minimizar).

Operadores
----------
Los operadores de variación son EXACTAMENTE los del AG (`cruce_spx`,
`mutacion_bitflip`, `reparar_greedy` de `core/operators.py`): ambos algoritmos
operan sobre el mismo espacio binario y la única diferencia metodológica está en
la selección y el reemplazo (docs/ARCHITECTURE.md).

Referencia: Tema 10 del curso CE UNI 2026 — README.md, formulacion del MCKP y algoritmos
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from algorithms.base import AlgoritmoEvolutivo
from config.params import Params
from core.operators import (
    cruce_spx,
    evaluar_poblacion,
    inicializar_heuristico,
    mutacion_bitflip,
    reparar_greedy,
)
from core.problem import Problema
from core.solution import Solucion

logger = logging.getLogger(__name__)

#: Pesos wᵢ de la ec. 5: −1 maximiza, +1 minimiza. (f1 riesgo, f2 costo).
PESOS_OBJETIVOS: Tuple[float, float] = (-1.0, +1.0)

#: Nombres de los objetivos, alineados con `PESOS_OBJETIVOS` y con API_SPEC.md.
NOMBRES_OBJETIVOS: Tuple[str, str] = ("f1_riesgo", "f2_costo")

#: Número de participantes del torneo binario de NSGA-II.
K_TORNEO_PARETO: int = 2


class NSGA2(AlgoritmoEvolutivo):
    """
    NSGA-II (variante elitista de NSGA) - Tema 10 CE UNI 2026.

    Ciclo de una generación (Deb et al., 2002):

        1. Generar N hijos: torneo binario con comparación aglomerada
           -> cruce_spx(pc) -> mutacion_bitflip(pm) -> reparar_greedy
        2. Combinar padres e hijos: Rt = Pt ∪ Qt  (tamaño 2N)  <- ELITISMO
        3. Ordenar Rt en frentes no dominados con `_fast_nds`
        4. Llenar P(t+1) frente por frente; el frente que no cabe completo se
           trunca conservando los individuos de mayor `crowding distance`

    Manejo de restricciones: se usa el principio de **dominancia con
    restricciones** de Deb (2002), no la penalización escalar del AG:

        i domina a j  si  (i factible y j infactible), o
                          (ambos infactibles y violacion(i) < violacion(j)), o
                          (ambos factibles e i domina a j por la ec. 5)

    Es preferible a penalizar porque no mezcla escalas: la presión hacia la
    factibilidad no distorsiona el compromiso entre f1 y f2 dentro de la región
    factible, y garantiza que si existe al menos una solución factible, el primer
    frente esté compuesto solo por soluciones factibles.

    Atributos:
        pop_size: Tamaño de la población.
        n_gen: Número de generaciones.
        pc: Probabilidad de cruce.
        pm: Probabilidad de mutación por bit.
        frente: Frente de Pareto factible resultante (tras `evolucionar()`).

    Referencia: Tema 10 del curso CE UNI 2026.
    """

    nombre: str = "NSGA-II"
    tema: str = "Tema 10"

    def __init__(
        self,
        problema: Problema,
        params: Optional[Params] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Configura NSGA-II a partir de los parámetros centralizados.

        Args:
            problema: Problema MCKP a optimizar.
            params: Parámetros del proyecto; None usa `Params.get()`.
            seed: Semilla de esta corrida; None usa `params.SEED`.
        """
        super().__init__(problema, params, seed)

        self.pop_size: int = int(self.params.POP_SIZE)
        self.n_gen: int = int(self.params.N_GEN)
        self.pc: float = float(self.params.PC)
        self.pm: float = float(self.params.PM)

        self.frente: List[Solucion] = []

        # Vectores de los dos objetivos, precomputados una sola vez.
        self._r_norm: np.ndarray = self.problema.df["R_n"].to_numpy(dtype=float)
        self._pesos: np.ndarray = np.array(PESOS_OBJETIVOS, dtype=float)

        logger.info(
            "  Parametros NSGA-II | pop=%d gen=%d pc=%.2f pm=%.4f | f1=suma(R_n*x) max, "
            "f2=suma(C*x) min",
            self.pop_size,
            self.n_gen,
            self.pc,
            self.pm,
        )

    # ----------------------------------------------------------- Objetivos

    def _objetivos(self, pop: np.ndarray) -> np.ndarray:
        """
        Evalúa los dos objetivos de cada individuo.

        Args:
            pop: Población de forma (N, n).

        Returns:
            Matriz (N, 2) con [f1, f2] en su escala natural (sin aplicar wᵢ):
            f1 = Σ R̃ᵢ·xᵢ (a maximizar) y f2 = Σ Cᵢ·xᵢ (a minimizar).

        Referencia: Tema 10 del curso CE UNI 2026.
        """
        matriz = np.asarray(pop, dtype=float)
        f1 = matriz @ self._r_norm
        f2 = matriz @ self.problema.C
        return np.column_stack([f1, f2])

    def _violaciones(self, pop: np.ndarray) -> np.ndarray:
        """
        Violación normalizada de restricciones de cada individuo.

        Args:
            pop: Población de forma (N, n).

        Returns:
            Vector de longitud N; 0 indica solución factible.
        """
        return np.array([self.problema.violacion(x) for x in pop], dtype=float)

    # --------------------------------------------- Ordenamiento no dominado

    def _fast_nds(
        self, fits: np.ndarray, viol: Optional[np.ndarray] = None
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Ordenamiento no dominado rápido (fast non-dominated sort).

        Aplica la dominancia de la ec. 5 de la Clase 10 en el espacio de
        minimización gᵢ = wᵢ·fᵢ, con w = (−1, +1): así una sola comparación
        "≤ en todo y < en algo" sirve para f1 (maximizar) y f2 (minimizar).

            x1 ≺ x2  sii  ∀i: wᵢ·fᵢ(x1) ≤ wᵢ·fᵢ(x2)  y  ∃j: wⱼ·fⱼ(x1) < wⱼ·fⱼ(x2)

        Si se pasa `viol`, la comparación se extiende con la dominancia con
        restricciones de Deb (2002): una solución factible domina a cualquier
        infactible, y entre dos infactibles domina la de menor violación.

        La matriz de dominancia se construye vectorizada por broadcasting; el
        pelado de frentes es el algoritmo O(M·N²) de Deb et al. (2002), frente al
        O(M·N³) del NSGA original.

        Args:
            fits: Matriz (N, 2) de objetivos en su escala natural.
            viol: Vector opcional de violaciones de longitud N.

        Returns:
            Tupla (fronts, ranks): `fronts[k]` es la lista de índices del frente
            k (k=0 es el frente de Pareto) y `ranks[i]` es el frente al que
            pertenece el individuo i.

        Referencia: Tema 10 del curso CE UNI 2026 (dominancia, frentes de Pareto).
        """
        g = np.asarray(fits, dtype=float) * self._pesos
        n = g.shape[0]
        if n == 0:
            return [], []

        # Dominancia por objetivos (ec. 5) en el espacio de minimizacion.
        no_peor = (g[:, None, :] <= g[None, :, :]).all(axis=-1)
        mejor_en_algo = (g[:, None, :] < g[None, :, :]).any(axis=-1)
        domina = no_peor & mejor_en_algo

        if viol is not None:
            factible = np.asarray(viol, dtype=float) <= 0.0
            v = np.asarray(viol, dtype=float)
            domina = (
                (factible[:, None] & ~factible[None, :])
                | (~factible[:, None] & ~factible[None, :] & (v[:, None] < v[None, :]))
                | (factible[:, None] & factible[None, :] & domina)
            )

        # Numero de individuos que dominan a cada uno.
        n_dominadores = domina.sum(axis=0).astype(int)

        fronts: List[List[int]] = []
        ranks: List[int] = [0] * n
        actual = np.flatnonzero(n_dominadores == 0).tolist()
        rango = 0

        while actual:
            fronts.append(actual)
            for i in actual:
                ranks[i] = rango
            siguiente: List[int] = []
            for i in actual:
                for j in np.flatnonzero(domina[i]):
                    n_dominadores[j] -= 1
                    if n_dominadores[j] == 0:
                        siguiente.append(int(j))
            actual = siguiente
            rango += 1

        return fronts, ranks

    # ------------------------------------------------------------ Crowding

    def _crowding(self, front: List[int], fits: np.ndarray) -> Dict[int, float]:
        """
        Crowding distance de los individuos de un frente.

        Estima la densidad local sumando, por objetivo, la distancia normalizada
        entre los dos vecinos inmediatos del individuo dentro del frente
        ordenado. Los extremos reciben distancia infinita, para preservar siempre
        las soluciones que definen los límites del frente.

        Cumple el rol que en el NSGA de primera generación cumplía el **fitness
        compartido** con radio `sigma_share`: mantener la población repartida a lo
        largo del frente en vez de amontonada. La ventaja es que no requiere
        calibrar ningún parámetro de nicho, porque la escala se toma del propio
        rango del frente en cada objetivo.

        Se calcula sobre los objetivos en su escala natural: la fórmula usa
        diferencias entre vecinos consecutivos, así que el signo de wᵢ no altera
        el resultado.

        Args:
            front: Índices de los individuos del frente.
            fits: Matriz (N, 2) de objetivos.

        Returns:
            Diccionario {índice: crowding distance}.

        Referencia: Tema 10 del curso CE UNI 2026 (crowding distance, diversidad).
        """
        distancias: Dict[int, float] = {i: 0.0 for i in front}
        if len(front) <= 2:
            return {i: float("inf") for i in front}

        matriz = np.asarray(fits, dtype=float)
        for objetivo in range(matriz.shape[1]):
            orden = sorted(front, key=lambda i: matriz[i, objetivo])
            distancias[orden[0]] = float("inf")
            distancias[orden[-1]] = float("inf")
            rango = matriz[orden[-1], objetivo] - matriz[orden[0], objetivo]
            if rango <= 0.0:
                continue
            for posicion in range(1, len(orden) - 1):
                indice = orden[posicion]
                if distancias[indice] == float("inf"):
                    continue
                anterior = matriz[orden[posicion - 1], objetivo]
                siguiente = matriz[orden[posicion + 1], objetivo]
                distancias[indice] += (siguiente - anterior) / rango

        return distancias

    # ------------------------------------------------------------- Selección

    def _torneo_pareto(
        self,
        pop: np.ndarray,
        ranks: List[int],
        crows: Dict[int, float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Torneo binario con el operador de comparación aglomerada (≺n).

        Se sortean dos individuos y gana:
          1. el de menor rango de frente (más cerca del frente de Pareto); y
          2. si empatan en rango, el de mayor crowding distance (el que está en
             la zona menos poblada del frente).

        Este criterio reemplaza al fitness escalar del AG: no hay un único valor
        que ordene la población, sino el par (rango, dispersión).

        Args:
            pop: Población de forma (N, n).
            ranks: Rango de frente de cada individuo.
            crows: Crowding distance por índice.
            rng: Generador aleatorio explícito.

        Returns:
            Copia del cromosoma ganador, de forma (n,) y dtype np.int8.

        Raises:
            ValueError: Si la población está vacía.

        Referencia: Tema 10 del curso CE UNI 2026 (seleccion por dominancia y diversidad).
        """
        n = len(pop)
        if n == 0:
            raise ValueError("La poblacion esta vacia")
        if n == 1:
            return np.array(pop[0], dtype=np.int8, copy=True)

        a, b = (int(i) for i in rng.choice(n, size=K_TORNEO_PARETO, replace=False))
        if ranks[a] < ranks[b]:
            ganador = a
        elif ranks[b] < ranks[a]:
            ganador = b
        else:
            ganador = a if crows.get(a, 0.0) >= crows.get(b, 0.0) else b
        return np.array(pop[ganador], dtype=np.int8, copy=True)

    # ------------------------------------------------------- Inicialización

    def _inicializar(self) -> np.ndarray:
        """
        Construye la población inicial con cobertura territorial completada.

        Parte de `inicializar_heuristico` (el mismo operador que el AG) y, en cada
        individuo, completa el mínimo por macroregión que exige R3 sorteando obras
        inactivas de la región con probabilidad proporcional a su ratio b/C; luego
        aplica `reparar_greedy` para no violar R1 ni R2.

        Motivo de la diferencia con el AG: la inicialización heurística del
        prototipo produce individuos casi vacíos (~0.45 bits activos medidos con
        n=326), de modo que el AG gasta decenas de generaciones solo en descubrir
        la factibilidad mediante mutación. El AG lo conserva por fidelidad al
        prototipo que generó las métricas de referencia; NSGA-II se construye
        desde cero, así que puede arrancar en la región factible y dedicar todo su
        presupuesto de generaciones a explorar el compromiso f1/f2, que es su
        objetivo real.

        Returns:
            Población de forma (pop_size, n) y dtype np.int8.

        Referencia: Tema 5 y Tema 10 del curso CE UNI 2026.
        """
        pop = inicializar_heuristico(self.problema, self.pop_size, self.rng)
        ratio = np.asarray(self.problema.ratio(), dtype=float)

        for j in range(pop.shape[0]):
            individuo = pop[j]
            for region, minimo in self.problema.m_r.items():
                de_la_region = np.flatnonzero(self.problema.mr == region)
                faltan = int(minimo) - int(individuo[de_la_region].sum())
                if faltan <= 0:
                    continue
                candidatos = de_la_region[individuo[de_la_region] == 0]
                if candidatos.size == 0:
                    continue
                cuantos = min(faltan, candidatos.size)
                pesos = ratio[candidatos]
                total = float(pesos.sum())
                probabilidad = pesos / total if total > 0 else None
                elegidos = self.rng.choice(
                    candidatos, size=cuantos, replace=False, p=probabilidad
                )
                individuo[elegidos] = 1
            pop[j] = reparar_greedy(individuo, self.problema)

        logger.debug(
            "Poblacion inicial NSGA-II | bits: media=%.2f | factibles=%.0f%%",
            float(pop.sum(axis=1).mean()),
            100.0 * self.fraccion_factible(pop),
        )
        return pop

    # ---------------------------------------------------------- Ciclo NSGA-II

    def evolucionar(self) -> List[Solucion]:
        """
        Ejecuta NSGA-II y devuelve el frente de Pareto factible.

        Returns:
            Lista de `Solucion` no dominadas y factibles, sin duplicados y sin
            dominados internos, ordenada por costo f2 ascendente. Si no se
            encontró ninguna solución factible, devuelve una lista vacía y lo
            registra como advertencia.

        Raises:
            ValueError: Si pop_size < 2 o n_gen < 0.

        Referencia: Tema 10 del curso CE UNI 2026.
        """
        if self.pop_size < 2:
            raise ValueError(f"pop_size debe ser >= 2, se recibio {self.pop_size}")
        if self.n_gen < 0:
            raise ValueError(f"n_gen debe ser >= 0, se recibio {self.n_gen}")

        pop = self._inicializar()
        fits = self._objetivos(pop)
        viol = self._violaciones(pop)
        fronts, ranks = self._fast_nds(fits, viol)
        crows = self._crowding_global(fronts, fits)

        for generacion in range(1, self.n_gen + 1):
            hijos = self._generar_hijos(pop, ranks, crows)

            # --- Elitismo: padres e hijos compiten juntos (Rt = Pt U Qt) -----
            combinada = np.vstack([pop, hijos])
            fits_c = self._objetivos(combinada)
            viol_c = self._violaciones(combinada)
            fronts_c, ranks_c = self._fast_nds(fits_c, viol_c)

            pop, ranks, crows = self._seleccion_por_frentes(
                combinada, fits_c, ranks_c, fronts_c
            )
            fits = self._objetivos(pop)
            viol = self._violaciones(pop)
            fronts, _ = self._fast_nds(fits, viol)

            self._registrar(fits, pop, fronts)
            self.log_progreso(generacion, self.n_gen)

        self.frente = self._construir_frente(pop, fits, fronts)
        logger.info(
            "  NSGA-II resultado | frente factible=%d soluciones | f1 max=%.4f | f2 min=%.4f",
            len(self.frente),
            max((self.f1(s) for s in self.frente), default=float("nan")),
            min((s.costo for s in self.frente), default=float("nan")),
        )
        return self.frente

    def f1(self, solucion: Solucion) -> float:
        """
        Riesgo cubierto por una solución: f1 = Σᵢ R̃ᵢ·xᵢ (objetivo a maximizar).

        No confundir con `Solucion.z`, que es la función objetivo escalar del AG
        (suma ponderada de los cinco atributos).

        Args:
            solucion: Solución a evaluar.

        Returns:
            Valor del primer objetivo.
        """
        return float(np.asarray(solucion.x, dtype=float) @ self._r_norm)

    def _generar_hijos(
        self, pop: np.ndarray, ranks: List[int], crows: Dict[int, float]
    ) -> np.ndarray:
        """
        Genera la descendencia Qt con los operadores del AG.

        Args:
            pop: Población actual Pt, de forma (N, n).
            ranks: Rango de frente de cada padre.
            crows: Crowding distance por índice.

        Returns:
            Descendencia de forma (pop_size, n) y dtype np.int8.

        Referencia: Tema 3 y Tema 10 del curso CE UNI 2026.
        """
        hijos: List[np.ndarray] = []
        while len(hijos) < self.pop_size:
            padre1 = self._torneo_pareto(pop, ranks, crows, self.rng)
            padre2 = self._torneo_pareto(pop, ranks, crows, self.rng)
            hijo1, hijo2 = cruce_spx(padre1, padre2, self.pc, self.rng)
            hijos.append(
                reparar_greedy(mutacion_bitflip(hijo1, self.pm, self.rng), self.problema)
            )
            hijos.append(
                reparar_greedy(mutacion_bitflip(hijo2, self.pm, self.rng), self.problema)
            )
        return np.array(hijos[: self.pop_size], dtype=np.int8)

    def _seleccion_por_frentes(
        self,
        combinada: np.ndarray,
        fits: np.ndarray,
        ranks: List[int],
        fronts: List[List[int]],
    ) -> Tuple[np.ndarray, List[int], Dict[int, float]]:
        """
        Selecciona los N supervivientes de la población combinada.

        Los frentes se agregan completos mientras caben; el primero que no cabe se
        trunca conservando los individuos de mayor crowding distance, que son los
        que están en las zonas menos exploradas del frente.

        Args:
            combinada: Población Rt = Pt ∪ Qt, de forma (2N, n).
            fits: Objetivos de `combinada`.
            ranks: Rango de frente de cada individuo de `combinada`.
            fronts: Frentes de `combinada`.

        Returns:
            Tupla (nueva población, rangos reindexados, crowding reindexado).

        Referencia: Tema 10 del curso CE UNI 2026 (elitismo y truncamiento por diversidad).
        """
        seleccionados: List[int] = []
        nuevos_ranks: List[int] = []
        nuevos_crows: Dict[int, float] = {}

        for frente in fronts:
            crow_frente = self._crowding(frente, fits)
            if len(seleccionados) + len(frente) <= self.pop_size:
                elegidos = list(frente)
            else:
                faltan = self.pop_size - len(seleccionados)
                elegidos = sorted(
                    frente, key=lambda i: crow_frente[i], reverse=True
                )[:faltan]
            for indice in elegidos:
                nuevos_crows[len(seleccionados)] = crow_frente[indice]
                nuevos_ranks.append(ranks[indice])
                seleccionados.append(indice)
            if len(seleccionados) >= self.pop_size:
                break

        nueva = np.array(combinada[seleccionados], dtype=np.int8)
        return nueva, nuevos_ranks, nuevos_crows

    def _crowding_global(
        self, fronts: List[List[int]], fits: np.ndarray
    ) -> Dict[int, float]:
        """
        Calcula el crowding distance de todos los frentes de una población.

        Args:
            fronts: Frentes de la población.
            fits: Objetivos de la población.

        Returns:
            Diccionario {índice: crowding distance} que cubre toda la población.
        """
        distancias: Dict[int, float] = {}
        for frente in fronts:
            distancias.update(self._crowding(frente, fits))
        return distancias

    def _registrar(
        self, fits: np.ndarray, pop: np.ndarray, fronts: List[List[int]]
    ) -> None:
        """
        Registra en el historial las métricas de la generación.

        El fitness escalar penalizado no interviene en la selección de NSGA-II;
        se registra solo para poder comparar la convergencia con el AG en la misma
        escala (figuras y análisis).

        Args:
            fits: Objetivos de la población.
            pop: Población actual.
            fronts: Frentes de la población.
        """
        fitness_escalar = evaluar_poblacion(pop, self.problema)
        self.registrar_generacion(
            fitness_escalar,
            self.fraccion_factible(pop),
            n_frente=float(len(fronts[0]) if fronts else 0),
            f1_max=float(fits[:, 0].max()),
            f2_min=float(fits[:, 1].min()),
        )

    def _construir_frente(
        self, pop: np.ndarray, fits: np.ndarray, fronts: List[List[int]]
    ) -> List[Solucion]:
        """
        Extrae el frente de Pareto final como lista de `Solucion`.

        Filtra a las soluciones factibles, elimina cromosomas duplicados y ordena
        por costo f2 ascendente. Con dominancia con restricciones, si existe al
        menos una solución factible el frente 0 solo contiene factibles, así que
        el filtro es una salvaguarda explícita más que una corrección.

        Args:
            pop: Población final.
            fits: Objetivos de la población final.
            fronts: Frentes de la población final.

        Returns:
            Lista de `Solucion` factibles, sin duplicados, ordenada por f2.
        """
        if not fronts:
            logger.warning("NSGA-II no produjo ningun frente")
            return []

        unicos: Dict[bytes, Solucion] = {}
        for indice in fronts[0]:
            cromosoma = pop[indice]
            if not self.problema.es_factible(cromosoma):
                continue
            clave = np.asarray(cromosoma, dtype=np.int8).tobytes()
            if clave not in unicos:
                unicos[clave] = Solucion(cromosoma, self.problema)

        frente = sorted(unicos.values(), key=lambda s: s.costo)
        if not frente:
            logger.warning(
                "El frente de Pareto no tiene soluciones factibles: revisar si el "
                "problema admite solucion (K, B y m_r compatibles) o aumentar n_gen"
            )
        return frente

    # ------------------------------------------------------------ Metadatos

    def pareto_a_dicts(self) -> List[Dict[str, object]]:
        """
        Serializa el frente para `GET /pareto` (docs/API_SPEC.md).

        Returns:
            Lista de diccionarios con f1_riesgo, f2_costo, n_obras y obras.
        """
        return [
            {
                NOMBRES_OBJETIVOS[0]: round(self.f1(s), 6),
                NOMBRES_OBJETIVOS[1]: round(s.costo, 6),
                "n_obras": s.n_seleccionadas,
                "obras": s.codigos,
            }
            for s in self.frente
        ]

    def metadatos(self) -> Dict[str, object]:
        """
        Metadatos de la corrida, extendidos con la configuración de NSGA-II.

        Returns:
            Diccionario serializable a JSON.
        """
        datos = super().metadatos()
        datos.update(
            {
                "pop_size": self.pop_size,
                "n_gen": self.n_gen,
                "pc": self.pc,
                "pm": self.pm,
                "objetivos": list(NOMBRES_OBJETIVOS),
                "pesos_objetivos": list(PESOS_OBJETIVOS),
                "n_frente_final": len(self.frente),
                "manejo_restricciones": "dominancia con restricciones (Deb, 2002)",
            }
        )
        return datos

    def __repr__(self) -> str:
        """Resumen de una línea de NSGA-II."""
        return (
            f"NSGA2(n={self.problema.n}, pop={self.pop_size}, gen={self.n_gen}, "
            f"pc={self.pc}, pm={self.pm}, seed={self.seed}, "
            f"frente={len(self.frente)})"
        )
