"""
Algoritmo Genético Canónico Binario para el MCKP.

AG Canonico Binario - Tema 3 del curso CE UNI 2026.

Ensambla los operadores puros de `core/operators.py` en el ciclo evolutivo
generacional clásico. Todos los parámetros (población, generaciones, pc, pm, k,
semilla) vienen de `config/params.py`; ninguno está hard-codeado aquí.

Ciclo de una generación:

    torneo(k) -> cruce_spx(pc) -> mutacion_bitflip(pm) -> reparar_greedy
    -> evaluar -> elitismo (1 individuo)

Referencia: Tema 2 y Tema 3 del curso CE UNI 2026
            README.md, formulacion del MCKP y algoritmos — docs/ARCHITECTURE.md
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from algorithms.base import AlgoritmoEvolutivo
from config.params import Params
from core.operators import (
    cruce_spx,
    evaluar_poblacion,
    inicializar_heuristico,
    mutacion_bitflip,
    reparar_greedy,
    torneo,
)
from core.problem import Problema
from core.solution import Solucion

logger = logging.getLogger(__name__)

#: Número de individuos de elite que sobreviven intactos a la siguiente generación.
N_ELITE: int = 1


class AG(AlgoritmoEvolutivo):
    """
    AG Canonico Binario - Tema 3 del curso CE UNI 2026.

    Configuración de los operadores (README.md, algoritmos implementados):

        Representación  : vector binario x ∈ {0,1}ⁿ, un bit por obra
        Inicialización  : heurística estocástica sesgada por el ratio b/C
        Selección       : torneo determinístico, k = K_TORNEO (3)
        Cruce           : un punto (SPX), pc = PC (0.85)
        Mutación        : inversión de bit, pm = PM (0.015)
        Reparación      : greedy por ratio b/C, restaura R1 y R2
        Elitismo        : el mejor individuo sobrevive intacto (1 individuo)
        Reemplazo       : generacional con elite

    Sobre el elitismo: como la elite se inserta en la nueva población sin mutar,
    el máximo de fitness de la población es monótono no decreciente entre
    generaciones. Con más de un individuo de elite la diversidad cae y aparece
    convergencia prematura (docs/ARCHITECTURE.md).

    Sobre la reparación: cada descendiente se repara antes de evaluarse, así que
    R1 (presupuesto) y R2 (capacidad) se cumplen por construcción. La única
    restricción que la búsqueda tiene que aprender por penalización es R3
    (cobertura territorial), que la reparación no toca porque exigiría añadir
    obras. En consecuencia, la mutación bit-flip es el único mecanismo capaz de
    AÑADIR obras al portafolio.

    Atributos:
        pop_size: Tamaño de la población.
        n_gen: Número de generaciones.
        pc: Probabilidad de cruce.
        pm: Probabilidad de mutación por bit.
        k_torneo: Tamaño del torneo.
        mejor: Mejor `Solucion` encontrada (disponible tras `evolucionar()`).

    Referencia: Tema 3 del curso CE UNI 2026.
    """

    nombre: str = "AG Binario"
    tema: str = "Tema 3"

    def __init__(
        self,
        problema: Problema,
        params: Optional[Params] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Configura el AG a partir de los parámetros centralizados.

        Los parámetros se copian a atributos de instancia para poder ajustarlos
        puntualmente (por ejemplo `ag.n_gen = 15` en un smoke test o en el
        análisis de sensibilidad) sin tocar el `.env` ni el Singleton compartido.

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
        self.k_torneo: int = int(self.params.K_TORNEO)

        self.mejor: Optional[Solucion] = None

        logger.info(
            "  Parametros AG | pop=%d gen=%d pc=%.2f pm=%.4f k=%d alpha=%.2f",
            self.pop_size,
            self.n_gen,
            self.pc,
            self.pm,
            self.k_torneo,
            self.problema.alpha,
        )

    def evolucionar(self) -> Solucion:
        """
        Ejecuta el ciclo evolutivo completo y devuelve la mejor solución hallada.

        Secuencia por generación:
          1. Se copia la elite a la nueva población (elitismo, sin mutar).
          2. Se seleccionan dos padres por torneo de tamaño k.
          3. Se cruzan con SPX y probabilidad pc.
          4. Cada hijo se muta bit a bit con probabilidad pm.
          5. Cada hijo se repara con greedy (restaura R1 y R2).
          6. Se evalúa la población y se actualiza la elite si mejoró.
          7. Se registra el historial y se loguea cada 100 generaciones.

        Returns:
            La mejor `Solucion` encontrada en toda la corrida (la elite final).

        Raises:
            ValueError: Si pop_size < 1 o n_gen < 0.

        Referencia: Tema 2 y Tema 3 del curso CE UNI 2026.
        """
        if self.pop_size < 1:
            raise ValueError(f"pop_size debe ser >= 1, se recibio {self.pop_size}")
        if self.n_gen < 0:
            raise ValueError(f"n_gen debe ser >= 0, se recibio {self.n_gen}")

        # --- Generación 0: inicialización heurística y evaluación ------------
        pop = inicializar_heuristico(self.problema, self.pop_size, self.rng)
        fit = evaluar_poblacion(pop, self.problema)

        indice_mejor = int(np.argmax(fit))
        elite = pop[indice_mejor].copy()
        elite_fit = float(fit[indice_mejor])

        # --- Ciclo generacional ---------------------------------------------
        for generacion in range(1, self.n_gen + 1):
            pop = self._nueva_generacion(pop, fit, elite)
            fit = evaluar_poblacion(pop, self.problema)

            indice_mejor = int(np.argmax(fit))
            if fit[indice_mejor] > elite_fit:
                elite_fit = float(fit[indice_mejor])
                elite = pop[indice_mejor].copy()

            self.registrar_generacion(fit, self.fraccion_factible(pop))
            self.log_progreso(generacion, self.n_gen)

        self.mejor = Solucion(elite, self.problema)
        logger.info(
            "  AG resultado | fitness=%.6f | obras=%d | costo=%.4f | factible=%s",
            self.mejor.fitness,
            self.mejor.n_seleccionadas,
            self.mejor.costo,
            self.mejor.factible,
        )
        return self.mejor

    def _nueva_generacion(
        self, pop: np.ndarray, fit: np.ndarray, elite: np.ndarray
    ) -> np.ndarray:
        """
        Construye la población de la siguiente generación.

        La elite ocupa la primera posición sin pasar por mutación ni reparación;
        el resto se genera por parejas (torneo, cruce, mutación, reparación) hasta
        completar `pop_size`, truncando el excedente si el tamaño es impar.

        Args:
            pop: Población actual, de forma (pop_size, n).
            fit: Fitness de la población actual.
            elite: Mejor cromosoma conocido hasta ahora.

        Returns:
            Nueva población de forma (pop_size, n) y dtype np.int8.

        Referencia: Tema 3 del curso CE UNI 2026 (reemplazo generacional con elitismo).
        """
        nueva: List[np.ndarray] = [elite.copy()]

        while len(nueva) < self.pop_size:
            padre1 = torneo(pop, fit, self.k_torneo, self.rng)
            padre2 = torneo(pop, fit, self.k_torneo, self.rng)
            hijo1, hijo2 = cruce_spx(padre1, padre2, self.pc, self.rng)
            nueva.append(
                reparar_greedy(mutacion_bitflip(hijo1, self.pm, self.rng), self.problema)
            )
            nueva.append(
                reparar_greedy(mutacion_bitflip(hijo2, self.pm, self.rng), self.problema)
            )

        return np.array(nueva[: self.pop_size], dtype=np.int8)

    def metadatos(self) -> Dict[str, object]:
        """
        Metadatos de la corrida, extendidos con los parámetros del AG.

        Returns:
            Diccionario serializable a JSON con la identidad de la corrida más
            pop_size, n_gen, pc, pm, k_torneo y n_elite.
        """
        datos = super().metadatos()
        datos.update(
            {
                "pop_size": self.pop_size,
                "n_gen": self.n_gen,
                "pc": self.pc,
                "pm": self.pm,
                "k_torneo": self.k_torneo,
                "n_elite": N_ELITE,
            }
        )
        return datos

    def __repr__(self) -> str:
        """Resumen de una línea del AG."""
        mejores = self.historial["mejor_fitness"]
        mejor = f"{max(mejores):.6f}" if mejores else "sin evaluar"
        return (
            f"AG(n={self.problema.n}, pop={self.pop_size}, gen={self.n_gen}, "
            f"pc={self.pc}, pm={self.pm}, k={self.k_torneo}, seed={self.seed}, "
            f"mejor={mejor})"
        )
