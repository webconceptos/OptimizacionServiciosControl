"""
Tests de NSGA-II (`algorithms/nsga2.py`).

El test de factibilidad del frente cubre DT-005 de docs/TECHNICAL_DEBT.md, que
advierte que la reparación solo atiende R1 y R2 y que por tanto el frente podría
contener soluciones que violan R3. La implementación lo resuelve con dominancia
con restricciones (Deb, 2002) más un filtro explícito; estos tests lo verifican.

Referencia: Tema 10 del curso CE UNI 2026
            los criterios de cobertura del proyecto
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from algorithms.nsga2 import PESOS_OBJETIVOS, NSGA2
from config.params import Params
from core.problem import Problema

#: Poblacion y generaciones de los tests de NSGA-II.
POP_TEST: int = 40
GEN_TEST: int = 30

#: Presupuesto de tiempo del test de rendimiento, en segundos.
LIMITE_SEGUNDOS: float = 30.0


def _nsga2(
    problema: Problema, params: Params, seed: int = 42, pop: int = POP_TEST, gen: int = GEN_TEST
) -> NSGA2:
    """
    Construye NSGA-II con los parámetros reducidos de los tests.

    Args:
        problema: Problema MCKP.
        params: Parámetros de prueba.
        seed: Semilla de la corrida.
        pop: Tamaño de población.
        gen: Número de generaciones.

    Returns:
        Instancia lista para ejecutar.
    """
    algoritmo = NSGA2(problema, params, seed=seed)
    algoritmo.pop_size = pop
    algoritmo.n_gen = gen
    return algoritmo


def _domina(objetivos_a: np.ndarray, objetivos_b: np.ndarray) -> bool:
    """
    Dominancia de Pareto según la ec. 5 de la Clase 03 (MOGA, Túpac).

    Aplica los pesos wᵢ = (−1, +1) para llevar f1 (maximizar) y f2 (minimizar) a un
    espacio de minimización común: a domina a b si es no peor en todo y mejor en
    algo.

    Args:
        objetivos_a: Par (f1, f2) de la primera solución.
        objetivos_b: Par (f1, f2) de la segunda.

    Returns:
        True si a domina a b.
    """
    pesos = np.asarray(PESOS_OBJETIVOS, dtype=float)
    g_a = np.asarray(objetivos_a, dtype=float) * pesos
    g_b = np.asarray(objetivos_b, dtype=float) * pesos
    return bool(np.all(g_a <= g_b) and np.any(g_a < g_b))


class TestFrenteDePareto:
    """Propiedades que el frente devuelto debe cumplir siempre."""

    def test_frente_no_vacio(self, problema_mediano: Problema, params_test: Params) -> None:
        """Con 120 obras debe encontrar al menos una solución no dominada."""
        frente = _nsga2(problema_mediano, params_test).evolucionar()
        assert len(frente) > 0

    def test_todas_las_soluciones_son_factibles(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """
        Cubre DT-005: ninguna solución del frente puede violar R3.

        La reparación greedy solo restaura R1 y R2; la factibilidad territorial la
        garantizan la dominancia con restricciones y el filtro final.
        """
        frente = _nsga2(problema_mediano, params_test).evolucionar()

        assert len(frente) > 0
        for solucion in frente:
            assert solucion.factible
            assert solucion.violacion == 0.0
            for region, minimo in problema_mediano.m_r.items():
                assert solucion.distribucion_territorial[region] >= minimo

    def test_no_hay_dominados_internos(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Ningún par del frente puede dominarse: es la definición de frente."""
        algoritmo = _nsga2(problema_mediano, params_test)
        frente = algoritmo.evolucionar()
        objetivos = [(algoritmo.f1(s), s.costo) for s in frente]

        for (i, obj_i), (j, obj_j) in itertools.permutations(enumerate(objetivos), 2):
            assert not _domina(np.array(obj_i), np.array(obj_j)), (
                f"la solucion {i} domina a la {j}: {obj_i} vs {obj_j}"
            )

    def test_frente_sin_cromosomas_duplicados(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Dos soluciones idénticas no aportan información al decisor."""
        frente = _nsga2(problema_mediano, params_test).evolucionar()
        assert len({s.x.tobytes() for s in frente}) == len(frente)

    def test_frente_ordenado_por_costo_ascendente(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """El frente se entrega ordenado por f2 para poder recorrerlo como curva."""
        frente = _nsga2(problema_mediano, params_test).evolucionar()
        costos = [s.costo for s in frente]
        assert costos == sorted(costos)

    def test_existe_compromiso_entre_objetivos(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Si hay más de una solución, cubrir más riesgo cuesta más."""
        algoritmo = _nsga2(problema_mediano, params_test)
        frente = algoritmo.evolucionar()
        if len(frente) < 2:
            pytest.skip("el frente tiene una sola solucion: no hay compromiso que medir")

        f1 = [algoritmo.f1(s) for s in frente]
        f2 = [s.costo for s in frente]
        assert f1[-1] > f1[0] and f2[-1] > f2[0]

    def test_pareto_a_dicts_es_serializable(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """La salida para GET /pareto usa tipos nativos."""
        import json

        algoritmo = _nsga2(problema_mediano, params_test)
        algoritmo.evolucionar()
        puntos = algoritmo.pareto_a_dicts()

        json.dumps(puntos)  # no debe lanzar
        assert len(puntos) == len(algoritmo.frente)
        assert set(puntos[0]) == {"f1_riesgo", "f2_costo", "n_obras", "obras"}


class TestOrdenamientoNoDominado:
    """`_fast_nds` y `_crowding` por separado."""

    def test_fast_nds_particiona_la_poblacion(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Los frentes cubren todos los índices exactamente una vez."""
        algoritmo = _nsga2(problema_mediano, params_test)
        rng = np.random.default_rng(0)
        objetivos = rng.random((25, 2)) * np.array([10.0, 5.0])

        frentes, rangos = algoritmo._fast_nds(objetivos)

        assert sum(len(f) for f in frentes) == 25
        assert sorted(i for f in frentes for i in f) == list(range(25))
        for numero, frente in enumerate(frentes):
            assert all(rangos[i] == numero for i in frente)

    def test_fast_nds_primer_frente_sin_dominados(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Ningún elemento del frente 0 es dominado por otro de la población."""
        algoritmo = _nsga2(problema_mediano, params_test)
        rng = np.random.default_rng(1)
        objetivos = rng.random((20, 2)) * np.array([10.0, 5.0])

        frentes, _ = algoritmo._fast_nds(objetivos)

        for i in frentes[0]:
            for j in range(len(objetivos)):
                assert not _domina(objetivos[j], objetivos[i])

    def test_dominancia_con_restricciones_prioriza_factibles(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Una solución factible domina a una infactible aunque sea peor en objetivos."""
        algoritmo = _nsga2(problema_mediano, params_test)
        # El indice 0 es mejor en objetivos, pero infactible.
        objetivos = np.array([[9.0, 1.0], [0.0, 9.0]])
        violaciones = np.array([2.0, 0.0])

        frentes, _ = algoritmo._fast_nds(objetivos, violaciones)
        assert frentes[0] == [1]

    def test_crowding_infinito_en_los_extremos(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Los extremos del frente reciben distancia infinita para no perderse."""
        algoritmo = _nsga2(problema_mediano, params_test)
        objetivos = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        distancias = algoritmo._crowding([0, 1, 2, 3], objetivos)

        assert distancias[0] == float("inf")
        assert distancias[3] == float("inf")
        assert np.isfinite(distancias[1]) and np.isfinite(distancias[2])

    def test_crowding_premia_al_aislado(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """En un frente asimétrico, el punto de la zona menos poblada gana."""
        algoritmo = _nsga2(problema_mediano, params_test)
        objetivos = np.array([[0.0, 0.0], [1.0, 1.0], [1.1, 1.1], [5.0, 5.0]])

        distancias = algoritmo._crowding([0, 1, 2, 3], objetivos)
        assert distancias[2] > distancias[1]


class TestRendimiento:
    """Coste computacional a la escala del caso de estudio."""

    def test_corre_en_tiempo_razonable(
        self, problema_grande: Problema, params_test: Params
    ) -> None:
        """Con n=326 y 50 generaciones debe terminar bien por debajo de 30 s."""
        algoritmo = _nsga2(problema_grande, params_test, pop=POP_TEST, gen=50)

        inicio = time.perf_counter()
        frente = algoritmo.evolucionar()
        duracion = time.perf_counter() - inicio

        assert duracion < LIMITE_SEGUNDOS, f"NSGA-II tardo {duracion:.1f} s"
        assert len(frente) > 0
