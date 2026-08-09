"""
Tests del AG binario (`algorithms/genetic.py`) y de los operadores puros.

Referencia: Tema 3 del curso CE UNI 2026
            docs/TECHNICAL_DEBT.md DT-008 (reproducibilidad de semillas)
            los criterios de cobertura del proyecto
"""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.benchmarks import aleatorio
from algorithms.genetic import AG
from config.params import Params
from core.operators import (
    cruce_spx,
    inicializar_heuristico,
    mutacion_bitflip,
    reparar_greedy,
    torneo,
)
from core.problem import Problema

#: Generaciones que basta darle al AG para alcanzar factibilidad con n=120
#: (medido: factible ya en 10 generaciones con poblacion 20).
GEN_TEST: int = 20

#: Poblacion de los tests, de los tests.
POP_TEST: int = 20


def _ag(problema: Problema, params: Params, seed: int, n_gen: int = GEN_TEST) -> AG:
    """
    Construye un AG con los parámetros reducidos de los tests.

    Args:
        problema: Problema MCKP.
        params: Parámetros de prueba.
        seed: Semilla de la corrida.
        n_gen: Generaciones a ejecutar.

    Returns:
        Instancia de `AG` lista para ejecutar.
    """
    algoritmo = AG(problema, params, seed=seed)
    algoritmo.pop_size = POP_TEST
    algoritmo.n_gen = n_gen
    return algoritmo


class TestAG:
    """Comportamiento del ciclo evolutivo completo."""

    def test_ag_retorna_solucion_factible(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Con 120 obras y 20 generaciones el AG ya entra en la region factible."""
        solucion = _ag(problema_mediano, params_test, seed=0).evolucionar()

        assert solucion.factible
        assert solucion.violacion == 0.0
        assert solucion.n_seleccionadas <= problema_mediano.K
        assert solucion.costo <= problema_mediano.B

    def test_ag_fitness_positivo(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """El fitness de la solución devuelta es positivo."""
        assert _ag(problema_mediano, params_test, seed=0).evolucionar().fitness > 0.0

    def test_ag_supera_al_aleatorio(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """El AG debe batir a la busqueda aleatoria con reparacion naive."""
        solucion_ag = _ag(problema_mediano, params_test, seed=42, n_gen=60).evolucionar()
        solucion_azar = aleatorio(problema_mediano, n_intentos=2000, seed=99)
        assert solucion_ag.fitness > solucion_azar.fitness

    def test_elitismo_monotono(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """
        La serie mejor_fitness no puede decrecer.

        La elite entra sin mutar en la nueva población, así que el máximo de la
        población es monótono no decreciente entre generaciones.
        """
        algoritmo = _ag(problema_mediano, params_test, seed=0, n_gen=40)
        algoritmo.evolucionar()
        mejores = algoritmo.historial["mejor_fitness"]

        assert len(mejores) == 40
        diferencias = np.diff(np.asarray(mejores))
        assert (diferencias >= -1e-12).all(), "el mejor fitness decrecio en alguna generacion"

    def test_semillas_distintas_dan_resultados_distintos(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Semillas diferentes exploran trayectorias diferentes."""
        soluciones = [
            _ag(problema_mediano, params_test, seed=s).evolucionar() for s in (1, 2, 3)
        ]
        cromosomas = {s.x.tobytes() for s in soluciones}
        assert len(cromosomas) > 1

    def test_misma_semilla_reproduce_el_resultado(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """
        Reproducibilidad exacta por semilla (DT-008).

        Se ejecutan de forma intercalada con una corrida de otra semilla en medio,
        para comprobar que no hay estado aleatorio global compartido.
        """
        primera = _ag(problema_mediano, params_test, seed=7).evolucionar()
        _ag(problema_mediano, params_test, seed=999).evolucionar()
        segunda = _ag(problema_mediano, params_test, seed=7).evolucionar()

        assert np.array_equal(primera.x, segunda.x)
        assert primera.fitness == pytest.approx(segunda.fitness)

    def test_historial_registra_las_cuatro_series(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Cada generación añade una entrada a las cuatro series del historial."""
        algoritmo = _ag(problema_mediano, params_test, seed=0)
        algoritmo.evolucionar()

        for clave in ("mejor_fitness", "media_fitness", "peor_fitness", "fraccion_factible"):
            assert len(algoritmo.historial[clave]) == GEN_TEST
        assert algoritmo.generaciones_ejecutadas == GEN_TEST
        assert all(0.0 <= v <= 1.0 for v in algoritmo.historial["fraccion_factible"])

    def test_no_muta_el_singleton_de_params(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Ajustar pop_size y n_gen en la instancia no toca la configuracion global."""
        singleton = Params.get()
        pop_antes, gen_antes = singleton.POP_SIZE, singleton.N_GEN

        _ag(problema_mediano, params_test, seed=0).evolucionar()

        assert (singleton.POP_SIZE, singleton.N_GEN) == (pop_antes, gen_antes)

    def test_ejecutar_cronometra_la_corrida(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """ejecutar() deja el tiempo en tiempo_seg."""
        algoritmo = _ag(problema_mediano, params_test, seed=0)
        algoritmo.ejecutar()
        assert algoritmo.tiempo_seg > 0.0


class TestClaseBase:
    """Servicios heredados de `AlgoritmoEvolutivo` (historial, metadatos, resumen)."""

    def test_guardar_historial_escribe_json_trazable(
        self, problema_mediano: Problema, params_test: Params, tmp_path
    ) -> None:
        """El JSON lleva metadatos y series con tipos nativos, legible sin encoding."""
        import json

        algoritmo = _ag(problema_mediano, params_test, seed=3)
        algoritmo.ejecutar()
        destino = tmp_path / "sub" / "historial.json"
        algoritmo.guardar_historial(destino)

        contenido = json.loads(destino.read_text())
        assert set(contenido) == {"metadatos", "historial"}
        assert contenido["metadatos"]["seed"] == 3
        assert contenido["metadatos"]["algoritmo"] == "AG Binario"
        assert len(contenido["historial"]["mejor_fitness"]) == GEN_TEST
        assert all(
            isinstance(v, (int, float)) for v in contenido["historial"]["mejor_fitness"]
        )

    def test_metadatos_incluye_los_parametros_del_ag(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """metadatos() extiende el de la base con la configuracion del AG."""
        algoritmo = _ag(problema_mediano, params_test, seed=0)
        datos = algoritmo.metadatos()

        assert datos["tema"] == "Tema 3"
        assert datos["pop_size"] == POP_TEST
        assert datos["n_gen"] == GEN_TEST
        assert set(datos["pesos"]) == {"R", "M", "P", "E", "G"}
        assert datos["capacidad_K"] == problema_mediano.K

    def test_resumen_y_reinicio_del_historial(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """resumen() refleja la corrida y reiniciar_historial() la borra."""
        algoritmo = _ag(problema_mediano, params_test, seed=0)
        algoritmo.ejecutar()

        resumen = algoritmo.resumen()
        assert resumen["generaciones"] == GEN_TEST
        assert resumen["mejor_fitness"] is not None
        assert "AG" in repr(algoritmo)

        algoritmo.reiniciar_historial()
        assert algoritmo.generaciones_ejecutadas == 0
        assert algoritmo.resumen()["mejor_fitness"] is None

    def test_log_progreso_no_falla_sin_historial(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Llamar al log antes de registrar nada no debe romper."""
        algoritmo = _ag(problema_mediano, params_test, seed=0)
        algoritmo.log_progreso(1, 10)  # no debe lanzar
        algoritmo.log_progreso(1, 10, cada=0)

    def test_rechaza_problema_invalido(self, params_test: Params) -> None:
        """Pasar algo que no es un Problema es el error mas facil de cometer."""
        with pytest.raises(TypeError, match="Problema"):
            AG("no soy un problema", params_test, seed=0)  # type: ignore[arg-type]

    def test_fraccion_factible_de_poblacion_vacia(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Sin individuos la fraccion es 0.0, no una division por cero."""
        algoritmo = _ag(problema_mediano, params_test, seed=0)
        assert algoritmo.fraccion_factible(np.empty((0, problema_mediano.n))) == 0.0

    def test_registrar_generacion_rechaza_fitness_vacio(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Registrar una generación sin fitness es un error de programación."""
        algoritmo = _ag(problema_mediano, params_test, seed=0)
        with pytest.raises(ValueError, match="vacio"):
            algoritmo.registrar_generacion([], 0.0)


class TestOperadores:
    """Operadores puros de `core/operators.py`."""

    def test_cruce_spx_produce_hijos_complementarios(self) -> None:
        """Con padres de ceros y unos, los hijos se complementan bit a bit."""
        rng = np.random.default_rng(0)
        padre1 = np.zeros(20, dtype=np.int8)
        padre2 = np.ones(20, dtype=np.int8)
        hijo1, hijo2 = cruce_spx(padre1, padre2, pc=1.0, rng=rng)

        assert np.all(hijo1 + hijo2 == 1)
        assert padre1.sum() == 0 and padre2.sum() == 20  # padres intactos

    def test_cruce_spx_con_pc_cero_copia_a_los_padres(self) -> None:
        """Con pc=0 nunca se cruza."""
        rng = np.random.default_rng(0)
        padre1 = np.zeros(20, dtype=np.int8)
        padre2 = np.ones(20, dtype=np.int8)
        for _ in range(50):
            hijo1, hijo2 = cruce_spx(padre1, padre2, pc=0.0, rng=rng)
            assert np.array_equal(hijo1, padre1) and np.array_equal(hijo2, padre2)

    def test_mutacion_bitflip_es_pura_y_respeta_la_tasa(self) -> None:
        """No muta la entrada y el número de inversiones tiende a pm*n."""
        rng = np.random.default_rng(0)
        x = np.zeros(500, dtype=np.int8)
        cambios = [int(mutacion_bitflip(x, 0.02, rng).sum()) for _ in range(200)]

        assert x.sum() == 0
        assert np.mean(cambios) == pytest.approx(0.02 * 500, rel=0.15)

    def test_reparar_greedy_restaura_R1_y_R2(self, problema_mediano: Problema) -> None:
        """Tras reparar, la solución cumple presupuesto y capacidad."""
        sucio = np.ones(problema_mediano.n, dtype=np.int8)
        reparado = reparar_greedy(sucio, problema_mediano)
        detalle = problema_mediano.violaciones_detalle(reparado)

        assert sucio.sum() == problema_mediano.n  # entrada intacta
        assert int(reparado.sum()) <= problema_mediano.K
        assert problema_mediano.costo(reparado) <= problema_mediano.B
        assert detalle["R1"] == 0.0 and detalle["R2"] == 0.0

    def test_torneo_favorece_al_mejor(self, problema_mediano: Problema) -> None:
        """Con k = tamano de poblacion, el torneo es elitista puro."""
        rng = np.random.default_rng(0)
        pop = inicializar_heuristico(problema_mediano, 10, rng)
        fit = np.arange(10, dtype=float)
        ganador = torneo(pop, fit, k=10, rng=rng)
        assert np.array_equal(ganador, pop[9])

    def test_torneo_rechaza_fit_desalineado(self, problema_mediano: Problema) -> None:
        """Un fitness de longitud distinta a la población es un error."""
        rng = np.random.default_rng(0)
        pop = inicializar_heuristico(problema_mediano, 10, rng)
        with pytest.raises(ValueError, match="coincidir"):
            torneo(pop, np.zeros(3), k=3, rng=rng)

    def test_inicializar_heuristico_es_reproducible(
        self, problema_mediano: Problema
    ) -> None:
        """El mismo generador produce la misma poblacion; otro, una distinta."""
        primera = inicializar_heuristico(problema_mediano, 15, np.random.default_rng(4))
        segunda = inicializar_heuristico(problema_mediano, 15, np.random.default_rng(4))
        otra = inicializar_heuristico(problema_mediano, 15, np.random.default_rng(5))

        assert np.array_equal(primera, segunda)
        assert not np.array_equal(primera, otra)
        assert primera.shape == (15, problema_mediano.n)
        assert primera.dtype == np.int8
