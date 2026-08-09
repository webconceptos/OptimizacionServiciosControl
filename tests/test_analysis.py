"""
Tests de `analysis/`: estadística, teoría de esquemas y sensibilidad.

Nota sobre el tamaño de muestra del contraste: el test de Wilcoxon signed-rank
bilateral con n corridas tiene un p mínimo de 2/2ⁿ, así que con 3 corridas el p
más pequeño posible es 0.25 y **nunca** puede ser significativo a 0.05. Por eso la
significancia se prueba sobre un conjunto sintético de 10 corridas (que es el
`N_RUNS` real del proyecto) y las corridas reales del AG solo se usan para
comprobar la forma del DataFrame.

Referencia: docs/ARCHITECTURE.md (Wilcoxon sobre t-test), Tema 4 (esquemas)
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from algorithms.benchmarks import greedy
from analysis.schema_theory import (
    analizar_building_blocks,
    calcular_supervivencia,
    resumen_building_blocks,
)
from analysis.sensitivity import (
    COLUMNAS_PARAMETROS,
    redistribuir_pesos,
    resumen_sensibilidad,
    resumen_sensibilidad_parametros,
    sensibilidad_parametros,
    sensibilidad_w1,
)
from analysis.statistics import (
    COLUMNAS_CORRIDAS,
    analisis_wilcoxon,
    multiples_corridas,
    tabla_comparacion,
)
from config.params import Params
from core.problem import Problema


def _df_runs(fitness: List[float]) -> pd.DataFrame:
    """
    Construye un DataFrame de corridas sintético con la forma de `multiples_corridas`.

    Args:
        fitness: Valores de fitness de cada corrida.

    Returns:
        DataFrame con las columnas de `COLUMNAS_CORRIDAS`.
    """
    n = len(fitness)
    return pd.DataFrame(
        {
            "corrida": range(1, n + 1),
            "seed": range(42, 42 + n),
            "fitness": fitness,
            "n_seleccionadas": [50] * n,
            "costo": [25.0] * n,
            "r_medio": [0.88] * n,
            "factible": [True] * n,
            "tiempo_seg": [1.0] * n,
        }
    )[COLUMNAS_CORRIDAS]


class TestMultiplesCorridas:
    """Ejecución de N corridas independientes del AG."""

    def test_devuelve_una_fila_por_corrida(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Con n_runs=3 salen 3 filas y las columnas acordadas."""
        params_test.POP_SIZE = 20
        params_test.N_GEN = 10
        df_runs = multiples_corridas(problema_mediano, params_test, n_runs=3)

        assert len(df_runs) == 3
        assert list(df_runs.columns) == COLUMNAS_CORRIDAS
        assert df_runs["fitness"].notna().all()
        assert (df_runs["tiempo_seg"] > 0).all()

    def test_usa_semillas_consecutivas_desde_la_base(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """La corrida i usa la semilla base + i, para poder reproducirla."""
        params_test.POP_SIZE = 20
        params_test.N_GEN = 5
        df_runs = multiples_corridas(problema_mediano, params_test, n_runs=3, seed=100)
        assert df_runs["seed"].tolist() == [100, 101, 102]

    def test_rechaza_n_runs_invalido(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Pedir cero corridas es un error de uso."""
        with pytest.raises(ValueError, match="n_runs"):
            multiples_corridas(problema_mediano, params_test, n_runs=0)


class TestWilcoxon:
    """Contraste de hipótesis contra el Greedy."""

    def test_significativo_cuando_el_ag_supera_al_greedy(self) -> None:
        """
        Con 10 corridas todas por encima del Greedy, el p exacto es 2/2^10.

        Es el valor de referencia del proyecto: 0.001953125.
        """
        greedy_fitness = 23.0
        df_runs = _df_runs([24.0 + i * 0.01 for i in range(10)])

        resumen = analisis_wilcoxon(df_runs, greedy_fitness)

        assert resumen["n_corridas_mejores"] == 10
        assert resumen["wilcoxon_p"] == pytest.approx(0.001953125)
        assert resumen["significativo"] is True
        assert resumen["mejora_media_pct"] > 0.0
        assert resumen["test_omitido"] is None

    def test_no_significativo_con_pocas_corridas(self) -> None:
        """
        Con 3 corridas el p minimo posible es 0.25: no puede ser significativo.

        Documenta por qué la significancia no se prueba con el N_RUNS reducido de
        los tests.
        """
        resumen = analisis_wilcoxon(_df_runs([24.0, 24.1, 24.2]), 23.0)
        assert resumen["wilcoxon_p"] == pytest.approx(0.25)
        assert resumen["significativo"] is False

    def test_no_significativo_cuando_el_ag_no_supera(self) -> None:
        """Si el AG queda por debajo, el unilateral lo delata aunque el bilateral no."""
        resumen = analisis_wilcoxon(_df_runs([20.0 + i * 0.01 for i in range(10)]), 23.0)

        assert resumen["n_corridas_mejores"] == 0
        assert resumen["mejora_media_pct"] < 0.0
        assert resumen["wilcoxon_p_unilateral"] > 0.05

    def test_diferencias_nulas_no_lanzan_excepcion(self) -> None:
        """Empatar con el Greedy en todas las corridas se reporta como p=1.0."""
        resumen = analisis_wilcoxon(_df_runs([23.0] * 5), 23.0)

        assert resumen["wilcoxon_p"] == 1.0
        assert resumen["significativo"] is False
        assert resumen["test_omitido"] is not None

    def test_una_sola_corrida_no_lanza_excepcion(self) -> None:
        """Con n=1 el signed-rank no aplica, pero los descriptivos siguen saliendo."""
        resumen = analisis_wilcoxon(_df_runs([24.0]), 23.0)

        assert np.isnan(resumen["wilcoxon_p"])
        assert resumen["significativo"] is False
        assert resumen["media"] == pytest.approx(24.0)
        assert resumen["std"] == 0.0

    def test_estadisticos_descriptivos(self) -> None:
        """media, mediana, minimo y maximo salen de la muestra."""
        resumen = analisis_wilcoxon(_df_runs([10.0, 20.0, 30.0]), 5.0)

        assert resumen["media"] == pytest.approx(20.0)
        assert resumen["mediana"] == pytest.approx(20.0)
        assert resumen["minimo"] == pytest.approx(10.0)
        assert resumen["maximo"] == pytest.approx(30.0)
        assert resumen["greedy"] == pytest.approx(5.0)

    def test_rechaza_dataframe_sin_fitness(self) -> None:
        """Sin la columna fitness no hay nada que contrastar."""
        with pytest.raises(ValueError, match="fitness"):
            analisis_wilcoxon(pd.DataFrame({"otra": [1, 2]}), 1.0)


class TestTablaComparacion:
    """Tabla comparativa de métodos."""

    def test_calcula_la_mejora_contra_greedy(self, solucion_factible) -> None:
        """La columna de mejora se mide contra la fila llamada Greedy."""
        tabla = tabla_comparacion(
            {"AG (propuesto)": solucion_factible, "Greedy": solucion_factible}
        )

        assert list(tabla["metodo"]) == ["AG (propuesto)", "Greedy"]
        assert tabla.loc[tabla["metodo"] == "Greedy", "mejora_vs_greedy_pct"].iloc[0] == 0.0

    def test_sin_fila_greedy_la_mejora_queda_en_nan(self) -> None:
        """Sin referencia no se inventa una mejora."""
        tabla = tabla_comparacion({"A": {"fitness": 1.0}, "B": {"fitness": 2.0}})
        assert tabla["mejora_vs_greedy_pct"].isna().all()


class TestTeoriaDeEsquemas:
    """Building blocks y cota de supervivencia (Tema 4)."""

    def test_factores_son_positivos(self, problema_mediano: Problema) -> None:
        """Todos los factores del teorema son positivos y el ranking es coherente."""
        df_bbs = analizar_building_blocks(problema_mediano, top_k=6)

        assert len(df_bbs) == 6 + 15 + 20  # C(6,1) + C(6,2) + C(6,3)
        assert (df_bbs["f_H"] > 0).all()
        assert (df_bbs["fitness_relativo"] > 0).all()
        assert (df_bbs["supervivencia"] > 0).all()
        assert (df_bbs["crecimiento"] > 0).all()
        assert (df_bbs["supervivencia"] <= 1.0).all()

    def test_crecimiento_es_el_producto_de_los_dos_factores(
        self, problema_mediano: Problema
    ) -> None:
        """crecimiento = K_G * K_S, la descomposicion de la ec. 25/30."""
        df_bbs = analizar_building_blocks(problema_mediano, top_k=5)
        assert np.allclose(
            df_bbs["crecimiento"], df_bbs["fitness_relativo"] * df_bbs["supervivencia"]
        )

    def test_ordenado_por_crecimiento_descendente(self, problema_mediano: Problema) -> None:
        """El ranking se entrega de mayor a menor factor de crecimiento."""
        df_bbs = analizar_building_blocks(problema_mediano, top_k=5)
        assert df_bbs["crecimiento"].is_monotonic_decreasing
        assert df_bbs["rank"].tolist() == list(range(1, len(df_bbs) + 1))

    def test_orden_uno_tiene_delta_cero(self, problema_mediano: Problema) -> None:
        """Un esquema de un solo simbolo definido no tiene longitud de definicion."""
        df_bbs = analizar_building_blocks(problema_mediano, top_k=5)
        assert (df_bbs.loc[df_bbs["o_H"] == 1, "delta_H"] == 0).all()

    def test_hay_esquemas_favorecidos(self, problema_mediano: Problema) -> None:
        """Las obras de mejor ratio deben formar bloques con crecimiento > 1."""
        df_bbs = analizar_building_blocks(problema_mediano, top_k=6)
        resumen = resumen_building_blocks(df_bbs)

        assert resumen["n_favorecidos"] > 0
        assert resumen["crecimiento_max"] > 1.0

    def test_supervivencia_coincide_con_la_formula(self) -> None:
        """K_S = (1 - pc*delta/(l-1)) * (1-pm)^o, con l = longitud del cromosoma."""
        l, pc, pm = 326, 0.85, 0.015
        for o_H, delta_H in ((1, 0), (2, 10), (3, 325)):
            esperado = (1 - pc * delta_H / (l - 1)) * (1 - pm) ** o_H
            assert calcular_supervivencia(o_H, delta_H, l, pc, pm) == pytest.approx(esperado)

    def test_supervivencia_decrece_con_el_orden_y_la_longitud(self) -> None:
        """Más símbolos definidos o más dispersos implican menos supervivencia."""
        l, pc, pm = 326, 0.85, 0.015
        por_orden = [calcular_supervivencia(o, 10, l, pc, pm) for o in (1, 2, 3, 4)]
        por_delta = [calcular_supervivencia(2, d, l, pc, pm) for d in (0, 50, 150, 325)]

        assert por_orden == sorted(por_orden, reverse=True)
        assert por_delta == sorted(por_delta, reverse=True)

    def test_supervivencia_rechaza_longitud_invalida(self) -> None:
        """Con l<2 no hay puntos de corte: es el error de pasar top_k como l."""
        with pytest.raises(ValueError, match="l debe ser"):
            calcular_supervivencia(1, 0, 1, 0.85, 0.015)

    def test_rechaza_top_k_mayor_que_n(self, problema_mediano: Problema) -> None:
        """No se pueden tomar más obras de las que hay."""
        with pytest.raises(ValueError, match="top_k"):
            analizar_building_blocks(problema_mediano, top_k=problema_mediano.n + 1)


class TestSensibilidad:
    """Barrido del peso w₁ del riesgo."""

    def test_redistribucion_conserva_proporciones_y_suma_uno(
        self, params_test: Params
    ) -> None:
        """El resto (1-w1) se reparte entre M,P,E,G sin alterar su proporcion relativa."""
        originales = dict(params_test.W)
        base = np.array([originales[a] for a in "MPEG"], dtype=float)
        base = base / base.sum()

        for w1 in (0.10, 0.40, 0.75):
            pesos = redistribuir_pesos(w1, originales)
            nuevos = np.array([pesos[a] for a in "MPEG"], dtype=float)

            assert sum(pesos.values()) == pytest.approx(1.0)
            assert pesos["R"] == pytest.approx(w1)
            assert np.allclose(nuevos / nuevos.sum(), base)

    def test_w1_original_reproduce_los_pesos_del_trabajo(self, params_test: Params) -> None:
        """Con w1 = el del .env, la redistribución devuelve los pesos originales."""
        originales = dict(params_test.W)
        assert redistribuir_pesos(originales["R"], originales) == pytest.approx(originales)

    def test_barrido_devuelve_una_fila_por_valor(
        self, df_mediano, params_test: Params
    ) -> None:
        """Cada w1 del rango produce una fila con su fitness."""
        df_sens = sensibilidad_w1(
            df_mediano, params_test, w1_range=[0.20, 0.40], pop_size=15, n_gen=10
        )

        assert len(df_sens) == 2
        assert {"w1", "fitness_optimo", "es_valor_usado", "similitud_vs_usado"} <= set(
            df_sens.columns
        )
        assert bool(df_sens["es_valor_usado"].sum()) == 1

    def test_similitud_es_uno_en_el_w1_usado(self, df_mediano, params_test: Params) -> None:
        """El portafolio de referencia es idéntico a sí mismo."""
        df_sens = sensibilidad_w1(
            df_mediano, params_test, w1_range=[0.20, 0.40], pop_size=15, n_gen=10
        )
        fila = df_sens.loc[df_sens["es_valor_usado"]]
        assert fila["similitud_vs_usado"].iloc[0] == pytest.approx(1.0)

    def test_resumen_expone_las_claves_de_la_api(
        self, df_mediano, params_test: Params
    ) -> None:
        """El resumen alimenta GET /analysis/sensitivity."""
        df_sens = sensibilidad_w1(
            df_mediano, params_test, w1_range=[0.30, 0.40], pop_size=15, n_gen=10
        )
        resumen = resumen_sensibilidad(df_sens)

        assert {"w1_rango", "fitness_optimo", "w1_usado", "fitness_en_w1_usado"} <= set(
            resumen
        )
        assert resumen["w1_usado"] == pytest.approx(params_test.W["R"])
        assert "advertencia" in resumen

    def test_rechaza_rango_vacio(self, df_mediano, params_test: Params) -> None:
        """Sin valores de w1 no hay barrido."""
        with pytest.raises(ValueError, match="w1_range"):
            sensibilidad_w1(df_mediano, params_test, w1_range=[])


class TestSensibilidadParametros:
    """
    Barrido de los parámetros del AG (pop_size, pc, pm).

    Se usa un grid mínimo, una sola semilla y 5 generaciones: lo que se valida es la
    mecánica del barrido (forma, columnas, marcado del valor usado, no mutación del
    Singleton), no la calidad de las configuraciones, que necesita el grid completo
    y varias semillas.
    """

    #: Grid minimo. pop_size incluye el valor de params_test (20) para poder
    #: comprobar el marcado de es_valor_usado.
    GRID = {"pop_size": (10, 20), "pc": (0.8,), "pm": (0.02,)}

    def _barrido(self, problema: Problema, params: Params) -> pd.DataFrame:
        """
        Ejecuta el barrido mínimo.

        Args:
            problema: Problema MCKP de prueba.
            params: Parámetros de prueba.

        Returns:
            DataFrame del barrido.
        """
        return sensibilidad_parametros(
            problema, params, seeds=(42,), grid=self.GRID, n_gen=5
        )

    def test_forma_y_columnas(self, problema_mediano: Problema, params_test: Params) -> None:
        """Una fila por (parametro, valor) y las columnas acordadas, en orden."""
        df_params = self._barrido(problema_mediano, params_test)

        assert len(df_params) == 2 + 1 + 1
        assert list(df_params.columns) == COLUMNAS_PARAMETROS
        assert df_params["fitness_medio"].notna().all()
        assert (df_params["tiempo_medio_seg"] > 0).all()
        # Con una sola semilla no hay dispersion que medir.
        assert (df_params["fitness_std"] == 0.0).all()

    def test_cubre_los_tres_parametros(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """El barrido recorre pop_size, pc y pm."""
        df_params = self._barrido(problema_mediano, params_test)

        assert set(df_params["parametro"]) == {"pop_size", "pc", "pm"}
        assert sorted(df_params.loc[df_params["parametro"] == "pop_size", "valor"]) == [
            10.0,
            20.0,
        ]

    def test_marca_el_valor_usado(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """
        es_valor_usado marca el valor configurado en params, y solo ese.

        Con params_test.POP_SIZE=20 se marca pop_size=20; pc=0.8 y pm=0.02 no
        coinciden con los del proyecto (0.85 y 0.015), así que no se marcan.
        """
        df_params = self._barrido(problema_mediano, params_test)
        marcadas = df_params.loc[df_params["es_valor_usado"]]

        assert len(marcadas) == 1
        assert marcadas["parametro"].iloc[0] == "pop_size"
        assert marcadas["valor"].iloc[0] == pytest.approx(float(params_test.POP_SIZE))
        assert not df_params.loc[df_params["parametro"] == "pc", "es_valor_usado"].any()

    def test_no_muta_el_singleton_ni_los_params(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Los overrides van en la instancia del AG, no en la configuracion."""
        singleton = Params.get()
        antes_global = (singleton.POP_SIZE, singleton.PC, singleton.PM, singleton.N_GEN)
        antes_test = (params_test.POP_SIZE, params_test.PC, params_test.PM, params_test.N_GEN)

        self._barrido(problema_mediano, params_test)

        assert (singleton.POP_SIZE, singleton.PC, singleton.PM, singleton.N_GEN) == antes_global
        assert (
            params_test.POP_SIZE,
            params_test.PC,
            params_test.PM,
            params_test.N_GEN,
        ) == antes_test

    def test_resumen_compara_mejor_contra_usado(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """El resumen expone el mejor valor y su comparación con el usado."""
        df_params = self._barrido(problema_mediano, params_test)
        resumen = resumen_sensibilidad_parametros(df_params)

        assert resumen["n_configuraciones"] == len(df_params)
        assert set(resumen["por_parametro"]) == {"pop_size", "pc", "pm"}
        assert "nota_metodologica" in resumen

        detalle = resumen["por_parametro"]["pop_size"]
        assert detalle["valor_usado"] == pytest.approx(float(params_test.POP_SIZE))
        assert detalle["mejor_valor"] in (10.0, 20.0)
        assert detalle["factor_tiempo_vs_usado"] > 0
        assert isinstance(detalle["el_usado_es_el_mejor"], bool)
        # Sin el valor del proyecto en el grid, no se puede comparar.
        assert resumen["por_parametro"]["pc"]["valor_usado"] is None

    def test_rechaza_entradas_invalidas(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """Semillas vacías, n_gen no positivo o un parámetro inexistente son errores."""
        with pytest.raises(ValueError, match="seeds"):
            sensibilidad_parametros(problema_mediano, params_test, seeds=(), grid=self.GRID)
        with pytest.raises(ValueError, match="n_gen"):
            sensibilidad_parametros(
                problema_mediano, params_test, seeds=(1,), grid=self.GRID, n_gen=0
            )
        with pytest.raises(ValueError, match="no soportados"):
            sensibilidad_parametros(
                problema_mediano, params_test, seeds=(1,), grid={"k_torneo": (2, 3)}, n_gen=5
            )
        with pytest.raises(ValueError, match="df_params esta vacio"):
            resumen_sensibilidad_parametros(pd.DataFrame())


class TestIntegracionGreedy:
    """El Greedy como referencia del contraste."""

    def test_greedy_es_determinista_y_factible(self, problema_mediano: Problema) -> None:
        """Dos llamadas dan el mismo portafolio: es lo que hace limpia la hipotesis nula."""
        primera = greedy(problema_mediano)
        segunda = greedy(problema_mediano)

        assert np.array_equal(primera.x, segunda.x)
        assert primera.factible
