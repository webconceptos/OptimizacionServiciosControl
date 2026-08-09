"""
Tests de `core/problem.py` y `core/solution.py`.

Cubren la función objetivo, la normalización y la detección de las tres
restricciones institucionales por separado.

Referencia: Tema 1 y Tema 5 del curso CE UNI 2026
            los criterios de cobertura del proyecto
"""

from __future__ import annotations

import numpy as np
import pytest

from config.params import Params
from core.problem import ATRIBUTOS, Problema
from core.solution import Solucion


class TestFuncionObjetivo:
    """Función objetivo y aptitud penalizada."""

    def test_fitness_solucion_vacia_es_negativo(
        self, problema_mediano: Problema, cromosoma_vacio: np.ndarray
    ) -> None:
        """Sin obras, Z=0 y solo queda la penalización de R3: el fitness es negativo."""
        fitness = problema_mediano.fitness(cromosoma_vacio)
        n_regiones = len(problema_mediano.m_r)
        esperado = -problema_mediano.alpha * n_regiones

        assert problema_mediano.Z(cromosoma_vacio) == 0.0
        assert fitness < 0.0
        assert fitness == pytest.approx(esperado)

    def test_fitness_factible_es_positivo(self, solucion_factible: Solucion) -> None:
        """Una solución factible no tiene penalización, así que su fitness es Z > 0."""
        assert solucion_factible.factible
        assert solucion_factible.fitness > 0.0
        assert solucion_factible.fitness == pytest.approx(solucion_factible.z)

    def test_z_es_suma_de_beneficios_seleccionados(
        self, solucion_factible: Solucion, problema_mediano: Problema
    ) -> None:
        """Z(X) coincide con la suma de los beneficios de las obras activas."""
        beneficio = np.asarray(problema_mediano.beneficio())
        esperado = float(beneficio[solucion_factible.indices].sum())
        assert solucion_factible.z == pytest.approx(esperado)

    def test_alpha_explicito_sobrescribe_el_de_params(
        self, problema_mediano: Problema, cromosoma_vacio: np.ndarray
    ) -> None:
        """El alpha del argumento manda sobre el de params."""
        con_defecto = problema_mediano.fitness(cromosoma_vacio)
        con_alpha_10 = problema_mediano.fitness(cromosoma_vacio, alpha=10.0)
        assert con_alpha_10 < con_defecto


class TestNormalizacion:
    """Normalización min-max de los atributos."""

    def test_atributos_normalizados_en_cero_uno(self, problema_mediano: Problema) -> None:
        """Cada atributo normalizado cae en [0, 1] y toca ambos extremos."""
        for atributo in ATRIBUTOS:
            valores = problema_mediano.df[f"{atributo}_n"].to_numpy(dtype=float)
            assert valores.min() == pytest.approx(0.0)
            assert valores.max() == pytest.approx(1.0)

    def test_no_muta_el_dataframe_original(self, df_mediano, params_test: Params) -> None:
        """Construir el problema no añade columnas al DataFrame que recibe."""
        columnas_antes = list(df_mediano.columns)
        Problema(df_mediano, params_test)
        assert list(df_mediano.columns) == columnas_antes

    def test_beneficio_es_de_solo_lectura(self, problema_mediano: Problema) -> None:
        """El vector de beneficios está cacheado: no debe poder mutarse por fuera."""
        with pytest.raises(ValueError):
            problema_mediano.beneficio()[0] = 99.0

    def test_presupuesto_es_fraccion_del_costo_total(
        self, problema_mediano: Problema, params_test: Params
    ) -> None:
        """B = sum(C) * PRESUPUESTO_PCT, en las unidades de la columna C (kS/)."""
        esperado = float(problema_mediano.C.sum()) * params_test.PRESUPUESTO_PCT
        assert problema_mediano.B == pytest.approx(esperado)


class TestRestricciones:
    """Detección de R1, R2 y R3 por separado."""

    def test_violacion_cero_si_es_factible(self, solucion_factible: Solucion) -> None:
        """Una solución factible tiene violación exactamente 0 en las tres."""
        detalle = solucion_factible.problema.violaciones_detalle(solucion_factible.x)
        assert solucion_factible.violacion == 0.0
        assert detalle["R1"] == 0.0
        assert detalle["R2"] == 0.0
        assert detalle["R3"] == 0.0

    def test_restriccion_R1_detectada_al_superar_el_presupuesto(
        self, problema_mediano: Problema
    ) -> None:
        """Seleccionar todas las obras excede B y R1 se activa."""
        todas = np.ones(problema_mediano.n, dtype=np.int8)
        detalle = problema_mediano.violaciones_detalle(todas)

        assert problema_mediano.costo(todas) > problema_mediano.B
        assert detalle["R1"] > 0.0
        assert not problema_mediano.es_factible(todas)

    def test_restriccion_R2_detectada_al_superar_K(
        self, problema_mediano: Problema
    ) -> None:
        """Activar mas de K obras baratas dispara R2 sin tocar R1."""
        mas_baratas = np.argsort(problema_mediano.C)[: problema_mediano.K + 10]
        x = np.zeros(problema_mediano.n, dtype=np.int8)
        x[mas_baratas] = 1
        detalle = problema_mediano.violaciones_detalle(x)

        assert int(x.sum()) == problema_mediano.K + 10
        assert detalle["R2"] > 0.0
        assert detalle["R2"] == pytest.approx(10 / problema_mediano.K)
        assert not problema_mediano.es_factible(x)

    def test_restriccion_R3_detectada_al_omitir_una_macroregion(
        self, solucion_factible: Solucion
    ) -> None:
        """Vaciar una macroregión de una solución factible activa solo R3."""
        problema = solucion_factible.problema
        region = sorted(problema.m_r)[0]
        x = solucion_factible.x.copy()
        x[problema.mr == region] = 0

        detalle = problema.violaciones_detalle(x)
        assert problema.distribucion_territorial(x)[region] == 0
        assert detalle["R3"] > 0.0
        assert detalle["R3_por_region"][region] == pytest.approx(1.0)
        assert detalle["R1"] == 0.0
        assert detalle["R2"] == 0.0
        assert not problema.es_factible(x)

    def test_violacion_total_suma_los_tres_terminos(
        self, problema_mediano: Problema
    ) -> None:
        """violacion() es la suma de los términos normalizados de R1, R2 y R3."""
        todas = np.ones(problema_mediano.n, dtype=np.int8)
        detalle = problema_mediano.violaciones_detalle(todas)
        assert problema_mediano.violacion(todas) == pytest.approx(
            detalle["R1"] + detalle["R2"] + detalle["R3"]
        )

    def test_problema_de_10_obras_es_infactible_por_construccion(
        self, problema_small: Problema
    ) -> None:
        """
        Documenta la limitación del fixture pequeño.

        Con 10 obras, R3 exige MIN_POR_REGION x 5 = 25 obras: ninguna solución puede
        ser factible. Si este test empieza a fallar, es que cambiaron m_r o
        MIN_POR_REGION y los fixtures hay que revisarlos.
        """
        exigidas = sum(problema_small.m_r.values())
        assert exigidas > problema_small.n
        assert not problema_small.es_factible(np.ones(problema_small.n, dtype=np.int8))
        assert problema_small.violacion(np.ones(problema_small.n, dtype=np.int8)) > 0.0


class TestSolucion:
    """Clase `Solucion`."""

    def test_copia_el_cromosoma(self, problema_mediano: Problema) -> None:
        """Mutar el array de entrada después no debe alterar la solución."""
        x = np.zeros(problema_mediano.n, dtype=np.int8)
        x[:5] = 1
        solucion = Solucion(x, problema_mediano)
        x[:] = 0
        assert solucion.n_seleccionadas == 5

    def test_rechaza_cromosoma_de_longitud_incorrecta(
        self, problema_mediano: Problema
    ) -> None:
        """Un cromosoma de otra longitud es un error de programación, no un caso válido."""
        with pytest.raises(ValueError, match="forma"):
            Solucion(np.zeros(5, dtype=np.int8), problema_mediano)

    def test_rechaza_cromosoma_no_binario(self, problema_mediano: Problema) -> None:
        """Solo se admiten valores 0 y 1."""
        with pytest.raises(ValueError, match="binarios"):
            Solucion(np.full(problema_mediano.n, 2, dtype=np.int8), problema_mediano)

    def test_r_medio_usa_el_riesgo_crudo(self, solucion_factible: Solucion) -> None:
        """r_medio es el promedio de R (no de R_n) de las obras elegidas."""
        riesgos = solucion_factible.problema.df["R"].to_numpy(dtype=float)
        esperado = float(riesgos[solucion_factible.indices].mean())
        assert solucion_factible.r_medio == pytest.approx(esperado)
        assert 0.0 <= solucion_factible.r_medio <= 1.0

    def test_r_medio_de_solucion_vacia_es_cero(
        self, problema_mediano: Problema, cromosoma_vacio: np.ndarray
    ) -> None:
        """Sin obras seleccionadas, r_medio es 0.0 y no NaN."""
        assert Solucion(cromosoma_vacio, problema_mediano).r_medio == 0.0

    def test_to_dict_es_serializable(self, solucion_factible: Solucion) -> None:
        """to_dict devuelve tipos nativos, listos para JSON y para la API."""
        import json

        datos = solucion_factible.to_dict()
        json.dumps(datos)  # no debe lanzar
        assert datos["n_seleccionadas"] == solucion_factible.n_seleccionadas
        assert len(datos["obras_seleccionadas"]) == solucion_factible.n_seleccionadas
        assert set(datos) >= {"fitness", "n_seleccionadas", "costo_total", "factible"}

    def test_obras_df_devuelve_las_seleccionadas(self, solucion_factible: Solucion) -> None:
        """obras_df tiene una fila por obra elegida y trae el beneficio."""
        sub = solucion_factible.obras_df()
        assert len(sub) == solucion_factible.n_seleccionadas
        assert "beneficio" in sub.columns
        assert sub["beneficio"].is_monotonic_decreasing
