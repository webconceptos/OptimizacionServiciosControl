"""
Representación de una solución del MCKP.

`Solucion` envuelve un cromosoma binario junto con el `Problema` que le da
sentido, y expone de forma perezosa las magnitudes que consumen el análisis, las
figuras y la API. No contiene lógica evolutiva: los algoritmos de `algorithms/`
producen cromosomas y los envuelven aquí al terminar.

Referencia: Tema 1 y Tema 5 del curso CE UNI 2026
            README.md, formulacion del MCKP — docs/API_SPEC.md (SolucionResponse)
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.problem import Problema

logger = logging.getLogger(__name__)

#: Columna con el código único de inversión de cada obra.
COLUMNA_CODIGO: str = "codigo"

#: Columna con el score de riesgo de la Parte 1.
COLUMNA_RIESGO: str = "R"


class Solucion:
    """
    Un portafolio de obras seleccionadas para control preventivo.

    El cromosoma se **copia** al construir: el ciclo evolutivo reutiliza y muta
    sus arrays, así que guardar una referencia haría que la solución cambiara
    sola después de devolverla.

    Las magnitudes derivadas son `cached_property`: se calculan la primera vez
    que se consultan y quedan memorizadas, lo que permite pasar la misma
    `Solucion` a varias figuras y endpoints sin recomputar el fitness.

    Atributos:
        x: Cromosoma binario (np.int8) de longitud problema.n.
        problema: Problema MCKP que define la aptitud y las restricciones.

    Referencia: Tema 1 y Tema 5 del curso CE UNI 2026.
    """

    def __init__(self, x: np.ndarray, problema: Problema, alpha: Optional[float] = None) -> None:
        """
        Envuelve un cromosoma con su problema.

        Args:
            x: Cromosoma binario de longitud problema.n.
            problema: Instancia de `Problema` sobre la que se evalúa.
            alpha: Coeficiente de penalización; None usa `problema.alpha`.

        Raises:
            ValueError: Si la longitud del cromosoma no coincide con problema.n,
                o si contiene valores distintos de 0 y 1.
        """
        xv = np.asarray(x)
        if xv.shape != (problema.n,):
            raise ValueError(
                f"El cromosoma debe tener forma ({problema.n},), se recibió {xv.shape}"
            )
        self.x: np.ndarray = xv.astype(np.int8, copy=True)
        if not np.all((self.x == 0) | (self.x == 1)):
            raise ValueError("El cromosoma solo admite valores binarios (0 o 1)")

        self.problema: Problema = problema
        self.alpha: float = problema.alpha if alpha is None else float(alpha)

    # --------------------------------------------------------- Propiedades

    @cached_property
    def fitness(self) -> float:
        """Aptitud penalizada: Z(X) − α·violacion(X)."""
        return self.problema.fitness(self.x, alpha=self.alpha)

    @cached_property
    def z(self) -> float:
        """Valor de la función objetivo sin penalización."""
        return self.problema.Z(self.x)

    @cached_property
    def violacion(self) -> float:
        """Violación normalizada total de R1, R2 y R3 (0 si es factible)."""
        return self.problema.violacion(self.x)

    @cached_property
    def factible(self) -> bool:
        """True si la solución cumple R1, R2 y R3 simultáneamente."""
        return self.problema.es_factible(self.x)

    @cached_property
    def n_seleccionadas(self) -> int:
        """Número de obras incluidas en el portafolio."""
        return int(self.x.sum())

    @cached_property
    def costo(self) -> float:
        """Costo total del servicio de control, en las unidades de la columna C."""
        return self.problema.costo(self.x)

    @cached_property
    def r_medio(self) -> float:
        """
        Riesgo promedio Rᵢ de las obras seleccionadas.

        Usa el Rᵢ crudo (no el normalizado), porque es la magnitud interpretable
        institucionalmente: P(Extrem. Riesgosa) del modelo de la Parte 1.
        Devuelve 0.0 si no hay ninguna obra seleccionada.
        """
        if self.n_seleccionadas == 0:
            return 0.0
        riesgos = self.problema.df[COLUMNA_RIESGO].to_numpy(dtype=float)
        return float(riesgos[self.indices].mean())

    @cached_property
    def indices(self) -> np.ndarray:
        """Índices (posiciones en el dataset) de las obras seleccionadas."""
        return np.flatnonzero(self.x == 1)

    @cached_property
    def codigos(self) -> List[str]:
        """Códigos CUI de las obras seleccionadas, en orden del dataset."""
        if COLUMNA_CODIGO not in self.problema.df.columns:
            return [str(i) for i in self.indices]
        return self.problema.df[COLUMNA_CODIGO].iloc[self.indices].astype(str).tolist()

    @cached_property
    def distribucion_territorial(self) -> Dict[int, int]:
        """Obras seleccionadas por macroregión (restricción R3)."""
        return self.problema.distribucion_territorial(self.x)

    # -------------------------------------------------------------- Métodos

    def obras_df(self, ordenar_por_beneficio: bool = True) -> pd.DataFrame:
        """
        Devuelve el subconjunto de obras seleccionadas.

        Args:
            ordenar_por_beneficio: Si True, ordena de mayor a menor beneficio
                unitario bᵢ (útil para el ranking de obras priorizadas); si es
                False conserva el orden del dataset.

        Returns:
            Copia del subconjunto seleccionado, con una columna extra `beneficio`
            con el bᵢ ponderado de cada obra.
        """
        sub = self.problema.df.iloc[self.indices].copy()
        sub["beneficio"] = np.asarray(self.problema.beneficio())[self.indices]
        if ordenar_por_beneficio:
            sub = sub.sort_values("beneficio", ascending=False)
        return sub

    def to_dict(self, incluir_obras: bool = True) -> Dict[str, object]:
        """
        Serializa la solución para la API y para `metricas.json`.

        Las claves siguen el contrato de `SolucionResponse` en docs/API_SPEC.md;
        `tiempo_seg` no se incluye porque lo aporta el algoritmo que la produjo.

        Args:
            incluir_obras: Si False omite la lista de códigos CUI, para respuestas
                compactas (por ejemplo, un frente de Pareto de 45 soluciones).

        Returns:
            Diccionario con tipos nativos de Python, serializable a JSON.
        """
        datos: Dict[str, object] = {
            "fitness": round(self.fitness, 6),
            "z": round(self.z, 6),
            "violacion": round(self.violacion, 6),
            "n_seleccionadas": self.n_seleccionadas,
            "costo_total": round(self.costo, 4),
            "r_medio": round(self.r_medio, 4),
            "factible": self.factible,
            "distribucion_territorial": {str(r): n for r, n in self.distribucion_territorial.items()},
        }
        if incluir_obras:
            datos["obras_seleccionadas"] = self.codigos
        return datos

    def copiar(self) -> "Solucion":
        """
        Crea una copia independiente de esta solución.

        Returns:
            Nueva `Solucion` con el mismo cromosoma y problema.
        """
        return Solucion(self.x, self.problema, alpha=self.alpha)

    def __len__(self) -> int:
        """Longitud del cromosoma (número de obras candidatas)."""
        return int(self.x.size)

    def __eq__(self, otro: object) -> bool:
        """Dos soluciones son iguales si comparten problema y cromosoma."""
        if not isinstance(otro, Solucion):
            return NotImplemented
        return otro.problema is self.problema and bool(np.array_equal(self.x, otro.x))

    def __hash__(self) -> int:
        """Hash del cromosoma, para deduplicar frentes de Pareto."""
        return hash(self.x.tobytes())

    def __repr__(self) -> str:
        """Resumen de una línea de la solución."""
        estado = "factible" if self.factible else f"infactible(v={self.violacion:.4f})"
        return (
            f"Solucion(fitness={self.fitness:.6f}, n={self.n_seleccionadas}, "
            f"costo={self.costo:.4f}, r_medio={self.r_medio:.4f}, {estado})"
        )
