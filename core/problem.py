"""
Definición del problema de optimización MCKP.

Encapsula la función objetivo multicriterio, las tres restricciones
institucionales y la función de aptitud con penalización estática. Es el único
módulo que conoce la formulación matemática del problema: los algoritmos de
`algorithms/` solo llaman a `fitness()` y `es_factible()`.

Referencia: Tema 1 y Tema 5 del curso CE UNI 2026
            (Tema 1: formulación del problema y función objetivo;
             Tema 5: manejo de restricciones por penalización y reparación)
            README.md, formulacion del MCKP — docs/ARCHITECTURE.md "core/problem.py"
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config.params import Params

logger = logging.getLogger(__name__)

#: Atributos que componen la función objetivo, en el orden de los pesos w₁..w₅.
ATRIBUTOS: List[str] = ["R", "M", "P", "E", "G"]

#: Sufijo de las columnas normalizadas que agrega esta clase (R_n, M_n, ...).
SUFIJO_NORMALIZADO: str = "_n"

#: Columna con la macroregión usada por la restricción territorial R3.
COLUMNA_REGION: str = "macroregion"

#: Columna con el costo del servicio de control usado por R1.
COLUMNA_COSTO: str = "C"


def _min_max(valores: np.ndarray, nombre: str) -> np.ndarray:
    """
    Escala un vector al rango [0, 1] por min-max.

    Args:
        valores: Vector de atributos crudos.
        nombre: Nombre de la columna, para el mensaje de advertencia.

    Returns:
        Vector normalizado en [0, 1]. Si la columna es constante devuelve ceros,
        porque un atributo sin variabilidad no aporta información discriminante.
    """
    minimo = float(np.min(valores))
    maximo = float(np.max(valores))
    rango = maximo - minimo
    if rango <= 0.0:
        # Mensajes de log en ASCII: la consola de Windows (cp1252) escapa los
        # caracteres no representables y vuelve la salida ilegible.
        logger.warning(
            "El atributo '%s' es constante (=%.6g): su contribucion normalizada sera 0",
            nombre,
            minimo,
        )
        return np.zeros_like(valores, dtype=float)
    return (valores - minimo) / rango


class Problema:
    """
    Problema MCKP de selección de obras públicas para control preventivo.

    Formulación (README.md, formulacion del MCKP):

        maximizar   Z(X) = Σᵢ (w₁·R̃ᵢ + w₂·M̃ᵢ + w₃·P̃ᵢ + w₄·Ẽᵢ + w₅·G̃ᵢ)·xᵢ
        sujeto a    R1: Σᵢ Cᵢ·xᵢ ≤ B = Σᵢ Cᵢ · PRESUPUESTO_PCT
                    R2: Σᵢ xᵢ ≤ K
                    R3: Σ{xᵢ : macroregión(i)=r} ≥ mᵣ   ∀r ∈ {1..5}
        con         xᵢ ∈ {0,1}

    La aptitud aplica penalización estática (Tema 5):

        fitness(X) = Z(X) − α · violacion(X)

    Los atributos R, M, P, E, G se normalizan por min-max al instanciar, sobre el
    dataset efectivamente recibido: si cambia el dataset, la escala se recalcula
    automáticamente y los pesos siguen siendo comparables entre sí.

    R3 es un supuesto institucional de política de control de la CGR, no un hecho
    derivado de los datos reales de la Parte 1 (ver `data/generator.py`).

    Atributos:
        df: Copia del dataset con las columnas normalizadas R_n..G_n añadidas.
        n: Número de obras candidatas (longitud del cromosoma).
        B: Presupuesto máximo de supervisión (R1), en las unidades de C.
        K: Máximo de obras seleccionables (R2).
        m_r: Mínimo de obras por macroregión (R3).
        alpha: Coeficiente de penalización estática por defecto.
        C: Vector de costos de control.
        mr: Vector de macroregiones.

    Referencia: Tema 1 y Tema 5 del curso CE UNI 2026.
    """

    def __init__(self, df: pd.DataFrame, params: Optional[Params] = None) -> None:
        """
        Construye el problema a partir del dataset y los parámetros.

        Args:
            df: Dataset de obras con las columnas R, M, P, E, G, C y macroregion.
            params: Parámetros del proyecto; None usa el Singleton `Params.get()`.

        Raises:
            ValueError: Si faltan columnas requeridas, el dataset está vacío o
                los costos no son positivos.
        """
        self.params: Params = params if params is not None else Params.get()

        if df is None or len(df) == 0:
            raise ValueError("El dataset de obras está vacío")
        requeridas = ATRIBUTOS + [COLUMNA_COSTO, COLUMNA_REGION]
        faltantes = [c for c in requeridas if c not in df.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas en el dataset: {faltantes}")

        self.df: pd.DataFrame = df.copy().reset_index(drop=True)
        self.n: int = len(self.df)

        # --- Normalización min-max de los atributos de la función objetivo ---
        for atributo in ATRIBUTOS:
            crudo = self.df[atributo].to_numpy(dtype=float)
            self.df[f"{atributo}{SUFIJO_NORMALIZADO}"] = _min_max(crudo, atributo)

        # Matriz (n x 5) de atributos normalizados, en el orden de los pesos.
        self._A: np.ndarray = np.column_stack(
            [self.df[f"{a}{SUFIJO_NORMALIZADO}"].to_numpy(dtype=float) for a in ATRIBUTOS]
        )

        # --- Vectores del problema -------------------------------------------
        self.C: np.ndarray = self.df[COLUMNA_COSTO].to_numpy(dtype=float)
        self.mr: np.ndarray = self.df[COLUMNA_REGION].to_numpy(dtype=int)
        if np.any(self.C <= 0):
            raise ValueError("Todos los costos C deben ser positivos")

        # --- R1: presupuesto como fracción del costo total del universo ------
        self.B: float = float(self.C.sum()) * float(self.params.PRESUPUESTO_PCT)

        # --- R2: capacidad operativa -----------------------------------------
        self.K: int = int(self.params.K_MAX)

        # --- R3: mínimo por macroregión --------------------------------------
        self.m_r: Dict[int, int] = dict(self.params.M_R)
        self._regiones: List[int] = sorted(self.m_r)
        # Matriz indicadora (n_regiones x n): conteos = _R_ind @ x en un solo paso.
        self._R_ind: np.ndarray = np.vstack(
            [(self.mr == r).astype(float) for r in self._regiones]
        )
        self._min_region: np.ndarray = np.array(
            [self.m_r[r] for r in self._regiones], dtype=float
        )
        if np.any(self._min_region <= 0):
            raise ValueError("Los mínimos por macroregión (m_r) deben ser positivos")

        self.alpha: float = float(self.params.ALPHA)

        # --- Pesos y beneficio unitario (cacheado) ---------------------------
        self._w: np.ndarray = np.array(
            [self.params.W[a] for a in ATRIBUTOS], dtype=float
        )
        self._recalcular_beneficio()

        self._validar_consistencia()
        logger.info(
            "Problema MCKP | n=%d | B=%.4f | K=%d | m_r=%s | alpha=%.2f | w=%s",
            self.n,
            self.B,
            self.K,
            self.m_r,
            self.alpha,
            np.round(self._w, 4).tolist(),
        )

    # ------------------------------------------------------------- Pesos

    @property
    def w(self) -> np.ndarray:
        """
        Vector de pesos (w₁..w₅) en el orden R, M, P, E, G.

        Returns:
            Copia del vector de pesos, para que mutarla no invalide la caché.
        """
        return self._w.copy()

    @w.setter
    def w(self, pesos: Sequence[float] | Dict[str, float]) -> None:
        """
        Reemplaza los pesos y recalcula el beneficio unitario cacheado.

        Lo usa el análisis de sensibilidad, que barre w₁ redistribuyendo el resto.

        Args:
            pesos: Secuencia de 5 pesos en el orden R,M,P,E,G, o dict por atributo.

        Raises:
            ValueError: Si la secuencia no tiene 5 elementos.
            KeyError: Si el dict trae claves fuera de {R,M,P,E,G}.
        """
        if isinstance(pesos, dict):
            desconocidas = set(pesos) - set(ATRIBUTOS)
            if desconocidas:
                raise KeyError(f"Claves de peso desconocidas: {sorted(desconocidas)}")
            nuevos = self._w.copy()
            for i, atributo in enumerate(ATRIBUTOS):
                if atributo in pesos:
                    nuevos[i] = float(pesos[atributo])
        else:
            nuevos = np.asarray(pesos, dtype=float)
            if nuevos.shape != (len(ATRIBUTOS),):
                raise ValueError(
                    f"Se esperaban {len(ATRIBUTOS)} pesos (R,M,P,E,G), "
                    f"se recibieron {nuevos.shape}"
                )
        self._w = nuevos
        self._recalcular_beneficio()
        logger.debug("Pesos actualizados: %s", np.round(self._w, 4).tolist())

    def _recalcular_beneficio(self) -> None:
        """Recalcula el beneficio unitario y el ratio beneficio/costo cacheados."""
        self._beneficio: np.ndarray = self._A @ self._w
        self._ratio: np.ndarray = self._beneficio / self.C

    # -------------------------------------------------- Función objetivo

    def beneficio(self) -> np.ndarray:
        """
        Beneficio unitario ponderado de cada obra.

        bᵢ = w₁·R̃ᵢ + w₂·M̃ᵢ + w₃·P̃ᵢ + w₄·Ẽᵢ + w₅·G̃ᵢ, con atributos normalizados.
        Se calcula una sola vez al instanciar (o al cambiar los pesos), porque el
        ciclo evolutivo lo consulta decenas de miles de veces.

        Returns:
            Vector de beneficios de longitud n. Es una vista de solo lectura: no
            modificar in-place (usar el setter de `w` para cambiar los pesos).

        Referencia: Tema 1 del curso CE UNI 2026 (función objetivo).
        """
        vista = self._beneficio.view()
        vista.flags.writeable = False
        return vista

    def ratio(self) -> np.ndarray:
        """
        Ratio beneficio/costo de cada obra, bᵢ/Cᵢ.

        Base del operador de reparación greedy y del benchmark Greedy.

        Returns:
            Vector de ratios de longitud n, de solo lectura.

        Referencia: Tema 5 del curso CE UNI 2026 (heurística de reparación).
        """
        vista = self._ratio.view()
        vista.flags.writeable = False
        return vista

    def Z(self, x: np.ndarray) -> float:
        """
        Valor de la función objetivo, SIN penalización.

        Args:
            x: Cromosoma binario de longitud n.

        Returns:
            Z(X) = Σᵢ bᵢ·xᵢ.

        Referencia: Tema 1 del curso CE UNI 2026.
        """
        return float(self._beneficio @ self._vector(x))

    def costo(self, x: np.ndarray) -> float:
        """
        Costo total del servicio de control de la solución.

        Args:
            x: Cromosoma binario de longitud n.

        Returns:
            Σᵢ Cᵢ·xᵢ, en las mismas unidades que la columna C.
        """
        return float(self.C @ self._vector(x))

    # --------------------------------------------------- Restricciones

    def violacion(self, x: np.ndarray) -> float:
        """
        Suma de las violaciones normalizadas de R1, R2 y R3.

        Cada término se normaliza por la magnitud de su propia restricción, de
        modo que restricciones con unidades distintas (miles de soles, número de
        obras) sean comparables y ningún término domine artificialmente:

            v_R1 = max(0, (ΣCᵢxᵢ − B) / B)
            v_R2 = max(0, (Σxᵢ − K) / K)
            v_R3 = Σᵣ max(0, (mᵣ − nᵣ(X)) / mᵣ)

        Args:
            x: Cromosoma binario de longitud n.

        Returns:
            Violación total ≥ 0. Es 0 si y solo si la solución es factible.

        Referencia: Tema 5 del curso CE UNI 2026 (manejo de restricciones).
        """
        xv = self._vector(x)
        v_r1 = max(0.0, (float(self.C @ xv) - self.B) / self.B)
        v_r2 = max(0.0, (float(xv.sum()) - self.K) / self.K)
        conteos = self._R_ind @ xv
        v_r3 = float(np.maximum(0.0, (self._min_region - conteos) / self._min_region).sum())
        return v_r1 + v_r2 + v_r3

    def violaciones_detalle(self, x: np.ndarray) -> Dict[str, float]:
        """
        Desglosa la violación por restricción, para diagnóstico y tests.

        Args:
            x: Cromosoma binario de longitud n.

        Returns:
            Diccionario con las claves R1, R2, R3 y total (violaciones
            normalizadas), más R3_por_region con el detalle por macroregión.
        """
        xv = self._vector(x)
        v_r1 = max(0.0, (float(self.C @ xv) - self.B) / self.B)
        v_r2 = max(0.0, (float(xv.sum()) - self.K) / self.K)
        conteos = self._R_ind @ xv
        por_region = np.maximum(0.0, (self._min_region - conteos) / self._min_region)
        return {
            "R1": v_r1,
            "R2": v_r2,
            "R3": float(por_region.sum()),
            "R3_por_region": {
                int(r): float(v) for r, v in zip(self._regiones, por_region)
            },
            "total": v_r1 + v_r2 + float(por_region.sum()),
        }

    def es_factible(self, x: np.ndarray) -> bool:
        """
        Indica si la solución cumple simultáneamente R1, R2 y R3.

        Args:
            x: Cromosoma binario de longitud n.

        Returns:
            True si no viola ninguna restricción.

        Referencia: Tema 5 del curso CE UNI 2026.
        """
        xv = self._vector(x)
        if float(self.C @ xv) > self.B:
            return False
        if float(xv.sum()) > self.K:
            return False
        return bool(np.all(self._R_ind @ xv >= self._min_region))

    def distribucion_territorial(self, x: np.ndarray) -> Dict[int, int]:
        """
        Cuenta las obras seleccionadas por macroregión.

        Args:
            x: Cromosoma binario de longitud n.

        Returns:
            Diccionario {macroregión: número de obras seleccionadas}.
        """
        conteos = self._R_ind @ self._vector(x)
        return {int(r): int(c) for r, c in zip(self._regiones, conteos)}

    # ------------------------------------------------ Función de aptitud

    def fitness(self, x: np.ndarray, alpha: Optional[float] = None) -> float:
        """
        Aptitud con penalización estática de las restricciones violadas.

        fitness(X) = Z(X) − α · violacion(X)

        Alcance real de α = 2.5 (matiza docs/ARCHITECTURE.md, "¿Por qué alpha=2.5?"):
        la penalización NO garantiza que toda solución infactible quede por debajo
        de la peor factible. Como Z crece con el número de obras seleccionadas
        mientras la violación está normalizada, seleccionar en exceso puede rendir
        más de lo que se penaliza: con n=326, X = todo unos da violacion ≈ 7.38 y
        fitness ≈ 92, muy por encima de una solución factible (≈ 24.5). Haría falta
        α ≳ 12 para dominar ese caso extremo.

        Lo que α = 2.5 sí hace, y es lo que necesita el AG:
          * penalizar con fuerza R3, la única restricción que la reparación greedy
            no corrige (dejar una macroregión sin cubrir cuesta hasta 2.5);
          * mantener la penalización en la escala de Z, sin aplastar el gradiente
            de búsqueda como haría una penalización enorme.

        La factibilidad de R1 y R2 la garantiza el operador de reparación greedy
        (Tema 5), que se aplica a cada descendiente antes de evaluarlo, de modo
        que las soluciones con exceso de costo u obras nunca llegan a competir.

        Args:
            x: Cromosoma binario de longitud n.
            alpha: Coeficiente de penalización. None usa `self.alpha`, tomado de
                `params.ALPHA` (2.5 por defecto en el `.env`).

        Returns:
            Valor de aptitud; puede ser negativo si la violación es grande.

        Referencia: Tema 1 y Tema 5 del curso CE UNI 2026.
        """
        coef = self.alpha if alpha is None else float(alpha)
        return self.Z(x) - coef * self.violacion(x)

    # ------------------------------------------------------------ Internos

    def _vector(self, x: np.ndarray) -> np.ndarray:
        """
        Valida y adapta un cromosoma para las operaciones vectoriales.

        Args:
            x: Cromosoma binario (int8, int, bool o float).

        Returns:
            El cromosoma como np.ndarray 1-D.

        Raises:
            ValueError: Si la longitud no coincide con n.
        """
        xv = np.asarray(x)
        if xv.shape != (self.n,):
            raise ValueError(
                f"El cromosoma debe tener forma ({self.n},), se recibió {xv.shape}"
            )
        return xv

    def _validar_consistencia(self) -> None:
        """Advierte si las restricciones son mutuamente inalcanzables."""
        minimo_exigido = int(self._min_region.sum())
        if self.K < minimo_exigido:
            logger.warning(
                "R2 y R3 son incompatibles: K=%d < suma(m_r)=%d (no existe solucion factible)",
                self.K,
                minimo_exigido,
            )
        for r, minimo in self.m_r.items():
            disponibles = int((self.mr == r).sum())
            if disponibles < minimo:
                logger.warning(
                    "R3 inalcanzable en la macroregion %d: hay %d obras y se exigen %d",
                    r,
                    disponibles,
                    minimo,
                )
        costo_minimo = float(np.sort(self.C)[:minimo_exigido].sum())
        if costo_minimo > self.B:
            logger.warning(
                "R1 y R3 son incompatibles: cubrir suma(m_r)=%d obras cuesta al menos "
                "%.4f y B=%.4f",
                minimo_exigido,
                costo_minimo,
                self.B,
            )

    def __repr__(self) -> str:
        """Resumen de una línea del problema."""
        return (
            f"Problema(n={self.n}, B={self.B:.4f}, K={self.K}, "
            f"m_r={list(self.m_r.values())}, alpha={self.alpha})"
        )
