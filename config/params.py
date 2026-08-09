"""
Parámetros centralizados del proyecto (patrón Singleton).

Este módulo es la única fuente de verdad de la configuración: lee todas las
variables desde el archivo `.env` con python-dotenv y las expone como atributos
tipados. Ningún otro módulo debe hard-codear parámetros del problema ni del
algoritmo.

Referencia: docs/ARCHITECTURE.md — "5. Parámetros centralizados"
            README.md — sección 3 (Problema de optimización MCKP)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Raíz del proyecto: config/params.py -> config/ -> ComputacionEvolutiva/
RAIZ_PROYECTO: Path = Path(__file__).resolve().parent.parent
RUTA_ENV: Path = RAIZ_PROYECTO / ".env"

#: Número de macroregiones de la restricción territorial R3 (Lima, Sur, Norte,
#: Centro, Oriente). Ver README.md, resultados de referencia.
N_MACROREGIONES: int = 5

#: Claves de los cinco atributos ponderados de la función objetivo Z.
CLAVES_PESOS: tuple[str, ...] = ("R", "M", "P", "E", "G")


def _leer_int(clave: str, defecto: int) -> int:
    """
    Lee una variable de entorno como entero.

    Args:
        clave: Nombre de la variable en `.env`.
        defecto: Valor usado si la variable no existe o no es convertible.

    Returns:
        El valor entero leído, o `defecto` si la conversión falla.
    """
    crudo = os.getenv(clave)
    if crudo is None or crudo.strip() == "":
        return defecto
    try:
        return int(float(crudo))
    except ValueError:
        logger.warning("Valor inválido para %s=%r; se usa el defecto %s", clave, crudo, defecto)
        return defecto


def _leer_float(clave: str, defecto: float) -> float:
    """
    Lee una variable de entorno como flotante.

    Args:
        clave: Nombre de la variable en `.env`.
        defecto: Valor usado si la variable no existe o no es convertible.

    Returns:
        El valor flotante leído, o `defecto` si la conversión falla.
    """
    crudo = os.getenv(clave)
    if crudo is None or crudo.strip() == "":
        return defecto
    try:
        return float(crudo)
    except ValueError:
        logger.warning("Valor inválido para %s=%r; se usa el defecto %s", clave, crudo, defecto)
        return defecto


def _leer_str(clave: str, defecto: str) -> str:
    """
    Lee una variable de entorno como cadena.

    Args:
        clave: Nombre de la variable en `.env`.
        defecto: Valor usado si la variable no existe o está vacía.

    Returns:
        La cadena leída sin espacios laterales, o `defecto`.
    """
    crudo = os.getenv(clave)
    if crudo is None or crudo.strip() == "":
        return defecto
    return crudo.strip()


class Params:
    """
    Contenedor Singleton de todos los parámetros del proyecto.

    No instanciar directamente con `Params()` en el código de aplicación: usar
    `Params.get()`, que devuelve siempre la misma instancia y evita releer el
    archivo `.env` en cada módulo.

    Los atributos son mutables a propósito, para permitir overrides puntuales en
    tests y smoke tests (por ejemplo `params.N_GEN = 10`) sin tocar el `.env`.

    Atributos del problema (README.md, formulacion del MCKP):
        N_OBRAS: Número de obras públicas del dataset (n del cromosoma).
        PRESUPUESTO_PCT: Fracción del costo total que define el presupuesto B (R1).
        K_MAX: Máximo de obras seleccionables — capacidad operativa K (R2).
        MIN_POR_REGION: Mínimo de obras por macroregión — m_r (R3).
        W1..W5: Pesos de R, M, P, E y G en la función objetivo Z.

    Atributos del algoritmo (README.md, algoritmos implementados):
        POP_SIZE: Tamaño de la población.
        N_GEN: Número de generaciones.
        PC: Probabilidad de cruce.
        PM: Probabilidad de mutación por bit.
        K_TORNEO: Tamaño del torneo de selección.
        ALPHA: Coeficiente de penalización estática del fitness.
        N_RUNS: Corridas independientes para el análisis estadístico.
        SEED: Semilla base de reproducibilidad.

    Atributos de infraestructura:
        API_PORT: Puerto de la API de la Parte 2 (8001; la Parte 1 usa 8000).
        PARTE1_API_URL: URL base de la API de la tesis (Parte 1).
    """

    #: Única instancia viva de la clase (patrón Singleton).
    _instancia: Optional["Params"] = None

    def __init__(self, ruta_env: Optional[Path] = None) -> None:
        """
        Carga los parámetros desde `.env` y los registra en el log.

        Args:
            ruta_env: Ruta alternativa al archivo `.env`. Si es None se usa el
                `.env` de la raíz del proyecto.
        """
        self.ruta_env: Path = Path(ruta_env) if ruta_env is not None else RUTA_ENV
        self.env_encontrado: bool = self.ruta_env.is_file()
        if self.env_encontrado:
            load_dotenv(self.ruta_env, override=True)
        else:
            # Caso normal en Docker: la imagen no incluye el .env y las variables
            # llegan por env_file, que las inyecta en el entorno del contenedor.
            logger.warning(
                "No se encontro %s; se leen las variables del entorno y, si faltan, "
                "los valores por defecto documentados.",
                self.ruta_env,
            )

        # --- Problema MCKP -------------------------------------------------
        self.N_OBRAS: int = _leer_int("N_OBRAS", 326)
        self.PRESUPUESTO_PCT: float = _leer_float("PRESUPUESTO_PCT", 0.35)
        self.K_MAX: int = _leer_int("K_MAX", 50)
        self.MIN_POR_REGION: int = _leer_int("MIN_POR_REGION", 5)

        # --- Pesos de la función objetivo Z --------------------------------
        self.W1: float = _leer_float("W1", 0.40)  # R: P(Extrem. Riesgosa), Parte 1
        self.W2: float = _leer_float("W2", 0.25)  # M: Monto viable
        self.W3: float = _leer_float("W3", 0.15)  # P: PIM
        self.W4: float = _leer_float("W4", 0.10)  # E: Nivel de ejecución
        self.W5: float = _leer_float("W5", 0.10)  # G: Cobertura geográfica

        # --- Algoritmo evolutivo -------------------------------------------
        self.POP_SIZE: int = _leer_int("POP_SIZE", 150)
        self.N_GEN: int = _leer_int("N_GEN", 400)
        self.PC: float = _leer_float("PC", 0.85)
        self.PM: float = _leer_float("PM", 0.015)
        self.K_TORNEO: int = _leer_int("K_TORNEO", 3)
        self.ALPHA: float = _leer_float("ALPHA", 2.5)
        self.N_RUNS: int = _leer_int("N_RUNS", 10)
        self.SEED: int = _leer_int("SEED", 42)

        # --- Infraestructura ------------------------------------------------
        self.API_PORT: int = _leer_int("API_PORT", 8001)
        self.PARTE1_API_URL: str = _leer_str("PARTE1_API_URL", "http://localhost:8000")

        # --- Territorial (R3) -----------------------------------------------
        self.N_MACROREGIONES: int = N_MACROREGIONES
        #: Override explícito de M_R; si es None, M_R se deriva de MIN_POR_REGION.
        self._m_r_override: Optional[Dict[int, int]] = None

        self._validar()
        self._loguear_parametros()

    # ------------------------------------------------------------------ API

    @classmethod
    def get(cls, ruta_env: Optional[Path] = None) -> "Params":
        """
        Devuelve la instancia Singleton, creándola en la primera llamada.

        Args:
            ruta_env: Ruta alternativa al `.env`. Solo se tiene en cuenta si la
                instancia todavía no existe (usar `recargar()` para forzarla).

        Returns:
            La única instancia de Params del proceso.
        """
        if cls._instancia is None:
            cls._instancia = cls(ruta_env=ruta_env)
        return cls._instancia

    @classmethod
    def recargar(cls, ruta_env: Optional[Path] = None) -> "Params":
        """
        Descarta la instancia actual y vuelve a leer el `.env`.

        Útil en tests y cuando se edita el `.env` en caliente.

        Args:
            ruta_env: Ruta alternativa al archivo `.env`.

        Returns:
            La nueva instancia Singleton.
        """
        cls._instancia = cls(ruta_env=ruta_env)
        return cls._instancia

    # ------------------------------------------------------- Derivados

    @property
    def W(self) -> Dict[str, float]:
        """
        Pesos de la función objetivo Z indexados por atributo.

        Returns:
            Diccionario {"R": W1, "M": W2, "P": W3, "E": W4, "G": W5}.
        """
        return {
            "R": self.W1,
            "M": self.W2,
            "P": self.W3,
            "E": self.W4,
            "G": self.W5,
        }

    @W.setter
    def W(self, pesos: Dict[str, float]) -> None:
        """
        Reescribe los pesos W1..W5 desde un diccionario por atributo.

        Args:
            pesos: Diccionario parcial o completo con claves R, M, P, E, G.

        Raises:
            KeyError: Si alguna clave no pertenece a {R, M, P, E, G}.
        """
        desconocidas = set(pesos) - set(CLAVES_PESOS)
        if desconocidas:
            raise KeyError(f"Claves de peso desconocidas: {sorted(desconocidas)}")
        for i, clave in enumerate(CLAVES_PESOS, start=1):
            if clave in pesos:
                setattr(self, f"W{i}", float(pesos[clave]))

    @property
    def M_R(self) -> Dict[int, int]:
        """
        Mínimo de obras exigido por macroregión — restricción territorial R3.

        Returns:
            Diccionario {macroregión: mínimo}, por defecto {1..5: MIN_POR_REGION}.
        """
        if self._m_r_override is not None:
            return dict(self._m_r_override)
        return {r: self.MIN_POR_REGION for r in range(1, self.N_MACROREGIONES + 1)}

    @M_R.setter
    def M_R(self, minimos: Dict[int, int]) -> None:
        """
        Fija mínimos territoriales distintos por macroregión.

        Args:
            minimos: Diccionario {macroregión: mínimo de obras}.
        """
        self._m_r_override = {int(r): int(m) for r, m in minimos.items()}

    def as_dict(self) -> Dict[str, object]:
        """
        Serializa los parámetros para logging, respuestas de API o metricas.json.

        Returns:
            Diccionario plano con todos los parámetros y los derivados W y M_R.
        """
        datos: Dict[str, object] = {
            clave: valor
            for clave, valor in vars(self).items()
            if not clave.startswith("_") and clave != "ruta_env"
        }
        datos["W"] = self.W
        datos["M_R"] = self.M_R
        return datos

    # ------------------------------------------------------------ Internos

    def _validar(self) -> None:
        """Advierte (sin abortar) sobre valores fuera de los rangos esperados."""
        suma_pesos = sum(self.W.values())
        if abs(suma_pesos - 1.0) > 1e-6:
            logger.warning("Los pesos W suman %.4f, se esperaba 1.0", suma_pesos)
        if not 0.0 <= self.PC <= 1.0:
            logger.warning("PC=%s fuera de [0, 1]", self.PC)
        if not 0.0 <= self.PM <= 1.0:
            logger.warning("PM=%s fuera de [0, 1]", self.PM)
        if not 0.0 < self.PRESUPUESTO_PCT <= 1.0:
            logger.warning("PRESUPUESTO_PCT=%s fuera de (0, 1]", self.PRESUPUESTO_PCT)
        if self.K_TORNEO < 2:
            logger.warning("K_TORNEO=%s: sin presión selectiva (se espera >= 2)", self.K_TORNEO)
        if self.K_MAX < self.MIN_POR_REGION * self.N_MACROREGIONES:
            logger.warning(
                "K_MAX=%s < MIN_POR_REGION*N_MACROREGIONES=%s: R2 y R3 son incompatibles",
                self.K_MAX,
                self.MIN_POR_REGION * self.N_MACROREGIONES,
            )

    def _loguear_parametros(self) -> None:
        """Registra en el log los parámetros cargados al instanciar."""
        origen = str(self.ruta_env) if self.env_encontrado else "el entorno (sin archivo .env)"
        logger.info("Params cargados desde %s", origen)
        logger.info(
            "Problema  | N_OBRAS=%d PRESUPUESTO_PCT=%.2f K_MAX=%d MIN_POR_REGION=%d",
            self.N_OBRAS,
            self.PRESUPUESTO_PCT,
            self.K_MAX,
            self.MIN_POR_REGION,
        )
        logger.info(
            "Pesos     | %s (suma=%.4f)",
            ", ".join(f"{k}={v:.2f}" for k, v in self.W.items()),
            sum(self.W.values()),
        )
        logger.info(
            "AG        | POP_SIZE=%d N_GEN=%d PC=%.2f PM=%.4f K_TORNEO=%d ALPHA=%.2f",
            self.POP_SIZE,
            self.N_GEN,
            self.PC,
            self.PM,
            self.K_TORNEO,
            self.ALPHA,
        )
        logger.info("Análisis  | N_RUNS=%d SEED=%d", self.N_RUNS, self.SEED)
        logger.info("API       | API_PORT=%d PARTE1_API_URL=%s", self.API_PORT, self.PARTE1_API_URL)
        logger.info("R3        | M_R=%s", self.M_R)

    def __repr__(self) -> str:
        """Resumen de una línea con los parámetros más relevantes."""
        return (
            f"Params(n={self.N_OBRAS}, K={self.K_MAX}, pop={self.POP_SIZE}, "
            f"gen={self.N_GEN}, pc={self.PC}, pm={self.PM}, alpha={self.ALPHA}, "
            f"seed={self.SEED})"
        )
