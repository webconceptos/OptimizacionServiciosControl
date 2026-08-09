"""
Interfaz común de los algoritmos evolutivos.

Define el contrato que cumplen tanto el AG canónico (Tema 3) como NSGA-II
(Tema 10): quien invoca no necesita saber qué algoritmo está corriendo, solo que
tiene un `evolucionar()` que devuelve una solución o un frente de soluciones.

Aquí vive todo lo que ambos algoritmos comparten y que no es un operador:
el generador aleatorio propio de la corrida, el historial de convergencia, el
cronometraje y la serialización del historial. Los operadores de variación viven
en `core/operators.py` como funciones puras.

Dependencias: solo `core/` y `config/`. Este módulo NO importa nada de
`algorithms/`, `analysis/` ni `api/`, para no invertir el diagrama de
dependencias de docs/ARCHITECTURE.md.

Referencia: Tema 2 del curso CE UNI 2026 (ciclo evolutivo generico)
            docs/ARCHITECTURE.md — "4. Algoritmos como clases con interfaz comun"
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from config.params import Params
from core.problem import Problema

if TYPE_CHECKING:  # Solo para anotaciones: evita una dependencia en runtime.
    from core.solution import Solucion

logger = logging.getLogger(__name__)

#: Series que todo algoritmo registra en cada generación.
CLAVES_HISTORIAL: Tuple[str, ...] = (
    "mejor_fitness",
    "media_fitness",
    "peor_fitness",
    "fraccion_factible",
)

#: Cada cuántas generaciones se emite una línea de progreso por defecto.
CADA_N_GENERACIONES: int = 100


class AlgoritmoEvolutivo(ABC):
    """
    Clase base abstracta de los algoritmos evolutivos del proyecto.

    Un subclase concreta solo tiene que implementar `evolucionar()`. Todo lo
    demás (RNG reproducible, historial, cronómetro, persistencia) lo hereda.

    Reproducibilidad: cada instancia crea su propio `np.random.Generator` a
    partir de `seed`. No se usan semillas globales, de modo que N corridas
    independientes con semillas distintas no interfieren entre sí ni dependen del
    orden en que se ejecuten (docs/TECHNICAL_DEBT.md, DT-008).

    Atributos de clase que conviene sobreescribir en cada subclase:
        nombre: Nombre legible del algoritmo, usado en logs y metadatos.
        tema: Tema del curso al que corresponde (p. ej. "Tema 3").

    Atributos de instancia:
        problema: Problema MCKP a optimizar.
        params: Parámetros del proyecto (Singleton de config/params.py).
        seed: Semilla efectiva de esta corrida.
        rng: Generador aleatorio propio de la corrida.
        historial: Series de convergencia por generación.
        tiempo_seg: Duración de la última llamada a `ejecutar()`.
        generaciones_ejecutadas: Número de generaciones registradas.

    Referencia: Tema 2 del curso CE UNI 2026.
    """

    #: Nombre legible del algoritmo (sobreescribir en la subclase).
    nombre: str = "AlgoritmoEvolutivo"

    #: Tema del curso al que corresponde el algoritmo.
    tema: str = "Tema 2"

    def __init__(
        self,
        problema: Problema,
        params: Optional[Params] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Inicializa el algoritmo con su problema, parámetros y semilla.

        Args:
            problema: Problema MCKP a optimizar.
            params: Parámetros del proyecto; None usa el Singleton `Params.get()`.
            seed: Semilla de esta corrida; None usa `params.SEED`.

        Raises:
            TypeError: Si `problema` no es una instancia de `Problema`.
        """
        if not isinstance(problema, Problema):
            raise TypeError(
                f"Se esperaba una instancia de Problema, se recibio {type(problema).__name__}"
            )

        self.problema: Problema = problema
        self.params: Params = params if params is not None else Params.get()
        self.seed: int = int(seed) if seed is not None else int(self.params.SEED)
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

        self.historial: Dict[str, List[float]] = {clave: [] for clave in CLAVES_HISTORIAL}
        self.tiempo_seg: float = 0.0
        self.generaciones_ejecutadas: int = 0

        logger.info(
            "%s (%s) | n=%d | seed=%d",
            self.nombre,
            self.tema,
            self.problema.n,
            self.seed,
        )

    # ------------------------------------------------------------- Contrato

    @abstractmethod
    def evolucionar(self) -> Union["Solucion", List["Solucion"]]:
        """
        Ejecuta el ciclo evolutivo completo.

        Returns:
            La mejor `Solucion` encontrada (algoritmos mono-objetivo, p. ej. AG),
            o la lista de `Solucion` del frente de Pareto (multiobjetivo, NSGA-II).

        Referencia: Tema 2 del curso CE UNI 2026 (ciclo evolutivo).
        """
        raise NotImplementedError

    def ejecutar(self) -> Union["Solucion", List["Solucion"]]:
        """
        Envoltorio de `evolucionar()` que cronometra la corrida.

        Deja la duración en `self.tiempo_seg`, que es lo que consumen el análisis
        estadístico (columna `tiempo_seg`) y las respuestas de la API.

        Returns:
            Lo mismo que `evolucionar()`.
        """
        inicio = time.perf_counter()
        resultado = self.evolucionar()
        self.tiempo_seg = time.perf_counter() - inicio
        logger.info(
            "%s terminado | generaciones=%d | tiempo=%.2f s",
            self.nombre,
            self.generaciones_ejecutadas,
            self.tiempo_seg,
        )
        return resultado

    # ------------------------------------------------------------ Historial

    def registrar_generacion(
        self,
        fit: Sequence[float],
        fraccion_factible: float,
        **extras: float,
    ) -> None:
        """
        Añade al historial las métricas de una generación.

        Args:
            fit: Vector de fitness de la población de esta generación.
            fraccion_factible: Proporción de individuos factibles, en [0, 1].
            **extras: Series adicionales propias del algoritmo (por ejemplo
                `n_frente` en NSGA-II). Se crean en el historial la primera vez
                que aparecen y se rellenan con NaN las generaciones previas, para
                que todas las series queden alineadas por índice.

        Raises:
            ValueError: Si el vector de fitness está vacío.
        """
        valores = np.asarray(fit, dtype=float)
        if valores.size == 0:
            raise ValueError("El vector de fitness esta vacio: no hay nada que registrar")

        self.historial["mejor_fitness"].append(float(valores.max()))
        self.historial["media_fitness"].append(float(valores.mean()))
        self.historial["peor_fitness"].append(float(valores.min()))
        self.historial["fraccion_factible"].append(float(fraccion_factible))
        self.generaciones_ejecutadas = len(self.historial["mejor_fitness"])

        for clave, valor in extras.items():
            serie = self.historial.setdefault(clave, [])
            # Alinea una serie nueva rellenando las generaciones anteriores.
            faltan = self.generaciones_ejecutadas - 1 - len(serie)
            if faltan > 0:
                serie.extend([float("nan")] * faltan)
            serie.append(float(valor))

    def fraccion_factible(self, pop: np.ndarray) -> float:
        """
        Calcula la proporción de individuos factibles de una población.

        Args:
            pop: Población de forma (pop_size, n).

        Returns:
            Fracción en [0, 1]; 0.0 si la población está vacía.
        """
        if len(pop) == 0:
            return 0.0
        factibles = sum(1 for individuo in pop if self.problema.es_factible(individuo))
        return factibles / len(pop)

    def reiniciar_historial(self) -> None:
        """Vacía el historial y el contador de generaciones, para reutilizar la instancia."""
        self.historial = {clave: [] for clave in CLAVES_HISTORIAL}
        self.generaciones_ejecutadas = 0
        self.tiempo_seg = 0.0

    def log_progreso(
        self,
        generacion: int,
        total: int,
        cada: int = CADA_N_GENERACIONES,
    ) -> None:
        """
        Emite una línea de progreso cada `cada` generaciones.

        Usa la última entrada del historial, así que debe llamarse después de
        `registrar_generacion()`. Los mensajes van en ASCII: la consola de
        Windows (cp1252) escapa los caracteres no representables.

        Args:
            generacion: Índice de la generación actual, base 1.
            total: Número total de generaciones previstas.
            cada: Periodicidad del log; si es <= 0 no registra nada.
        """
        if cada <= 0 or not self.historial["mejor_fitness"]:
            return
        if generacion % cada != 0 and generacion != total:
            return
        logger.info(
            "  %s | gen %d/%d | mejor=%.6f | media=%.6f | factibles=%.0f%%",
            self.nombre,
            generacion,
            total,
            self.historial["mejor_fitness"][-1],
            self.historial["media_fitness"][-1],
            100.0 * self.historial["fraccion_factible"][-1],
        )

    # -------------------------------------------------------- Serialización

    def metadatos(self) -> Dict[str, object]:
        """
        Metadatos de la corrida, para acompañar al historial.

        Las subclases pueden extenderlo (llamando a `super().metadatos()`) para
        añadir sus propios parámetros.

        Returns:
            Diccionario serializable a JSON con la identidad de la corrida.
        """
        return {
            "algoritmo": self.nombre,
            "tema": self.tema,
            "seed": self.seed,
            "n_obras": self.problema.n,
            "generaciones": self.generaciones_ejecutadas,
            "tiempo_seg": round(self.tiempo_seg, 4),
            "presupuesto_B": round(self.problema.B, 6),
            "capacidad_K": self.problema.K,
            "m_r": {str(r): m for r, m in self.problema.m_r.items()},
            "alpha": self.problema.alpha,
            "pesos": {
                atributo: float(peso)
                for atributo, peso in zip(["R", "M", "P", "E", "G"], self.problema.w)
            },
        }

    def guardar_historial(self, path: Union[str, Path]) -> None:
        """
        Serializa el historial de convergencia a un archivo JSON.

        El archivo contiene dos claves: `metadatos` (identidad y parámetros de la
        corrida, imprescindibles para que el historial sea trazable) e
        `historial` (las series por generación). Se escribe con escapes ASCII
        para poder leerlo con `open()` sin declarar encoding en Windows.

        Args:
            path: Ruta del archivo JSON de salida; se crean los directorios
                intermedios que falten.

        Raises:
            OSError: Si el archivo no se puede escribir.
        """
        destino = Path(path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        contenido = {
            "metadatos": self.metadatos(),
            "historial": {
                clave: [float(v) for v in serie] for clave, serie in self.historial.items()
            },
        }
        destino.write_text(json.dumps(contenido, indent=2), encoding="utf-8")
        logger.info(
            "Historial de %s guardado | %d generaciones | %s",
            self.nombre,
            self.generaciones_ejecutadas,
            destino,
        )

    def resumen(self) -> Dict[str, object]:
        """
        Resumen compacto del estado de la corrida, para logs y reportes.

        Returns:
            Diccionario con el mejor fitness alcanzado, las generaciones
            ejecutadas y el tiempo. Los valores de fitness son None si todavía no
            se registró ninguna generación.
        """
        mejores = self.historial["mejor_fitness"]
        return {
            "algoritmo": self.nombre,
            "seed": self.seed,
            "generaciones": self.generaciones_ejecutadas,
            "mejor_fitness": float(max(mejores)) if mejores else None,
            "fitness_final": float(mejores[-1]) if mejores else None,
            "tiempo_seg": round(self.tiempo_seg, 4),
        }

    def __repr__(self) -> str:
        """Resumen de una línea del algoritmo."""
        mejores = self.historial["mejor_fitness"]
        mejor = f"{max(mejores):.6f}" if mejores else "sin evaluar"
        return (
            f"{type(self).__name__}(n={self.problema.n}, seed={self.seed}, "
            f"gen={self.generaciones_ejecutadas}, mejor={mejor})"
        )
