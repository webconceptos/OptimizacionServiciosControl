"""
Validación end-to-end del proyecto.

Ejercita el pipeline completo en escala reducida y reporta PASS/FAIL por
chequeo. Está pensado como comprobación rápida antes de entregar o desplegar:
verifica que las piezas encajan, no la calidad de los resultados (para eso están
`pytest tests/` y `python main.py --mode full`).

Uso:
    python validate.py

Código de salida: 0 si todos los chequeos pasan, 1 si alguno falla, de modo que
sirve en un pipeline de CI o encadenado con `&&`.

Referencia: README.md, comandos de uso
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

#: Ancho de las líneas del reporte.
ANCHO: int = 78

#: Parámetros del AG en la validación (reducidos frente a la corrida real).
POP_VALIDACION: int = 50
GEN_VALIDACION: int = 50

#: Parámetros de NSGA-II en la validación.
POP_NSGA2: int = 60
GEN_NSGA2: int = 60

#: Soluciones mínimas que se le exigen al frente de Pareto.
MIN_SOLUCIONES_PARETO: int = 5

#: Corridas del análisis estadístico en la validación.
N_CORRIDAS: int = 3

#: Obras del dataset de validación (la escala del caso de estudio).
N_OBRAS: int = 326

#: Carpeta donde se escribe la figura de prueba.
SALIDA_PRUEBA: Path = Path("outputs") / "validate"


@dataclass
class Chequeo:
    """
    Resultado de un chequeo individual.

    Atributos:
        nombre: Descripción corta del chequeo.
        ok: True si pasó.
        detalle: Evidencia medida o motivo del fallo.
        segundos: Duración del chequeo.
    """

    nombre: str
    ok: bool
    detalle: str = ""
    segundos: float = 0.0


@dataclass
class Validador:
    """
    Ejecuta chequeos, los reporta y acumula el estado global.

    Atributos:
        chequeos: Resultados en el orden en que se ejecutaron.
        contexto: Objetos que los chequeos se pasan entre sí (dataset, problema,
            soluciones), para no recalcularlos.
    """

    chequeos: List[Chequeo] = field(default_factory=list)
    contexto: Dict[str, Any] = field(default_factory=dict)

    def ejecutar(self, nombre: str, funcion: Callable[[], str]) -> bool:
        """
        Corre un chequeo y registra su resultado.

        Cualquier excepción se captura y cuenta como FAIL: la validación completa
        debe terminar y reportar todo, no abortar en el primer fallo.

        Args:
            nombre: Descripción del chequeo.
            funcion: Callable que devuelve el detalle a mostrar. Si lanza o si
                alguna de sus aserciones falla, el chequeo se marca como FAIL.

        Returns:
            True si el chequeo pasó.
        """
        inicio = time.perf_counter()
        try:
            detalle = funcion()
            chequeo = Chequeo(nombre, True, detalle, time.perf_counter() - inicio)
        except Exception as exc:  # noqa: BLE001 - se reporta, no se propaga
            detalle = f"{type(exc).__name__}: {exc}"
            chequeo = Chequeo(nombre, False, detalle, time.perf_counter() - inicio)
            traceback.print_exc(limit=3)

        self.chequeos.append(chequeo)
        marca = "PASS" if chequeo.ok else "FAIL"
        print(f" [{marca}] {nombre:<44} {chequeo.detalle}")
        return chequeo.ok

    @property
    def todos_ok(self) -> bool:
        """True si no hay ningún chequeo fallido."""
        return all(chequeo.ok for chequeo in self.chequeos)


def _chequeo_dataset(validador: Validador) -> str:
    """
    Genera el dataset y construye el problema MCKP.

    Args:
        validador: Validador que guarda el contexto compartido.

    Returns:
        Detalle con el tamaño del dataset y los parámetros del problema.
    """
    from config.params import Params
    from core.problem import Problema
    from data.generator import generar_obras

    params = Params()
    df = generar_obras(n=N_OBRAS, seed=int(params.SEED))
    problema = Problema(df, params)

    assert len(df) == N_OBRAS, f"se esperaban {N_OBRAS} obras, hay {len(df)}"
    assert not df[["R", "M", "P", "E", "G", "C"]].isna().any().any(), "hay valores nulos"
    assert sorted(df["macroregion"].unique()) == [1, 2, 3, 4, 5], "faltan macroregiones"
    assert problema.B > 0, "el presupuesto B debe ser positivo"

    validador.contexto.update(params=params, df=df, problema=problema)
    return f"n={len(df)} | B={problema.B:.4f} kS/ | K={problema.K} | R medio={df['R'].mean():.4f}"


def _chequeo_ag(validador: Validador) -> str:
    """
    Ejecuta el AG y exige fitness positivo y solución factible.

    Args:
        validador: Validador con el problema en contexto.

    Returns:
        Detalle con el fitness, las obras y el tiempo.
    """
    from algorithms.genetic import AG

    problema = validador.contexto["problema"]
    params = validador.contexto["params"]

    algoritmo = AG(problema, params, seed=int(params.SEED))
    algoritmo.pop_size = POP_VALIDACION
    algoritmo.n_gen = GEN_VALIDACION
    solucion = algoritmo.ejecutar()

    assert solucion.fitness > 0, f"fitness no positivo: {solucion.fitness}"
    assert solucion.factible, f"solucion infactible, violacion={solucion.violacion:.4f}"
    assert len(algoritmo.historial["mejor_fitness"]) == GEN_VALIDACION

    validador.contexto.update(solucion_ag=solucion, algoritmo_ag=algoritmo)
    return (
        f"fitness={solucion.fitness:.4f} obras={solucion.n_seleccionadas} "
        f"costo={solucion.costo:.2f} kS/ ({algoritmo.tiempo_seg:.2f} s)"
    )


def _chequeo_nsga2(validador: Validador) -> str:
    """
    Ejecuta NSGA-II y exige un frente factible con más de 5 soluciones.

    Args:
        validador: Validador con el problema en contexto.

    Returns:
        Detalle con el tamaño del frente y el rango de objetivos.
    """
    from algorithms.nsga2 import NSGA2

    problema = validador.contexto["problema"]
    params = validador.contexto["params"]

    algoritmo = NSGA2(problema, params, seed=int(params.SEED))
    algoritmo.pop_size = POP_NSGA2
    algoritmo.n_gen = GEN_NSGA2
    frente = algoritmo.ejecutar()

    assert len(frente) > MIN_SOLUCIONES_PARETO, (
        f"el frente tiene {len(frente)} soluciones, se exigen mas de {MIN_SOLUCIONES_PARETO}"
    )
    assert all(s.factible for s in frente), "hay soluciones infactibles en el frente"

    f1 = [algoritmo.f1(s) for s in frente]
    f2 = [s.costo for s in frente]
    validador.contexto.update(frente=frente, algoritmo_nsga2=algoritmo)
    return (
        f"{len(frente)} soluciones factibles | f1 max={max(f1):.2f} "
        f"f2 min={min(f2):.2f} kS/ ({algoritmo.tiempo_seg:.2f} s)"
    )


def _chequeo_greedy(validador: Validador) -> str:
    """
    Ejecuta el Greedy y exige que sea factible y determinístico.

    Args:
        validador: Validador con el problema en contexto.

    Returns:
        Detalle con el fitness y la cobertura territorial.
    """
    from algorithms.benchmarks import greedy

    problema = validador.contexto["problema"]
    solucion = greedy(problema)
    repetida = greedy(problema)

    assert solucion.factible, f"greedy infactible, violacion={solucion.violacion:.4f}"
    assert np.array_equal(solucion.x, repetida.x), "greedy no es deterministico"

    validador.contexto["solucion_greedy"] = solucion
    territorial = solucion.distribucion_territorial
    return (
        f"fitness={solucion.fitness:.4f} obras={solucion.n_seleccionadas} "
        f"territorial={list(territorial.values())}"
    )


def _chequeo_wilcoxon(validador: Validador) -> str:
    """
    Ejecuta N corridas del AG y el contraste de Wilcoxon sin error.

    No exige significancia: con 3 corridas el p mínimo posible del signed-rank
    bilateral es 0.25, así que exigirla sería un chequeo imposible de pasar. Lo que
    se valida es que el análisis corre y devuelve todas sus claves.

    Args:
        validador: Validador con el problema en contexto.

    Returns:
        Detalle con la media, el p-valor y la factibilidad de las corridas.
    """
    from analysis.statistics import analisis_wilcoxon, multiples_corridas

    problema = validador.contexto["problema"]
    params = validador.contexto["params"]
    solucion_greedy = validador.contexto["solucion_greedy"]

    params_reducidos = type(params)()
    params_reducidos.POP_SIZE = POP_VALIDACION
    params_reducidos.N_GEN = GEN_VALIDACION

    df_runs = multiples_corridas(problema, params_reducidos, n_runs=N_CORRIDAS)
    resumen = analisis_wilcoxon(df_runs, solucion_greedy.fitness)

    assert len(df_runs) == N_CORRIDAS, f"se esperaban {N_CORRIDAS} corridas"
    claves = {"media", "std", "mediana", "wilcoxon_stat", "wilcoxon_p", "significativo"}
    assert claves <= set(resumen), f"faltan claves en el resumen: {claves - set(resumen)}"
    assert np.isfinite(resumen["media"]), "la media no es finita"

    validador.contexto.update(df_runs=df_runs, estadisticas=resumen)
    return (
        f"{N_CORRIDAS} corridas: media={resumen['media']:.4f} p={resumen['wilcoxon_p']:.4f} "
        f"factibles={resumen['n_runs_factibles']}/{N_CORRIDAS}"
    )


def _chequeo_esquemas(validador: Validador) -> str:
    """
    Analiza los building blocks y exige al menos uno con crecimiento > 1.

    Args:
        validador: Validador con el problema en contexto.

    Returns:
        Detalle con el número de esquemas favorecidos y el crecimiento máximo.
    """
    from analysis.schema_theory import analizar_building_blocks

    problema = validador.contexto["problema"]
    df_bbs = analizar_building_blocks(problema, top_k=8)

    favorecidos = df_bbs.loc[df_bbs["crecimiento"] > 1.0]
    assert len(favorecidos) > 0, "ningun esquema tiene crecimiento > 1"
    assert (df_bbs["supervivencia"] <= 1.0).all(), "supervivencia fuera de [0, 1]"

    validador.contexto["df_bbs"] = df_bbs
    return (
        f"{len(favorecidos)}/{len(df_bbs)} esquemas con crecimiento>1 | "
        f"maximo={df_bbs['crecimiento'].max():.4f}"
    )


def _chequeo_figura(validador: Validador) -> str:
    """
    Genera una figura de prueba y verifica que el archivo exista y pese.

    Args:
        validador: Validador con el historial del AG en contexto.

    Returns:
        Detalle con la ruta y el tamaño del PNG.
    """
    from visualization import plots

    algoritmo = validador.contexto["algoritmo_ag"]
    SALIDA_PRUEBA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA_PRUEBA / "validate_convergencia.png"
    if destino.exists():
        destino.unlink()

    plots.plot_convergencia(algoritmo.historial, destino)

    assert destino.is_file(), f"no se creo la figura {destino}"
    tamano = destino.stat().st_size
    assert tamano > 1024, f"la figura pesa solo {tamano} bytes"

    import matplotlib.pyplot as plt

    assert not plt.get_fignums(), "quedaron figuras de matplotlib abiertas"
    return f"{destino.name} ({tamano / 1024:.1f} KB)"


def main() -> int:
    """
    Ejecuta todos los chequeos y reporta el resumen final.

    Returns:
        0 si todos pasaron, 1 si alguno falló.
    """
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s", stream=sys.stderr
    )
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")

    print("=" * ANCHO)
    print(" VALIDACION END-TO-END - Optimizacion Evolutiva de Obras Publicas")
    print("=" * ANCHO)

    inicio = time.perf_counter()
    validador = Validador()
    pasos: List[tuple[str, Callable[[], str]]] = [
        ("Dataset y problema MCKP", lambda: _chequeo_dataset(validador)),
        (f"AG (pop={POP_VALIDACION}, gen={GEN_VALIDACION}) factible y fitness>0", lambda: _chequeo_ag(validador)),
        (f"NSGA-II con frente > {MIN_SOLUCIONES_PARETO} soluciones", lambda: _chequeo_nsga2(validador)),
        ("Greedy factible y deterministico", lambda: _chequeo_greedy(validador)),
        (f"{N_CORRIDAS} corridas + Wilcoxon sin error", lambda: _chequeo_wilcoxon(validador)),
        ("Esquemas con crecimiento > 1", lambda: _chequeo_esquemas(validador)),
        ("Figura de prueba escrita en disco", lambda: _chequeo_figura(validador)),
    ]

    for nombre, funcion in pasos:
        validador.ejecutar(nombre, funcion)

    duracion = time.perf_counter() - inicio
    n_ok = sum(1 for chequeo in validador.chequeos if chequeo.ok)
    total = len(validador.chequeos)

    print("=" * ANCHO)
    if validador.todos_ok:
        print(f" RESULTADO: PASS - {n_ok}/{total} chequeos en {duracion:.1f} s")
    else:
        print(f" RESULTADO: FAIL - {n_ok}/{total} chequeos en {duracion:.1f} s")
        for chequeo in validador.chequeos:
            if not chequeo.ok:
                print(f"   FALLO: {chequeo.nombre} -> {chequeo.detalle}")
    print("=" * ANCHO)

    return 0 if validador.todos_ok else 1


if __name__ == "__main__":
    sys.exit(main())
