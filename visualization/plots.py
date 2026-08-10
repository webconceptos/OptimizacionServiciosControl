"""
Las nueve figuras del informe.

Una función por figura del informe, todas con la misma forma:
reciben los datos ya calculados y una ruta, escriben el PNG y cierran la figura.
No calculan nada: el análisis vive en `analysis/` y los algoritmos en
`algorithms/`.

Sin dependencias internas (docs/ARCHITECTURE.md, diagrama de dependencias): este
módulo solo importa numpy, pandas, matplotlib y la paleta. Los objetos de dominio
(`Solucion`, `Problema`) se consumen por duck typing a través de los protocolos
`SolucionLike` y `ProblemaLike`, así que no hay import de `core/`.

UNIDADES: todos los montos se rotulan en **miles S/ (kS/)**, la escala real de la
columna C del dataset (sum(C) ≈ 209.7 kS/, B ≈ 73.4 kS/). El rótulo
"74,370 kS/" de docs/DATA_DICTIONARY.md tiene un factor 1000 de más y no se usa.

Referencia: README.md, figuras del informe
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from visualization.palette import COLORES, DPI, UNIDAD_MONETARIA, hexc

logger = logging.getLogger(__name__)

#: Umbral de Rᵢ que la Parte 1 asocia a la clase "Extrem. Riesgosa".
UMBRAL_EXTREMO: float = 0.70

#: Nivel de significancia usado en las anotaciones estadísticas.
ALPHA: float = 0.05

#: Umbral del factor de crecimiento de un esquema (Teorema de Esquemas).
UMBRAL_CRECIMIENTO: float = 1.0

#: Nombres de las macroregiones, para las etiquetas territoriales.
NOMBRES_MACROREGIONES: Dict[int, str] = {
    1: "Lima",
    2: "Sur",
    3: "Norte",
    4: "Centro",
    5: "Oriente",
}


class ProblemaLike(Protocol):
    """Contrato mínimo que las figuras necesitan de un `Problema`."""

    n: int
    B: float
    K: int
    C: np.ndarray
    mr: np.ndarray
    m_r: Dict[int, int]
    df: pd.DataFrame

    def beneficio(self) -> np.ndarray: ...


class SolucionLike(Protocol):
    """Contrato mínimo que las figuras necesitan de una `Solucion`."""

    x: np.ndarray
    problema: ProblemaLike
    fitness: float
    costo: float
    factible: bool
    n_seleccionadas: int
    r_medio: float
    indices: np.ndarray


def _guardar(fig: plt.Figure, path: Union[str, Path], descripcion: str) -> None:
    """
    Escribe la figura en disco y la cierra.

    Args:
        fig: Figura de matplotlib a guardar.
        path: Ruta del PNG de salida; se crean los directorios que falten.
        descripcion: Nombre de la figura, para el log.
    """
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura guardada | %s | %s", descripcion, destino)


def _titulo(fig: plt.Figure, texto: str, subtitulo: Optional[str] = None) -> None:
    """
    Coloca el título (y opcionalmente el subtítulo) de una figura.

    Args:
        fig: Figura a titular.
        texto: Título principal.
        subtitulo: Línea secundaria, típicamente la referencia al Tema del curso.
    """
    completo = texto if subtitulo is None else f"{texto}\n{subtitulo}"
    fig.suptitle(completo, fontsize=12, fontweight="bold", color=COLORES["navy"])


# =========================================================================
# Figura 1 — Convergencia del AG
# =========================================================================


def plot_convergencia(historial: Mapping[str, Sequence[float]], path: Union[str, Path]) -> None:
    """
    Curva de convergencia del AG: fitness y fracción factible por generación.

    Args:
        historial: Diccionario de `AlgoritmoEvolutivo.historial`, con las series
            mejor_fitness, media_fitness, peor_fitness y fraccion_factible.
        path: Ruta del PNG de salida.

    Raises:
        ValueError: Si el historial no tiene la serie `mejor_fitness`.

    Referencia: Tema 2 y Tema 3 del curso CE UNI 2026.
    """
    if "mejor_fitness" not in historial or len(historial["mejor_fitness"]) == 0:
        raise ValueError("El historial no tiene la serie 'mejor_fitness'")

    mejor = np.asarray(historial["mejor_fitness"], dtype=float)
    media = np.asarray(historial.get("media_fitness", mejor), dtype=float)
    peor = np.asarray(historial.get("peor_fitness", mejor), dtype=float)
    factible = np.asarray(historial.get("fraccion_factible", np.ones_like(mejor)), dtype=float)
    generaciones = np.arange(1, mejor.size + 1)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"hspace": 0.08, "height_ratios": [3, 1]}
    )
    _titulo(
        fig,
        "Curva de Convergencia - AG Binario (MCKP)",
        "Fitness penalizado y fraccion factible por generacion",
    )

    ax1.plot(generaciones, mejor, color=COLORES["blue"], lw=2, label="Mejor fitness")
    ax1.plot(generaciones, media, color=COLORES["teal"], lw=1.5, ls="--", label="Fitness promedio")
    ax1.plot(generaciones, peor, color=COLORES["gray"], lw=1, ls=":", label="Peor fitness")
    ax1.fill_between(generaciones, peor, mejor, alpha=0.07, color=COLORES["blue"])
    ax1.set_ylabel("Fitness Z (penalizado)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y")

    ax2.fill_between(generaciones, factible, alpha=0.35, color=COLORES["green"])
    ax2.plot(generaciones, factible, color=COLORES["green"], lw=2)
    ax2.axhline(1.0, color=COLORES["green"], lw=1, ls="--", alpha=0.4)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Generacion")
    ax2.set_ylabel("Fraccion\nfactible")
    ax2.grid(axis="y")

    _guardar(fig, path, "fig1 convergencia")


# =========================================================================
# Figura 2 — Comparación de métodos
# =========================================================================


def plot_comparacion(
    resultados: Union[pd.DataFrame, Mapping[str, Any]], path: Union[str, Path]
) -> None:
    """
    Comparación AG vs Greedy vs Aleatorio en fitness, obras y costo.

    Args:
        resultados: DataFrame de `analysis.statistics.tabla_comparacion` (columnas
            metodo, fitness, n_seleccionadas, costo) o diccionario
            {nombre: Solucion}.
        path: Ruta del PNG de salida.

    Raises:
        ValueError: Si no hay métodos que comparar.

    Referencia: README.md, metodos de comparacion.
    """
    df = _normalizar_comparacion(resultados)
    if len(df) == 0:
        raise ValueError("No hay metodos que comparar")

    paneles = [
        ("fitness", "Fitness Z (funcion objetivo)", "{:.4f}"),
        ("n_seleccionadas", "Obras seleccionadas", "{:.0f}"),
        ("costo", f"Costo de supervision [{UNIDAD_MONETARIA}]", "{:.2f}"),
    ]
    colores = [COLORES["blue"], COLORES["green"], COLORES["gray"], COLORES["gold"]]

    fig, axes = plt.subplots(1, len(paneles), figsize=(13, 4.5))
    _titulo(fig, "Comparacion de Metodos - AG vs Greedy vs Aleatorio")

    for ax, (columna, titulo, formato) in zip(np.atleast_1d(axes), paneles):
        valores = df[columna].to_numpy(dtype=float)
        barras = ax.bar(
            df["metodo"], valores, color=colores[: len(df)], width=0.55, edgecolor="white"
        )
        ax.set_title(titulo, fontsize=10, color=COLORES["blue"])
        for barra, valor in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() * 1.02,
                formato.format(valor),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=COLORES["navy"],
            )
        ax.set_ylim(0, float(np.nanmax(valores)) * 1.22)
        ax.grid(axis="y")
        ax.tick_params(axis="x", labelrotation=15)

    fig.tight_layout()
    _guardar(fig, path, "fig2 comparacion")


def _normalizar_comparacion(resultados: Union[pd.DataFrame, Mapping[str, Any]]) -> pd.DataFrame:
    """
    Convierte la entrada de `plot_comparacion` a un DataFrame homogéneo.

    Args:
        resultados: DataFrame ya tabulado o mapeo {nombre: Solucion}.

    Returns:
        DataFrame con las columnas metodo, fitness, n_seleccionadas y costo.
    """
    if isinstance(resultados, pd.DataFrame):
        return resultados.copy()
    filas = [
        {
            "metodo": nombre,
            "fitness": float(solucion.fitness),
            "n_seleccionadas": int(solucion.n_seleccionadas),
            "costo": float(solucion.costo),
        }
        for nombre, solucion in resultados.items()
    ]
    return pd.DataFrame(filas)


# =========================================================================
# Figura 3 — Distribución territorial
# =========================================================================


def plot_territorial(solucion: SolucionLike, path: Union[str, Path]) -> None:
    """
    Distribución territorial del portafolio y su composición por clase de riesgo.

    Panel izquierdo: obras seleccionadas por macroregión, con la línea del mínimo
    mᵣ que exige R3. Panel derecho: reparto por clase de riesgo, que es el output
    del Random Forest de la Parte 1.

    Args:
        solucion: Solución a graficar; aporta el cromosoma y su problema.
        path: Ruta del PNG de salida.

    Referencia: Tema 5 del curso CE UNI 2026 (restriccion territorial R3).
    """
    problema = solucion.problema
    seleccionadas = problema.df.iloc[np.asarray(solucion.indices)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _titulo(
        fig,
        "Distribucion Territorial y por Clase de Riesgo - Obras Priorizadas",
        f"R3: minimo {min(problema.m_r.values())} obras por macroregion (supuesto de politica CGR)",
    )

    conteos = seleccionadas["macroregion"].value_counts().sort_index()
    etiquetas = [f"{r} - {NOMBRES_MACROREGIONES.get(int(r), '?')}" for r in conteos.index]
    paleta = [COLORES["blue"], COLORES["teal"], COLORES["gold"], COLORES["green"], COLORES["gray"]]
    barras = ax1.barh(etiquetas, conteos.to_numpy(), color=paleta[: len(conteos)], edgecolor="white")
    for barra, valor in zip(barras, conteos.to_numpy()):
        ax1.text(
            valor + 0.3,
            barra.get_y() + barra.get_height() / 2,
            str(int(valor)),
            va="center",
            fontsize=10,
            fontweight="bold",
            color=COLORES["navy"],
        )
    minimo = min(problema.m_r.values())
    ax1.axvline(
        minimo,
        color=COLORES["red"],
        lw=1.5,
        ls="--",
        alpha=0.8,
        label=f"Minimo requerido ({minimo})",
    )
    ax1.set_xlabel("Numero de obras seleccionadas")
    ax1.set_title("Por macroregion", fontsize=10, color=COLORES["blue"])
    ax1.legend(fontsize=9)
    ax1.grid(axis="x")

    if "clase_riesgo" in seleccionadas.columns:
        clases = seleccionadas["clase_riesgo"].value_counts()
        colores_clase = [COLORES["gold"], COLORES["teal"], COLORES["blue"], COLORES["gray"]]
        _, _, autotextos = ax2.pie(
            clases.to_numpy(),
            labels=list(clases.index),
            autopct="%1.0f%%",
            colors=colores_clase[: len(clases)],
            startangle=140,
            pctdistance=0.75,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for texto in autotextos:
            texto.set_fontsize(9)
            texto.set_fontweight("bold")
        ax2.set_title(
            "Por clase de riesgo\n(output Parte 1 - modelo RF)", fontsize=10, color=COLORES["blue"]
        )
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Sin columna 'clase_riesgo'", ha="center", color=COLORES["gray"])

    fig.tight_layout()
    _guardar(fig, path, "fig3 territorial")


# =========================================================================
# Figura 4 — Top obras priorizadas
# =========================================================================


def plot_top_obras(
    solucion: SolucionLike, path: Union[str, Path], top_n: int = 20
) -> None:
    """
    Ranking de las obras priorizadas por score de impacto bᵢ.

    Args:
        solucion: Solución a graficar.
        path: Ruta del PNG de salida.
        top_n: Número de obras a mostrar.

    Raises:
        ValueError: Si top_n < 1.

    Referencia: Tema 1 del curso CE UNI 2026 (funcion objetivo por obra).
    """
    if top_n < 1:
        raise ValueError(f"top_n debe ser >= 1, se recibio {top_n}")

    problema = solucion.problema
    beneficio = np.asarray(problema.beneficio(), dtype=float)
    indices = np.asarray(solucion.indices)
    if indices.size == 0:
        raise ValueError("La solucion no tiene obras seleccionadas")

    mejores = indices[np.argsort(beneficio[indices])[::-1]][:top_n]
    sub = problema.df.iloc[mejores]
    scores = beneficio[mejores]

    etiquetas = [
        f"{fila.codigo}\n({getattr(fila, 'sector', '')})" for fila in sub.itertuples()
    ]
    colores_barra = [
        COLORES["gold"] if r > UMBRAL_EXTREMO else COLORES["blue"] for r in sub["R"]
    ]

    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.42 * len(mejores) + 1.5)))
    _titulo(
        fig,
        f"Top {len(mejores)} Obras Priorizadas - Score de Impacto b(i)",
        "Contribucion de cada obra a la funcion objetivo",
    )

    barras = ax.barh(
        etiquetas[::-1], scores[::-1], color=colores_barra[::-1], edgecolor="white"
    )
    for barra, fila in zip(barras, list(sub.itertuples())[::-1]):
        ax.text(
            barra.get_width() + max(scores) * 0.01,
            barra.get_y() + barra.get_height() / 2,
            f"Ri={fila.R:.2f}  M=S/{fila.M:.1f}M  C={fila.C:.2f} kS/",
            va="center",
            fontsize=7.5,
            color=COLORES["gray"],
        )
    ax.set_xlabel("Score de impacto b(i) = suma ponderada de atributos normalizados")
    ax.set_xlim(0, float(max(scores)) * 1.35)
    ax.grid(axis="x")
    ax.legend(
        handles=[
            Patch(color=COLORES["gold"], label=f"Riesgo extremo (Ri > {UMBRAL_EXTREMO:.2f}) - RF Parte 1"),
            Patch(color=COLORES["blue"], label=f"Riesgo medio/bajo (Ri <= {UMBRAL_EXTREMO:.2f})"),
        ],
        fontsize=9,
        loc="lower right",
    )

    fig.tight_layout()
    _guardar(fig, path, "fig4 top obras")


# =========================================================================
# Figura 5 — Espacio riesgo / costo
# =========================================================================


def plot_scatter_riesgo_costo(solucion: SolucionLike, path: Union[str, Path]) -> None:
    """
    Espacio Rᵢ vs costo de supervisión: obras priorizadas frente al universo.

    Args:
        solucion: Solución a graficar.
        path: Ruta del PNG de salida.

    Referencia: Tema 7 del curso CE UNI 2026 (estructura de la instancia MCKP).
    """
    problema = solucion.problema
    x = np.asarray(solucion.x, dtype=bool)
    df = problema.df

    fig, ax = plt.subplots(figsize=(10, 6))
    _titulo(
        fig,
        "Espacio Riesgo - Costo de Supervision: Priorizadas vs No Priorizadas",
    )

    ax.scatter(
        df.loc[~x, "C"],
        df.loc[~x, "R"],
        c=COLORES["gray"],
        alpha=0.30,
        s=28,
        label=f"No priorizadas (n={int((~x).sum())})",
    )
    ax.scatter(
        df.loc[x, "C"],
        df.loc[x, "R"],
        c=COLORES["gold"],
        alpha=0.9,
        s=62,
        edgecolors=COLORES["navy"],
        linewidths=0.7,
        label=f"Priorizadas (n={int(x.sum())})",
    )
    ax.axhline(
        UMBRAL_EXTREMO,
        color=COLORES["red"],
        lw=1.2,
        ls="--",
        alpha=0.7,
        label=f"Umbral Extrem. Riesgosa (Ri={UMBRAL_EXTREMO:.2f}, Parte 1)",
    )
    ax.set_xlabel(f"Costo estimado de supervision [{UNIDAD_MONETARIA}]")
    ax.set_ylabel("Ri = P(Extrem. Riesgosa) - output RF Parte 1")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.6)

    fig.tight_layout()
    _guardar(fig, path, "fig5 scatter riesgo-costo")


# =========================================================================
# Figura 6 — Sensibilidad de w1
# =========================================================================


def plot_sensibilidad(
    df_sens: pd.DataFrame,
    path: Union[str, Path],
    w1_usado: Optional[float] = None,
) -> None:
    """
    Estabilidad del portafolio frente al peso del riesgo w₁.

    El eje principal es la **similitud del portafolio** (índice de Jaccard) frente
    al obtenido con el w₁ del trabajo: es lo que responde si la recomendación
    depende del peso elegido.

    Deliberadamente NO se presenta el fitness como criterio para elegir w₁. Cada
    w₁ define una función objetivo distinta, así que sus óptimos no son
    comparables y su crecimiento monótono es un artefacto de escala, no una señal
    de que un w₁ alto sea mejor. El fitness se dibuja en un eje secundario tenue,
    solo como contexto, y la figura lo advierte explícitamente.

    Args:
        df_sens: DataFrame de `analysis.sensitivity.sensibilidad_w1`, con las
            columnas w1, fitness_optimo y, si están disponibles,
            similitud_vs_usado, r_medio y es_valor_usado.
        path: Ruta del PNG de salida.
        w1_usado: Valor de w₁ empleado en el trabajo; None lo deduce de la columna
            `es_valor_usado`.

    Raises:
        ValueError: Si el DataFrame está vacío o no tiene la columna `w1`.

    Referencia: README.md, formulacion del MCKP (pesos de la funcion objetivo).
    """
    if df_sens is None or len(df_sens) == 0:
        raise ValueError("df_sens esta vacio: no hay barrido que graficar")
    if "w1" not in df_sens.columns:
        raise ValueError("df_sens no tiene la columna 'w1'")

    w1 = df_sens["w1"].to_numpy(dtype=float)
    if w1_usado is None and "es_valor_usado" in df_sens.columns:
        marcadas = df_sens.loc[df_sens["es_valor_usado"], "w1"]
        w1_usado = float(marcadas.iloc[0]) if len(marcadas) else None

    tiene_similitud = "similitud_vs_usado" in df_sens.columns and df_sens[
        "similitud_vs_usado"
    ].notna().any()

    fig, ax = plt.subplots(figsize=(10, 5.8))
    _titulo(
        fig,
        "Sensibilidad al Peso del Riesgo w1 - Estabilidad del Portafolio",
        "La eleccion de w1 es normativa: esta figura NO identifica un w1 optimo",
    )

    if tiene_similitud:
        similitud = df_sens["similitud_vs_usado"].to_numpy(dtype=float)
        ax.plot(
            w1,
            similitud,
            color=COLORES["blue"],
            lw=2.5,
            marker="o",
            ms=6,
            label="Similitud del portafolio (Jaccard) vs w1 usado",
        )
        ax.fill_between(w1, similitud, 0, alpha=0.10, color=COLORES["blue"])
        ax.set_ylabel("Similitud del portafolio (Jaccard)")
        ax.set_ylim(0, 1.05)
    else:
        ax.plot(w1, df_sens["fitness_optimo"], color=COLORES["blue"], lw=2.5, marker="o", ms=6)
        ax.set_ylabel("Fitness optimo (NO comparable entre w1)")

    if "r_medio" in df_sens.columns:
        ax.plot(
            w1,
            df_sens["r_medio"].to_numpy(dtype=float),
            color=COLORES["green"],
            lw=1.8,
            ls="-.",
            marker="s",
            ms=4,
            label="Riesgo medio Ri de las obras elegidas",
        )

    if w1_usado is not None:
        ax.axvline(
            w1_usado,
            color=COLORES["gold"],
            lw=2,
            ls="--",
            label=f"w1 usado en el trabajo ({w1_usado:.2f})",
        )

    if "fitness_optimo" in df_sens.columns:
        ax2 = ax.twinx()
        ax2.plot(
            w1,
            df_sens["fitness_optimo"].to_numpy(dtype=float),
            color=COLORES["gray"],
            lw=1.2,
            ls=":",
            alpha=0.55,
        )
        ax2.set_ylabel(
            "Fitness optimo (escala no comparable)", color=COLORES["gray"], fontsize=9
        )
        ax2.tick_params(axis="y", labelcolor=COLORES["gray"], labelsize=8)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(COLORES["gray"])
        ax2.grid(False)

    ax.set_xlabel("Peso w1 del riesgo de corrupcion Ri")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y")

    nota = (
        "El fitness (linea punteada) crece con w1 solo por efecto de escala: cada w1\n"
        "define una funcion objetivo distinta. La robustez se lee en la similitud."
    )
    ax.text(
        0.99,
        0.02,
        nota,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=COLORES["gray"],
        bbox={"facecolor": COLORES["lgray"], "edgecolor": COLORES["gray"], "alpha": 0.85},
    )

    fig.tight_layout()
    _guardar(fig, path, "fig6 sensibilidad w1")


# =========================================================================
# Figura 10 — Sensibilidad de los parámetros del AG
# =========================================================================


def plot_sensibilidad_parametros(
    df_params: pd.DataFrame,
    path: Union[str, Path],
    n_gen: Optional[int] = None,
    n_seeds: Optional[int] = None,
) -> None:
    """
    Sensibilidad del AG a población, probabilidad de cruce y de mutación.

    Un panel por parámetro: el fitness medio con su banda de dispersión sobre el eje
    principal y el tiempo medio por corrida sobre el eje secundario, para que se vea
    a la vez qué gana y qué cuesta cada configuración. El valor usado en el trabajo
    queda marcado con una línea vertical y el mejor valor encontrado con un punto
    destacado.

    A diferencia de la figura 6, aquí el fitness SÍ es comparable entre valores: la
    función objetivo no cambia, solo la configuración del algoritmo.

    Args:
        df_params: DataFrame de `analysis.sensitivity.sensibilidad_parametros`.
        path: Ruta del PNG de salida.
        n_gen: Generaciones usadas en el barrido, para el subtítulo.
        n_seeds: Número de semillas promediadas, para el subtítulo.

    Raises:
        ValueError: Si el DataFrame está vacío o le faltan columnas.

    Referencia: Tema 3 del curso CE UNI 2026 (parametros del AG canonico).
    """
    if df_params is None or len(df_params) == 0:
        raise ValueError("df_params esta vacio: no hay barrido que graficar")
    requeridas = ["parametro", "valor", "fitness_medio", "fitness_std", "tiempo_medio_seg"]
    faltantes = [c for c in requeridas if c not in df_params.columns]
    if faltantes:
        raise ValueError(f"df_params no tiene las columnas requeridas: {faltantes}")

    etiquetas = {
        "pop_size": "Tamano de poblacion",
        "pc": "Probabilidad de cruce pc",
        "pm": "Probabilidad de mutacion pm",
    }
    parametros = [p for p in ("pop_size", "pc", "pm") if p in set(df_params["parametro"])]
    parametros += [p for p in sorted(set(df_params["parametro"])) if p not in parametros]

    partes = []
    if n_gen is not None:
        partes.append(f"n_gen fijo = {n_gen}")
    if n_seeds is not None:
        partes.append(f"{n_seeds} semillas por configuracion")
    subtitulo = "Barrido one-factor-at-a-time"
    if partes:
        subtitulo += " | " + " | ".join(partes)

    fig, axes = plt.subplots(1, len(parametros), figsize=(5.0 * len(parametros), 5.2))
    _titulo(fig, "Sensibilidad a los Parametros del AG - Calidad frente a Costo", subtitulo)

    for ax, parametro in zip(np.atleast_1d(axes), parametros):
        grupo = df_params.loc[df_params["parametro"] == parametro].sort_values("valor")
        valores = grupo["valor"].to_numpy(dtype=float)
        fitness = grupo["fitness_medio"].to_numpy(dtype=float)
        desviacion = grupo["fitness_std"].to_numpy(dtype=float)
        tiempos = grupo["tiempo_medio_seg"].to_numpy(dtype=float)

        ax.errorbar(
            valores,
            fitness,
            yerr=desviacion,
            color=COLORES["blue"],
            lw=2.2,
            marker="o",
            ms=7,
            capsize=4,
            label="Fitness medio +- std",
            zorder=3,
        )
        ax.fill_between(
            valores, fitness - desviacion, fitness + desviacion, alpha=0.12, color=COLORES["blue"]
        )

        mejor = grupo.loc[grupo["fitness_medio"].idxmax()]
        ax.scatter(
            [float(mejor["valor"])],
            [float(mejor["fitness_medio"])],
            marker="*",
            s=340,
            color=COLORES["green"],
            edgecolors=COLORES["navy"],
            linewidths=0.9,
            zorder=5,
            label=f"Mejor: {_formatear(parametro, float(mejor['valor']))}",
        )

        if "es_valor_usado" in grupo.columns and grupo["es_valor_usado"].any():
            usado = float(grupo.loc[grupo["es_valor_usado"], "valor"].iloc[0])
            ax.axvline(
                usado,
                color=COLORES["gold"],
                lw=2,
                ls="--",
                zorder=2,
                label=f"Usado: {_formatear(parametro, usado)}",
            )

        ax.set_xlabel(etiquetas.get(parametro, parametro))
        ax.set_ylabel("Fitness medio")
        ax.grid(axis="y")
        ax.legend(fontsize=8, loc="lower right")

        eje_tiempo = ax.twinx()
        eje_tiempo.plot(
            valores,
            tiempos,
            color=COLORES["gray"],
            lw=1.4,
            ls=":",
            marker="s",
            ms=4,
            alpha=0.8,
        )
        eje_tiempo.set_ylabel(
            "Tiempo medio por corrida (s)", color=COLORES["gray"], fontsize=9
        )
        eje_tiempo.tick_params(axis="y", labelcolor=COLORES["gray"], labelsize=8)
        eje_tiempo.spines["right"].set_visible(True)
        eje_tiempo.spines["right"].set_color(COLORES["gray"])
        eje_tiempo.set_ylim(0, float(tiempos.max()) * 1.35)
        eje_tiempo.grid(False)

    fig.tight_layout()
    _guardar(fig, path, "fig10 sensibilidad de parametros")


def _formatear(parametro: str, valor: float) -> str:
    """
    Formatea el valor de un parámetro según su naturaleza.

    Args:
        parametro: Nombre del parámetro.
        valor: Valor a formatear.

    Returns:
        Cadena con los decimales adecuados: entero para la población, tres
        decimales para las probabilidades.
    """
    if parametro == "pop_size":
        return f"{int(valor)}"
    return f"{valor:.3f}".rstrip("0").rstrip(".")


# =========================================================================
# Figura 7 — Múltiples corridas y Wilcoxon
# =========================================================================


def plot_multiples_corridas(
    df_runs: pd.DataFrame,
    stats_dict: Mapping[str, Any],
    path: Union[str, Path],
) -> None:
    """
    Dispersión de N corridas del AG frente al Greedy, con el test de Wilcoxon.

    Args:
        df_runs: DataFrame de `analysis.statistics.multiples_corridas`.
        stats_dict: Diccionario de `analysis.statistics.analisis_wilcoxon`.
        path: Ruta del PNG de salida.

    Raises:
        ValueError: Si `df_runs` está vacío o no tiene la columna `fitness`.

    Referencia: docs/ARCHITECTURE.md — Wilcoxon signed-rank sobre t-test.
    """
    if df_runs is None or len(df_runs) == 0:
        raise ValueError("df_runs esta vacio: no hay corridas que graficar")
    if "fitness" not in df_runs.columns:
        raise ValueError("df_runs no tiene la columna 'fitness'")

    fitness = df_runs["fitness"].to_numpy(dtype=float)
    f_greedy = float(stats_dict.get("greedy", np.nan))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.5]})
    _titulo(
        fig,
        f"Analisis Estadistico de {len(fitness)} Corridas Independientes del AG",
        "Prueba de Wilcoxon signed-rank contra el Greedy deterministico",
    )

    caja = ax1.boxplot(
        [fitness],
        widths=0.45,
        patch_artist=True,
        medianprops={"color": COLORES["navy"], "lw": 2},
        flierprops={"markeredgecolor": COLORES["red"]},
    )
    for parche in caja["boxes"]:
        parche.set_facecolor(hexc(COLORES["blue"]))
        parche.set_alpha(0.35)
    dispersion = np.random.default_rng(0).uniform(-0.07, 0.07, fitness.size)
    ax1.scatter(
        1 + dispersion,
        fitness,
        color=COLORES["blue"],
        s=45,
        zorder=3,
        edgecolors="white",
        label="Corridas del AG",
    )
    if np.isfinite(f_greedy):
        ax1.axhline(
            f_greedy,
            color=COLORES["red"],
            lw=1.8,
            ls="--",
            label=f"Greedy (deterministico) = {f_greedy:.4f}",
        )
    ax1.set_xticks([1])
    ax1.set_xticklabels(["AG"])
    ax1.set_ylabel("Fitness Z (penalizado)")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(axis="y")

    ax2.plot(
        df_runs.get("corrida", np.arange(1, fitness.size + 1)),
        fitness,
        color=COLORES["blue"],
        lw=1.6,
        marker="o",
        ms=6,
        label="Fitness por corrida",
    )
    if np.isfinite(f_greedy):
        ax2.axhline(f_greedy, color=COLORES["red"], lw=1.8, ls="--", label="Greedy")
    media = float(stats_dict.get("media", fitness.mean()))
    ax2.axhline(media, color=COLORES["green"], lw=1.4, ls="-.", label=f"Media AG = {media:.4f}")
    ax2.set_xlabel("Corrida")
    ax2.set_ylabel("Fitness Z (penalizado)")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="y")

    ax2.text(
        0.02,
        0.97,
        _texto_wilcoxon(stats_dict, fitness.size),
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        color=COLORES["navy"],
        bbox={"facecolor": COLORES["lgray"], "edgecolor": COLORES["gray"], "alpha": 0.9},
    )

    fig.tight_layout()
    _guardar(fig, path, "fig7 multiples corridas")


def _texto_wilcoxon(stats_dict: Mapping[str, Any], n: int) -> str:
    """
    Redacta el bloque de texto con el resultado del contraste.

    Args:
        stats_dict: Salida de `analisis_wilcoxon`.
        n: Número de corridas.

    Returns:
        Texto multilínea en ASCII, apto para anotar en la figura.
    """
    p_valor = stats_dict.get("wilcoxon_p", float("nan"))
    significativo = bool(stats_dict.get("significativo", False))
    lineas = [
        f"n corridas    = {n}",
        f"media +- std  = {stats_dict.get('media', float('nan')):.4f} +- {stats_dict.get('std', float('nan')):.4f}",
        f"mediana       = {stats_dict.get('mediana', float('nan')):.4f}",
        f"rango         = [{stats_dict.get('minimo', float('nan')):.4f}, {stats_dict.get('maximo', float('nan')):.4f}]",
        f"mejora media  = {stats_dict.get('mejora_media_pct', float('nan')):+.2f} %",
        f"Wilcoxon stat = {stats_dict.get('wilcoxon_stat', float('nan')):.4f}",
        f"p-valor       = {p_valor:.6f}",
        f"p < {ALPHA:.2f}      -> {'SIGNIFICATIVO' if significativo else 'no significativo'}",
    ]
    if stats_dict.get("test_omitido"):
        lineas.append("test omitido (ver log)")
    return "\n".join(lineas)


# =========================================================================
# Figura 8 — Frente de Pareto (NSGA-II)
# =========================================================================


def plot_pareto(
    pareto: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
    path: Union[str, Path],
    punto_ag: Optional[Tuple[float, float]] = None,
    punto_greedy: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Frente de Pareto de NSGA-II: riesgo cubierto frente a costo.

    Args:
        pareto: Frente ya serializado, como DataFrame o secuencia de mapeos con
            las claves `f1_riesgo` y `f2_costo` (salida de
            `NSGA2.pareto_a_dicts()`).
        path: Ruta del PNG de salida.
        punto_ag: Par (f1, f2) de la solución del AG, para situarla en el espacio
            de objetivos.
        punto_greedy: Par (f1, f2) de la solución Greedy.

    Raises:
        ValueError: Si el frente está vacío o le faltan las claves de objetivos.

    Referencia: Tema 10 del curso CE UNI 2026 (frente de Pareto, NSGA-II).
    """
    df = pd.DataFrame(list(pareto)) if not isinstance(pareto, pd.DataFrame) else pareto.copy()
    if len(df) == 0:
        raise ValueError("El frente de Pareto esta vacio: no hay nada que graficar")
    faltantes = [c for c in ("f1_riesgo", "f2_costo") if c not in df.columns]
    if faltantes:
        raise ValueError(f"El frente no tiene las columnas de objetivos: {faltantes}")

    df = df.sort_values("f2_costo")
    f1 = df["f1_riesgo"].to_numpy(dtype=float)
    f2 = df["f2_costo"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    _titulo(
        fig,
        f"Frente de Pareto NSGA-II - {len(df)} Soluciones No Dominadas Factibles",
        "f1 = riesgo cubierto (max), f2 = costo (min)",
    )

    ax.step(f2, f1, where="post", color=COLORES["teal"], lw=1.2, alpha=0.6, zorder=1)
    tamanos = df["n_obras"].to_numpy(dtype=float) * 1.6 if "n_obras" in df.columns else 55
    dispersion = ax.scatter(
        f2,
        f1,
        c=f1 / np.maximum(f2, 1e-9),
        cmap="viridis",
        s=tamanos,
        edgecolors=COLORES["navy"],
        linewidths=0.6,
        zorder=3,
        label=f"Frente de Pareto (n={len(df)})",
    )
    barra = fig.colorbar(dispersion, ax=ax, pad=0.02)
    barra.set_label("Eficiencia f1/f2 (riesgo por kS/)", fontsize=9)

    if punto_ag is not None:
        ax.scatter(
            [punto_ag[1]],
            [punto_ag[0]],
            marker="*",
            s=420,
            color=COLORES["gold"],
            edgecolors=COLORES["navy"],
            linewidths=1.0,
            zorder=5,
            label="Solucion del AG (mono-objetivo)",
        )
    if punto_greedy is not None:
        ax.scatter(
            [punto_greedy[1]],
            [punto_greedy[0]],
            marker="D",
            s=110,
            color=COLORES["red"],
            edgecolors=COLORES["navy"],
            linewidths=0.8,
            zorder=5,
            label="Solucion Greedy",
        )

    ax.set_xlabel(f"f2 = Costo de supervision [{UNIDAD_MONETARIA}] (minimizar)")
    ax.set_ylabel("f1 = Riesgo cubierto, suma(R_n * x) (maximizar)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.6)

    fig.tight_layout()
    _guardar(fig, path, "fig8 pareto nsga2")


# =========================================================================
# Figura 9 — Building blocks (Teoría de Esquemas)
# =========================================================================


def plot_esquemas(
    df_bbs: pd.DataFrame,
    path: Union[str, Path],
    n: Optional[int] = None,
    pc: Optional[float] = None,
    pm: Optional[float] = None,
    top_n: int = 15,
) -> None:
    """
    Building blocks y su factor de crecimiento esperado (Teorema de Esquemas).

    Panel izquierdo: los `top_n` esquemas de mayor crecimiento, con la línea del
    umbral 1.0 que separa amplificación de extinción. Panel derecho: supervivencia
    frente a la longitud de definición δ(H) por orden o(H), que es lo que explica
    por qué los bloques dispersos no sobreviven al cruce de un punto.

    Args:
        df_bbs: DataFrame de `analysis.schema_theory.analizar_building_blocks`.
        path: Ruta del PNG de salida.
        n: Longitud del cromosoma, para el subtítulo.
        pc: Probabilidad de cruce usada, para el subtítulo.
        pm: Probabilidad de mutación usada, para el subtítulo.
        top_n: Número de esquemas a mostrar en el panel izquierdo.

    Raises:
        ValueError: Si el DataFrame está vacío o le faltan columnas.

    Referencia: Tema 4 del curso CE UNI 2026 (Teorema de Esquemas, cota de Goldberg).
    """
    if df_bbs is None or len(df_bbs) == 0:
        raise ValueError("df_bbs esta vacio: no hay esquemas que graficar")
    faltantes = [c for c in ("o_H", "delta_H", "crecimiento", "supervivencia") if c not in df_bbs.columns]
    if faltantes:
        raise ValueError(f"df_bbs no tiene las columnas requeridas: {faltantes}")

    partes = []
    if n is not None:
        partes.append(f"l={n}")
    if pc is not None:
        partes.append(f"pc={pc:.2f}")
    if pm is not None:
        partes.append(f"pm={pm:.4f}")
    subtitulo = "Cota inferior de Goldberg"
    if partes:
        subtitulo += " | " + " ".join(partes)

    colores_orden = {1: COLORES["blue"], 2: COLORES["teal"], 3: COLORES["gold"]}
    mejores = df_bbs.head(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    _titulo(fig, "Building Blocks - Factor de Crecimiento Esperado K_G * K_S", subtitulo)

    etiquetas = [
        f"#{int(fila.rank)} o={int(fila.o_H)} d={int(fila.delta_H)}"
        if "rank" in mejores.columns
        else f"o={int(fila.o_H)} d={int(fila.delta_H)}"
        for fila in mejores.itertuples()
    ]
    colores_barra = [
        colores_orden.get(int(o), COLORES["gray"]) for o in mejores["o_H"]
    ]
    barras = ax1.barh(
        etiquetas[::-1],
        mejores["crecimiento"].to_numpy()[::-1],
        color=colores_barra[::-1],
        edgecolor="white",
    )
    for barra, valor in zip(barras, mejores["crecimiento"].to_numpy()[::-1]):
        ax1.text(
            valor + 0.02,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.3f}",
            va="center",
            fontsize=8,
            color=COLORES["navy"],
        )
    ax1.axvline(
        UMBRAL_CRECIMIENTO,
        color=COLORES["red"],
        lw=1.8,
        ls="--",
        label=f"Umbral de amplificacion ({UMBRAL_CRECIMIENTO:.1f})",
    )
    ax1.set_xlabel("Crecimiento esperado por generacion")
    ax1.set_title(f"Top {len(mejores)} esquemas", fontsize=10, color=COLORES["blue"])
    # Holgura a la derecha para las etiquetas de valor, y leyenda FUERA del area de
    # datos: dentro tapaba las etiquetas de las ultimas barras.
    ax1.set_xlim(0, float(mejores["crecimiento"].max()) * 1.12)
    ax1.legend(
        handles=[
            Patch(color=colores_orden.get(o, COLORES["gray"]), label=f"orden o(H)={o}")
            for o in sorted(df_bbs["o_H"].unique())
        ]
        + [Patch(color=COLORES["red"], label="umbral 1.0")],
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=False,
    )
    ax1.grid(axis="x")

    for orden, grupo in df_bbs.groupby("o_H"):
        ax2.scatter(
            grupo["delta_H"],
            grupo["supervivencia"],
            color=colores_orden.get(int(orden), COLORES["gray"]),
            s=38,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label=f"o(H)={int(orden)}",
        )
    ax2.set_xlabel("delta(H) - longitud de definicion del esquema")
    ax2.set_ylabel("K_S - supervivencia al cruce y la mutacion")
    ax2.set_title(
        "El cruce de un punto destruye los bloques dispersos",
        fontsize=10,
        color=COLORES["blue"],
    )
    if pc is not None:
        ax2.axhline(
            1.0 - pc,
            color=COLORES["red"],
            lw=1.2,
            ls=":",
            alpha=0.8,
            label=f"cota 1-pc = {1.0 - pc:.2f}",
        )
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.6)

    fig.tight_layout()
    _guardar(fig, path, "fig9 esquemas")
