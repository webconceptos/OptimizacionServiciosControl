"""
Carga y preparación del dataset de obras públicas.

Este módulo es la **única** interfaz entre la Parte 2 (CE) y la Parte 1 (tesis):

    cargar_desde_api()  → GET {base_url}/obras/risk-scores  (Parte 1)
    cargar_csv()        → dataset exportado a disco
    preparar_dataset()  → valida, tipa y ordena las columnas antes de core/

No se normaliza aquí: la normalización min-max de R, M, P, E, G la aplica
`core/problem.py` al instanciar `Problema`.

Referencia: README.md, dependencia con la Parte 1 — docs/ARCHITECTURE.md "1. Separación Parte 1 / Parte 2"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from data.generator import (
    COLUMNAS_BASE,
    ESTADOS,
    MACROREGIONES,
    NIVELES_GOBIERNO,
    PROBS_ESTADOS,
    SECTORES,
    probs_nivel_gobierno,
)

logger = logging.getLogger(__name__)

#: Columnas imprescindibles para construir el Problema MCKP.
COLUMNAS_REQUERIDAS: List[str] = ["codigo", "R", "M", "P", "E", "G", "C", "macroregion"]

#: Columnas descriptivas: si faltan se completan con valores por defecto.
COLUMNAS_OPCIONALES: List[str] = [
    "clase_riesgo",
    "sector",
    "nivel_gobierno",
    "departamento",
    "estado",
]

#: Endpoint de la Parte 1 que expone los scores de riesgo del Random Forest.
RUTA_RISK_SCORES: str = "/obras/risk-scores"

#: Umbrales de clase de riesgo usados al reconstruir `clase_riesgo` desde R.
UMBRAL_EXTREMO: float = 0.60
UMBRAL_MEDIO: float = 0.30


class ErrorAPIParte1(ConnectionError):
    """
    La API de la Parte 1 no está disponible o respondió de forma inesperada.

    Hereda de `ConnectionError` para que el llamador pueda capturar ambos casos
    con un único `except ConnectionError` y caer al generador sintético.
    """


def cargar_csv(path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset de obras desde un archivo CSV.

    Args:
        path: Ruta del archivo CSV.

    Returns:
        DataFrame preparado y validado (ver `preparar_dataset`).

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas requeridas o hay valores fuera de rango.
    """
    ruta = Path(path)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el CSV de obras: {ruta}")

    df = pd.read_csv(ruta)
    logger.info("CSV cargado | %s | filas=%d columnas=%d", ruta, len(df), df.shape[1])
    return preparar_dataset(df)


def cargar_desde_api(base_url: str = "http://localhost:8000", timeout: float = 10.0) -> pd.DataFrame:
    """
    Obtiene los scores de riesgo Rᵢ desde la API de la tesis (Parte 1).

    Llama a `GET {base_url}/obras/risk-scores`. La API de la Parte 1 vive en
    C:\\IA_Investigacion\\Deteccion_Corrupcion\\ y corre por defecto en el
    puerto 8000 (la Parte 2 usa el 8001).

    Args:
        base_url: URL base de la API de la Parte 1.
        timeout: Segundos de espera antes de abortar la petición.

    Returns:
        DataFrame preparado con las columnas de `COLUMNAS_BASE`.

    Raises:
        ErrorAPIParte1: Si la API no está disponible, agota el timeout, responde
            con un código de error o devuelve un cuerpo sin la clave "obras".
            Es un `ConnectionError`, por lo que el llamador puede degradar a
            `data.generator.generar_obras()`.
    """
    import httpx  # Import local: la API de Parte 1 es opcional.

    url = f"{base_url.rstrip('/')}{RUTA_RISK_SCORES}"
    try:
        respuesta = httpx.get(url, timeout=timeout)
        respuesta.raise_for_status()
        cuerpo = respuesta.json()
    except (httpx.HTTPError, ConnectionError, ValueError) as exc:
        raise ErrorAPIParte1(
            f"API Parte 1 no disponible en {url} ({type(exc).__name__}: {exc}). "
            "Usar el generador sintético: data.generator.generar_obras()"
        ) from exc

    obras = cuerpo.get("obras") if isinstance(cuerpo, dict) else None
    if not obras:
        raise ErrorAPIParte1(
            f"Respuesta inesperada de {url}: falta la clave 'obras' o está vacía. "
            "Usar el generador sintético: data.generator.generar_obras()"
        )

    df = pd.DataFrame(obras)
    # La Parte 1 nombra la clase predicha como "clase"; la Parte 2 usa "clase_riesgo".
    if "clase" in df.columns and "clase_riesgo" not in df.columns:
        df = df.rename(columns={"clase": "clase_riesgo"})

    logger.info("Scores de riesgo obtenidos de la Parte 1 | %s | obras=%d", url, len(df))
    return preparar_dataset(df)


def completar_columnas_faltantes(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Completa columnas ausentes que la Parte 1 no expone (G, C y contextuales).

    La API de la Parte 1 devuelve Rᵢ y algunos atributos presupuestales, pero no
    el factor de cobertura geográfica G ni el costo de supervisión C, que son
    propios de la Parte 2. Se derivan con las mismas reglas de
    `data/generator.py` y se registra un warning, porque son valores simulados.

    Args:
        df: DataFrame parcial proveniente de la Parte 1 o de un CSV incompleto.
        seed: Semilla para las columnas que requieren sorteo.

    Returns:
        Copia del DataFrame con todas las columnas de `COLUMNAS_BASE` presentes.
    """
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(seed)
    completadas: List[str] = []

    if "macroregion" not in df.columns:
        df["macroregion"] = rng.integers(1, 6, n)
        completadas.append("macroregion")

    if "M" not in df.columns or df["M"].isna().all():
        df["M"] = np.clip(rng.lognormal(2.8, 1.2, n), 0.3, 800.0).round(2)
        completadas.append("M")

    if "P" not in df.columns or df["P"].isna().all():
        df["P"] = np.clip(df["M"].to_numpy() * rng.uniform(0.55, 1.35, n), 0.1, 900.0).round(2)
        completadas.append("P")

    if "E" not in df.columns or df["E"].isna().all():
        df["E"] = np.clip(rng.beta(1.8, 2.5, n), 0.01, 0.99).round(4)
        completadas.append("E")

    if "nivel_gobierno" not in df.columns:
        # Distribución real de la Parte 1 (data/calibracion_parte1.json).
        codigos = rng.choice([0, 1, 2], n, p=list(probs_nivel_gobierno()))
        df["nivel_gobierno"] = [NIVELES_GOBIERNO[int(v)] for v in codigos]
        completadas.append("nivel_gobierno")

    if "G" not in df.columns or df["G"].isna().all():
        # G depende del nivel de gobierno: mayor cobertura en obras nacionales.
        rangos = {"Nacional": (4, 2, 0.50, 0.99), "Regional": (3, 3, 0.25, 0.85)}
        valores = np.empty(n, dtype=float)
        for i, nivel in enumerate(df["nivel_gobierno"].astype(str)):
            a, b, lo, hi = rangos.get(nivel, (2, 4, 0.05, 0.55))
            valores[i] = np.clip(rng.beta(a, b), lo, hi)
        df["G"] = valores.round(4)
        completadas.append("G")

    if "C" not in df.columns or df["C"].isna().all():
        df["C"] = np.clip(df["M"].to_numpy() * rng.uniform(0.006, 0.022, n), 0.5, 60.0).round(3)
        completadas.append("C")

    if "clase_riesgo" not in df.columns:
        df["clase_riesgo"] = pd.cut(
            df["R"],
            bins=[-np.inf, UMBRAL_MEDIO, UMBRAL_EXTREMO, np.inf],
            labels=["Bajo Riesgo", "Med/Alt Riesgosa", "Extrem. Riesgosa"],
        ).astype(str)
        completadas.append("clase_riesgo")

    if "sector" not in df.columns:
        df["sector"] = [str(rng.choice(SECTORES)) for _ in range(n)]
        completadas.append("sector")

    if "departamento" not in df.columns:
        df["departamento"] = [
            str(rng.choice(MACROREGIONES[int(m)])) for m in df["macroregion"].to_numpy()
        ]
        completadas.append("departamento")

    if "estado" not in df.columns:
        df["estado"] = rng.choice(ESTADOS, n, p=list(PROBS_ESTADOS))
        completadas.append("estado")

    if completadas:
        logger.warning(
            "Columnas ausentes completadas con valores simulados: %s", ", ".join(completadas)
        )
    return df


def preparar_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Valida, tipa y ordena el dataset antes de entregarlo a `core/problem.py`.

    Pasos:
      1. Completa columnas ausentes (ver `completar_columnas_faltantes`).
      2. Verifica que estén las columnas de `COLUMNAS_REQUERIDAS`.
      3. Convierte los atributos numéricos y descarta filas con nulos en ellos.
      4. Elimina duplicados por `codigo`.
      5. Descarta columnas normalizadas preexistentes (R_n..G_n): las recalcula
         `core/problem.py` sobre el dataset efectivo.
      6. Valida rangos según docs/DATA_DICTIONARY.md.

    Args:
        df: DataFrame crudo (CSV, API de la Parte 1 o generador).
        seed: Semilla usada al completar columnas ausentes.

    Returns:
        DataFrame con las columnas de `COLUMNAS_BASE`, índice reiniciado y
        **sin** columnas normalizadas.

    Raises:
        ValueError: Si el DataFrame está vacío, faltan columnas requeridas tras
            el completado, no queda ninguna fila válida, o hay valores fuera de
            los rangos documentados.
    """
    if df is None or len(df) == 0:
        raise ValueError("El dataset de obras está vacío")

    df = completar_columnas_faltantes(df, seed=seed)

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {faltantes}")

    # --- Tipado ----------------------------------------------------------
    numericas = ["R", "M", "P", "E", "G", "C"]
    for col in numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["macroregion"] = pd.to_numeric(df["macroregion"], errors="coerce").astype("Int64")
    df["codigo"] = df["codigo"].astype(str)

    filas_previas = len(df)
    df = df.dropna(subset=numericas + ["macroregion"])
    if len(df) < filas_previas:
        logger.warning("Descartadas %d filas con valores nulos", filas_previas - len(df))
    if len(df) == 0:
        raise ValueError("No quedó ninguna fila válida tras descartar nulos")

    duplicados = int(df["codigo"].duplicated().sum())
    if duplicados:
        logger.warning("Descartadas %d obras con código CUI duplicado", duplicados)
        df = df.drop_duplicates(subset="codigo", keep="first")

    df["macroregion"] = df["macroregion"].astype(int)

    # --- Se descartan columnas normalizadas preexistentes ------------------
    normalizadas = [c for c in df.columns if c.endswith("_n")]
    if normalizadas:
        logger.debug("Descartadas columnas normalizadas del origen: %s", normalizadas)
        df = df.drop(columns=normalizadas)

    # --- Validación de rangos (docs/DATA_DICTIONARY.md) -------------------
    _validar_rangos(df)

    columnas = [c for c in COLUMNAS_BASE if c in df.columns]
    df = df[columnas].reset_index(drop=True)

    logger.info(
        "Dataset preparado | n=%d | R medio=%.4f | C total=%.2f kS/ | macroregiones=%s",
        len(df),
        df["R"].mean(),
        df["C"].sum(),
        sorted(df["macroregion"].unique().tolist()),
    )
    return df


def _validar_rangos(df: pd.DataFrame) -> None:
    """
    Verifica que los atributos estén dentro de los rangos documentados.

    Args:
        df: DataFrame ya tipado.

    Raises:
        ValueError: Si algún atributo sale de su rango válido.
    """
    errores: List[str] = []
    for col in ("R", "E", "G"):
        fuera = int(((df[col] < 0.0) | (df[col] > 1.0)).sum())
        if fuera:
            errores.append(f"{col}: {fuera} valores fuera de [0, 1]")
    for col in ("M", "P", "C"):
        fuera = int((df[col] <= 0).sum())
        if fuera:
            errores.append(f"{col}: {fuera} valores <= 0")
    fuera_macro = int((~df["macroregion"].isin(range(1, 6))).sum())
    if fuera_macro:
        errores.append(f"macroregion: {fuera_macro} valores fuera de [1, 5]")
    if errores:
        raise ValueError("Dataset inválido — " + "; ".join(errores))


def cargar_obras(
    csv_path: Optional[str | Path] = None,
    base_url: Optional[str] = None,
    n: int = 326,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Carga las obras con degradación en cascada: CSV → API Parte 1 → generador.

    Args:
        csv_path: Si se indica y existe, se usa como primera fuente.
        base_url: URL de la API de la Parte 1; si es None se omite ese intento.
        n: Número de obras a generar si no hay ninguna fuente disponible.
        seed: Semilla del generador sintético.

    Returns:
        DataFrame preparado con las columnas de `COLUMNAS_BASE`.
    """
    from data.generator import generar_obras

    if csv_path is not None and Path(csv_path).is_file():
        return cargar_csv(csv_path)

    if base_url:
        try:
            return cargar_desde_api(base_url)
        except ConnectionError as exc:
            logger.warning("%s", exc)

    logger.info("Se usa el generador sintético calibrado (n=%d, seed=%d)", n, seed)
    return preparar_dataset(generar_obras(n=n, seed=seed), seed=seed)
