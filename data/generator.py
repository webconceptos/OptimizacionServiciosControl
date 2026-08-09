"""
Generador del dataset sintético de obras públicas (calibrado con la Parte 1).

Rᵢ = P(Extrem. Riesgosa) es el output del Random Forest de la tesis (Parte 1).
Cuando la API de la Parte 1 no está disponible, este módulo produce un dataset
cuya **distribución de clases y de nivel de gobierno se lee de la auditoría
real** `data/calibracion_parte1.json` (generada por `scripts/auditar_parte1.py`),
en lugar de proporciones inventadas:

    45.4 % Extremadamente Riesgosa → Beta(7, 2) clipada a [0.60, 0.99]
    28.8 % Med/Alt Riesgosa        → Beta(3, 3) clipada a [0.30, 0.70]
    25.8 % Bajo Riesgo             → Beta(2, 7) clipada a [0.01, 0.40]

Ninguna de esas proporciones está hard-codeada: se resuelven en tiempo de
ejecución desde el JSON de calibración. Si el archivo no existe, el generador
falla con `CalibracionNoDisponible` pidiendo ejecutar la auditoría primero.

Alcance de la calibración (importante para la sección de limitaciones):

* **Sí calibrado con datos reales**: proporciones de las 3 clases de riesgo
  (desde `y_riesgo_obra` con MAPEO={0:0,1:0,2:1,3:2}) y distribución de
  `nivel_gobierno` (59.8 / 29.1 / 11.0 % Nacional / Regional / Local). Esto
  último corrige el sesgo del atributo G, que se condiciona al nivel de
  gobierno: la versión previa asumía 15/30/55 y subestimaba la cobertura
  geográfica del universo real, dominado por obras nacionales.
* **Deliberadamente NO calibrado — Betas de Rᵢ**: se conservan las Beta suaves
  por clase (2/7, 3/3, 7/2). El Rᵢ real medido en la auditoría (media 0.4591,
  std 0.3592, fuertemente bimodal) es *parcialmente in-sample* — 260 de las 326
  obras fueron de entrenamiento (test_size=0.2) —, por lo que su dispersión no
  es representativa de un escenario out-of-sample. Ajustar las Beta a esa forma
  importaría el sobreajuste del RF al dataset de la Parte 2.
* **Sintético — `macroregion` y la restricción R3**: la macroregión (1..5) no
  existe en el dataset real, donde el 65.03 % de las obras son
  MULTIDEPARTAMENTALES y no hay 5 grupos territoriales balanceados. R3
  (mínimo de obras por macroregión) es un **supuesto institucional de política
  de control de la CGR**, no un hecho derivado de los datos: modela la
  obligación de cobertura territorial del ente de control, y así debe
  interpretarse en el informe.
* **Sintéticos — M, P, E, C y `sector`**: son supuestos propios de la Parte 2
  (rangos INFOBRAS / SIAF / CGR). El dataset real no contiene esos atributos,
  de modo que no pueden validarse contra él.

El DataFrame devuelto **no está normalizado**: la normalización min-max de
R, M, P, E y G es responsabilidad de `core/problem.py`, que la aplica al
instanciar `Problema` sobre el dataset efectivamente cargado.

AVISO DE REPRODUCIBILIDAD: desde la recalibración, `generar_obras(326, 42)`
**ya no reproduce** el dataset de la version v1 del prototipo ni las métricas de
referencia de la version v1 (24.1835 ± 0.0977, Greedy 23.6860, etc.). Es el
comportamiento esperado: esas métricas se recomputan con el dataset recalibrado.

Referencia: Tema 7 del curso CE UNI 2026 (instancia del problema MCKP NP-Hard).
            README.md, formulacion del MCKP y dependencia con la Parte 1
            data/calibracion_parte1.json — evidencia de calibración
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Raíz del proyecto: data/generator.py -> data/ -> raíz.
RAIZ_PROYECTO: Path = Path(__file__).resolve().parent.parent

#: Nombre del archivo con la evidencia de calibración.
NOMBRE_CALIBRACION: str = "calibracion_parte1.json"

#: Ubicaciones donde se busca la calibración, por orden de preferencia. `data/` es
#: la ubicación actual, versionada junto al código para que el proyecto sea
#: reproducible por sí solo; `docs/` se sigue aceptando por compatibilidad con la
#: ubicación anterior.
RUTAS_CALIBRACION: Tuple[Path, ...] = (
    RAIZ_PROYECTO / "data" / NOMBRE_CALIBRACION,
    RAIZ_PROYECTO / "docs" / NOMBRE_CALIBRACION,
)


def ruta_calibracion() -> Path:
    """
    Localiza el archivo de calibración.

    Returns:
        La primera ubicación de `RUTAS_CALIBRACION` que exista; si no existe
        ninguna, la preferida, para que el mensaje de error apunte a ella.
    """
    for candidata in RUTAS_CALIBRACION:
        if candidata.is_file():
            return candidata
    return RUTAS_CALIBRACION[0]


#: Ubicación efectiva de la calibración al importar el módulo.
RUTA_CALIBRACION: Path = ruta_calibracion()


class CalibracionNoDisponible(FileNotFoundError):
    """
    No se encontró (o no es utilizable) `data/calibracion_parte1.json`.

    El generador se niega a producir datos con proporciones inventadas: sin la
    evidencia de la auditoría contra la Parte 1 no hay con qué calibrarlo.
    """


@lru_cache(maxsize=4)
def cargar_calibracion(ruta: Optional[Path] = None) -> Dict[str, object]:
    """
    Carga la evidencia de calibración de la Parte 1 (resultado cacheado).

    Args:
        ruta: Ruta alternativa al JSON de calibración; None busca en las
            ubicaciones de `RUTAS_CALIBRACION`.

    Returns:
        Diccionario con el contenido de `data/calibracion_parte1.json`.

    Raises:
        CalibracionNoDisponible: Si el archivo no existe o no es JSON válido.
    """
    ruta_json = Path(ruta) if ruta is not None else ruta_calibracion()
    if not ruta_json.is_file():
        raise CalibracionNoDisponible(
            f"No se encontró la calibración de la Parte 1: {ruta_json}\n"
            "  El generador no usa proporciones hard-codeadas: necesita la evidencia\n"
            "  de la auditoría contra el dataset real de la Parte 1.\n"
            "  Genera la calibracion primero:\n"
            "      python scripts/auditar_parte1.py"
        )
    try:
        return json.loads(ruta_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibracionNoDisponible(
            f"La calibración {ruta_json} no se pudo leer ({type(exc).__name__}: {exc}).\n"
            "  Regenérala con: python scripts/auditar_parte1.py"
        ) from exc


def _normalizar(valores: Sequence[float], contexto: str) -> Tuple[float, ...]:
    """
    Normaliza un vector de proporciones para que sume exactamente 1.

    Args:
        valores: Proporciones o porcentajes, en cualquier escala positiva.
        contexto: Nombre del campo, usado en el mensaje de error.

    Returns:
        Tupla de proporciones que suma 1.0.

    Raises:
        CalibracionNoDisponible: Si la suma no es positiva.
    """
    total = float(sum(valores))
    if total <= 0:
        raise CalibracionNoDisponible(
            f"Las proporciones de '{contexto}' en la calibración suman {total}; "
            "regenera data/calibracion_parte1.json con scripts/auditar_parte1.py"
        )
    return tuple(float(v) / total for v in valores)


def probs_clases(ruta: Optional[Path] = None) -> Tuple[float, float, float]:
    """
    Proporciones REALES de las 3 clases de riesgo, según la Parte 1.

    Se leen de `fracciones_clase_reales` del JSON de calibración; si esa clave
    no está, se derivan de `conteo_clase_reales`. Nunca están hard-codeadas.

    Args:
        ruta: Ruta alternativa al JSON de calibración.

    Returns:
        Tupla (p_bajo, p_med_alt, p_extrema) que suma 1.0, en el orden de
        `CLASES_RIESGO` (código 0, 1, 2). Valor real: ≈ (0.258, 0.288, 0.454).

    Raises:
        CalibracionNoDisponible: Si el JSON no existe o no trae las clases.
    """
    calib = cargar_calibracion(ruta)
    fracciones = calib.get("fracciones_clase_reales") or {}
    if not fracciones:
        conteos = calib.get("conteo_clase_reales") or {}
        if not conteos:
            raise CalibracionNoDisponible(
                "La calibración no trae 'fracciones_clase_reales' ni 'conteo_clase_reales'; "
                "regenérala con: python scripts/auditar_parte1.py"
            )
        fracciones = conteos
    faltantes = [e for e in CLASES_RIESGO.values() if e not in fracciones]
    if faltantes:
        raise CalibracionNoDisponible(
            f"La calibración no cubre las clases {faltantes}; "
            "regenérala con: python scripts/auditar_parte1.py"
        )
    orden = [fracciones[CLASES_RIESGO[codigo]] for codigo in sorted(CLASES_RIESGO)]
    return _normalizar(orden, "clases de riesgo")  # type: ignore[return-value]


def probs_nivel_gobierno(ruta: Optional[Path] = None) -> Tuple[float, float, float]:
    """
    Distribución REAL de nivel de gobierno, según la Parte 1.

    Se lee de `contexto.nivel_gobierno.real_pct_normalizado`. Corrige el sesgo
    del atributo G, que el generador condiciona al nivel de gobierno.

    Args:
        ruta: Ruta alternativa al JSON de calibración.

    Returns:
        Tupla (p_nacional, p_regional, p_local) que suma 1.0, en el orden de
        `NIVELES_GOBIERNO`. Valor real: ≈ (0.598, 0.291, 0.110).

    Raises:
        CalibracionNoDisponible: Si el JSON no trae la distribución.
    """
    calib = cargar_calibracion(ruta)
    contexto = calib.get("contexto") or {}
    nivel = contexto.get("nivel_gobierno") or {} if isinstance(contexto, dict) else {}
    pct = nivel.get("real_pct_normalizado") or {} if isinstance(nivel, dict) else {}
    faltantes = [n for n in NIVELES_GOBIERNO if n not in pct]
    if faltantes:
        raise CalibracionNoDisponible(
            "La calibración no trae 'contexto.nivel_gobierno.real_pct_normalizado' "
            f"para {faltantes}; regenérala con: python scripts/auditar_parte1.py"
        )
    return _normalizar([pct[n] for n in NIVELES_GOBIERNO], "nivel de gobierno")  # type: ignore[return-value]


#: Etiqueta de clase por código interno (2 = mayor riesgo).
CLASES_RIESGO: Dict[int, str] = {
    0: "Bajo Riesgo",
    1: "Med/Alt Riesgosa",
    2: "Extrem. Riesgosa",
}

#: Parámetros Beta y recorte de Rᵢ por clase (docs/DATA_DICTIONARY.md).
BETA_POR_CLASE: Dict[int, tuple[float, float, float, float]] = {
    2: (7.0, 2.0, 0.60, 0.99),
    1: (3.0, 3.0, 0.30, 0.70),
    0: (2.0, 7.0, 0.01, 0.40),
}

#: Sectores de intervención (INFOBRAS).
SECTORES: List[str] = [
    "Transporte",
    "Salud",
    "Educación",
    "Saneamiento",
    "Energía",
    "Agricultura",
]

#: Niveles de gobierno por código interno (0 = Nacional).
NIVELES_GOBIERNO: List[str] = ["Nacional", "Regional", "Local"]

#: Departamentos por macroregión del Perú (docs/DATA_DICTIONARY.md).
#: SINTÉTICO: la macroregión no existe en el dataset real de la Parte 1 (ver
#: docstring del módulo y `data/calibracion_parte1.json` → "macroregion").
MACROREGIONES: Dict[int, List[str]] = {
    1: ["Lima", "Ica", "Callao"],
    2: ["Arequipa", "Moquegua", "Tacna"],
    3: ["La Libertad", "Áncash", "Cajamarca", "Lambayeque"],
    4: ["Cusco", "Puno", "Apurímac", "Ayacucho"],
    5: ["Loreto", "Ucayali", "San Martín", "Huánuco", "Madre de Dios"],
}

#: Nombre corto de cada macroregión, para etiquetas y reportes.
NOMBRES_MACROREGIONES: Dict[int, str] = {
    1: "Lima",
    2: "Sur",
    3: "Norte",
    4: "Centro",
    5: "Oriente",
}

#: Estados de avance de una obra (INFOBRAS).
ESTADOS: List[str] = ["En ejecución", "Paralizado", "En liquidación", "Por iniciar"]

#: Probabilidad de cada estado.
PROBS_ESTADOS: tuple[float, float, float, float] = (0.55, 0.15, 0.15, 0.15)

#: Columnas del dataset sin normalizar, en el orden canónico.
COLUMNAS_BASE: List[str] = [
    "codigo",
    "R",
    "M",
    "P",
    "E",
    "G",
    "C",
    "clase_riesgo",
    "sector",
    "nivel_gobierno",
    "macroregion",
    "departamento",
    "estado",
]

#: Prefijo y offset del Código Único de Inversión simulado.
_CUI_OFFSET: int = 2_025_000


def _generar_scores_riesgo(clases: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Genera el score Rᵢ de cada obra según su clase de riesgo.

    Se conservan a propósito las Beta suaves por clase (2/7, 3/3, 7/2) del
    prototipo: el Rᵢ real medido en la auditoría es parcialmente in-sample
    (260 de 326 obras fueron de entrenamiento), así que su forma bimodal
    refleja el sobreajuste del RF y no un escenario out-of-sample.

    Se recorre obra por obra (en lugar de vectorizar por clase) para mantener la
    estructura de consumo del generador aleatorio del prototipo
    del prototipo previo. La equivalencia numérica con el dataset de aquella
    version quedó rota a propósito al recalibrar
    las proporciones de clase con los valores reales de la Parte 1.

    Args:
        clases: Vector de códigos de clase (0, 1, 2) por obra.
        rng: Generador aleatorio ya sembrado.

    Returns:
        Vector de scores Rᵢ ∈ [0.01, 0.99] de longitud len(clases).
    """
    r = np.zeros(len(clases), dtype=float)
    for i, c in enumerate(clases):
        a, b, lo, hi = BETA_POR_CLASE[int(c)]
        r[i] = np.clip(rng.beta(a, b), lo, hi)
    return r


def _garantizar_cobertura_macroregiones(macro_ids: np.ndarray, n_macro: int = 5) -> np.ndarray:
    """
    Asegura que cada macroregión aparezca al menos una vez en el dataset.

    Con n grande el muestreo uniforme ya cubre las 5 macroregiones; en datasets
    pequeños (tests, smoke tests) puede faltar alguna, lo que dejaría la
    restricción territorial R3 mal definida. La reparación es determinística
    (no consume números aleatorios) para no alterar la reproducibilidad.

    Args:
        macro_ids: Vector de macroregiones asignadas.
        n_macro: Número total de macroregiones esperadas.

    Returns:
        El vector con al menos una obra por macroregión (si len(macro_ids) lo permite).
    """
    if len(macro_ids) < n_macro:
        return macro_ids
    macro_ids = macro_ids.copy()
    for region in range(1, n_macro + 1):
        if np.any(macro_ids == region):
            continue
        # Se reasigna una obra de la macroregión más poblada.
        valores, cuentas = np.unique(macro_ids, return_counts=True)
        mayoritaria = valores[int(np.argmax(cuentas))]
        idx = int(np.where(macro_ids == mayoritaria)[0][0])
        macro_ids[idx] = region
        logger.debug("Macroregión %d ausente: se reasigna la obra %d", region, idx)
    return macro_ids


def _elegir_por_elemento(opciones: Sequence[str], veces: int, rng: np.random.Generator) -> List[str]:
    """
    Realiza `veces` sorteos escalares independientes sobre `opciones`.

    Args:
        opciones: Valores candidatos.
        veces: Número de sorteos.
        rng: Generador aleatorio ya sembrado.

    Returns:
        Lista de valores elegidos, de longitud `veces`.
    """
    return [str(rng.choice(opciones)) for _ in range(veces)]


def generar_obras(n: int = 326, seed: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético de obras públicas calibrado con la Parte 1.

    Rᵢ = P(Extrem.Riesgosa) es el output del RF de la tesis (Parte 1); aquí se
    simula con mezclas Beta cuyas proporciones de clase se **leen de la
    auditoría real** `data/calibracion_parte1.json` (≈ 25.8 / 28.8 / 45.4 %),
    igual que la distribución de `nivel_gobierno` (≈ 59.8 / 29.1 / 11.0 %). Los
    demás atributos siguen rangos realistas de INFOBRAS, SIAF, Invierte.pe y CGR
    según docs/DATA_DICTIONARY.md.

    `macroregion` es SINTÉTICA y la restricción R3 que la usa es un supuesto
    institucional de política de control de la CGR, no un hecho de los datos
    reales (ver el docstring del módulo).

    El resultado **no incluye** las columnas normalizadas R_n..G_n: la
    normalización min-max se aplica en `core/problem.py`.

    Args:
        n: Número de obras a generar (326 en el caso de estudio).
        seed: Semilla del generador, para reproducibilidad.

    Returns:
        DataFrame con las columnas de `COLUMNAS_BASE`:
        codigo, R, M, P, E, G, C, clase_riesgo, sector, nivel_gobierno,
        macroregion, departamento, estado.

    Raises:
        ValueError: Si n < 1.
        CalibracionNoDisponible: Si falta `data/calibracion_parte1.json`
            (ejecutar `python scripts/auditar_parte1.py`).

    Referencia: Tema 7 del curso CE UNI 2026 (instancia MCKP NP-Hard).
    """
    if n < 1:
        raise ValueError(f"n debe ser >= 1, se recibió {n}")

    # Proporciones reales de la Parte 1: fallan explícitamente si falta la auditoría.
    p_clases = probs_clases()
    p_nivel_gobierno = probs_nivel_gobierno()

    rng = np.random.default_rng(seed)

    # --- Clases de riesgo y score Rᵢ (output del RF de la Parte 1) ----------
    # Códigos en orden ascendente: 0=Bajo, 1=Med/Alt, 2=Extrema.
    clases = rng.choice([0, 1, 2], size=n, p=list(p_clases))
    ri = _generar_scores_riesgo(clases, rng)

    # --- Monto viable M (millones S/), lognormal con cola larga -------------
    mi = np.clip(rng.lognormal(mean=2.8, sigma=1.2, size=n), 0.3, 800.0)

    # --- PIM P: ligado a M con variabilidad presupuestal (SIAF) -------------
    pi = np.clip(mi * rng.uniform(0.55, 1.35, n), 0.1, 900.0)

    # --- Nivel de ejecución acumulada E ∈ [0, 1] (INFOBRAS) -----------------
    ei = np.clip(rng.beta(1.8, 2.5, n), 0.01, 0.99)

    # --- Cobertura geográfica G: mayor en obras nacionales ------------------
    nivel_gob = rng.choice([0, 1, 2], n, p=list(p_nivel_gobierno))
    g_nacional = np.clip(rng.beta(4, 2, n), 0.50, 0.99)
    g_regional = np.clip(rng.beta(3, 3, n), 0.25, 0.85)
    g_local = np.clip(rng.beta(2, 4, n), 0.05, 0.55)
    gi = np.where(nivel_gob == 0, g_nacional, np.where(nivel_gob == 1, g_regional, g_local))

    # --- Costo del servicio de control C (miles S/), proporcional al monto --
    ci = np.clip(mi * rng.uniform(0.006, 0.022, n), 0.5, 60.0)

    # --- Variables contextuales --------------------------------------------
    # El orden de estos sorteos importa: replica el flujo del prototipo.
    macro_ids = _garantizar_cobertura_macroregiones(rng.integers(1, 6, n))
    departamentos = [str(rng.choice(MACROREGIONES[int(m)])) for m in macro_ids]
    sectores = _elegir_por_elemento(SECTORES, n, rng)
    estados = rng.choice(ESTADOS, n, p=list(PROBS_ESTADOS))

    df = pd.DataFrame(
        {
            "codigo": [f"CUI-{_CUI_OFFSET + i:07d}" for i in range(n)],
            "R": np.round(ri, 4),
            "M": np.round(mi, 2),
            "P": np.round(pi, 2),
            "E": np.round(ei, 4),
            "G": np.round(gi, 4),
            "C": np.round(ci, 3),
            "clase_riesgo": [CLASES_RIESGO[int(c)] for c in clases],
            "sector": sectores,
            "nivel_gobierno": [NIVELES_GOBIERNO[int(v)] for v in nivel_gob],
            "macroregion": macro_ids.astype(int),
            "departamento": departamentos,
            "estado": estados,
        }
    )[COLUMNAS_BASE]

    logger.info(
        "Dataset generado | n=%d seed=%d | R medio=%.4f | C total=%.2f kS/ | clases=%s",
        n,
        seed,
        df["R"].mean(),
        df["C"].sum(),
        df["clase_riesgo"].value_counts().to_dict(),
    )
    logger.info(
        "Calibración real (Parte 1) | clases=%s | nivel_gobierno=%s | fuente=%s",
        tuple(round(p, 4) for p in p_clases),
        tuple(round(p, 4) for p in p_nivel_gobierno),
        RUTA_CALIBRACION.name,
    )
    return df
