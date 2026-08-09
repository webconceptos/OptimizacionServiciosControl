"""
Auditoría de calibración: datos REALES de la Parte 1 vs supuestos del generador.

Este script produce la evidencia metodológica que respalda (o cuestiona) los
supuestos de `data/generator.py`. Compara el dataset real de la tesis
(Parte 1, Random Forest sobre 326 obras) contra las distribuciones que el
generador sintético asume, y deja el resultado en `data/calibracion_parte1.json`
para citarlo en la sección de limitaciones del informe.

**Solo lectura sobre la Parte 1**: no modifica nada de
`C:\\IA_Investigacion\\Deteccion_Corrupcion\\` ni `data/generator.py`.

Uso:
    python scripts/auditar_parte1.py

Variables de entorno:
    PARTE1_DATASET_PATH  Ruta al parquet del dataset real (override del defecto).
    PARTE1_MODEL_PATH    Ruta al .pkl del RF de 3 clases (override del defecto).

Referencia: README.md, arquitectura de dos capas y dependencia con la Parte 1 — docs/DATA_DICTIONARY.md
            docs/TECHNICAL_DEBT.md (limitaciones del dataset sintético)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# La consola de Windows suele usar cp1252: evita UnicodeEncodeError al imprimir.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("auditar_parte1")

# Raíz de este proyecto: scripts/auditar_parte1.py -> scripts/ -> ComputacionEvolutiva/
RAIZ_PROYECTO: Path = Path(__file__).resolve().parent.parent

#: Carpeta de la Parte 1 (tesis de maestría), hermana de este proyecto.
RAIZ_PARTE1: Path = RAIZ_PROYECTO.parent / "Deteccion_Corrupcion"

#: Dataset real de 326 obras y 61 features usado para entrenar el RF.
RUTA_DATASET_DEFECTO: Path = RAIZ_PARTE1 / "data" / "processed" / "dataset_obra_v4_model.parquet"

#: Pipeline RF de 3 clases (variante final, notebook 08).
RUTA_MODELO_DEFECTO: Path = RAIZ_PARTE1 / "models" / "obra_v4" / "pipeline_rf_obra_3clases_final.pkl"

#: Salida de la auditoría.
RUTA_SALIDA: Path = RAIZ_PROYECTO / "data" / "calibracion_parte1.json"

#: Columna objetivo del dataset real (4 niveles originales).
COLUMNA_TARGET: str = "y_riesgo_obra"

#: Fusión de los 4 niveles originales a 3 clases (notebook 08 de la Parte 1).
MAPEO: Dict[int, int] = {0: 0, 1: 0, 2: 1, 3: 2}

#: Etiquetas de las 3 clases, en orden de código.
ETIQUETAS_CLASE: List[str] = ["Bajo Riesgo", "Med/Alt Riesgosa", "Extrem. Riesgosa"]

#: Supuestos del generador ANTES de recalibrar (baseline v1), en porcentaje.
#: Se conservan como línea base histórica: son el término de comparación que
#: evidencia la brecha que se corrigio al recalibrar. NO reflejan el generador actual,
#: que lee estas proporciones de `data/calibracion_parte1.json`.
SUPUESTO_CLASES_PCT: Dict[str, float] = {
    "Bajo Riesgo": 24.2,
    "Med/Alt Riesgosa": 27.6,
    "Extrem. Riesgosa": 48.2,
}

#: Supuesto de nivel de gobierno del generador v1 (baseline), en porcentaje.
SUPUESTO_NIVEL_GOBIERNO_PCT: Dict[str, float] = {
    "Nacional": 15.0,
    "Regional": 30.0,
    "Local": 55.0,
}

#: Sectores que enumera `data/generator.py` (SECTORES).
SECTORES_GENERADOR: List[str] = [
    "Transporte",
    "Salud",
    "Educación",
    "Saneamiento",
    "Energía",
    "Agricultura",
]

#: Media de Rᵢ del dataset sintético v1 (pre-recalibración), seed=42, n=326.
R_MEDIO_GENERADOR: float = 0.5672

#: Desviación tolerada, en puntos porcentuales, antes de exigir recalibración.
UMBRAL_DESVIACION_PP: float = 2.0

#: Etiqueta del dataset real que agrupa obras sin un departamento único.
ETIQUETA_MULTIDEPARTAMENTAL: str = "MULTIDEPARTAMENTAL"

ANCHO: int = 78


def _titulo(texto: str) -> None:
    """
    Imprime un encabezado de sección.

    Args:
        texto: Título a mostrar.
    """
    print("\n" + "=" * ANCHO)
    print(f" {texto}")
    print("=" * ANCHO)


def resolver_ruta_dataset() -> Path:
    """
    Determina la ruta del dataset real de la Parte 1.

    Returns:
        Ruta al parquet, tomada de PARTE1_DATASET_PATH o del valor por defecto.

    Raises:
        SystemExit: Si el archivo no existe, con un mensaje que explica cómo
            apuntar a la ubicación correcta.
    """
    override = os.getenv("PARTE1_DATASET_PATH")
    ruta = Path(override) if override else RUTA_DATASET_DEFECTO

    if not ruta.is_file():
        origen = "PARTE1_DATASET_PATH" if override else "ruta por defecto"
        raise SystemExit(
            "\nERROR: no se encontró el dataset real de la Parte 1.\n"
            f"  Ruta buscada ({origen}): {ruta}\n\n"
            "  Esta auditoría necesita el parquet de 326 obras de la tesis\n"
            "  (C:\\IA_Investigacion\\Deteccion_Corrupcion\\).\n"
            "  Si está en otra ubicación, indícala así:\n"
            '    PowerShell : $env:PARTE1_DATASET_PATH = "D:\\ruta\\dataset_obra_v4_model.parquet"\n'
            '    Bash       : export PARTE1_DATASET_PATH="/ruta/dataset_obra_v4_model.parquet"\n'
        )
    return ruta


def cargar_dataset_real(ruta: Path) -> pd.DataFrame:
    """
    Lee el dataset real de la Parte 1 (solo lectura).

    Args:
        ruta: Ruta al parquet.

    Returns:
        DataFrame con las 326 obras y sus 61 features + target.

    Raises:
        SystemExit: Si el parquet no puede leerse o falta la columna objetivo.
    """
    try:
        df = pd.read_parquet(ruta)
    except Exception as exc:  # pyarrow ausente, archivo corrupto, etc.
        raise SystemExit(
            f"\nERROR: no se pudo leer {ruta}\n  {type(exc).__name__}: {exc}\n"
            "  Verifica que pyarrow esté instalado: pip install -r requirements.txt\n"
        ) from exc

    if COLUMNA_TARGET not in df.columns:
        raise SystemExit(
            f"\nERROR: el dataset {ruta.name} no tiene la columna objetivo "
            f"'{COLUMNA_TARGET}'.\n  Columnas disponibles: {len(df.columns)}\n"
        )

    logger.info("Dataset real leído | %s | filas=%d columnas=%d", ruta.name, len(df), df.shape[1])
    return df


def distribucion_clases(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Calcula la distribución de las 3 clases de riesgo desde el target original.

    Aplica MAPEO = {0:0, 1:0, 2:1, 3:2} para fusionar los 4 niveles originales
    del dataset de la tesis en las 3 clases del modelo final.

    Args:
        df: Dataset real con la columna `y_riesgo_obra`.

    Returns:
        Tupla (conteos por etiqueta, porcentajes por etiqueta).

    Raises:
        SystemExit: Si el target contiene niveles fuera de MAPEO.
    """
    y = pd.to_numeric(df[COLUMNA_TARGET], errors="coerce").dropna().astype(int)
    desconocidos = sorted(set(y.unique()) - set(MAPEO))
    if desconocidos:
        raise SystemExit(
            f"\nERROR: '{COLUMNA_TARGET}' contiene niveles no previstos en MAPEO: {desconocidos}\n"
            f"  MAPEO esperado: {MAPEO}\n"
        )

    y3 = y.map(MAPEO)
    total = len(y3)
    conteos = {etiqueta: int((y3 == codigo).sum()) for codigo, etiqueta in enumerate(ETIQUETAS_CLASE)}
    porcentajes = {etiqueta: round(100.0 * n / total, 2) for etiqueta, n in conteos.items()}

    logger.info("Distribución de clases (real, n=%d): %s", total, porcentajes)
    return conteos, porcentajes


def calcular_ri_real(df: pd.DataFrame) -> Optional[Dict[str, object]]:
    """
    Calcula Rᵢ = P(Extrem. Riesgosa) con el RF real de la Parte 1.

    Paso opcional: si scikit-learn o joblib no están disponibles, o el .pkl no
    puede cargarse (versión de sklearn incompatible, archivo ausente), se omite
    con un aviso y la auditoría continúa.

    Args:
        df: Dataset real con las 61 features del modelo.

    Returns:
        Diccionario con estadísticos de Rᵢ y metadatos del modelo, o None si el
        paso se omitió.
    """
    try:
        import joblib  # noqa: F401  (import diferido: dependencia opcional)
        from sklearn.pipeline import Pipeline  # noqa: F401
    except ImportError as exc:
        logger.warning(
            "PASO OMITIDO (Ri real): falta scikit-learn o joblib (%s). "
            "Instala con: pip install -r requirements.txt",
            exc,
        )
        return None

    override = os.getenv("PARTE1_MODEL_PATH")
    ruta_modelo = Path(override) if override else RUTA_MODELO_DEFECTO
    if not ruta_modelo.is_file():
        logger.warning("PASO OMITIDO (Ri real): no se encontró el modelo en %s", ruta_modelo)
        return None

    try:
        import warnings

        import joblib

        with warnings.catch_warnings():
            # sklearn avisa si el .pkl se serializó con otra versión.
            warnings.simplefilter("ignore")
            pipeline = joblib.load(ruta_modelo)

        features = _features_del_modelo(ruta_modelo, df)
        faltantes = [f for f in features if f not in df.columns]
        if faltantes:
            logger.warning(
                "PASO OMITIDO (Ri real): al dataset le faltan %d features del modelo: %s",
                len(faltantes),
                faltantes[:5],
            )
            return None

        proba = pipeline.predict_proba(df[features])
        idx_extremo = _indice_clase_extrema(pipeline)
        ri = np.asarray(proba)[:, idx_extremo].astype(float)
    except Exception as exc:  # noqa: BLE001 — el paso es best-effort
        logger.warning(
            "PASO OMITIDO (Ri real): no se pudo evaluar el modelo (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return None

    stats: Dict[str, object] = {
        "modelo": str(ruta_modelo),
        "n": int(ri.size),
        "media": round(float(ri.mean()), 4),
        "std": round(float(ri.std(ddof=1)), 4),
        "min": round(float(ri.min()), 4),
        "q25": round(float(np.percentile(ri, 25)), 4),
        "mediana": round(float(np.percentile(ri, 50)), 4),
        "q75": round(float(np.percentile(ri, 75)), 4),
        "max": round(float(ri.max()), 4),
        "media_generador": R_MEDIO_GENERADOR,
        "desviacion_media_vs_generador": round(float(ri.mean()) - R_MEDIO_GENERADOR, 4),
        "advertencia": (
            "predict_proba se evalúa sobre las 326 obras, de las cuales 260 fueron "
            "de entrenamiento (test_size=0.2): los scores son parcialmente in-sample "
            "y su dispersión está subestimada respecto a un escenario out-of-sample."
        ),
    }
    logger.info("Ri real calculado | media=%.4f std=%.4f", stats["media"], stats["std"])
    return stats


def _features_del_modelo(ruta_modelo: Path, df: pd.DataFrame) -> List[str]:
    """
    Obtiene la lista de features esperada por el pipeline.

    Prefiere el `*_meta.json` que acompaña al modelo; si no existe, cae a las
    columnas del dataset distintas del target y de los identificadores.

    Args:
        ruta_modelo: Ruta del .pkl del modelo.
        df: Dataset real, usado como respaldo.

    Returns:
        Lista de nombres de columnas en el orden esperado por el modelo.
    """
    ruta_meta = ruta_modelo.with_name(f"{ruta_modelo.stem}_meta.json")
    if ruta_meta.is_file():
        meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
        features = meta.get("features")
        if features:
            return list(features)
    excluidas = {COLUMNA_TARGET, "IDENTIFICADOR_OBRA"}
    return [c for c in df.columns if c not in excluidas]


def _indice_clase_extrema(pipeline: object) -> int:
    """
    Localiza la columna de `predict_proba` que corresponde a "Extrem. Riesgosa".

    Args:
        pipeline: Pipeline de sklearn ya cargado.

    Returns:
        Índice de la columna de la clase de mayor riesgo (código 2). Si el
        pipeline no expone `classes_`, se asume la última columna.
    """
    clases = getattr(pipeline, "classes_", None)
    if clases is None:
        return -1
    clases_lista = list(clases)
    codigo_extremo = len(ETIQUETAS_CLASE) - 1  # 2
    if codigo_extremo in clases_lista:
        return clases_lista.index(codigo_extremo)
    if ETIQUETAS_CLASE[-1] in clases_lista:
        return clases_lista.index(ETIQUETAS_CLASE[-1])
    return len(clases_lista) - 1


def distribucion_contextual(df: pd.DataFrame) -> Dict[str, object]:
    """
    Registra la distribución real de nivel de gobierno y sector.

    Args:
        df: Dataset real de la Parte 1.

    Returns:
        Diccionario con conteos y porcentajes de `obra_ctx_nivel_gobierno` y
        `obra_ctx_sector`, más la comparación con los supuestos del generador.
    """
    resultado: Dict[str, object] = {}

    col_nivel = "obra_ctx_nivel_gobierno"
    if col_nivel in df.columns:
        conteos = df[col_nivel].astype(str).value_counts(dropna=False)
        total = int(conteos.sum())
        pct = {k: round(100.0 * v / total, 2) for k, v in conteos.items()}
        # El generador usa las etiquetas cortas Nacional/Regional/Local.
        pct_corto = {
            corto: round(
                100.0
                * sum(v for k, v in conteos.items() if corto.upper() in str(k).upper())
                / total,
                2,
            )
            for corto in SUPUESTO_NIVEL_GOBIERNO_PCT
        }
        resultado["nivel_gobierno"] = {
            "real_conteo": {str(k): int(v) for k, v in conteos.items()},
            "real_pct": pct,
            "real_pct_normalizado": pct_corto,
            "supuesto_generador_pct": SUPUESTO_NIVEL_GOBIERNO_PCT,
            "desviaciones_pp": {
                k: round(pct_corto.get(k, 0.0) - v, 2)
                for k, v in SUPUESTO_NIVEL_GOBIERNO_PCT.items()
            },
        }
    else:
        logger.warning("El dataset real no tiene la columna %s", col_nivel)

    col_sector = "obra_ctx_sector"
    if col_sector in df.columns:
        conteos = df[col_sector].astype(str).value_counts(dropna=False)
        total = int(conteos.sum())
        resultado["sector"] = {
            "real_conteo": {str(k): int(v) for k, v in conteos.items()},
            "real_pct": {str(k): round(100.0 * v / total, 2) for k, v in conteos.items()},
            "n_categorias_real": int(conteos.size),
            "categorias_generador": SECTORES_GENERADOR,
            "nota": (
                f"El dataset real usa {conteos.size} categorías de sector (taxonomía SEACE, "
                f"p. ej. TRABAJO, OTROS, ORDEN PÚBLICO Y SEGURIDAD) frente a las "
                f"{len(SECTORES_GENERADOR)} del generador, que además las asigna de forma "
                "uniforme. Las taxonomías no son comparables una a una: el sector del "
                "dataset sintético es ilustrativo y no debe usarse para inferencia sectorial."
            ),
        }
    else:
        logger.warning("El dataset real no tiene la columna %s", col_sector)

    return resultado


def nota_macroregion(df: pd.DataFrame) -> Dict[str, object]:
    """
    Documenta que la macroregión del dataset sintético es una construcción propia.

    El dataset real no tiene macroregiones: la mayoría de obras son
    MULTIDEPARTAMENTALES, por lo que no existen 5 grupos territoriales
    balanceados como los que asume la restricción R3.

    Args:
        df: Dataset real de la Parte 1.

    Returns:
        Diccionario con el porcentaje multidepartamental, el conteo por
        departamento y la nota metodológica.
    """
    col = "obra_ctx_departamento"
    info: Dict[str, object] = {
        "es_sintetica": True,
        "nota": (
            "La columna 'macroregion' (1..5) NO existe en el dataset real: es una "
            "construcción de la Parte 2 para dar contenido a la restricción territorial "
            "R3. El dataset real solo tiene 'obra_ctx_departamento', dominado por obras "
            "MULTIDEPARTAMENTALES, de modo que no hay 5 macroregiones balanceadas ni una "
            "asignación unívoca obra->macroregión. R3 debe leerse como un escenario "
            "institucional plausible, no como una restricción observada."
        ),
    }

    if col not in df.columns:
        logger.warning("El dataset real no tiene la columna %s", col)
        info["pct_multidepartamental"] = None
        return info

    serie = df[col].astype(str).str.upper()
    pct_multi = round(100.0 * serie.str.contains(ETIQUETA_MULTIDEPARTAMENTAL).mean(), 2)
    conteos = df[col].astype(str).value_counts(dropna=False)
    info["pct_multidepartamental"] = pct_multi
    info["n_departamentos_distintos"] = int(conteos.size)
    info["departamentos_top10"] = {str(k): int(v) for k, v in conteos.head(10).items()}
    logger.info("Departamento: %.2f%% MULTIDEPARTAMENTAL en el dataset real", pct_multi)
    return info


def evaluar_desviaciones(
    reales_pct: Dict[str, float], supuestos_pct: Dict[str, float]
) -> Tuple[Dict[str, float], float, str]:
    """
    Compara distribuciones y emite el veredicto de calibración.

    Args:
        reales_pct: Porcentajes observados en el dataset real.
        supuestos_pct: Porcentajes asumidos por `data/generator.py`.

    Returns:
        Tupla (desviaciones en puntos porcentuales, desviación máxima absoluta,
        veredicto: "aceptable" o "recalibrar").
    """
    desviaciones = {
        etiqueta: round(reales_pct.get(etiqueta, 0.0) - supuesto, 2)
        for etiqueta, supuesto in supuestos_pct.items()
    }
    maxima = max(abs(v) for v in desviaciones.values()) if desviaciones else 0.0
    veredicto = "aceptable" if maxima <= UMBRAL_DESVIACION_PP else "recalibrar"
    return desviaciones, round(maxima, 2), veredicto


def imprimir_reporte(informe: Dict[str, object]) -> None:
    """
    Imprime en consola la tabla comparativa REAL vs generador.

    Args:
        informe: Diccionario completo de la auditoría.
    """
    _titulo("AUDITORIA DE CALIBRACION — Parte 1 (real) vs data/generator.py")
    print(f" Dataset real : {informe['dataset_real']}")
    print(f" n obras      : {informe['n_obras']}")
    print(f" Target       : {COLUMNA_TARGET}  MAPEO={MAPEO}")
    print(f" Umbral desv. : {UMBRAL_DESVIACION_PP:.1f} puntos porcentuales")

    _titulo("1. DISTRIBUCION DE CLASES DE RIESGO")
    print(f" {'Clase':<20}{'Real N':>8}{'Real %':>9}{'Gener. %':>10}{'Desv. pp':>10}   Estado")
    print(" " + "-" * (ANCHO - 2))
    conteos = informe["conteo_clase_reales"]
    reales = informe["proporciones_clase_reales"]
    supuestos = informe["supuestos_generador"]["clases_pct"]
    desviaciones = informe["desviaciones_clase_pp"]
    for etiqueta in ETIQUETAS_CLASE:
        d = desviaciones[etiqueta]
        estado = "DESVIA" if abs(d) > UMBRAL_DESVIACION_PP else "ok"
        print(
            f" {etiqueta:<20}{conteos[etiqueta]:>8}{reales[etiqueta]:>9.2f}"
            f"{supuestos[etiqueta]:>10.2f}{d:>+10.2f}   {estado}"
        )
    print(" " + "-" * (ANCHO - 2))
    print(f" Desviacion maxima: {informe['desviacion_maxima_pp']:+.2f} pp")

    ri = informe.get("ri_real")
    _titulo("2. SCORE Ri = P(Extrem. Riesgosa) — modelo RF real")
    if ri:
        print(
            f" media={ri['media']:.4f}  std={ri['std']:.4f}  min={ri['min']:.4f}  "
            f"max={ri['max']:.4f}"
        )
        print(f" cuartiles: q25={ri['q25']:.4f}  mediana={ri['mediana']:.4f}  q75={ri['q75']:.4f}")
        print(
            f" media generador sintetico v1 = {ri['media_generador']:.4f}  ->  "
            f"desviacion {ri['desviacion_media_vs_generador']:+.4f}"
        )
        print(f" [!] {ri['advertencia']}")
    else:
        print(f" PASO OMITIDO: {informe.get('ri_real_motivo_omision')}")

    _titulo("3. NIVEL DE GOBIERNO")
    nivel = informe.get("contexto", {}).get("nivel_gobierno")
    if nivel:
        print(f" {'Nivel':<20}{'Real %':>9}{'Gener. %':>10}{'Desv. pp':>10}   Estado")
        print(" " + "-" * (ANCHO - 2))
        for k, supuesto in nivel["supuesto_generador_pct"].items():
            real = nivel["real_pct_normalizado"].get(k, 0.0)
            d = nivel["desviaciones_pp"][k]
            estado = "DESVIA" if abs(d) > UMBRAL_DESVIACION_PP else "ok"
            print(f" {k:<20}{real:>9.2f}{supuesto:>10.2f}{d:>+10.2f}   {estado}")
        print(" " + "-" * (ANCHO - 2))
        print(" Etiquetas reales: " + ", ".join(f"{k}={v}" for k, v in nivel["real_conteo"].items()))
    else:
        print(" No disponible en el dataset real.")

    _titulo("4. SECTOR")
    sector = informe.get("contexto", {}).get("sector")
    if sector:
        print(f" Categorias reales: {sector['n_categorias_real']} | generador: "
              f"{len(SECTORES_GENERADOR)}")
        top = list(sector["real_pct"].items())[:6]
        print(" Top real: " + ", ".join(f"{k} {v:.1f}%" for k, v in top))
        print(f" [!] {sector['nota']}")
    else:
        print(" No disponible en el dataset real.")

    _titulo("5. MACROREGION (restriccion R3)")
    macro = informe["macroregion"]
    print(f" Multidepartamental en el dataset real: {macro.get('pct_multidepartamental')}%")
    print(f" [!] {macro['nota']}")

    _titulo(f"VEREDICTO: {str(informe['veredicto']).upper()}")
    for linea in informe["conclusiones"]:
        print(f" - {linea}")
    print(f"\n Evidencia guardada en: {informe['ruta_salida']}")
    print("=" * ANCHO)


def construir_conclusiones(informe: Dict[str, object]) -> List[str]:
    """
    Redacta las conclusiones accionables de la auditoría.

    Args:
        informe: Diccionario parcial con las métricas ya calculadas.

    Returns:
        Lista de conclusiones en texto plano, para consola y JSON.
    """
    conclusiones: List[str] = []
    desviaciones = informe["desviaciones_clase_pp"]
    fuera = {k: v for k, v in desviaciones.items() if abs(v) > UMBRAL_DESVIACION_PP}

    if fuera:
        detalle = ", ".join(f"{k} {v:+.2f} pp" for k, v in fuera.items())
        conclusiones.append(
            f"Clases de riesgo: {len(fuera)} de {len(desviaciones)} exceden el umbral "
            f"de {UMBRAL_DESVIACION_PP:.1f} pp ({detalle}). Ajustar PROBS_CLASES en "
            "data/generator.py a las proporciones reales."
        )
    else:
        conclusiones.append(
            f"Clases de riesgo: todas las desviaciones estan dentro de "
            f"{UMBRAL_DESVIACION_PP:.1f} pp; la calibracion de PROBS_CLASES es adecuada."
        )

    nivel = informe.get("contexto", {}).get("nivel_gobierno")
    if nivel:
        peores = {k: v for k, v in nivel["desviaciones_pp"].items() if abs(v) > UMBRAL_DESVIACION_PP}
        if peores:
            detalle = ", ".join(f"{k} {v:+.2f} pp" for k, v in peores.items())
            conclusiones.append(
                f"Nivel de gobierno: PROBS_NIVEL_GOBIERNO esta invertido respecto a la "
                f"realidad ({detalle}). Afecta al atributo G (cobertura geografica), que "
                "el generador condiciona al nivel de gobierno."
            )

    sector = informe.get("contexto", {}).get("sector")
    if sector:
        conclusiones.append(
            f"Sector: taxonomias no comparables ({sector['n_categorias_real']} categorias "
            f"reales vs {len(SECTORES_GENERADOR)} sinteticas asignadas uniformemente); "
            "es una variable ilustrativa, no calibrada."
        )

    conclusiones.append(
        "Macroregion y R3: sinteticas. "
        f"{informe['macroregion'].get('pct_multidepartamental')}% de las obras reales son "
        "MULTIDEPARTAMENTALES, por lo que no existen 5 macroregiones balanceadas."
    )

    if informe.get("ri_real"):
        ri = informe["ri_real"]
        conclusiones.append(
            f"Ri real (in-sample): media {ri['media']:.4f} vs {ri['media_generador']:.4f} del "
            f"generador ({ri['desviacion_media_vs_generador']:+.4f}); std real {ri['std']:.4f}."
        )
    else:
        conclusiones.append(
            "Ri real no evaluado (paso omitido): la comparacion de la distribucion de scores "
            "queda pendiente."
        )

    conclusiones.append(
        "Los atributos M, P, E, G y C del generador son supuestos propios de la Parte 2 "
        "(rangos INFOBRAS/SIAF/CGR): el dataset real no los contiene y no pueden validarse aqui."
    )
    return conclusiones


def main() -> int:
    """
    Ejecuta la auditoría completa y escribe `data/calibracion_parte1.json`.

    Returns:
        0 si la auditoría se completó (independientemente del veredicto).
    """
    ruta_dataset = resolver_ruta_dataset()
    df = cargar_dataset_real(ruta_dataset)

    conteos, reales_pct = distribucion_clases(df)
    desviaciones, maxima, veredicto = evaluar_desviaciones(reales_pct, SUPUESTO_CLASES_PCT)
    ri = calcular_ri_real(df)
    contexto = distribucion_contextual(df)
    macro = nota_macroregion(df)

    informe: Dict[str, object] = {
        "generado_en": datetime.now().isoformat(timespec="seconds"),
        "descripcion": (
            "Auditoria de calibracion del generador sintetico de la Parte 2 contra el "
            "dataset real de la Parte 1. Porcentajes en puntos porcentuales (0-100)."
        ),
        "dataset_real": str(ruta_dataset),
        "n_obras": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "target": COLUMNA_TARGET,
        "mapeo_3clases": {str(k): v for k, v in MAPEO.items()},
        "etiquetas_clase": ETIQUETAS_CLASE,
        "conteo_clase_reales": conteos,
        "proporciones_clase_reales": reales_pct,
        "fracciones_clase_reales": {k: round(v / 100.0, 4) for k, v in reales_pct.items()},
        "supuestos_generador": {
            "clases_pct": SUPUESTO_CLASES_PCT,
            "nivel_gobierno_pct": SUPUESTO_NIVEL_GOBIERNO_PCT,
            "sectores": SECTORES_GENERADOR,
            "r_medio_sintetico": R_MEDIO_GENERADOR,
            "version_generador": "v1 (pre-recalibracion, baseline historico)",
            "fuente": (
                "data/generator.py ANTES de la recalibracion. Desde entonces, el "
                "generador lee 'fracciones_clase_reales' y "
                "'contexto.nivel_gobierno.real_pct_normalizado' de este mismo archivo; "
                "estos supuestos v1 se conservan como termino de comparacion."
            ),
        },
        "desviaciones_clase_pp": desviaciones,
        "desviacion_maxima_pp": maxima,
        "umbral_desviacion_pp": UMBRAL_DESVIACION_PP,
        "veredicto": veredicto,
        "ri_real": ri,
        "contexto": contexto,
        "macroregion": macro,
        "ruta_salida": str(RUTA_SALIDA),
    }
    if ri is None:
        informe["ri_real_motivo_omision"] = (
            "No se pudo cargar o evaluar el pipeline RF de 3 clases (ver los WARNING del log)."
        )
    informe["conclusiones"] = construir_conclusiones(informe)

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=True: los acentos quedan como escapes \uXXXX, de modo que el
    # JSON se puede leer con open() sin declarar encoding (cp1252 en Windows).
    RUTA_SALIDA.write_text(json.dumps(informe, indent=2), encoding="utf-8")
    logger.info("Evidencia escrita en %s", RUTA_SALIDA)

    imprimir_reporte(informe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
