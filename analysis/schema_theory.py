"""
Teoría de Esquemas aplicada al MCKP de obras públicas.

Teorema Fundamental de los AG (cota inferior de Goldberg) - Tema 4, Clase 03 (Tupac) - CE UNI 2026.

Un **esquema** H es una plantilla sobre el cromosoma con símbolos definidos y
comodines `*`. Aquí un esquema representa un **building block**: un subconjunto de
obras que aparecen juntas en el portafolio.

    H = 1 * * * 1 * * ... *      (obras 0 y 4 seleccionadas; el resto, indiferente)

Notación (Clase 03 — Teoría de Esquemas, Túpac):

    o(H)      número de símbolos definidos del esquema            (ec. 3/13)
    delta(H)  distancia entre los dos símbolos definidos más
              alejados                                            (ec. 4/13)
    l         LONGITUD DEL CROMOSOMA = problema.n (número de obras).
              Es el denominador l−1 del término de cruce; no confundir con
              `top_k` ni con el orden del esquema.

Teorema fundamental, en la forma multiplicativa de la ec. 25/30:

    m(H, t+1) >= m(H, t) · K_G · K_S

    K_G = f(H) / f_media          factor de ganancia por selección
    K_S = (1 − pc·delta(H)/(l−1)) · (1 − pm)^o(H)     factor de supervivencia

`crecimiento = K_G · K_S` es el factor esperado por generación: si es > 1 el
esquema se amplifica en la población, si es < 1 se extingue.

Sobre la cota inferior: `calcular_supervivencia` implementa las ecs. 20 y 24 de
la clase, que **omiten** el término `(1 − Pr(H,t))` de la ec. 23. Ese término
recoge que el cruce puede NO destruir el esquema si el otro padre también lo
contiene. Omitirlo hace la expresión pesimista, es decir, una **cota inferior**
del crecimiento real: un esquema que crece según esta fórmula crece con
seguridad, pero uno que no crece según ella todavía podría sobrevivir.

Referencia: Tema 4 del curso CE UNI 2026 — README.md, correspondencia con los temas
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.problem import Problema

logger = logging.getLogger(__name__)

#: Órdenes de esquema que se analizan (o(H) = 1, 2 y 3).
ORDENES_ANALIZADOS: Sequence[int] = (1, 2, 3)

#: Umbral del factor de crecimiento a partir del cual el esquema es favorecido.
UMBRAL_FAVORECIDO: float = 1.0

#: Columnas del DataFrame de building blocks, en orden.
COLUMNAS_BB: List[str] = [
    "rank",
    "o_H",
    "delta_H",
    "f_H",
    "f_media",
    "fitness_relativo",
    "supervivencia",
    "crecimiento",
    "favorecido",
    "posiciones",
    "obras",
    "descripcion",
]


def calcular_supervivencia(
    o_H: int, delta_H: int, l: int, pc: float, pm: float
) -> float:
    """
    Factor de supervivencia K_S de un esquema (cota inferior de Goldberg).

    K_S = (1 − pc·delta(H)/(l−1)) · (1 − pm)^o(H)

    El primer factor es la probabilidad de que el cruce de un punto NO corte
    dentro de la longitud de definición del esquema: hay l−1 puntos de corte
    posibles y delta(H) de ellos caen dentro de H. El segundo es la probabilidad
    de que ninguno de los o(H) símbolos definidos sea invertido por la mutación.

    Es una **cota inferior** (ecs. 20 y 24 de la Clase 03): omite el término
    `(1 − Pr(H,t))` de la ec. 23, que corrige por los cruces en que el segundo
    padre también contiene H y por tanto el esquema sobrevive igual. Con esa
    omisión, el valor devuelto subestima la supervivencia real.

    Args:
        o_H: Orden del esquema, o(H) = número de símbolos definidos.
        delta_H: Longitud de definición, delta(H) = distancia entre los dos
            símbolos definidos más alejados.
        l: Longitud del cromosoma (número de obras, `problema.n`). El denominador
            es l−1, el número de puntos de corte posibles.
        pc: Probabilidad de cruce.
        pm: Probabilidad de mutación por bit.

    Returns:
        Factor de supervivencia K_S en [0, 1].

    Raises:
        ValueError: Si l < 2 (no hay puntos de corte), o_H < 0, delta_H < 0, o si
            pc o pm caen fuera de [0, 1].

    Referencia: Tema 4 del curso CE UNI 2026 (ecs. 20 y 24, Clase 03).
    """
    if l < 2:
        raise ValueError(
            f"l debe ser >= 2 para que existan puntos de corte, se recibio l={l}. "
            "Recordar que l es la longitud del cromosoma (problema.n), no top_k"
        )
    if o_H < 0 or delta_H < 0:
        raise ValueError(f"o_H y delta_H deben ser >= 0, se recibio ({o_H}, {delta_H})")
    if not 0.0 <= pc <= 1.0:
        raise ValueError(f"pc debe estar en [0, 1], se recibio {pc}")
    if not 0.0 <= pm <= 1.0:
        raise ValueError(f"pm debe estar en [0, 1], se recibio {pm}")
    if delta_H > l - 1:
        logger.warning(
            "delta_H=%d excede l-1=%d: se recorta el termino de cruce a 0", delta_H, l - 1
        )

    prob_no_cortar = max(0.0, 1.0 - pc * delta_H / (l - 1))
    prob_no_mutar = (1.0 - pm) ** o_H
    return float(prob_no_cortar * prob_no_mutar)


def analizar_building_blocks(
    problema: Problema,
    top_k: int = 10,
    pc: Optional[float] = None,
    pm: Optional[float] = None,
    ordenes: Sequence[int] = ORDENES_ANALIZADOS,
) -> pd.DataFrame:
    """
    Identifica los building blocks del MCKP y su factor de crecimiento esperado.

    Procedimiento:

      1. Se toman las `top_k` obras de mayor ratio beneficio/costo: son las
         candidatas naturales a formar bloques constructivos, porque son las que
         el operador de reparación greedy nunca descarta.
      2. Se generan todos los esquemas de orden o ∈ `ordenes` combinando esas
         obras (C(top_k, o) esquemas por orden). Cada esquema fija esas o
         posiciones en 1 y deja el resto como comodín.
      3. Para cada esquema se calculan o(H), delta(H), f(H), K_S y el crecimiento
         K_G·K_S.

    Estimación de f(H): se usa la **media del beneficio unitario bᵢ de las obras
    definidas en H**, y f_media es la media de bᵢ sobre las n obras del universo.
    Es un proxy determinístico (sin ruido de Monte Carlo) que está justificado por
    la estructura aditiva de la función objetivo: como Z(X) = Σ bᵢ·xᵢ, la aptitud
    esperada de un individuo que contiene H se descompone en la contribución de
    las posiciones fijas más un término común a todos los esquemas, aportado por
    los comodines. Por tanto f(H)/f_media ordena los esquemas igual que la
    definición poblacional del teorema, sin depender de una población concreta.
    La lectura de fitness_relativo = 1.68 es "las obras de este bloque valen 1.68
    veces la obra promedio del universo".

    Args:
        problema: Problema MCKP; aporta ratio(), beneficio(), n y los parámetros.
        top_k: Número de obras de mayor ratio con las que se arman los esquemas.
        pc: Probabilidad de cruce; None usa `problema.params.PC`.
        pm: Probabilidad de mutación por bit; None usa `problema.params.PM`.
        ordenes: Órdenes de esquema a analizar; por defecto (1, 2, 3).

    Returns:
        DataFrame ordenado por `crecimiento` descendente, con las columnas de
        `COLUMNAS_BB`: rank, o_H, delta_H, f_H, f_media, fitness_relativo,
        supervivencia, crecimiento, favorecido, posiciones, obras y descripcion.

    Raises:
        ValueError: Si top_k < 1, si top_k excede el número de obras, o si algún
            orden solicitado no es positivo.

    Referencia: Tema 4 del curso CE UNI 2026 (Teorema de Esquemas, building blocks).
    """
    if top_k < 1:
        raise ValueError(f"top_k debe ser >= 1, se recibio {top_k}")
    if top_k > problema.n:
        raise ValueError(
            f"top_k={top_k} excede el numero de obras del problema (n={problema.n})"
        )
    if any(o < 1 for o in ordenes):
        raise ValueError(f"Los ordenes deben ser >= 1, se recibio {tuple(ordenes)}")

    # l es la LONGITUD DEL CROMOSOMA, no top_k ni el orden del esquema.
    l = problema.n
    prob_cruce = float(problema.params.PC if pc is None else pc)
    prob_mutacion = float(problema.params.PM if pm is None else pm)

    ratio = np.asarray(problema.ratio(), dtype=float)
    beneficio = np.asarray(problema.beneficio(), dtype=float)
    f_media = float(beneficio.mean())

    # Posiciones de las top_k obras por ratio b/C, en coordenadas del cromosoma.
    top = np.argsort(ratio)[::-1][:top_k]
    codigos = (
        problema.df["codigo"].astype(str).to_numpy()
        if "codigo" in problema.df.columns
        else np.array([str(i) for i in range(problema.n)])
    )

    logger.info(
        "Building blocks | l=%d (longitud del cromosoma) | top_k=%d | ordenes=%s | "
        "pc=%.2f pm=%.4f | f_media=%.6f",
        l,
        top_k,
        tuple(ordenes),
        prob_cruce,
        prob_mutacion,
        f_media,
    )

    filas: List[Dict[str, Any]] = []
    for orden in ordenes:
        if orden > top_k:
            logger.warning(
                "Se omite el orden %d: excede top_k=%d (no hay combinaciones)", orden, top_k
            )
            continue
        for combinacion in itertools.combinations(sorted(int(i) for i in top), orden):
            posiciones = list(combinacion)
            o_H = len(posiciones)
            # delta(H): distancia entre los dos simbolos definidos mas alejados.
            delta_H = int(max(posiciones) - min(posiciones))
            f_H = float(beneficio[posiciones].mean())
            fitness_relativo = f_H / f_media if f_media > 0 else float("nan")
            supervivencia = calcular_supervivencia(
                o_H, delta_H, l, prob_cruce, prob_mutacion
            )
            crecimiento = fitness_relativo * supervivencia
            filas.append(
                {
                    "o_H": o_H,
                    "delta_H": delta_H,
                    "f_H": f_H,
                    "f_media": f_media,
                    "fitness_relativo": fitness_relativo,
                    "supervivencia": supervivencia,
                    "crecimiento": crecimiento,
                    "favorecido": bool(crecimiento > UMBRAL_FAVORECIDO),
                    "posiciones": posiciones,
                    "obras": [codigos[i] for i in posiciones],
                    "descripcion": _describir(o_H, delta_H, crecimiento),
                }
            )

    if not filas:
        raise ValueError(
            f"No se generó ningún esquema con top_k={top_k} y ordenes={tuple(ordenes)}"
        )

    df = pd.DataFrame(filas).sort_values("crecimiento", ascending=False, kind="stable")
    df.insert(0, "rank", range(1, len(df) + 1))
    df = df[COLUMNAS_BB].reset_index(drop=True)

    favorecidos = int(df["favorecido"].sum())
    logger.info(
        "Building blocks | %d esquemas analizados | %d con crecimiento > %.1f | "
        "maximo=%.4f (o_H=%d, delta_H=%d)",
        len(df),
        favorecidos,
        UMBRAL_FAVORECIDO,
        float(df["crecimiento"].iloc[0]),
        int(df["o_H"].iloc[0]),
        int(df["delta_H"].iloc[0]),
    )
    if favorecidos == 0:
        logger.warning(
            "Ningun esquema supera el umbral de crecimiento: con pc=%.2f y delta(H) "
            "grande, el cruce de un punto destruye los bloques dispersos",
            prob_cruce,
        )
    return df


def _describir(o_H: int, delta_H: int, crecimiento: float) -> str:
    """
    Redacta la interpretación de un esquema para el informe y la API.

    Args:
        o_H: Orden del esquema.
        delta_H: Longitud de definición del esquema.
        crecimiento: Factor de crecimiento esperado K_G·K_S.

    Returns:
        Descripción en una línea.
    """
    if o_H == 1:
        sujeto = "obra individual de alto ratio beneficio/costo"
    else:
        sujeto = f"bloque de {o_H} obras de alto ratio beneficio/costo"
    destino = "se amplifica" if crecimiento > UMBRAL_FAVORECIDO else "se extingue"
    return (
        f"Esquema de orden {o_H} - {sujeto}, delta(H)={delta_H}; "
        f"{destino} (crecimiento={crecimiento:.4f})"
    )


def resumen_building_blocks(df_bbs: pd.DataFrame) -> Dict[str, Any]:
    """
    Resume el análisis de esquemas para `GET /analysis/schemas` y `metricas_ext.json`.

    Args:
        df_bbs: DataFrame devuelto por `analizar_building_blocks`.

    Returns:
        Diccionario con el conteo de esquemas, cuántos son favorecidos, el
        crecimiento máximo y el desglose por orden o(H).

    Raises:
        ValueError: Si el DataFrame está vacío.
    """
    if df_bbs is None or len(df_bbs) == 0:
        raise ValueError("df_bbs esta vacio: no hay esquemas que resumir")

    por_orden = {
        int(orden): {
            "n_esquemas": int(len(grupo)),
            "n_favorecidos": int(grupo["favorecido"].sum()),
            "crecimiento_max": float(grupo["crecimiento"].max()),
            "crecimiento_medio": float(grupo["crecimiento"].mean()),
            "supervivencia_media": float(grupo["supervivencia"].mean()),
        }
        for orden, grupo in df_bbs.groupby("o_H")
    }
    return {
        "n_esquemas": int(len(df_bbs)),
        "n_favorecidos": int(df_bbs["favorecido"].sum()),
        "umbral_favorecido": UMBRAL_FAVORECIDO,
        "crecimiento_max": float(df_bbs["crecimiento"].max()),
        "f_media": float(df_bbs["f_media"].iloc[0]),
        "por_orden": por_orden,
        "interpretacion": (
            "Esquemas con crecimiento > 1.0 son amplificados por seleccion; el valor "
            "es una cota inferior (Goldberg) porque omite el termino (1 - Pr(H,t)) de "
            "la ec. 23 de la Clase 03"
        ),
    }
