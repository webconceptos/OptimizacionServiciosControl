"""
Paleta institucional y estilo base de las figuras.

Centraliza los colores y los `rcParams` de matplotlib para que las nueve figuras
del informe tengan un aspecto homogéneo. Importar este módulo ya aplica el estilo:

    from visualization.palette import COLORES, hexc

El backend se fija en "Agg" antes de importar pyplot, porque las figuras se
generan sin interfaz gráfica (Docker, CI, `main.py --mode full`).

Referencia: README.md, figuras del informe
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import matplotlib

# Backend no interactivo: las figuras se escriben a disco, nunca se muestran.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (debe ir despues de matplotlib.use)

logger = logging.getLogger(__name__)

#: Paleta institucional del proyecto, en formato #RRGGBB.
COLORES: Dict[str, str] = {
    "navy": "#0B2D4E",   # titulos y texto destacado
    "blue": "#065A82",   # serie principal (AG)
    "teal": "#0D8FBF",   # serie secundaria
    "gold": "#F0A500",   # enfasis / riesgo alto
    "green": "#0D7A5F",  # factibilidad, benchmarks positivos
    "gray": "#64748B",   # series de apoyo, anotaciones
    "lgray": "#F0F7FB",  # fondos y rellenos suaves
    "red": "#CC2200",    # umbrales, restricciones, alertas
}

#: Color de fondo de figuras y ejes.
FONDO: str = "#F8FBFD"

#: Color de las rejillas.
COLOR_REJILLA: str = "#D0D0D0"

#: Resolución de guardado de todas las figuras.
DPI: int = 150

#: Unidad monetaria de todas las magnitudes de costo del proyecto.
#: La columna C del dataset esta en miles de soles: sum(C) ~ 209.7 y B ~ 73.4.
#: No usar el rotulo "74,370 kS/" de docs/DATA_DICTIONARY.md: tiene un factor
#: 1000 de mas respecto a la escala real de la columna.
UNIDAD_MONETARIA: str = "miles S/ (kS/)"


def hexc(h: str) -> Tuple[float, float, float]:
    """
    Convierte un color hexadecimal a una tupla RGB normalizada.

    Args:
        h: Color en formato "#RRGGBB" o "RRGGBB".

    Returns:
        Tupla (r, g, b) con cada componente en [0, 1], como espera matplotlib.

    Raises:
        ValueError: Si la cadena no tiene 6 dígitos hexadecimales.
    """
    limpio = h.lstrip("#").strip()
    if len(limpio) != 6:
        raise ValueError(f"Color hexadecimal invalido: {h!r} (se esperaban 6 digitos)")
    try:
        return tuple(int(limpio[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"Color hexadecimal invalido: {h!r}") from exc


def aplicar_estilo() -> None:
    """
    Aplica los `rcParams` base del proyecto.

    Se ejecuta automáticamente al importar el módulo; se expone como función para
    poder reaplicarlo si algún otro código cambia el estilo global.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": FONDO,
            "axes.facecolor": FONDO,
            "savefig.facecolor": FONDO,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORES["gray"],
            "axes.labelcolor": COLORES["navy"],
            "axes.titlecolor": COLORES["navy"],
            "text.color": COLORES["navy"],
            "xtick.color": COLORES["gray"],
            "ytick.color": COLORES["gray"],
            "grid.color": COLOR_REJILLA,
            "grid.linewidth": 0.5,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.dpi": 100,
            "savefig.dpi": DPI,
            "legend.framealpha": 0.7,
        }
    )


aplicar_estilo()
logger.debug("Paleta institucional aplicada | %d colores | backend=%s", len(COLORES), matplotlib.get_backend())
