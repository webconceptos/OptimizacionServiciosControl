"""
Schemas Pydantic de entrada y salida.

Contratos de datos compartidos entre el dominio (`core/`, `algorithms/`,
`analysis/`) y la capa HTTP (`api/`). No contienen lógica de optimización: solo
validan, documentan y serializan.

Referencia: docs/DATA_DICTIONARY.md — "Esquema Pydantic (API)"
            docs/API_SPEC.md — contratos de cada endpoint
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Claves válidas del diccionario de pesos de la función objetivo Z.
CLAVES_PESOS: tuple[str, ...] = ("R", "M", "P", "E", "G")


# =========================================================================
# Entrada de datos (dataset de obras)
# =========================================================================


class ObraInput(BaseModel):
    """
    Una obra pública candidata a servicio de control.

    `R` es el output del Random Forest de la Parte 1: P(Extrem. Riesgosa).
    """

    model_config = ConfigDict(extra="ignore")

    codigo: str = Field(..., description="Código Único de Inversión (CUI)")
    R: float = Field(..., ge=0.0, le=1.0, description="Score riesgo de corrupción (output Parte 1)")
    M: float = Field(..., gt=0, description="Monto viable (M S/)")
    P: float = Field(..., gt=0, description="PIM (M S/)")
    E: float = Field(..., ge=0.0, le=1.0, description="Nivel de ejecución")
    G: float = Field(..., ge=0.0, le=1.0, description="Cobertura geográfica")
    C: float = Field(..., gt=0, description="Costo supervisión (kS/)")
    macroregion: int = Field(..., ge=1, le=5, description="Macroregión del Perú (1..5)")


class ObraContextual(ObraInput):
    """Obra con los atributos descriptivos opcionales del dataset completo."""

    clase_riesgo: Optional[str] = Field(None, description="Clase predicha por el RF (Parte 1)")
    sector: Optional[str] = Field(None, description="Sector de intervención")
    nivel_gobierno: Optional[str] = Field(None, description="Nacional | Regional | Local")
    departamento: Optional[str] = Field(None, description="Departamento de ubicación")
    estado: Optional[str] = Field(None, description="Estado de avance de la obra")


class ObraRiskScore(BaseModel):
    """
    Registro tal como lo devuelve la API de la Parte 1 en `/obras/risk-scores`.

    Los campos G, C y contextuales pueden no venir en la respuesta: se completan
    en `data/loader.py` antes de construir el `Problema`.
    """

    model_config = ConfigDict(extra="allow")

    codigo: str
    R: float = Field(..., ge=0.0, le=1.0)
    clase: Optional[str] = Field(None, description="Clase de riesgo predicha")
    M: Optional[float] = Field(None, gt=0)
    P: Optional[float] = Field(None, gt=0)
    E: Optional[float] = Field(None, ge=0.0, le=1.0)
    G: Optional[float] = Field(None, ge=0.0, le=1.0)
    C: Optional[float] = Field(None, gt=0)
    departamento: Optional[str] = None
    macroregion: Optional[int] = Field(None, ge=1, le=5)


class RiskScoresResponse(BaseModel):
    """Envoltorio de la respuesta de `GET /obras/risk-scores` de la Parte 1."""

    obras: List[ObraRiskScore]


# =========================================================================
# Entrada de la optimización
# =========================================================================


class OptimizeRequest(BaseModel):
    """Cuerpo de `POST /optimize`."""

    model_config = ConfigDict(extra="forbid")

    algoritmo: Literal["AG", "NSGA2"] = Field("AG", description="Algoritmo a ejecutar")
    n_runs: int = Field(1, ge=1, le=20, description="Corridas independientes")
    seed: int = Field(42, description="Semilla base de reproducibilidad")
    pesos: Optional[Dict[str, float]] = Field(
        None, description="Pesos {R,M,P,E,G}; si es None se usan los del .env"
    )

    @field_validator("pesos")
    @classmethod
    def _validar_pesos(cls, pesos: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """
        Valida que las claves de pesos pertenezcan a {R, M, P, E, G} y sean >= 0.

        Args:
            pesos: Diccionario de pesos recibido, o None.

        Returns:
            El diccionario validado, o None.

        Raises:
            ValueError: Si hay claves desconocidas o pesos negativos.
        """
        if pesos is None:
            return None
        desconocidas = set(pesos) - set(CLAVES_PESOS)
        if desconocidas:
            raise ValueError(f"Claves de peso desconocidas: {sorted(desconocidas)}")
        negativos = {k: v for k, v in pesos.items() if v < 0}
        if negativos:
            raise ValueError(f"Los pesos no pueden ser negativos: {negativos}")
        return pesos


class ParetoQuery(BaseModel):
    """Parámetros de consulta de `GET /pareto`."""

    model_config = ConfigDict(extra="forbid")

    pop: int = Field(100, ge=10, le=500, description="Tamaño de población NSGA-II")
    gen: int = Field(200, ge=10, le=1000, description="Número de generaciones")


# =========================================================================
# Salida de la optimización
# =========================================================================


class SolucionResponse(BaseModel):
    """Portafolio óptimo devuelto por `POST /optimize` con n_runs=1."""

    fitness: float = Field(..., description="Fitness alcanzado (Z penalizado)")
    n_seleccionadas: int = Field(..., ge=0, description="Obras seleccionadas")
    costo_total: float = Field(..., ge=0, description="Costo de supervisión total (kS/)")
    factible: bool = Field(..., description="Cumple R1, R2 y R3")
    tiempo_seg: float = Field(..., ge=0, description="Tiempo de cómputo (s)")
    obras_seleccionadas: List[str] = Field(default_factory=list, description="Códigos CUI")
    distribucion_territorial: Dict[str, int] = Field(
        default_factory=dict, description="Obras por macroregión"
    )


class CorridaItem(BaseModel):
    """Resultado de una corrida individual del AG."""

    corrida: int = Field(..., ge=1)
    seed: int
    fitness: float
    n_seleccionadas: Optional[int] = Field(None, ge=0)
    costo: Optional[float] = Field(None, ge=0)
    r_medio: Optional[float] = Field(None, ge=0.0, le=1.0)
    factible: bool
    tiempo_seg: Optional[float] = Field(None, ge=0)


class EstadisticasResponse(BaseModel):
    """
    Estadísticas de múltiples corridas y prueba de Wilcoxon.

    Cubre `POST /optimize` con n_runs>1 y `GET /analysis/runs`.
    """

    media: float
    std: float
    mediana: float
    minimo: float
    maximo: float
    greedy: float
    mejora_media_pct: float
    wilcoxon_stat: float
    wilcoxon_p: float
    significativo: bool = Field(..., description="True si wilcoxon_p < 0.05")
    n_runs_factibles: Optional[int] = Field(None, ge=0)
    tiempo_total_seg: Optional[float] = Field(None, ge=0)
    corridas: List[CorridaItem] = Field(default_factory=list)
    mejor_solucion: Optional[SolucionResponse] = None


class ParetoPoint(BaseModel):
    """Una solución no dominada del frente de Pareto (NSGA-II)."""

    f1_riesgo: float = Field(..., description="Objetivo 1: riesgo cubierto (maximizar)")
    f2_costo: float = Field(..., description="Objetivo 2: costo en kS/ (minimizar)")
    n_obras: int = Field(..., ge=0)
    obras: List[str] = Field(default_factory=list, description="Códigos CUI")


class ParetoResponse(BaseModel):
    """Respuesta de `GET /pareto`."""

    n_soluciones: int = Field(..., ge=0)
    tiempo_seg: float = Field(..., ge=0)
    soluciones: List[ParetoPoint] = Field(default_factory=list)


class BuildingBlock(BaseModel):
    """Esquema H analizado por Teoría de Esquemas (Tema 4)."""

    rank: int = Field(..., ge=1)
    o_H: int = Field(..., ge=1, description="Orden del esquema")
    delta_H: int = Field(..., ge=0, description="Longitud de definición del esquema")
    f_H: float = Field(..., description="Fitness medio del esquema")
    fitness_relativo: float = Field(..., description="f(H) / f media de la población")
    crecimiento: float = Field(..., description="Factor de crecimiento esperado")
    descripcion: Optional[str] = None


class SchemasResponse(BaseModel):
    """Respuesta de `GET /analysis/schemas`."""

    building_blocks: List[BuildingBlock] = Field(default_factory=list)
    umbral_favorecido: float = Field(0.10, description="Umbral de proporción inicial del esquema")
    interpretacion: str = Field(
        "Esquemas con crecimiento > 1.0 son amplificados por selección",
        description="Lectura del resultado en términos del Teorema de Esquemas",
    )


class SensitivityResponse(BaseModel):
    """Respuesta de `GET /analysis/sensitivity`."""

    w1_rango: List[float] = Field(default_factory=list)
    fitness_optimo: List[float] = Field(default_factory=list)
    w1_usado: float = Field(..., ge=0.0, le=1.0)
    fitness_en_w1_usado: Optional[float] = None


# =========================================================================
# Operacionales
# =========================================================================


class HealthResponse(BaseModel):
    """Respuesta de `GET /health`."""

    status: Literal["ok", "degraded"] = "ok"
    version: str = "1.0.0"
    n_obras: int = Field(..., ge=0)
    timestamp: str = Field(..., description="Marca de tiempo ISO 8601")


class ErrorResponse(BaseModel):
    """Cuerpo de error uniforme de la API."""

    error: str = Field(..., description="Descripción del error")
    detail: Optional[str] = Field(None, description="Traceback o contexto adicional")
