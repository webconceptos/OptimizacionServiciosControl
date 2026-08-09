# Optimización Evolutiva de Servicios de Control

Selección óptima de obras públicas para servicios de control preventivo mediante
**Algoritmo Genético binario** y **NSGA-II**. Trabajo final del curso de
Computación Evolutiva (Parte 2 de una investigación de dos capas).

---

## 1. Problema

El ente de control debe elegir, de un universo de 326 obras públicas, un
subconjunto al que dedicar sus servicios de control, con un presupuesto y una
capacidad operativa limitados. Es un **MCKP** (Multi-Constraint Knapsack Problem),
NP-Hard.

### Variables de decisión

```
xᵢ ∈ {0, 1}   para cada obra i = 1..326
X = (x₁, ..., x₃₂₆)   →   cromosoma binario de 326 bits
```

### Función objetivo — maximizar Z

```
Z(X) = Σᵢ (w₁·R̃ᵢ + w₂·M̃ᵢ + w₃·P̃ᵢ + w₄·Ẽᵢ + w₅·G̃ᵢ) · xᵢ
```

| Peso | Valor | Atributo |
|---|---|---|
| w₁ | 0.40 | `R` = probabilidad de riesgo de corrupción (entrada de la Capa 1) |
| w₂ | 0.25 | `M` = monto viable de inversión |
| w₃ | 0.15 | `P` = presupuesto institucional modificado (PIM) |
| w₄ | 0.10 | `E` = nivel de ejecución acumulada |
| w₅ | 0.10 | `G` = factor de cobertura geográfica |

Los cinco atributos se normalizan por min-max a [0,1] al construir el problema,
para que los pesos sean comparables entre sí.

### Restricciones institucionales

```
R1 (presupuestal)  : Σᵢ Cᵢ·xᵢ ≤ B = Σᵢ Cᵢ · 0.35
R2 (capacidad)     : Σᵢ xᵢ ≤ K = 50 obras
R3 (territorial)   : Σ{xᵢ : macroregión(i) = r} ≥ mᵣ = 5   ∀r ∈ {1..5}
```

Con el dataset del trabajo, **B = 73.3887 kS/** (35 % de 209.682 kS/). Todos los
montos del proyecto están en **miles de soles (kS/)**, la escala de la columna `C`.

R1 y R2 se garantizan por un operador de reparación greedy; R3 se induce por
penalización estática, `fitness(X) = Z(X) − α·violación(X)` con α = 2.5.

> R3 es un **supuesto institucional de política de control**, no un hecho
> observado: en el universo real la mayoría de obras son multidepartamentales y no
> existen cinco grupos territoriales balanceados.

---

## 2. Arquitectura de dos capas

```
┌──────────────────────────────────────────────────────────────┐
│  CAPA 1 (antecedente, no incluida en este repositorio)      │
│  Random Forest de 3 clases sobre 326 obras y 61 variables.   │
│  Salida: Rᵢ = P(riesgo extremo) ∈ [0,1] por obra.            │
└───────────────────────────┬──────────────────────────────────┘
                            │  Rᵢ
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  CAPA 2 (este repositorio)                                   │
│  MCKP resuelto con AG binario y NSGA-II biobjetivo.          │
│  Salida: portafolio X* de 50 obras + frente de Pareto.       │
└──────────────────────────────────────────────────────────────┘
```

Este repositorio contiene únicamente la Capa 2. Cuando la API de la Capa 1 no está
disponible, el dataset se genera con `data/generator.py`, cuyas proporciones de
clase y de nivel de gobierno **no están inventadas**: se leen de
`data/calibracion_parte1.json`, la evidencia obtenida al auditar el dataset real.

---

## 3. Algoritmos

| Algoritmo | Tema del curso | Descripción |
|---|---|---|
| AG binario | Tema 3 | Inicialización heurística por ratio b/C, torneo k=3, cruce de un punto pc=0.85, mutación bit-flip pm=0.015, reparación greedy, elitismo de 1 individuo |
| NSGA-II | Tema 10 | Biobjetivo: maximizar f₁ = Σ R̃ᵢxᵢ, minimizar f₂ = Σ Cᵢxᵢ. Fast non-dominated sort, crowding distance, dominancia con restricciones |
| Greedy | Referencia | Cobertura territorial primero, luego relleno por ratio b/C. Determinístico |
| Aleatorio | Referencia | Mejor de 10 000 cromosomas con reparación no informada |

Análisis complementarios: prueba de **Wilcoxon signed-rank** sobre 10 corridas
independientes, **Teoría de Esquemas** (Tema 4) con la cota inferior de Goldberg, y
sensibilidad del peso w₁ y de los parámetros del algoritmo.

---

## 4. Requisitos

- **Python 3.11**
- Sin dependencias del sistema: todas las librerías se instalan con `pip`.

---

## 5. Instalación

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env      # opcional: sin .env se usan los mismos valores
```

---

## 6. Reproducción

```bash
# Pipeline completo: dataset, AG (10 corridas), Wilcoxon, NSGA-II, benchmarks,
# esquemas, sensibilidad de w1, 9 figuras y metricas.json / metricas_ext.json
python main.py --mode full                        # ~60 s

# Igual, mas el barrido de parametros del AG (pop_size, pc, pm) y la figura 10.
# Son 39 corridas extra, por eso no corre por defecto.
python main.py --mode full --con-sensibilidad-parametros   # ~130 s

# Modos individuales
python main.py --mode ag --seed 42                # una corrida del AG
python main.py --mode nsga2                       # frente de Pareto
python main.py --mode analysis --n-runs 10        # corridas + Wilcoxon

# Validacion end-to-end (7 chequeos PASS/FAIL) y suite de tests
python validate.py
pytest tests/ -v
```

Los resultados se escriben en `outputs/`: diez figuras PNG, `metricas.json`,
`metricas_ext.json` y los CSV de corridas, esquemas y sensibilidad.

### API REST

```bash
python main.py --mode api          # http://localhost:8001/docs
# o con contenedores
docker compose up -d               # publica el puerto 8001
```

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servicio y número de obras cargadas |
| `POST /optimize` | Ejecuta el AG (1 o N corridas) o NSGA-II |
| `GET /pareto` | Frente de Pareto de NSGA-II |
| `GET /analysis/runs` | Estadísticas de N corridas y prueba de Wilcoxon |
| `GET /analysis/schemas` | Building blocks (Teoría de Esquemas) |
| `GET /analysis/sensitivity` | Sensibilidad del peso w₁ |

---

## 7. Estructura

```
config/params.py         Parametros centralizados (lee .env), patron Singleton
data/generator.py        Generador del dataset, calibrado con la Capa 1
data/loader.py           Carga en cascada: CSV -> API Capa 1 -> generador
data/schemas.py          Esquemas de entrada y salida de la API
core/problem.py          MCKP: normalizacion, funcion objetivo, restricciones
core/solution.py         Solucion: fitness, factibilidad, serializacion
core/operators.py        Operadores evolutivos como funciones puras
algorithms/base.py       Interfaz comun de los algoritmos evolutivos
algorithms/genetic.py    AG binario canonico
algorithms/nsga2.py      NSGA-II biobjetivo
algorithms/benchmarks.py Greedy y busqueda aleatoria
analysis/statistics.py   Multiples corridas y prueba de Wilcoxon
analysis/schema_theory.py Teoria de Esquemas y factor de crecimiento
analysis/sensitivity.py  Sensibilidad de w1 y de los parametros del AG
visualization/           Paleta institucional y las diez figuras
api/                     Aplicacion REST y sus routers
scripts/auditar_parte1.py Auditoria de calibracion contra el dataset de la Capa 1
tests/                   Suite de pruebas (91 tests)
main.py                  Punto de entrada por linea de comandos
validate.py              Validacion end-to-end
```

### Parámetros configurables

Todos viven en `.env` (ver `.env.example`) y se leen a través de
`config/params.py`; ninguno está escrito en el código:

`N_OBRAS`, `PRESUPUESTO_PCT`, `K_MAX`, `MIN_POR_REGION`, `POP_SIZE`, `N_GEN`,
`PC`, `PM`, `K_TORNEO`, `ALPHA`, `N_RUNS`, `SEED`, `W1`…`W5`, `API_PORT`,
`PARTE1_API_URL`.

---

## 8. Resultados

Dataset de 326 obras, semillas 42–51, `pop=150`, `gen=400`.

### AG — 10 corridas independientes

| Métrica | Valor |
|---|---|
| Media ± desviación | **24.8662 ± 0.0404** |
| Mediana | 24.8769 |
| Rango | [24.7857, 24.9297] |
| Greedy (determinístico) | 24.5416 |
| Mejora media sobre Greedy | +1.32 % |
| Wilcoxon signed-rank | stat = 0.0, **p = 0.001953** → significativo (p < 0.05) |
| Corridas factibles | 10 / 10 |
| Corridas mejores que Greedy | 10 / 10 |

### Comparación de métodos

| Método | Fitness Z | Obras | Costo (kS/) | Factible |
|---|---|---|---|---|
| AG (propuesto) | 24.9297 | 50 | 28.47 | sí |
| Greedy | 24.5416 | 50 | 25.03 | sí |
| Aleatorio | 19.7343 | 50 | 43.05 | sí |

### NSGA-II

**51 soluciones** de Pareto factibles y no dominadas, con f₁ hasta 45.4938 y f₂
desde 12.50 kS/, entre 25 y 50 obras por solución, en 2.9 s.

### Teoría de Esquemas

83 de 175 esquemas presentan factor de crecimiento > 1. Los de orden 1 son
favorecidos en su totalidad (10/10), frente a 27/45 de orden 2 y 46/120 de orden 3:
al estar las obras de mejor ratio dispersas en el cromosoma, δ(H) crece y el cruce
de un punto destruye los bloques largos.

---

## 9. Nota sobre la calibración

`data/calibracion_parte1.json` viene incluido y es lo único que el generador
necesita para reproducir el dataset. **Regenerarlo** con
`python scripts/auditar_parte1.py` requiere el dataset y el modelo de la Capa 1,
que no forman parte de este repositorio; el script indica con un mensaje explícito
qué falta y cómo apuntar a otra ubicación.

Para ese paso opcional hacen falta las dependencias adicionales ya incluidas en
`requirements.txt` (`pyarrow`, `scikit-learn`, `joblib`).

---

## 10. Correspondencia con los temas del curso

| Módulo | Tema | Concepto |
|---|---|---|
| `core/problem.py` | T1, T5 | Función objetivo y penalización estática |
| `core/operators.py` | T3 | Torneo, cruce de un punto, mutación bit-flip, reparación |
| `algorithms/genetic.py` | T2, T3 | Ciclo evolutivo y AG canónico |
| `analysis/schema_theory.py` | T4 | Teorema de esquemas y building blocks |
| `algorithms/nsga2.py` | T10 | Dominancia, crowding distance, frente de Pareto |
| `data/generator.py` | T7 | Instancia del problema MCKP NP-Hard |
| `analysis/statistics.py` | Síntesis | Múltiples corridas y contraste de hipótesis |

---

## Autoría

**Fernando Garcia Atuncar**
Universidad Nacional de Ingeniería — Facultad de Ingeniería Industrial y de
Sistemas, Unidad de Postgrado. 2026.

Licencia MIT (ver `LICENSE`).
