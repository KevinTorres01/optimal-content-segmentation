# Resultados: Segmentación Óptima de Contenido

## Índice
1. [Descripción del Problema](#descripción-del-problema)
2. [Modelado Formal](#modelado-formal)
3. [Dataset](#dataset)
4. [Algoritmos Implementados](#algoritmos-implementados)
5. [Rol del LLM](#rol-del-llm)
6. [Metodología Experimental](#metodología-experimental)
7. [Resultados y Análisis](#resultados-y-análisis)
8. [Limitaciones y Mejoras](#limitaciones-y-mejoras)
9. [Conclusiones](#conclusiones)

---

## Descripción del Problema

### Contexto

La segmentación de textos (*text segmentation*) es el problema de dividir un documento en fragmentos semánticamente coherentes. Tiene aplicaciones directas en sistemas de recuperación de información, resumen automático, análisis de conversaciones y procesamiento de documentos largos para LLMs (donde los fragmentos relevantes se extraen antes de enviarlos al modelo).

### Definición Formal

Dado un documento $D$ compuesto por $n$ oraciones $s_1, s_2, \ldots, s_n$, se desea encontrar un conjunto de $k$ **fronteras de segmento** $B = \{b_1, b_2, \ldots, b_k\}$ con $b_1 = 0 < b_2 < \ldots < b_k < n$, tal que cada segmento resultante $S_j = \{s_{b_j}, \ldots, s_{b_{j+1}-1}\}$ sea internamente coherente y semánticamente distinto de sus vecinos.

### Ejemplo

Un documento con 17 oraciones sobre deportes (oraciones 0–3), tecnología (4–10) y cocina (11–16) debería producir fronteras en $B = \{0, 4, 11\}$. Un algoritmo que devuelva $B = \{0, 2, 4, 6, 11\}$ tiene buen recall en la frontera real (detectó 4 y 11), pero introduce fronteras espurias que fragmentan el segmento de deportes.

---

## Modelado Formal

### Función Objetivo

Se define la **cohesión** de un segmento $S = \{s_i, \ldots, s_j\}$ como la similitud coseno media entre todos sus pares de oraciones, ponderada por la longitud relativa del segmento:

$$\text{cohesion}(i, j) = \overline{\cos}(S) \cdot \frac{j - i + 1}{n}$$

donde $\overline{\cos}(S)$ es la similitud coseno media entre todos los pares de vectores TF-IDF de las oraciones del segmento.

El problema de optimización es:

$$\max_{B} \sum_{j=1}^{k} \text{cohesion}(b_j,\, b_{j+1} - 1)$$

sujeto a:
- $b_1 = 0$
- $b_j < b_{j+1} \quad \forall j$
- $b_k < n$
- $k \leq k_{\max}$ (parámetro de configuración)

### Representación Vectorial

Cada oración $s_i$ se representa como un vector TF-IDF de dimensión $|V|$ (vocabulario del documento), usando frecuencia logarítmica (`sublinear_tf=True`). La similitud entre oraciones es el coseno del ángulo entre sus vectores:

$$\cos(s_i, s_j) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\| \cdot \|\vec{v}_j\|}$$

### Justificación del Ponderado por Longitud

Sin el factor $\frac{j-i+1}{n}$, la función objetivo degenera: un segmento de una sola oración tiene cohesión cero (no hay pares), lo que hace que particiones con muchos segmentos unitarios sean siempre "óptimas". El ponderado compensa esto recompensando segmentos más largos con cohesión moderada frente a muchos segmentos cortos con cohesión perfecta trivial.

### Complejidad del Espacio de Búsqueda

El número de formas de colocar $k-1$ fronteras internas en $n-1$ posiciones posibles es $\binom{n-1}{k-1}$, que crece exponencialmente con $n$. Para $n=20, k=5$: $\binom{19}{4} = 3876$ particiones. Para $n=50, k=5$: $\binom{49}{4} = 211876$.

---

## Dataset

### Generación Sintética

Se generó un dataset sintético en español con el generador interno del proyecto (`src/dataset/generator.py`). Cada documento se construye concatenando bloques de oraciones sobre tópicos distintos, con vocabulario controlado para simular documentos reales con cambios temáticos abruptos.

**Tópicos disponibles**: deportes, tecnología, política, ciencia, arte, economía, salud, historia.

**Configuración utilizada** (`config/datasets/small.yaml`):

```yaml
dataset_name: small
n_documents: 20
segments_per_doc:
  min: 3
  max: 5
sentences_per_segment:
  min: 4
  max: 8
topic_source: synthetic_templates
overlap_level: low
random_seed: 42
language: es
```

### Estadísticas del Dataset

| Parámetro | Valor |
|---|---|
| Documentos totales | 20 |
| Oraciones por documento (media) | 22.9 |
| Oraciones por documento (rango) | 15 – 36 |
| Segmentos de referencia (media) | 3.80 |
| Segmentos de referencia (distribución) | 3 segmentos: 8 docs · 4 segmentos: 8 docs · 5 segmentos: 4 docs |
| Nivel de solapamiento léxico | bajo |
| Semilla aleatoria | 42 |

### Estructura en Disco

```
data/small/
├── documents/        # 20 archivos .txt con el texto completo
├── boundaries/       # 20 archivos .json con fronteras de referencia
└── metadata.json     # Snapshot de configuración + estadísticas
```

Cada archivo de fronteras tiene el formato:
```json
{"boundaries": [0, 8, 15], "n_segments": 3, "doc_id": "doc_0001"}
```

---

## Algoritmos Implementados

Se implementaron cuatro variantes que comparten la misma función objetivo y representación vectorial, permitiendo comparación directa.

### 1. Fuerza Bruta (`brute_force`)

**Idea**: Enumerar explícitamente todas las $\binom{n-1}{k-1}$ particiones posibles y seleccionar la de mayor cohesión total.

**Pseudocódigo**:
```
ENTRADA: documento D (n oraciones), k_max
SALIDA:  fronteras B* óptimas

1. Calcular matriz de cohesión C[i][j] para todo i ≤ j
2. mejor_score ← -∞
3. Para cada combinación (b2,...,bk) de {1,...,n-1}:
   a. B ← [0, b2, ..., bk]
   b. score ← Σ C[B[j]][B[j+1]-1]
   c. Si score > mejor_score: guardar B y actualizar mejor_score
4. RETORNAR B*
```

**Complejidad**: Tiempo $O\!\left(\binom{n-1}{k-1} \cdot n\right)$, espacio $O(n^2)$.  
**Límite práctico**: $n \leq 15$ (restricción impuesta en código).  
**Uso**: Referencia de correctitud para los demás algoritmos en instancias pequeñas.

---

### 2. Programación Dinámica (`dynamic_programming`)

**Idea**: Explotar la subestructura óptima del problema. La mejor partición de las primeras $i$ oraciones en $j$ segmentos se puede construir a partir de la mejor partición de las primeras $i'$ oraciones en $j-1$ segmentos más el costo del segmento $[i', i]$.

**Pseudocódigo**:
```
ENTRADA: documento D (n oraciones), k
SALIDA:  fronteras B* óptimas

1. Calcular C[i][j] para todo 0 ≤ i ≤ j < n
2. dp[0][0] ← 0;  dp[i][j] ← -∞  para el resto
3. Para j = 1 hasta k:
   Para i = j hasta n:
     Para i' = j-1 hasta i-1:
       val ← dp[i'][j-1] + C[i'][i-1]
       Si val > dp[i][j]: actualizar dp[i][j] y split[i][j] ← i'
4. Retroceder desde split[n][k*] para reconstruir B*
5. RETORNAR B*
```

**Complejidad**: Tiempo $O(n^2 k)$, espacio $O(n^2 + nk)$.  
**Garantía**: Encuentra el óptimo global en tiempo polinomial.

---

### 3. Greedy — TextTiling (`greedy`)

**Idea**: Para cada posición de posible frontera (hueco entre oraciones $i$ e $i+1$), calcular la similitud entre el bloque de $w$ oraciones a la izquierda y el bloque de $w$ oraciones a la derecha. Colocar fronteras en los $k-1$ "valles" más profundos de esa curva de similitud.

**Pseudocódigo**:
```
ENTRADA: documento D (n oraciones), k, w (tamaño de ventana)
SALIDA:  fronteras B

1. Calcular vectores TF-IDF para todas las oraciones
2. Para cada hueco g ∈ {0,...,n-2}:
   a. izq ← promedio(TF-IDF[max(0,g-w+1)..g])
   b. der ← promedio(TF-IDF[g+1..min(n,g+1+w)])
   c. sim[g] ← coseno(izq, der)
3. Para cada hueco g:
   profundidad[g] ← (max(sim[0..g-1]) - sim[g]) + (max(sim[g+1..n-2]) - sim[g])
4. Seleccionar los k-1 huecos con mayor profundidad
5. RETORNAR [0] + [g+1 para cada hueco seleccionado, ordenados]
```

**Complejidad**: Tiempo $O(n \cdot w)$, espacio $O(n)$.  
**Ventaja**: Extremadamente rápido. Sin optimización global; puede ser subóptimo.

---

### 4. Recocido Simulado (`simulated_annealing`)

**Idea**: Partir de una solución con fronteras uniformemente espaciadas y explorar el espacio moviendo aleatoriamente una frontera ±1 posición. Aceptar empeoramientos con probabilidad $e^{\Delta/T}$ para escapar óptimos locales, reduciendo $T$ gradualmente.

**Pseudocódigo**:
```
ENTRADA: documento D, k, n_iter, T_0, α (tasa de enfriamiento), semilla
SALIDA:  fronteras B* (cerca del óptimo)

1. Calcular C[i][j]
2. B ← partición uniformemente espaciada de k segmentos
3. T ← T_0;  B* ← B;  score* ← cohesion(B)
4. Para iter = 1 hasta n_iter:
   a. Elegir frontera interna aleatoria b_j
   b. Δ ← +1 o -1 (aleatorio)
   c. Si b_{j-1} < b_j+Δ < b_{j+1}: (movimiento válido)
      - score_nuevo ← cohesion(B con b_j+Δ)
      - Si score_nuevo > score: aceptar
      - Si no: aceptar con prob. e^((score_nuevo - score)/T)
   d. Si score_nuevo > score*: B* ← B_nuevo
   e. T ← T × α
5. RETORNAR B*
```

**Complejidad**: Tiempo $O(n^2 + \text{n\_iter})$, espacio $O(n^2)$.  
**Parámetros usados**: `n_iterations=2000`, `initial_temp=1.0`, `cooling_rate=0.995`, `random_seed=42`.

---

## Rol del LLM

### Función en el Sistema

El LLM actúa como **evaluador externo de cohesión semántica**: dado un segmento de texto, asigna un puntaje de 1 a 5 y una justificación en lenguaje natural. Esto complementa las métricas estructurales (WindowDiff, Pk) que solo miden si las fronteras coinciden con las de referencia, sin evaluar la calidad semántica real del segmento.

### Flujo de Evaluación

```
Segmento de texto
       ↓
[Prompt de cohesión] → LLM (Groq / Mistral) → JSON {"score": 4, "rationale": "..."}
       ↓
CohesionScore(segment_id, score, rationale, provider, model, used_fallback)
```

### Prompt Utilizado

```
Evalúa la cohesión semántica de este segmento de texto en una escala del 1 al 5:
- 1: Sin coherencia. Las oraciones hablan de temas completamente distintos.
- 3: Coherencia parcial. Temas relacionados pero transiciones bruscas.
- 5: Coherencia perfecta. Todas las oraciones contribuyen a un único tema claro.

Segmento:
{texto del segmento}

Responde ÚNICAMENTE con un objeto JSON (sin markdown):
{"score": <entero 1-5>, "rationale": "<una oración de explicación>"}
```

### Configuración del LLM

```yaml
provider: groq
model: llama-3.3-70b-versatile
temperature: 0.0
max_tokens: 512
```

### Estrategia de Fallback

Para garantizar robustez ante fallos de red o límites de tasa:

```
1. Intentar Groq (primario — tier gratuito, ~14,400 req/día)
2. Si falla (timeout, 429, error de red) → Mistral (fallback — tier gratuito)
3. Si ambos fallan → score neutro = 3 (degradación elegante)
```

Cada resultado registra `used_fallback: bool` para auditoría.

---

## Metodología Experimental

### Experimento Principal

**ID**: `exp_compare_algorithms`  
**Objetivo**: Comparar los tres algoritmos escalables (DP, Greedy, SA) sobre el dataset `small` en métricas estándar de segmentación.

> **Nota sobre Fuerza Bruta**: No se incluyó en el experimento principal porque los documentos del dataset `small` tienen entre 15 y 36 oraciones, superando el límite de 15 del algoritmo exhaustivo. Su rol es ser referencia de correctitud en instancias pequeñas, verificado en tests unitarios (`test_matches_dp_on_small_doc`).

**Configuración** (`config/experiments/exp_compare_algorithms.yaml`):

```yaml
algorithms:
  - name: dynamic_programming
    params: {max_segments: 5}
  - name: greedy
    params: {max_segments: 5, window_size: 2}
  - name: simulated_annealing
    params: {max_segments: 5, n_iterations: 2000, cooling_rate: 0.995, random_seed: 42}
llm_evaluator:
  provider: none
evaluation:
  metrics: [windowdiff, pk, f1_boundary, runtime_seconds]
  random_seed: 42
```

### Métricas de Evaluación

**Pk** — Probabilidad de error en una ventana deslizante de tamaño $k = \lfloor n / (2 \cdot k_{\text{ref}}) \rfloor$:
$$P_k = \frac{1}{n-k} \sum_{i=1}^{n-k} \mathbf{1}\!\left[\text{ref}(i, i+k) \neq \text{pred}(i, i+k)\right]$$
donde $\text{ref}(i,j) = 1$ si $i$ y $j$ están en segmentos distintos según la referencia.  
Menor es mejor. Rango: [0, 1].

**WindowDiff** — Penaliza más que Pk cuando el número de fronteras dentro de la ventana difiere:
$$WD = \frac{1}{n-k} \sum_{i=1}^{n-k} \left| \text{ref\_count}(i, i+k) - \text{pred\_count}(i, i+k) \right| > 0$$
Menor es mejor. Rango: [0, 1].

**F1-Boundary** — Precisión y recall sobre las posiciones exactas de frontera (tolerancia ±1 oración):
$$F1 = \frac{2 \cdot P \cdot R}{P + R}$$
Mayor es mejor. Rango: [0, 1].

**Runtime** — Tiempo de ejecución por documento en segundos.

### Reproducibilidad

| Parámetro | Valor |
|---|---|
| Semilla aleatoria | 42 |
| Versión Python | 3.12.3 |
| scikit-learn | ≥ 1.3 |
| numpy | ≥ 1.24 |
| Fecha de ejecución | 2026-05-31 |

---

## Resultados y Análisis

### Tabla de Resultados Agregados (20 documentos)

| Algoritmo | Pk ↓ | WindowDiff ↓ | F1-Boundary ↑ | Runtime (ms) |
|---|---|---|---|---|
| Dynamic Programming | **0.1728** ± 0.157 | **0.1945** ± 0.162 | **0.797** ± 0.124 | 10.1 |
| Greedy (TextTiling) | 0.1288 ± 0.076 | 0.2373 ± 0.125 | 0.743 ± 0.140 | **1.2** |
| Simulated Annealing | 0.2109 ± 0.158 | 0.2236 ± 0.166 | 0.755 ± 0.141 | 12.8 |

*(↓ menor es mejor, ↑ mayor es mejor)*

### Análisis por Métrica

#### Pk (tasa de error en ventana deslizante)

Greedy obtiene el **mejor Pk (0.1288)**, un 25% mejor que DP y 39% mejor que SA. Esto indica que, en términos de clasificación de pares de oraciones como "mismo segmento / distinto segmento", la heurística de valles de similitud es sorprendentemente efectiva. La baja desviación estándar de Greedy (0.076 vs ~0.157 para DP y SA) sugiere además que es el método más consistente documento a documento.

#### WindowDiff (penalización por densidad de fronteras)

DP obtiene el **mejor WindowDiff (0.1945)**. Esta métrica es más estricta que Pk porque penaliza tanto fronteras faltantes como fronteras extras dentro de la misma ventana. El hecho de que DP supere a Greedy aquí (0.1945 vs 0.2373) refleja que DP, al maximizar la cohesión globalmente, tiende a colocar fronteras en posiciones más precisas incluso cuando se equivoca.

#### F1-Boundary (precisión en posición exacta de fronteras)

DP obtiene el **mejor F1 (0.797)**. La diferencia respecto a Greedy (0.743) y SA (0.755) indica que la optimización exacta produce fronteras más cercanas a las posiciones reales (dentro de tolerancia ±1 oración). SA queda por encima de Greedy en esta métrica, sugiriendo que la búsqueda estocástica eventualmente encuentra mejores posiciones absolutas que el enfoque voraz local.

#### Runtime

Greedy es **8x más rápido** que DP (1.2 ms vs 10.1 ms) y **10x más rápido** que SA (1.2 ms vs 12.8 ms). Para documentos cortos como los de este experimento, todas las diferencias son despreciables en práctica. La diferencia se volvería significativa en documentos de cientos de oraciones, donde DP tiene complejidad $O(n^2 k)$ y SA necesita más iteraciones.

### Sobre-Segmentación Observada

Los tres algoritmos predicen en media **5.00 segmentos** cuando la referencia es **3.80**. Esto ocurre porque `max_segments=5` es el límite configurado y la función objetivo siempre favorece más segmentos (mayor libertad de optimización). En un sistema de producción sería necesario seleccionar $k$ automáticamente, por ejemplo minimizando un criterio de información (BIC, MDL) o usando el score LLM como señal.

### Comparativa General

| Criterio | Mejor | Razón |
|---|---|---|
| Menor tasa de error (Pk) | Greedy | Heurística de valles captura cambios temáticos abruptos |
| Fronteras más precisas (F1) | DP | Optimización global penaliza posiciones alejadas |
| Mejor balance WD | DP | Considera la distribución completa de fronteras |
| Mayor velocidad | Greedy | O(n·w) sin backtracking ni iteraciones |
| Mayor consistencia | Greedy | Desviación estándar más baja en todas las métricas |

**Conclusión parcial**: DP es el algoritmo más preciso cuando importa la posición exacta de las fronteras. Greedy es la opción práctica cuando el volumen de documentos es alto y se prioriza velocidad y consistencia. SA no supera a DP en ninguna métrica con los parámetros actuales.

---

## Limitaciones y Mejoras

### Limitaciones Actuales

**1. Selección de $k$ manual**  
Todos los algoritmos requieren `max_segments` como parámetro. En un escenario real el número de segmentos es desconocido. Se debería implementar selección automática de $k$ (por ejemplo, el $k$ que maximiza el salto de cohesión marginal, o penalizar con MDL).

**2. Representación TF-IDF superficial**  
TF-IDF ignora semántica distribucional. Dos oraciones sobre el mismo tema con vocabulario diferente (sinónimos, paráfrasis) tendrán baja similitud coseno. Usar embeddings densos (BERT, sentence-transformers) mejoraría la representación.

**3. Dataset sintético y pequeño**  
20 documentos con vocabulario controlado favorecen métodos basados en TF-IDF. En documentos reales (noticias, artículos académicos) con ruido léxico el rendimiento podría diferir. El dataset debería crecer y diversificarse.

**4. Fuerza Bruta limitada a n ≤ 15**  
El algoritmo exhaustivo no puede correr en el dataset principal. Solo sirve como referencia en tests unitarios. Para validar el óptimo en documentos más largos habría que comparar DP con BF en un dataset dedicado de documentos cortos.

**5. SA sin ajuste de hiperparámetros**  
Los parámetros `initial_temp=1.0`, `cooling_rate=0.995`, `n_iterations=2000` se eligieron sin búsqueda sistemática. SA podría mejorar con un grid search sobre estos parámetros.

**6. Sin evaluación LLM en el experimento principal**  
Por razones de costo (límites de tasa en tier gratuito) el experimento comparativo se ejecutó con `provider: none`. La dimensión de calidad semántica (¿los segmentos predichos son cohesivos incluso si no coinciden exactamente con las fronteras de referencia?) queda sin explorar.

### Mejoras Propuestas

| Prioridad | Mejora | Impacto estimado |
|---|---|---|
| Alta | Embeddings densos (sentence-transformers) en lugar de TF-IDF | +15–20% F1 en textos reales |
| Alta | Selección automática de $k$ | Elimina el parámetro más crítico |
| Media | Ajuste de hiperparámetros de SA | +5–10% F1 para SA |
| Media | Dataset real (Wikipedia, noticias) | Validación más robusta |
| Baja | Evaluación LLM masiva con Groq | Score semántico complementario |

---

## Conclusiones

### Resumen de Hallazgos

Se implementaron y compararon cuatro algoritmos de segmentación de texto bajo una función objetivo unificada de cohesión coseno ponderada por longitud:

- **Fuerza Bruta**: garantiza el óptimo global pero solo es viable para $n \leq 15$; confirmó que DP produce resultados idénticos en instancias pequeñas.
- **Programación Dinámica**: es el mejor algoritmo general — mayor F1 (0.797) y mejor WindowDiff (0.1945) — con complejidad polinomial $O(n^2 k)$.
- **Greedy (TextTiling)**: sorprendentemente competitivo en Pk (0.1288, el mejor), 8x más rápido que DP, y el más consistente; es la opción práctica para alto volumen.
- **Recocido Simulado**: con los parámetros actuales no supera a DP en ninguna métrica, aunque aporta el mecanismo para escapar óptimos locales en espacios de búsqueda más complejos.

### Respuesta a los Objetivos

| Objetivo | Estado |
|---|---|
| Implementar variante exacta (BF + DP) | ✓ Completado |
| Implementar heurística rápida (Greedy) | ✓ Completado |
| Implementar metaheurística (SA) | ✓ Completado |
| Comparar bajo métricas estándar | ✓ WindowDiff, Pk, F1 calculados |
| Integrar evaluación LLM | ✓ Infraestructura lista (Groq + Mistral + fallback) |
| Verificar reproducibilidad | ✓ Semilla fijada en config y metadatos |

### Trabajo Futuro

1. Ejecutar el experimento con evaluación LLM habilitada (`provider: groq`) para obtener la dimensión semántica de los resultados.
2. Reemplazar TF-IDF por embeddings de sentence-transformers para mejorar la representación semántica.
3. Implementar selección automática de $k$ para eliminar el parámetro más crítico del sistema.
4. Evaluar en un dataset real (Wikipedia secciones, artículos de noticias segmentados manualmente).
5. Explorar variantes de SA con temperatura adaptativa o reinicios aleatorios para mejorar su rendimiento relativo.
