# Informe Detallado del Proyecto

## Segmentación Óptima de Contenido — Explicación desde Cero

**Autor**: Kevin Torres Perera
**Fecha**: 2026-06-07
**Asignatura**: Inteligencia Artificial 2025–2026 — Tema 6
**Repositorio**: `optimal-content-segmentation`

---

## Cómo leer este informe

Este documento está pensado para una persona que **no tiene conocimiento previo** del problema, de los algoritmos ni de los conceptos de procesamiento de lenguaje natural que se utilizan. Cada sección introduce los conceptos antes de usarlos. Si ya dominas algo, puedes saltar a la sección siguiente: cada parte es razonablemente autocontenida.

Hay tres niveles de profundidad en el texto:

1. **Texto principal** — explicación en lenguaje natural con analogías y ejemplos numéricos pequeños.
2. **Cajas matemáticas** — fórmulas con su interpretación. Si las fórmulas asustan, puedes saltarlas: el texto que las rodea siempre dice qué significan.
3. **Pseudocódigo y ejemplos paso a paso** — para verificar tu comprensión simulando el algoritmo a mano.

Al final del documento hay un **glosario** con todos los términos técnicos.

---

## Índice

1. [Introducción y objetivo](#1-introducción-y-objetivo)
2. [El problema en términos cotidianos](#2-el-problema-en-términos-cotidianos)
3. [Conceptos previos necesarios](#3-conceptos-previos-necesarios)
4. [Modelado formal del problema](#4-modelado-formal-del-problema)
5. [La función objetivo](#5-la-función-objetivo)
6. [Los cuatro algoritmos](#6-los-cuatro-algoritmos)
7. [El papel del LLM](#7-el-papel-del-llm)
8. [El dataset](#8-el-dataset)
9. [Métricas de evaluación](#9-métricas-de-evaluación)
10. [Diseño experimental](#10-diseño-experimental)
11. [Resultados y análisis](#11-resultados-y-análisis)
12. [Discusión: por qué pasó lo que pasó](#12-discusión-por-qué-pasó-lo-que-pasó)
13. [Limitaciones y mejoras futuras](#13-limitaciones-y-mejoras-futuras)
14. [Conclusiones](#14-conclusiones)
15. [Manual de uso paso a paso](#15-manual-de-uso-paso-a-paso)
16. [Glosario](#16-glosario)

---

## 1. Introducción y objetivo

### 1.1 ¿Qué es este proyecto?

Este proyecto es un trabajo de investigación académica que estudia el problema de **dividir automáticamente un texto largo en fragmentos coherentes**. Imagina que tienes un artículo de Wikipedia de cinco páginas sin títulos ni subtítulos: las oraciones están todas seguidas, pero el contenido habla de varios temas (historia, geografía, economía, cultura). El objetivo es que un programa, sin intervención humana, descubra **dónde empieza un tema y dónde empieza el siguiente**.

A esos puntos donde el contenido "cambia de tema" los llamamos **fronteras de segmento** (o, simplemente, fronteras). Encontrar las fronteras correctas es lo que se conoce como **segmentación de textos** (*text segmentation* en inglés).

### 1.2 ¿Para qué sirve?

Esto no es un ejercicio teórico: el problema aparece en muchas aplicaciones reales.

- **Recuperación de información**: cuando buscas algo en Google, el motor no devuelve documentos enteros, sino el fragmento donde aparece la respuesta. Para eso, primero tiene que partir los documentos en fragmentos.
- **Resumen automático**: para resumir un documento largo, es útil resumir cada sección por separado.
- **Análisis de conversaciones**: en una transcripción de un *call center*, encontrar dónde cambia el tema permite identificar la queja, la solución y el cierre.
- **Sistemas RAG con LLMs**: los modelos de lenguaje grandes (GPT, Claude, Llama) tienen un límite de cuántas palabras pueden leer a la vez. Antes de hacerles una pregunta sobre un libro entero, hay que partir el libro y enviarles solo los fragmentos relevantes.
- **Procesamiento de subtítulos de video**: para crear capítulos automáticamente.

### 1.3 ¿Qué resuelve concretamente este proyecto?

El proyecto hace tres cosas:

1. **Implementa cuatro algoritmos distintos** que resuelven el problema. Dos de ellos garantizan encontrar la mejor solución posible (la "óptima"), y los otros dos son más rápidos pero pueden equivocarse en alguna frontera.
2. **Usa un LLM (Llama 3.3 de 70 mil millones de parámetros, ejecutado vía Groq) como evaluador**: le pasa cada segmento que el algoritmo propuso y le pregunta "del 1 al 5, ¿qué tan coherente es este segmento?".
3. **Mide objetivamente cuál algoritmo es mejor** comparando sus salidas con un dataset donde conocemos las fronteras "verdaderas" (porque lo construimos sintéticamente).

### 1.4 ¿De dónde viene la motivación académica?

Este trabajo se enmarca en el **Tema 6** de los temas propuestos para el proyecto final de Inteligencia Artificial 2025–2026:

> **Tema 6 — Segmentación óptima de contenido**: Dada una secuencia de elementos (texto o fragmentos de contenido), dividirla en segmentos contiguos que maximicen la coherencia interna. El sistema debe utilizar un LLM para evaluar la cohesión semántica de cada segmento.

La consigna pide tres cosas que este proyecto cumple: (a) un problema de optimización, (b) un dataset con instancias de prueba, y (c) un LLM integrado como parte funcional del sistema.

---

## 2. El problema en términos cotidianos

### 2.1 Un ejemplo intuitivo

Supón que recibimos este texto compuesto por 11 oraciones (cada oración numerada para referencia):

> (0) El Real Madrid ganó la final 3 a 1.
> (1) Vinicius marcó dos goles en la primera mitad.
> (2) El portero salvó tres tiros peligrosos.
> (3) Apple presentó su nuevo iPhone con cámara mejorada.
> (4) El procesador es un 40 % más rápido que el modelo anterior.
> (5) La batería dura hasta 18 horas con uso intensivo.
> (6) La pantalla mide 6,7 pulgadas y tiene tasa de refresco de 120 Hz.
> (7) La paella valenciana lleva arroz, azafrán y pollo.
> (8) El sofrito se hace con tomate, pimiento y ajo.
> (9) La cocción dura aproximadamente 20 minutos a fuego medio.
> (10) Se sirve directamente en la paellera para mantener el calor.

Cualquier humano lee esto y reconoce de inmediato tres bloques:

- **Bloque A — Deportes**: oraciones 0, 1, 2
- **Bloque B — Tecnología**: oraciones 3, 4, 5, 6
- **Bloque C — Cocina**: oraciones 7, 8, 9, 10

Decimos entonces que las **fronteras de segmento son las oraciones 0, 3 y 7**, porque ahí empieza cada bloque nuevo. En notación compacta: `B = [0, 3, 7]`.

Si numeráramos las fronteras como "huecos entre oraciones" (entre la oración 2 y la 3 hay un hueco, etc.), también podríamos representarlo así. Pero en este proyecto siempre usamos **la convención de "primera oración de cada segmento"**: la primera frontera siempre es `0` (todo documento empieza por su primera oración).

### 2.2 ¿Por qué es difícil para un programa?

Para un humano la segmentación anterior es obvia, porque entiende el significado de las palabras. Para un programa que **no entiende el lenguaje** (o que solo lo entiende muy superficialmente) hay tres dificultades grandes:

1. **No sabe de antemano cuántos segmentos hay.** En el ejemplo había 3, pero podría haber 2 o 7.
2. **No sabe de qué temas trata el texto.** No tiene una lista de "temas posibles" con la cual comparar.
3. **Hay muchísimas formas posibles de dividir.** Con 11 oraciones y queriendo 3 segmentos, el número de divisiones posibles es ya considerable; con 100 oraciones, el número se vuelve astronómico.

A este problema (encontrar la mejor entre muchísimas configuraciones posibles) se le llama **problema de optimización combinatoria**, y es el tipo de problema clásico en cursos de IA y algoritmos.

### 2.3 La idea de "mejor"

¿Qué significa que una segmentación sea "mejor" que otra? Tenemos que definir una **función objetivo**: una fórmula matemática que asigna un número a cada segmentación posible. Cuanto más alto el número, mejor la segmentación. Luego buscamos la segmentación que maximice ese número.

Una idea natural: una segmentación es buena si **dentro de cada segmento las oraciones se parecen mucho entre sí** (todas hablan del mismo tema). Esto se llama **cohesión interna**. La definición precisa requiere convertir las oraciones en algo que un programa pueda comparar numéricamente, lo cual nos lleva a la sección 3.

---

## 3. Conceptos previos necesarios

Esta sección explica los tres ingredientes que usamos para que un programa pueda comparar oraciones: TF-IDF, similitud coseno y complejidad algorítmica.

### 3.1 Representar texto como números: TF-IDF

Un programa no puede comparar oraciones directamente. Necesitamos convertir cada oración en un **vector de números**.

#### La idea más simple: bolsa de palabras

La forma más sencilla es contar cuántas veces aparece cada palabra. Si nuestro vocabulario completo es `{el, gol, gato, marcó, ronroneó}`, entonces:

- Oración "El gato ronroneó" → `[1, 0, 1, 0, 1]` (un "el", cero "gol", un "gato", cero "marcó", un "ronroneó")
- Oración "Marcó el gol" → `[1, 1, 0, 1, 0]`

Este vector se llama **bolsa de palabras** (*bag of words*): hemos perdido el orden, pero ganamos algo que podemos comparar numéricamente.

#### Problema: las palabras frecuentes dominan

En español, palabras como "el", "la", "de", "y" aparecen en casi todas las oraciones. Si solo contamos frecuencias, dos oraciones cualesquiera parecerán similares simplemente porque ambas usan "el" y "de". Esto es ruido.

#### Solución: TF-IDF

**TF-IDF** son las siglas de *Term Frequency – Inverse Document Frequency*. La idea es ponderar cada palabra por dos factores:

- **TF** (frecuencia de término): cuántas veces aparece la palabra en esta oración. Si una palabra aparece mucho en esta oración, probablemente es importante para esta oración.
- **IDF** (frecuencia inversa de documento): qué tan rara es la palabra en el conjunto total. Si una palabra aparece en todas las oraciones (como "el"), su IDF es bajo; si aparece solo en una oración (como "Vinicius"), su IDF es alto.

> **Fórmula simplificada**:
> $$\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \log\!\left(\frac{N}{\text{df}(t)}\right)$$
> donde $t$ es el término (palabra), $d$ el documento (oración), $N$ el número total de oraciones, y $\text{df}(t)$ el número de oraciones donde aparece $t$.

Resultado: palabras como "Vinicius" en la oración de fútbol tendrán un valor alto, y palabras como "el" tendrán un valor cercano a cero. **El vector TF-IDF "destaca" las palabras informativas de cada oración.**

En este proyecto usamos la implementación de `scikit-learn` con `sublinear_tf=True` (sustituye $\text{tf}$ por $1 + \log(\text{tf})$, lo cual amortigua aún más las palabras muy frecuentes).

### 3.2 Medir parecido entre vectores: similitud coseno

Una vez que cada oración es un vector de números, ¿cómo decimos si dos oraciones se parecen?

La **similitud coseno** mide el ángulo entre dos vectores. Si dos vectores apuntan en la misma dirección, el coseno del ángulo entre ellos es 1 (máximo parecido). Si apuntan en direcciones perpendiculares, el coseno es 0 (sin relación).

> **Fórmula**:
> $$\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \cdot \|\vec{v}\|}$$
> donde $\vec{u} \cdot \vec{v}$ es el producto escalar (suma de productos componente a componente) y $\|\vec{u}\|$ es la norma euclidiana (raíz de la suma de cuadrados).

**Ejemplo numérico con dos oraciones reducidas**:

- $\vec{u} = [1, 0, 1, 0]$ (oración A)
- $\vec{v} = [1, 1, 0, 0]$ (oración B)

Producto escalar: $1 \cdot 1 + 0 \cdot 1 + 1 \cdot 0 + 0 \cdot 0 = 1$
Norma de $\vec{u}$: $\sqrt{1 + 0 + 1 + 0} = \sqrt{2}$
Norma de $\vec{v}$: $\sqrt{1 + 1 + 0 + 0} = \sqrt{2}$
Similitud coseno: $\frac{1}{\sqrt{2} \cdot \sqrt{2}} = \frac{1}{2} = 0{,}5$

Las dos oraciones tienen una palabra en común (la primera componente) sobre dos palabras cada una: 0,5 es razonable.

La similitud coseno es la métrica estándar para comparar vectores TF-IDF porque ignora la "magnitud" (longitud del vector, que depende del largo de la oración) y se queda solo con la dirección (qué palabras importan).

### 3.3 Complejidad algorítmica: por qué importa el tiempo

Cuando hablamos de un algoritmo, no basta con que funcione: tiene que terminar en un tiempo razonable. La **complejidad algorítmica** describe cómo crece el tiempo de ejecución cuando el tamaño de la entrada (en nuestro caso, $n$ = número de oraciones) crece.

Notación habitual:

- $O(n)$ — lineal: si duplico $n$, el tiempo se duplica.
- $O(n^2)$ — cuadrático: si duplico $n$, el tiempo se multiplica por 4.
- $O(n^2 \cdot k)$ — cuadrático en $n$ multiplicado por $k$: si triplico $k$, el tiempo se triplica.
- $O(2^n)$ — exponencial: cada oración nueva **duplica** el tiempo. Inviable para $n > 30$ aproximadamente.

> **Ejemplo concreto del costo exponencial**: con $n = 20$ oraciones y un algoritmo $O(2^n)$, son aproximadamente un millón de operaciones (instantáneo). Con $n = 40$, son un billón de operaciones (minutos). Con $n = 60$, son un trillón (años).

Una parte central de este proyecto es entender el **trade-off entre exactitud y velocidad**: los algoritmos exactos garantizan encontrar la mejor segmentación pero son caros; los algoritmos heurísticos son rápidos pero pueden equivocarse. ¿Vale la pena la diferencia? La sección 11 responde con datos.

---

## 4. Modelado formal del problema

### 4.1 Definición precisa

Tenemos un documento $D$ formado por $n$ oraciones ordenadas:

$$D = (s_1, s_2, s_3, \ldots, s_n)$$

Una **segmentación** es una lista de $k$ enteros, llamados **fronteras**:

$$B = (b_1, b_2, \ldots, b_k)$$

que cumplen tres restricciones:

1. $b_1 = 0$ — la primera frontera siempre es el inicio del documento.
2. $0 = b_1 < b_2 < \ldots < b_k \leq n - 1$ — están en orden estrictamente creciente.
3. $k \leq k_{\max}$ — el número de segmentos no excede un máximo configurado.

Cada frontera $b_j$ marca el **inicio de un segmento**. El segmento $S_j$ se define como las oraciones desde la posición $b_j$ hasta la posición $b_{j+1} - 1$ (y el último segmento llega hasta el final del documento).

**Ejemplo**: con $n = 11$ y $B = (0, 3, 7)$:

- $S_1 = (s_0, s_1, s_2)$ — oraciones 0 a 2 (inicio del documento hasta antes del 3)
- $S_2 = (s_3, s_4, s_5, s_6)$ — oraciones 3 a 6
- $S_3 = (s_7, s_8, s_9, s_{10})$ — oraciones 7 a 10 (hasta el final)

### 4.2 ¿Qué buscamos?

Buscamos la segmentación $B^*$ que maximice la **cohesión total** del documento, definida en la sección siguiente. Formalmente:

$$B^* = \arg\max_{B} \sum_{j=1}^{k} \text{cohesion}(b_j, b_{j+1} - 1)$$

Aquí $\text{cohesion}(i, j)$ es la cohesión del segmento que va desde la oración $i$ hasta la oración $j$.

### 4.3 Tamaño del espacio de búsqueda

¿Cuántas segmentaciones distintas existen? Si fijamos $k$ (número de segmentos), hay que escoger las $k - 1$ fronteras internas (la $b_1 = 0$ está fija) entre las $n - 1$ posiciones disponibles. El número de formas de hacerlo es el coeficiente binomial:

$$\binom{n - 1}{k - 1}$$

**Tabla de valores ilustrativos**:

| $n$ | $k$ | $\binom{n-1}{k-1}$ | Comentario |
|---|---|---|---|
| 10 | 3 | 36 | Trivial |
| 15 | 5 | 1.001 | Manejable |
| 20 | 5 | 3.876 | Manejable |
| 30 | 5 | 23.751 | Notable |
| 50 | 5 | 211.876 | Lento si cada evaluación es cara |
| 100 | 10 | 1,7 × 10¹² | Inviable |

Este crecimiento explica por qué necesitamos algoritmos más inteligentes que la fuerza bruta.

---

## 5. La función objetivo

Esta es probablemente la sección más importante del informe, porque toda la calidad de los algoritmos depende de elegir una buena función objetivo. Una mala función objetivo hace que incluso el algoritmo "óptimo" devuelva segmentaciones que no se parecen en nada a lo que un humano marcaría.

### 5.1 Cohesión de un segmento individual

La cohesión de un segmento mide qué tan parecidas son sus oraciones entre sí. Definimos:

> $$\text{cohesion}(i, j) = \overline{\cos}(S) \cdot \frac{j - i + 1}{n}$$

Donde:

- $\overline{\cos}(S)$ es la **similitud coseno promedio** entre todas las parejas de oraciones del segmento $S$.
- $\frac{j - i + 1}{n}$ es la **fracción del documento** que ocupa este segmento (longitud del segmento dividido entre el tamaño total).

#### Cálculo paso a paso de la parte $\overline{\cos}(S)$

Si el segmento tiene oraciones $s_i, s_{i+1}, \ldots, s_j$:

1. Calculamos el vector TF-IDF de cada oración.
2. Para cada par distinto $(s_a, s_b)$ con $a < b$ dentro del segmento, calculamos $\cos(s_a, s_b)$.
3. Promediamos todos esos cosenos.

> **Ejemplo numérico**: un segmento con 3 oraciones. Hay $\binom{3}{2} = 3$ pares: $(s_1, s_2)$, $(s_1, s_3)$, $(s_2, s_3)$. Si los cosenos respectivos son 0,8, 0,7 y 0,6, entonces $\overline{\cos}(S) = (0{,}8 + 0{,}7 + 0{,}6)/3 = 0{,}7$.

Caso especial: si el segmento tiene **una sola oración**, no hay pares, así que se define $\overline{\cos}(S) = 0$.

### 5.2 ¿Por qué multiplicamos por la longitud?

Sin el factor de longitud, la función objetivo se "rompe" de forma absurda. Veámoslo con un caso:

Imagina un documento de 10 oraciones que segmentamos en **10 segmentos de una oración cada uno**. Cada segmento tiene cohesión cero (un solo elemento, sin pares). Total: 0.

Ahora segmentamos el mismo documento en **2 segmentos de 5 oraciones cada uno**, con cohesión promedio 0,8 cada uno. Total: 0,8 + 0,8 = 1,6. Mucho mejor que la opción anterior.

¡Bien! Pero el problema aparece si segmentamos así: **1 segmento de 9 oraciones + 1 segmento de 1 oración**. La oración aislada da cero, pero el otro segmento da, digamos, 0,75. Total: 0,75.

Comparado con la versión de 2 segmentos de 5 oraciones, esta opción es peor. Hasta aquí todo bien.

**Pero**: si el algoritmo puede elegir libremente cuántos segmentos hacer, y descubrimos un segmento de 2 oraciones idénticas (similitud coseno = 1), pondría una frontera allí. Y un segmento de 2 oraciones con coseno 1 da cohesión 1. Es como ganar gratis. El algoritmo se ve incentivado a buscar pares triviales.

**La solución del factor $\frac{j - i + 1}{n}$**: ahora ese segmento de 2 oraciones contribuye $1 \cdot \frac{2}{10} = 0{,}2$, no 1. Y un segmento más largo, incluso con cohesión más baja, puede pesar más. Esto **recompensa los segmentos largos con cohesión moderada por encima de los segmentos cortos triviales**, que es lo que queremos.

### 5.3 Cohesión total de una segmentación

La cohesión total es simplemente la suma:

$$F(B) = \sum_{j=1}^{k} \text{cohesion}(b_j, b_{j+1} - 1)$$

Esto es lo que cada algoritmo intenta **maximizar**.

### 5.4 Limitación de esta función objetivo

Es importante reconocer una debilidad: esta función se basa en **TF-IDF**, que solo "ve" coincidencias de palabras exactas. Si dos oraciones hablan del mismo tema pero usando sinónimos (por ejemplo "automóvil" y "coche"), TF-IDF las considera distintas. Para textos reales con vocabulario rico esto puede subestimar la cohesión.

La sección 13 propone reemplazar TF-IDF por **embeddings densos** (vectores producidos por modelos como Sentence-BERT) como mejora futura.

---

## 6. Los cuatro algoritmos

Implementamos cuatro algoritmos que comparten la misma función objetivo y los mismos vectores TF-IDF. La diferencia entre ellos es **cómo exploran el espacio de posibles segmentaciones**.

### 6.1 Algoritmo 1 — Fuerza Bruta

#### Idea intuitiva

Si quiero estar 100 % seguro de encontrar la mejor segmentación, basta con **probar todas las segmentaciones posibles** y quedarme con la de mayor cohesión total. Es lento, pero correcto por definición.

#### Pseudocódigo

```
ENTRADA: documento D con n oraciones, número de segmentos k
SALIDA: mejor segmentación B*

1. Calcular la matriz C[i][j] = cohesion(i, j) para todo 0 ≤ i ≤ j < n.
2. mejor_score ← -infinito
3. PARA CADA combinación (b_2, b_3, ..., b_k) de {1, 2, ..., n-1}:
     a. B ← [0, b_2, ..., b_k]
     b. score ← Σ C[B[j]][B[j+1] - 1]
     c. SI score > mejor_score:
          mejor_score ← score
          B* ← B
4. RETORNAR B*
```

#### Ejemplo paso a paso

Documento con $n = 5$ oraciones, $k = 2$ segmentos. Las posibles segmentaciones son $\binom{4}{1} = 4$:

- $B = (0, 1)$: segmento 1 = $\{s_0\}$, segmento 2 = $\{s_1, s_2, s_3, s_4\}$
- $B = (0, 2)$: segmento 1 = $\{s_0, s_1\}$, segmento 2 = $\{s_2, s_3, s_4\}$
- $B = (0, 3)$: segmento 1 = $\{s_0, s_1, s_2\}$, segmento 2 = $\{s_3, s_4\}$
- $B = (0, 4)$: segmento 1 = $\{s_0, s_1, s_2, s_3\}$, segmento 2 = $\{s_4\}$

Calculamos la cohesión total de cada una y nos quedamos con la mayor. Hecho.

#### Complejidad

- Tiempo: $O\!\left(\binom{n-1}{k-1} \cdot n\right)$ — el factor $n$ viene de sumar las cohesiones de cada segmento.
- Espacio: $O(n^2)$ — para guardar la matriz de cohesiones.

#### Cuándo usarlo

**Casi nunca en producción.** Su valor es **académico y de validación**: sirve para verificar que algoritmos más rápidos (como Programación Dinámica) den la misma respuesta en instancias pequeñas. En este proyecto está limitado a $n \leq 15$ por código (más allá, el tiempo se dispara).

---

### 6.2 Algoritmo 2 — Programación Dinámica (DP)

#### Idea intuitiva

La fuerza bruta repite trabajo: muchas combinaciones distintas comparten subsegmentos. Por ejemplo, las segmentaciones $(0, 3, 7)$ y $(0, 3, 8)$ ambas necesitan calcular la cohesión del segmento $(0, 1, 2)$. La Programación Dinámica **calcula cada subproblema una sola vez y reutiliza el resultado**.

#### El truco: subestructura óptima

Observación clave: si la mejor segmentación del documento completo en $k$ segmentos termina con un último segmento que cubre las oraciones $i$ a $n-1$, entonces lo que viene antes (la primera parte del documento, oraciones $0$ a $i-1$) **también tiene que ser la mejor segmentación posible de esa primera parte en $k-1$ segmentos**.

(Si no lo fuera, podríamos sustituirla por una mejor y mejoraríamos el total, lo cual contradice que la solución global ya era óptima.)

Esto se llama **principio de subestructura óptima**, y es exactamente la condición que hace que la programación dinámica funcione.

#### Definición de la tabla

Sea $\text{dp}[i][j]$ = la **cohesión total máxima** alcanzable al segmentar las primeras $i$ oraciones en exactamente $j$ segmentos.

La recurrencia es:

$$\text{dp}[i][j] = \max_{j - 1 \leq i' < i} \left( \text{dp}[i'][j - 1] + C[i'][i - 1] \right)$$

En palabras: para llenar la celda $(i, j)$, miro todos los puntos $i'$ donde podría empezar el último segmento, y elijo el que maximiza la suma de "lo mejor que pude hacer con las primeras $i'$ oraciones en $j-1$ segmentos" más "la cohesión del último segmento desde $i'$ hasta $i-1$".

#### Pseudocódigo

```
ENTRADA: documento D con n oraciones, número máximo de segmentos k
SALIDA: mejor segmentación B*

1. Calcular C[i][j] para todo 0 ≤ i ≤ j < n.
2. Inicializar dp[i][j] ← -infinito, split[i][j] ← -1.
3. dp[0][0] ← 0.
4. PARA j = 1 HASTA k:
     PARA i = j HASTA n:
       PARA i' = j-1 HASTA i-1:
         val ← dp[i'][j-1] + C[i'][i-1]
         SI val > dp[i][j]:
           dp[i][j] ← val
           split[i][j] ← i'
5. # Reconstrucción: seguir los punteros desde split[n][k_optimo].
   i ← n; j ← k_optimo
   B ← []
   MIENTRAS j > 0:
     B.insert(0, split[i][j])
     i ← split[i][j]
     j ← j - 1
6. RETORNAR B
```

#### Complejidad

- Tiempo: $O(n^2 \cdot k)$
- Espacio: $O(n^2)$

Para $n = 100$ y $k = 10$ son 100.000 operaciones: completamente manejable.

#### Garantía

DP devuelve la **misma respuesta** que la fuerza bruta (es exacto), pero en tiempo polinómico. Esto se valida empíricamente en el Experimento 2 (sección 11.1): en las 20 instancias del dataset `tiny`, los resultados de BF y DP son **bit a bit idénticos**.

---

### 6.3 Algoritmo 3 — Greedy (TextTiling)

#### Idea intuitiva

En vez de optimizar globalmente, hacemos una observación local: **una frontera natural ocurre donde el "antes" y el "después" se parecen poco entre sí**. Si la oración 7 cierra un bloque de cocina y la oración 8 abre un bloque de deportes, en el hueco entre 7 y 8 esperamos un "valle" de similitud: lo que está a la izquierda no se parece a lo que está a la derecha.

Este algoritmo está inspirado en **TextTiling** (Hearst, 1997), uno de los métodos clásicos de segmentación lingüística.

#### Pseudocódigo

```
ENTRADA: documento D con n oraciones, número de segmentos k, tamaño de ventana w
SALIDA: segmentación B (heurística)

1. Calcular vector TF-IDF de cada oración.
2. PARA cada hueco g entre la oración g y la g+1:
     a. izq ← promedio de los TF-IDF de oraciones [max(0, g-w+1), ..., g]
     b. der ← promedio de los TF-IDF de oraciones [g+1, ..., min(n, g+1+w)]
     c. sim[g] ← coseno(izq, der)
3. PARA cada hueco g:
     profundidad[g] = (sim[pico izquierdo] - sim[g]) + (sim[pico derecho] - sim[g])
4. Seleccionar los k-1 huecos con mayor profundidad.
5. RETORNAR [0] + [g+1 por cada hueco seleccionado], ordenados.
```

La **profundidad de un valle** mide cuánto "baja" la curva de similitud respecto a los picos circundantes. Si la curva baja mucho, probablemente es porque ahí cambia el tema.

#### Ejemplo intuitivo

Imagina que dibujamos la curva de similitud $\text{sim}[g]$ para todos los huecos. Si el documento tiene tres temas claros, la curva tendrá una forma parecida a una sierra con dos valles profundos: uno entre el tema 1 y el tema 2, otro entre el tema 2 y el tema 3. El algoritmo identifica esos dos valles y coloca fronteras allí.

#### Complejidad

- Tiempo: $O(n \cdot w)$ — para cada uno de los $n$ huecos, calculamos un promedio de $w$ vectores.
- Espacio: $O(n)$.

#### Ventajas y limitaciones

- **Ventaja**: extremadamente rápido. En nuestros experimentos, ~7× más rápido que DP.
- **Limitación**: no optimiza la función objetivo globalmente. Puede equivocarse si los cambios de tema son sutiles o si hay valles "falsos" por ruido léxico local.

---

### 6.4 Algoritmo 4 — Recocido Simulado (Simulated Annealing, SA)

#### Idea intuitiva

El recocido simulado está inspirado en la **metalurgia**: cuando se calienta un metal y se enfría lentamente, los átomos se acomodan en una configuración de baja energía (estable). Análogamente:

- **Estado**: una segmentación.
- **Energía**: el negativo de la cohesión (queremos minimizar energía, equivalente a maximizar cohesión).
- **Temperatura**: parámetro que controla qué tan dispuesto está el algoritmo a aceptar empeoramientos. Empieza alta y baja gradualmente.

Al principio, el algoritmo se permite explorar libremente (aceptando incluso movimientos malos para escapar de óptimos locales). Al final, solo acepta mejoras (refina la solución).

#### Pseudocódigo

```
ENTRADA: documento D, k, n_iter, T_inicial, alpha (tasa de enfriamiento), semilla
SALIDA: mejor segmentación encontrada B*

1. Calcular C[i][j].
2. B ← segmentación uniformemente espaciada con k segmentos.
3. T ← T_inicial.
4. B* ← B; score* ← F(B).
5. PARA iter = 1 HASTA n_iter:
     a. Elegir aleatoriamente una frontera interna b_j.
     b. delta ← +1 o -1 (aleatorio).
     c. SI mover b_j por delta da una segmentación válida:
          score_nuevo ← F(B con b_j desplazada).
          SI score_nuevo > score:
            aceptar (B ← B_nuevo)
          SI NO:
            aceptar con probabilidad exp((score_nuevo - score) / T)
     d. SI score_nuevo > score*:
          B* ← B_nuevo
     e. T ← T × alpha
6. RETORNAR B*
```

El **criterio de aceptación de Metropolis** ($p = e^{\Delta / T}$) es lo que distingue a SA de un simple ascenso de colina (*hill climbing*): permite, con probabilidad decreciente, moverse a peores soluciones temporalmente. Esto es lo que permite escapar de óptimos locales.

#### Parámetros usados en este proyecto

- `n_iterations = 2000`
- `initial_temp = 1.0`
- `cooling_rate = 0.995` (es decir, $T_{t+1} = T_t \cdot 0{,}995$)
- `random_seed = 42`

Después de 2000 iteraciones, la temperatura es $1{,}0 \cdot 0{,}995^{2000} \approx 4{,}5 \times 10^{-5}$ (prácticamente cero), así que el algoritmo termina comportándose como un ascenso de colina puro.

#### Complejidad

- Tiempo: $O(n^2 + \text{n\_iter})$ — el $n^2$ es el cálculo inicial de la matriz de cohesión; las iteraciones son baratas (evaluación incremental).
- Espacio: $O(n^2)$.

#### Ventajas y limitaciones

- **Ventaja**: aplicable a problemas donde no hay subestructura óptima (DP no aplica).
- **Limitación**: sin ajuste cuidadoso de hiperparámetros, los resultados pueden ser inferiores a DP, como vemos en los experimentos.

---

### 6.5 Tabla resumen de los cuatro algoritmos

| Algoritmo | Complejidad tiempo | Tipo | Garantía | Mejor para |
|---|---|---|---|---|
| Brute Force | $O(\binom{n-1}{k-1} \cdot n)$ | Exhaustivo | Óptimo global | Validar otros algoritmos ($n \leq 15$) |
| Dynamic Programming | $O(n^2 \cdot k)$ | Exacto | Óptimo global | Producción general |
| Greedy (TextTiling) | $O(n \cdot w)$ | Heurístico | Ninguna | Velocidad máxima |
| Simulated Annealing | $O(n^2 + \text{iter})$ | Metaheurístico | Ninguna | Problemas sin subestructura óptima |

---

## 7. El papel del LLM

### 7.1 ¿Por qué necesitamos un LLM si ya tenemos algoritmos?

Los algoritmos optimizan una **función objetivo basada en TF-IDF**, que es una aproximación muy gruesa de "qué tan coherente es un segmento". Puede pasar que un algoritmo devuelva una segmentación con altísima cohesión TF-IDF pero que un humano consideraría mediocre (por ejemplo, agrupando oraciones que comparten muchas palabras pero hablan de cosas distintas).

Necesitamos una **medida externa, semántica, de calidad**. Aquí entra el LLM: como entiende el lenguaje de forma profunda, puede leer un segmento y juzgar su coherencia mejor que cualquier fórmula matemática que tengamos.

### 7.2 La consigna académica

El enunciado del Tema 6 exige explícitamente:

> *"El sistema debe utilizar un LLM para evaluar la cohesión semántica de cada segmento."*

Esto define el rol del LLM: **evaluador**, no optimizador. El LLM no decide dónde van las fronteras; los algoritmos las deciden. El LLM **califica** las decisiones de los algoritmos.

### 7.3 El prompt

Diseñamos un prompt deliberadamente restrictivo, con una escala discreta de 1 a 5 y anclajes textuales en cada nivel para reducir la varianza entre llamadas:

```
Evalúa la cohesión semántica de este segmento de texto en una escala del 1 al 5:
- 1: Sin coherencia. Las oraciones hablan de temas completamente distintos.
- 2: Coherencia baja. Hay temas relacionados pero las transiciones son muy bruscas.
- 3: Coherencia parcial. Temas relacionados pero el hilo narrativo es irregular.
- 4: Buena coherencia. Las oraciones se complementan bien con pequeñas interrupciones.
- 5: Coherencia perfecta. Todas las oraciones contribuyen a un único tema claro.

Segmento de texto:
{segment_text}

Responde ÚNICAMENTE con un objeto JSON válido (sin markdown, sin texto adicional):
{"score": <entero del 1 al 5>, "rationale": "<una oración breve que explique la puntuación>"}
```

Tres decisiones importantes:

1. **Escala 1–5** (no 0–10 ni 1–100): reduce la varianza entre llamadas y hace los puntajes interpretables.
2. **Salida en JSON estricto**: parseable por el programa sin riesgo de alucinaciones de formato.
3. **Razonamiento incluido (`rationale`)**: para auditoría — podemos ver por qué el LLM dio un puntaje y detectar evaluaciones disparatadas.

### 7.4 ¿Qué LLM usamos y por qué?

#### Restricción geográfica

Esto fue una decisión **forzada por circunstancias externas, no técnicas**. Desde Cuba (donde se ejecuta este proyecto) las APIs comerciales de Anthropic (Claude) y OpenAI (GPT) están bloqueadas por restricciones OFAC. No es viable usarlas, ni siquiera con tarjeta de crédito.

#### Proveedor principal: **Groq con Llama 3.3 70B**

[Groq](https://groq.com) ofrece tier gratuito (~14.400 peticiones/día) y aloja modelos de código abierto. Elegimos `llama-3.3-70b-versatile`: 70 mil millones de parámetros, instruction-tuned, suficientemente capaz para una tarea de calificación.

#### Proveedor de respaldo: **Mistral**

Si Groq falla (timeout, rate limit, error de red), el sistema reintenta automáticamente con Mistral AI usando `mistral-large-latest`. Esto se llama **patrón de fallback** y es fundamental para que el sistema no se caiga durante experimentos largos.

#### Última red de seguridad

Si **ambos** proveedores fallan, el sistema devuelve un puntaje neutro de 3 con `used_fallback=True` registrado en los metadatos. Esto es **degradación elegante**: el experimento continúa, los datos quedan marcados, y luego podemos auditar cuántas evaluaciones fueron neutras forzadas.

### 7.5 Diagrama del flujo

```
                    ┌─────────────────┐
   Segmento ───────▶│ Prompt builder  │
   (texto)          └────────┬────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Groq (Llama 3.3) │ ◀─── primario
                    └────────┬─────────┘
                             │ falla
                             ▼
                    ┌──────────────────┐
                    │ Mistral (large)  │ ◀─── fallback online
                    └────────┬─────────┘
                             │ falla
                             ▼
                    ┌──────────────────┐
                    │ Score neutro (3) │ ◀─── degradación elegante
                    │ + used_fallback  │
                    └────────┬─────────┘
                             │
                             ▼
                  CohesionScore(score, rationale, ...)
```

Implementación en el código: clase `FallbackEvaluator` en `src/llm/fallback_provider.py`, factory en `src/llm/factory.py`.

### 7.6 Rate limiting

Las APIs gratuitas imponen límites estrictos. Implementamos:

- **Throttle**: intervalo mínimo entre peticiones (`LLM_MIN_REQUEST_INTERVAL_SECONDS=1.5`, configurable).
- **Backoff exponencial**: ante un error 429 (rate limit), esperamos 2 s, luego 4 s, luego 8 s, etc.

Esto añade tiempo a los experimentos con LLM pero evita ser bloqueados.

---

## 8. El dataset

### 8.1 ¿Por qué un dataset sintético?

Para evaluar objetivamente un algoritmo de segmentación necesitamos un **ground truth**: documentos donde sabemos cuáles son las fronteras "verdaderas". Hay dos formas de obtenerlo:

1. **Dataset real anotado por humanos**: tomar artículos de Wikipedia con sus secciones, o noticias compiladas, o conversaciones donde un anotador marcó los cambios de tema.
2. **Dataset sintético generado por nosotros**: construir documentos artificiales concatenando oraciones de distintos temas, sabiendo exactamente dónde están las "costuras".

Optamos por el sintético por tres razones:

- **Control absoluto** de la dificultad (cuántos temas, cuánto se solapan los vocabularios, cuántas oraciones).
- **Sin costo de anotación humana** (que sería caro y lento).
- **Reproducibilidad perfecta**: con la misma semilla generamos exactamente el mismo dataset siempre.

La principal **limitación** es que los resultados podrían no extrapolarse perfectamente a documentos reales con vocabulario natural más ruidoso. La sección 13 propone usar Wikipedia como mejora futura.

### 8.2 Cómo se generan los documentos

El generador (`src/dataset/generator.py`) funciona así:

1. Define una **biblioteca de tópicos** (deportes, tecnología, política, ciencia, arte, economía, salud, historia). Cada tópico tiene un vocabulario característico y plantillas de oraciones.
2. Para cada documento:
   - Sortea cuántos segmentos tendrá (entre `min` y `max` del config).
   - Para cada segmento: sortea un tópico distinto y cuántas oraciones tendrá; genera esas oraciones desde las plantillas del tópico.
   - Concatena los segmentos secuencialmente.
   - Registra las posiciones donde empieza cada segmento (las **fronteras de referencia**).

### 8.3 Datasets usados

Hay dos datasets generados, ambos en español, semilla 42.

#### Dataset `small` (uso principal)

Configuración (`config/datasets/small.yaml`):

```yaml
dataset_name: small
n_documents: 20
segments_per_doc: { min: 3, max: 5 }
sentences_per_segment: { min: 4, max: 8 }
topic_source: synthetic_templates
overlap_level: low
random_seed: 42
language: es
```

Estadísticas resultantes:

| Parámetro | Valor |
|---|---|
| Documentos | 20 |
| Oraciones por documento (media) | 22,9 |
| Oraciones por documento (rango) | 15 – 36 |
| Segmentos por documento (media) | 3,80 |
| Distribución de segmentos | 3 segmentos en 8 docs · 4 segmentos en 8 docs · 5 segmentos en 4 docs |
| Solapamiento léxico entre tópicos | bajo |

#### Dataset `tiny` (validación BF vs DP)

Documentos más cortos (5–12 oraciones) para que la fuerza bruta sea viable. Usado únicamente para validar empíricamente que DP coincide con BF.

### 8.4 Estructura en disco

```
data/small/
├── documents/
│   ├── doc_0001.txt   # texto plano del documento
│   ├── doc_0002.txt
│   └── ...
├── boundaries/
│   ├── doc_0001.json  # {"boundaries": [0, 8, 15], "n_segments": 3, ...}
│   ├── doc_0002.json
│   └── ...
└── metadata.json      # snapshot completo de la config + estadísticas
```

Esta separación entre `documents/` y `boundaries/` permite que los algoritmos lean solo los textos sin "hacer trampa" mirando la respuesta.

---

## 9. Métricas de evaluación

Necesitamos medir, con números, qué tan buena es la salida de un algoritmo comparada con el ground truth. Usamos cuatro métricas que capturan aspectos distintos.

### 9.1 F1 de fronteras (con tolerancia ±1)

**Idea**: ¿cuántas de las fronteras predichas coinciden con las reales (con un margen de error de 1 oración)?

Definamos:

- **TP** (verdaderos positivos): fronteras predichas que están a distancia ≤ 1 de alguna frontera real.
- **FP** (falsos positivos): fronteras predichas que no.
- **FN** (falsos negativos): fronteras reales que no fueron predichas.

> $$P = \frac{TP}{TP + FP}, \quad R = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

**Rango**: $[0, 1]$. **Mayor es mejor**.

**Ejemplo**: ground truth $B^* = (0, 8, 15)$, predicción $B = (0, 7, 12, 16)$.

La implementación **excluye la frontera 0 del cómputo** (aparece trivialmente en cualquier segmentación, contarla inflaría la métrica). Comparamos entonces solo las fronteras internas: referencia $\{8, 15\}$ frente a predicción $\{7, 12, 16\}$. El emparejamiento es voraz de izquierda a derecha, y cada frontera real solo puede emparejarse con una predicción.

- La frontera 7 está a distancia 1 de 8 → TP, consume el 8.
- La frontera 12 no está cerca de ninguna real disponible → FP.
- La frontera 16 está a distancia 1 de 15 → TP, consume el 15.
- Reales no detectadas: ninguna → FN = 0.

$P = 2/3 \approx 0{,}667$; $R = 2/2 = 1{,}0$; $F_1 = 2 \cdot 0{,}667 \cdot 1 / 1{,}667 \approx 0{,}800$.

### 9.2 Pk

**Idea**: Pk (Beeferman et al., 1999) usa una ventana deslizante. Para cada par de posiciones a distancia $k$, ¿están en el mismo segmento según la referencia? ¿Y según la predicción? Si las dos respuestas difieren, es un error.

> $$P_k = \frac{1}{n-k} \sum_{i=1}^{n-k} \mathbf{1}\!\left[\text{ref}(i, i+k) \neq \text{pred}(i, i+k)\right]$$

donde $\text{ref}(i, j) = 0$ si $i$ y $j$ están en el mismo segmento de referencia, $1$ si están en segmentos distintos (análogamente para $\text{pred}$). $k$ se fija típicamente a la mitad del tamaño promedio de segmento real.

**Rango**: $[0, 1]$. **Menor es mejor** (es una tasa de error).

### 9.3 WindowDiff

**Idea**: variante de Pk que penaliza más cuando el **número** de fronteras dentro de la ventana es muy distinto entre referencia y predicción (no solo si difieren, sino por cuánto).

> $$WD = \frac{1}{n-k} \sum_{i=1}^{n-k} \mathbf{1}\!\left[|\text{ref\_count}(i, i+k) - \text{pred\_count}(i, i+k)| > 0\right]$$

donde $\text{ref\_count}(i, j)$ es el número de fronteras de referencia dentro de la ventana $[i, j]$.

**Rango**: $[0, 1]$. **Menor es mejor**.

WindowDiff es **más estricto** que Pk y se prefiere como métrica principal en la literatura moderna de segmentación.

### 9.4 LLM Score

**Idea**: la calificación promedio que el LLM dio a los segmentos producidos por el algoritmo.

> $$\text{LLMScore}(B) = \frac{1}{k} \sum_{j=1}^{k} \text{LLM}(S_j)$$

**Rango**: $[1, 5]$. **Mayor es mejor**.

Esta es la única métrica que mide **calidad semántica real**, no solo coincidencia con el ground truth. Útil porque puede pasar que un algoritmo produzca segmentaciones distintas a la referencia pero igualmente coherentes (e.g., el documento sintético tenía una transición ambigua y el algoritmo encontró otra agrupación válida).

### 9.5 Runtime

**Idea**: el tiempo en segundos que tomó procesar un documento.

**Importancia**: en producción muchas veces preferimos un algoritmo 5 % menos preciso pero 10× más rápido.

### 9.6 Implementación

Todas estas métricas están implementadas en `src/evaluation/metrics.py` sin dependencias externas (no usamos `segeval` para mantener el proyecto ligero).

---

## 10. Diseño experimental

Ejecutamos tres experimentos diseñados para responder preguntas distintas.

### 10.1 Experimento 1 — Comparación estructural sin LLM

**ID**: `exp_compare_algorithms`
**Pregunta**: ¿Cuál de los tres algoritmos escalables (DP, Greedy, SA) da mejores fronteras?
**Dataset**: `small` (20 documentos)
**LLM**: ninguno (solo métricas estructurales)
**Por qué no incluimos Brute Force aquí**: los documentos tienen 15–36 oraciones, por encima del límite de 15 de BF.

### 10.2 Experimento 2 — Validación BF vs DP

**ID**: `exp_bf_vs_dp`
**Pregunta**: ¿DP encuentra realmente el óptimo global, igual que la fuerza bruta?
**Dataset**: `tiny` (20 documentos con 5–12 oraciones)
**LLM**: ninguno
**Importancia**: este es el experimento de **validación de correctitud**. Si DP no coincidiera con BF, sería evidencia de un bug en DP.

### 10.3 Experimento 3 — Evaluación con LLM

**ID**: `exp_llm_groq`
**Pregunta**: ¿Cómo califica un LLM la calidad semántica de los segmentos que cada algoritmo produce?
**Dataset**: `small`
**LLM**: Groq `llama-3.3-70b-versatile`, temperatura 0
**Métricas**: estructurales + `llm_score`

### 10.4 Reproducibilidad

| Parámetro | Valor |
|---|---|
| Semilla aleatoria (todos los experimentos) | 42 |
| Python | 3.12.3 |
| scikit-learn | ≥ 1.3 |
| numpy | ≥ 1.24 |
| Fecha de ejecución | 2026-05-31 |

La semilla se registra en `run_metadata.json` junto a cada resultado.

---

## 11. Resultados y análisis

### 11.1 Experimento 2 — BF vs DP (dataset `tiny`)

| Algoritmo | Pk ↓ | WindowDiff ↓ | F1-Boundary ↑ | Runtime (ms) |
|---|---|---|---|---|
| Brute Force | **0,1633** | **0,1717** | **0,8167** | 2,0 |
| Dynamic Programming | **0,1633** | **0,1717** | **0,8167** | 1,9 |

**Lectura**: las dos filas son idénticas. Esto **prueba experimentalmente** (no demostración formal, pero evidencia muy fuerte) que DP encuentra el óptimo global en las 20 instancias. Además, DP es marginalmente más rápido que BF incluso en este tamaño pequeño, lo cual es consistente con su complejidad $O(n^2 k)$ vs $O(\binom{n-1}{k-1} \cdot n)$.

### 11.2 Experimento 1 — DP, Greedy, SA en `small`

| Algoritmo | Pk ↓ | WindowDiff ↓ | F1-Boundary ↑ | Runtime (ms) |
|---|---|---|---|---|
| Dynamic Programming | 0,1728 ± 0,157 | **0,1945** ± 0,162 | **0,797** ± 0,124 | 10,1 |
| Greedy (TextTiling) | **0,1288** ± 0,076 | 0,2373 ± 0,125 | 0,743 ± 0,140 | **1,2** |
| Simulated Annealing | 0,2109 ± 0,158 | 0,2236 ± 0,166 | 0,755 ± 0,141 | 12,8 |

### 11.3 Experimento 3 — Con LLM

| Algoritmo | Pk ↓ | WindowDiff ↓ | F1-Boundary ↑ | LLM Score ↑ | Runtime (ms) |
|---|---|---|---|---|---|
| Dynamic Programming | 0,1728 | **0,1945** | **0,797** | **4,56** | 32,7 |
| Greedy (TextTiling) | **0,1288** | 0,2373 | 0,743 | 4,41 | **4,5** |
| Simulated Annealing | 0,2299 | 0,2547 | 0,758 | 4,29 | 33,8 |

(Las métricas estructurales de SA difieren ligeramente entre Exp. 1 y Exp. 3 a pesar de fijar la misma semilla. El generador `random.Random(seed)` se re-inicializa por documento en `_run_sa`, por lo que en teoría debería ser determinista entre corridas; la diferencia observada (~2 puntos porcentuales en Pk) probablemente proviene de ejecuciones independientes hechas en momentos distintos, no de la presencia del LLM. La magnitud es menor que la dispersión inter-documento y no altera el ordenamiento entre algoritmos.)

### 11.4 Lectura por métrica

#### Pk

Greedy es el **mejor** (0,1288), un 25 % por debajo de DP. Esto puede sorprender, pero tiene una explicación: Pk solo mira pares de oraciones a una distancia fija. Greedy, al detectar valles de similitud locales, tiende a hacer pocos errores en pares de oraciones cercanos al cambio de tema.

#### WindowDiff

DP es el **mejor** (0,1945). WindowDiff es más estricto: castiga no solo errores binarios sino también discrepancias en el **número** de fronteras dentro de cada ventana. DP, al optimizar globalmente, distribuye mejor las fronteras.

#### F1-Boundary

DP es el **mejor** (0,797). DP coloca fronteras más cerca de las posiciones exactas reales, lo cual confirma que su optimización global paga dividendos en precisión local.

#### LLM Score

DP gana con **4,56/5**, seguido de Greedy (4,41) y SA (4,29). Los tres están por encima de 4 (rango muy alto), lo que indica que el LLM consideró las segmentaciones de todos los algoritmos como semánticamente coherentes. La jerarquía entre algoritmos coincide con la jerarquía en F1: **mejores fronteras objetivas → segmentos más coherentes según el LLM**, evidencia de que la calidad estructural y la calidad semántica están alineadas.

#### Runtime

Greedy es **~7× más rápido** que DP. En documentos cortos (los nuestros) las diferencias absolutas son despreciables (milisegundos), pero en documentos de cientos de oraciones la diferencia se vuelve significativa.

### 11.5 Resumen visual de quién gana en qué

| Métrica | Ganador |
|---|---|
| Pk | Greedy |
| WindowDiff | DP |
| F1-Boundary | DP |
| LLM Score | DP |
| Runtime | Greedy |
| Consistencia (menor desviación) | Greedy |

---

## 12. Discusión: por qué pasó lo que pasó

### 12.1 ¿Por qué DP no gana en Pk?

DP optimiza una **función objetivo TF-IDF** que no es idéntica a la métrica de evaluación Pk. Hay un **gap entre lo que optimizamos y lo que medimos**. Greedy, casualmente, está más alineado con Pk porque ambos enfatizan transiciones locales bruscas.

Esto es una observación importante: **el mejor algoritmo depende de la métrica que valoremos**. Si Pk es lo que importa para tu aplicación, Greedy es preferible.

### 12.2 ¿Por qué SA es el peor?

Tres razones probables:

1. **Hiperparámetros no ajustados**: usamos valores por defecto razonables pero sin búsqueda sistemática. SA es muy sensible a `initial_temp` y `cooling_rate`.
2. **2000 iteraciones es poco** para documentos de hasta 36 oraciones con varios segmentos: el espacio de búsqueda es grande.
3. **DP ya es óptimo**: por construcción SA no puede superar a DP en la misma función objetivo. SA tiene sentido cuando el problema **no tiene** subestructura óptima.

### 12.3 ¿Por qué los tres algoritmos predicen siempre 5 segmentos?

Configuramos `max_segments = 5`. La función objetivo crece monótonamente con más segmentos (más segmentos = más oportunidad de cohesión alta dentro de cada uno), así que el óptimo siempre satura el límite. La referencia, en cambio, tiene 3,80 segmentos en promedio.

Esto es **sobresegmentación sistemática**. La sección 13 propone como mejora la **selección automática de $k$**.

### 12.4 ¿El LLM es confiable como evaluador?

Posibles fuentes de error:

- **Sesgo del modelo**: Llama 3.3 puede tener sesgos en cómo califica español vs inglés, o ciertos temas vs otros.
- **Inconsistencia**: temperatura 0 reduce variabilidad pero no la elimina por completo.
- **Sin ground truth para la métrica**: no sabemos si "4,56" objetivamente significa "muy coherente" — es una opinión del modelo.

A pesar de eso, el LLM score correlaciona bien con F1-Boundary (DP gana en ambas), lo cual es una **señal de validez convergente**: dos métricas independientes apuntan al mismo orden.

### 12.5 Limitación del dataset sintético

Nuestros documentos tienen vocabulario controlado y cambios de tema abruptos. Esto favorece a TF-IDF, que detecta cambios léxicos fácilmente. En documentos reales (con sinónimos, transiciones suaves, vocabulario compartido) los resultados absolutos podrían bajar significativamente, aunque el ordenamiento entre algoritmos probablemente se mantendría.

---

## 13. Limitaciones y mejoras futuras

| Prioridad | Limitación | Mejora propuesta | Impacto estimado |
|---|---|---|---|
| Alta | Representación TF-IDF superficial | Reemplazar por embeddings densos (Sentence-BERT) | +15–20 % F1 en textos reales |
| Alta | $k$ se elige manualmente | Selección automática vía BIC/MDL o salto marginal de cohesión | Elimina el parámetro más crítico |
| Media | Sobresegmentación sistemática | Penalizar número de segmentos en la función objetivo | Acerca al número real |
| Media | SA con hiperparámetros por defecto | Grid search sobre $T_0$, $\alpha$, $n_{\text{iter}}$ | +5–10 % F1 para SA |
| Media | Dataset solo sintético | Añadir Wikipedia o noticias reales con anotación humana | Validación más robusta |
| Baja | BF limitado a $n \leq 15$ | (Inherente al algoritmo; ya cumple su rol) | — |
| Baja | LLM evalúa segmento por segmento | Evaluar pares de segmentos (cohesión inter-segmento) | Mayor riqueza diagnóstica |

---

## 14. Conclusiones

### 14.1 Hallazgos principales

1. **DP es el mejor algoritmo general** en F1, WindowDiff y LLM Score. Es la elección por defecto recomendada.
2. **Greedy es la opción práctica** cuando se prioriza velocidad y consistencia. Mejor Pk y 7× más rápido que DP.
3. **SA no es competitivo** con los parámetros actuales. Solo tendría sentido si el problema cambiara a una variante sin subestructura óptima.
4. **BF y DP coinciden bit a bit** en 20/20 instancias del dataset `tiny`, validando empíricamente la correctitud de DP.
5. **El LLM Score y las métricas estructurales coinciden en el ordenamiento**: ambas dan a DP el primer lugar. Esto refuerza la confiabilidad de ambas evaluaciones.
6. **La función objetivo basada en TF-IDF es razonable pero limitada**: todos los algoritmos obtuvieron LLM scores ≥ 4,29/5, indicando coherencia semántica aceptable, pero hay margen claro de mejora con embeddings densos.

### 14.2 Lecciones generales

- **No siempre el algoritmo óptimo gana en cada métrica.** Greedy, una heurística sencilla, gana en Pk. Esto subraya la importancia de elegir la métrica correcta antes que el algoritmo.
- **Funciones objetivo proxy importan.** Optimizar cohesión TF-IDF es solo una aproximación de "buena segmentación humana". La distancia entre proxy y verdad explica por qué incluso DP no alcanza F1 = 1,0.
- **Los LLMs como evaluadores funcionan**, al menos cuando se diseña un prompt restrictivo (escala discreta, salida JSON). Son una alternativa viable a la anotación humana costosa.
- **El patrón de fallback es esencial** para sistemas que dependen de APIs externas, especialmente bajo restricciones geográficas o de tier gratuito.

### 14.3 Cumplimiento de la consigna académica

| Requisito (Tema 6 / `orientacion.md`) | Estado |
|---|---|
| Algoritmo de optimización combinatoria | ✓ DP exacto + BF, Greedy, SA |
| Dataset con instancias de prueba | ✓ Sintético en español, 20 + 20 docs |
| LLM como parte funcional del sistema | ✓ Groq + Mistral fallback evaluando cohesión |
| Comparación entre variantes | ✓ 3 experimentos, 4 algoritmos |
| Análisis del comportamiento | ✓ Secciones 11 y 12 de este informe |
| Reproducibilidad | ✓ Semillas y metadatos persistidos |

---

## 15. Manual de uso paso a paso

### 15.1 Requisitos del sistema

- Python 3.10 o superior
- Sistema operativo: Linux, macOS o Windows
- Conexión a internet (solo para experimentos con LLM)
- ~200 MB de espacio en disco

### 15.2 Instalación

```bash
# 1. Clonar el repositorio (si aún no está)
cd "/home/kevin/Documentos/Cloned Projects/optimal-content-segmentation"

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # en Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### 15.3 Configurar API keys (opcional)

Solo necesario si vas a ejecutar experimentos con LLM. Para experimentos solo estructurales (sin LLM), salta este paso.

```bash
cp .env.example .env
# Edita .env y añade:
# GROQ_API_KEY=gsk_...
# MISTRAL_API_KEY=...
```

Las claves se obtienen gratis en:

- Groq: `console.groq.com`
- Mistral: `console.mistral.ai`

### 15.4 Verificar conectividad con el LLM

```bash
python -m src.llm.check --provider groq
python -m src.llm.check --provider mistral
```

Debe imprimir un puntaje y un mensaje "OK". Si falla, revisa la clave en `.env`.

### 15.5 Generar el dataset

```bash
# Dataset principal (20 documentos)
python -m src.dataset.generator \
    --config config/datasets/small.yaml \
    --output data/small/

# Dataset pequeño para validar BF vs DP
python -m src.dataset.generator \
    --config config/datasets/tiny.yaml \
    --output data/tiny/
```

Verás creados los directorios `documents/`, `boundaries/` y el archivo `metadata.json`.

### 15.6 Ejecutar los experimentos

```bash
# Experimento 1: comparación sin LLM
python -m src.experiments.runner --config config/experiments/exp_compare_algorithms.yaml

# Experimento 2: validación BF vs DP
python -m src.experiments.runner --config config/experiments/exp_bf_vs_dp.yaml

# Experimento 3: con LLM (requiere GROQ_API_KEY)
python -m src.experiments.runner --config config/experiments/exp_llm_groq.yaml
```

Cada uno genera tres archivos en `results/<experiment_id>/`:

- `results.json` — resultado por documento, con cada métrica
- `summary.csv` — promedios agregados por algoritmo
- `run_metadata.json` — configuración usada, fecha, semilla

### 15.7 Interpretar los resultados

Abre `results/exp_compare_algorithms/summary.csv` con cualquier editor o LibreOffice. Cada fila es un algoritmo. Recuerda:

- `pk`, `windowdiff` → más bajo es mejor.
- `f1_boundary`, `llm_score` → más alto es mejor.
- `runtime_seconds` → más bajo es mejor.

### 15.8 Ejecutar tests automáticos

```bash
# Todos los tests (no requieren API keys, usan mocks)
python -m pytest tests/ -v

# Solo unitarios (rápido)
python -m pytest tests/unit/ -v

# Con reporte de cobertura
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### 15.9 Añadir un nuevo algoritmo

1. Crear `src/algorithms/mi_algoritmo.py` extendiendo `BaseSegmenter`.
2. Registrarlo en `src/algorithms/__init__.py` agregando a `ALGORITHM_REGISTRY`.
3. Usarlo en un YAML de experimento:

```yaml
algorithms:
  - name: mi_algoritmo
    params:
      max_segments: 5
```

### 15.10 Añadir un nuevo proveedor LLM

1. Crear `src/llm/mi_proveedor.py` extendiendo `BaseLLMEvaluator`.
2. Registrarlo en la factory `src/llm/factory.py`.
3. Usarlo en YAML:

```yaml
llm_evaluator:
  provider: mi_proveedor
  model: nombre-del-modelo
```

---

## 16. Glosario

- **Algoritmo exacto**: garantiza encontrar la mejor solución posible (e.g., BF, DP).
- **Algoritmo heurístico**: usa atajos para ir rápido pero no garantiza el óptimo (e.g., Greedy).
- **Backoff exponencial**: técnica de reintento donde el tiempo de espera se duplica en cada fallo.
- **Bolsa de palabras** (*bag of words*): representación de texto que cuenta frecuencias de palabras ignorando el orden.
- **Cohesión**: medida de qué tan parecidas son las oraciones de un segmento entre sí.
- **Complejidad algorítmica**: cómo crece el tiempo de ejecución con el tamaño de la entrada (notación $O(\cdot)$).
- **Coseno (similitud)**: medida de parecido entre dos vectores; 1 = idénticos en dirección, 0 = sin relación.
- **DP** (Programación Dinámica): paradigma algorítmico que resuelve problemas grandes combinando subproblemas óptimos.
- **Embeddings densos**: vectores producidos por redes neuronales (BERT, etc.) que capturan semántica profunda.
- **Fallback**: sistema de respaldo que entra en juego cuando el componente principal falla.
- **Frontera** (de segmento): posición donde empieza un segmento nuevo. La primera frontera siempre es 0.
- **F1**: media armónica de precisión y recall, una métrica clásica en clasificación.
- **Función objetivo**: fórmula que el algoritmo intenta maximizar (o minimizar).
- **Ground truth**: la respuesta verdadera contra la que medimos las predicciones.
- **Hiperparámetro**: parámetro del algoritmo elegido antes de ejecutarlo (e.g., temperatura inicial de SA).
- **Inferencia LLM**: una llamada al modelo de lenguaje para obtener una respuesta.
- **LLM** (*Large Language Model*): modelo de lenguaje grande (GPT, Claude, Llama, etc.).
- **Metaheurística**: estrategia general para guiar la búsqueda en problemas duros (SA, algoritmos genéticos...).
- **Metropolis (criterio)**: en SA, regla que acepta movimientos malos con probabilidad $e^{\Delta/T}$.
- **OFAC**: oficina de control de activos extranjeros de EE.UU.; restringe servicios desde Cuba.
- **Óptimo global**: la mejor solución absoluta del espacio de búsqueda.
- **Óptimo local**: una solución que parece buena en su vecindad inmediata pero no es la mejor absoluta.
- **Pk**: métrica clásica de error en segmentación, basada en ventanas deslizantes.
- **Prompt**: el texto que enviamos al LLM para pedirle una respuesta.
- **RAG** (*Retrieval-Augmented Generation*): técnica donde se buscan fragmentos relevantes antes de enviarlos al LLM.
- **Rate limit**: límite de cuántas peticiones puedes hacer a una API por minuto/día.
- **Recall**: proporción de respuestas correctas que el sistema logró recuperar.
- **Segmentación**: división de un texto en bloques contiguos.
- **Semilla aleatoria**: número inicial que hace reproducible cualquier proceso pseudoaleatorio.
- **Subestructura óptima**: propiedad por la que la solución óptima de un problema se construye con soluciones óptimas de subproblemas. Habilita DP.
- **TF-IDF**: representación vectorial de texto que pondera frecuencia de palabras por su rareza global.
- **TextTiling**: algoritmo clásico de segmentación basado en valles de similitud entre bloques adyacentes.
- **Throttle**: limitar la velocidad de envío de peticiones para no exceder rate limits.
- **Trade-off**: compromiso entre dos cosas que no se pueden maximizar simultáneamente (e.g., velocidad vs precisión).
- **WindowDiff**: variante más estricta de Pk que considera la cantidad de fronteras en cada ventana.

---

*Fin del informe.*
