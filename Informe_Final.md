# Informe Final del Proyecto

## Segmentación Óptima de Contenido

Autor: Kevin Torres Perera
Fecha: 2026-06-11
Asignatura: Inteligencia Artificial 2025–2026 — Tema 6
Repositorio: [github.com/KevinTorres01/optimal-content-segmentation](https://github.com/KevinTorres01/optimal-content-segmentation)
Versión: Documento de entrega.

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

### 1.1 ¿Qué implementa este proyecto?

Este proyecto implementa un sistema de segmentación automática de textos: divide un documento largo en fragmentos coherentes detectando dónde cambia el tema. Imagina un artículo sin títulos donde el contenido salta entre varios temas. El sistema identifica automáticamente dónde empieza cada tema nuevo.

Lo relevante es que el proyecto implementa cuatro algoritmos distintos de optimización (dos exactos, dos heurísticos), los evalúa contra un dataset controlado, y utiliza un LLM como evaluador de coherencia semántica.

### 1.2 ¿Para qué sirve?

Esto no es un ejercicio teórico: el problema aparece en muchas aplicaciones reales.

- Recuperación de información: cuando buscas algo en Google, el motor no devuelve documentos enteros, sino el fragmento donde aparece la respuesta. Para eso, primero tiene que partir los documentos en fragmentos.
- Resumen automático: para resumir un documento largo, es útil resumir cada sección por separado.
- Análisis de conversaciones: en una transcripción de un *call center*, encontrar dónde cambia el tema permite identificar la queja, la solución y el cierre.
- Sistemas RAG con LLMs: los modelos de lenguaje grandes (GPT, Claude, Llama) tienen un límite de cuántas palabras pueden leer a la vez. Antes de hacerles una pregunta sobre un libro entero, hay que partir el libro y enviarles solo los fragmentos relevantes.
- Procesamiento de subtítulos de video: para crear capítulos automáticamente.

### 1.3 ¿Qué resuelve concretamente este proyecto?

El proyecto hace tres cosas:

1. Implementa cuatro algoritmos distintos que resuelven el problema. Dos de ellos garantizan encontrar la mejor solución posible (la "óptima"), y los otros dos son más rápidos pero pueden equivocarse en alguna frontera.
2. Usa un LLM (Llama 3.3 de 70 mil millones de parámetros, ejecutado vía Groq) como evaluador: le pasa cada segmento que el algoritmo propuso y le pregunta "del 1 al 5, ¿qué tan coherente es este segmento?".
3. Mide objetivamente cuál algoritmo es mejor comparando sus salidas con un dataset donde conocemos las fronteras "verdaderas" (porque lo construimos sintéticamente).

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

- Bloque A — Deportes: oraciones 0, 1, 2
- Bloque B — Tecnología: oraciones 3, 4, 5, 6
- Bloque C — Cocina: oraciones 7, 8, 9, 10

Decimos entonces que las fronteras de segmento son las oraciones 0, 3 y 7, porque ahí empieza cada bloque nuevo. En notación compacta: `B = [0, 3, 7]`.

Si numeráramos las fronteras como "huecos entre oraciones" (entre la oración 2 y la 3 hay un hueco, etc.), también podríamos representarlo así. Pero en este proyecto siempre usamos la convención de "primera oración de cada segmento": la primera frontera siempre es `0` (todo documento empieza por su primera oración).

### 2.2 ¿Por qué es difícil para un programa?

Para un humano la segmentación anterior es obvia, porque entiende el significado de las palabras. Para un programa que no entiende el lenguaje (o que solo lo entiende muy superficialmente) hay tres dificultades grandes:

1. No sabe de antemano cuántos segmentos hay. En el ejemplo había 3, pero podría haber 2 o 7.
2. No sabe de qué temas trata el texto. No tiene una lista de "temas posibles" con la cual comparar.
3. Hay muchísimas formas posibles de dividir. Con 11 oraciones y queriendo 3 segmentos, el número de divisiones posibles es ya considerable; con 100 oraciones, el número se vuelve astronómico.

A este problema (encontrar la mejor entre muchísimas configuraciones posibles) se le llama problema de optimización combinatoria.

### 2.3 La idea de "mejor"

¿Qué significa que una segmentación sea "mejor" que otra? Tenemos que definir una función objetivo: una fórmula matemática que asigna un número a cada segmentación posible. Cuanto más alto el número, mejor la segmentación. Luego buscamos la segmentación que maximice ese número.

Una idea natural: una segmentación es buena si dentro de cada segmento las oraciones se parecen mucho entre sí (todas hablan del mismo tema). Esto se llama cohesión interna. La definición precisa requiere convertir las oraciones en algo que un programa pueda comparar numéricamente, lo cual nos lleva a la sección 3.

---

## 3. Conceptos previos necesarios

Esta sección explica los tres ingredientes que usamos para que un programa pueda comparar oraciones: TF-IDF, similitud coseno y complejidad algorítmica.

### 3.1 Representar texto como números: TF-IDF

Un programa no puede comparar oraciones directamente. Necesitamos convertir cada oración en un vector de números.

#### La idea más simple: bolsa de palabras

La forma más sencilla es contar cuántas veces aparece cada palabra. Si nuestro vocabulario completo es `{el, gol, gato, marcó, ronroneó}`, entonces:

- Oración "El gato ronroneó" → `[1, 0, 1, 0, 1]` (un "el", cero "gol", un "gato", cero "marcó", un "ronroneó")
- Oración "Marcó el gol" → `[1, 1, 0, 1, 0]`

Este vector se llama bolsa de palabras (*bag of words*): hemos perdido el orden, pero ganamos algo que podemos comparar numéricamente.

#### Problema: las palabras frecuentes dominan

En español, palabras como "el", "la", "de", "y" aparecen en casi todas las oraciones. Si solo contamos frecuencias, dos oraciones cualesquiera parecerán similares simplemente porque ambas usan "el" y "de". Esto es ruido.

#### Solución: TF-IDF

TF-IDF son las siglas de *Term Frequency – Inverse Document Frequency*. La idea es ponderar cada palabra por dos factores:

- TF (frecuencia de término): cuántas veces aparece la palabra en esta oración. Si una palabra aparece mucho en esta oración, probablemente es importante para esta oración.
- IDF (frecuencia inversa de documento): qué tan rara es la palabra en el conjunto total. Si una palabra aparece en todas las oraciones (como "el"), su IDF es bajo; si aparece solo en una oración (como "Vinicius"), su IDF es alto.

> Fórmula simplificada:
> $$\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \log\!\left(\frac{N}{\text{df}(t)}\right)$$
> donde $t$ es el término (palabra), $d$ el documento (oración), $N$ el número total de oraciones, y $\text{df}(t)$ el número de oraciones donde aparece $t$.

Resultado: palabras como "Vinicius" en la oración de fútbol tendrán un valor alto, y palabras como "el" tendrán un valor cercano a cero. El vector TF-IDF "destaca" las palabras informativas de cada oración.

En este proyecto usamos la implementación de `scikit-learn` con `sublinear_tf=True` (sustituye $\text{tf}$ por $1 + \log(\text{tf})$, lo cual amortigua aún más las palabras muy frecuentes).

### 3.2 Medir parecido entre vectores: similitud coseno

Una vez que cada oración es un vector de números, ¿cómo decimos si dos oraciones se parecen?

La similitud coseno mide el ángulo entre dos vectores. Si dos vectores apuntan en la misma dirección, el coseno del ángulo entre ellos es 1 (máximo parecido). Si apuntan en direcciones perpendiculares, el coseno es 0 (sin relación).

> Fórmula:
> $$\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \cdot \|\vec{v}\|}$$
> donde $\vec{u} \cdot \vec{v}$ es el producto escalar (suma de productos componente a componente) y $\|\vec{u}\|$ es la norma euclidiana (raíz de la suma de cuadrados).

Ejemplo numérico con dos oraciones reducidas:

- $\vec{u} = [1, 0, 1, 0]$ (oración A)
- $\vec{v} = [1, 1, 0, 0]$ (oración B)

Producto escalar: $1 \cdot 1 + 0 \cdot 1 + 1 \cdot 0 + 0 \cdot 0 = 1$
Norma de $\vec{u}$: $\sqrt{1 + 0 + 1 + 0} = \sqrt{2}$
Norma de $\vec{v}$: $\sqrt{1 + 1 + 0 + 0} = \sqrt{2}$
Similitud coseno: $\frac{1}{\sqrt{2} \cdot \sqrt{2}} = \frac{1}{2} = 0{,}5$

Las dos oraciones tienen una palabra en común (la primera componente) sobre dos palabras cada una: 0,5 es razonable.

La similitud coseno es la métrica estándar para comparar vectores TF-IDF porque ignora la "magnitud" (longitud del vector, que depende del largo de la oración) y se queda solo con la dirección (qué palabras importan).

### 3.3 Complejidad algorítmica: por qué importa el tiempo

Cuando hablamos de un algoritmo, no basta con que funcione: tiene que terminar en un tiempo razonable. La complejidad algorítmica describe cómo crece el tiempo de ejecución cuando el tamaño de la entrada (en nuestro caso, $n$ = número de oraciones) crece.

Notación habitual:

- $O(n)$ — lineal: si duplico $n$, el tiempo se duplica.
- $O(n^2)$ — cuadrático: si duplico $n$, el tiempo se multiplica por 4.
- $O(n^2 \cdot k)$ — cuadrático en $n$ multiplicado por $k$: si triplico $k$, el tiempo se triplica.
- $O(2^n)$ — exponencial: cada oración nueva duplica el tiempo. Inviable para $n > 30$ aproximadamente.

> Ejemplo concreto del costo exponencial: con $n = 20$ oraciones y un algoritmo $O(2^n)$, son aproximadamente un millón de operaciones (instantáneo). Con $n = 40$, son un billón de operaciones (minutos). Con $n = 60$, son un trillón (años).

Una parte central de este proyecto es entender el trade-off entre exactitud y velocidad: los algoritmos exactos garantizan encontrar la mejor segmentación pero son caros; los algoritmos heurísticos son rápidos pero pueden equivocarse. ¿Vale la pena la diferencia? La sección 11 responde con datos.

---

## 4. Modelado formal del problema

### 4.1 Definición precisa

Tenemos un documento $D$ formado por $n$ oraciones ordenadas:

$$D = (s_1, s_2, s_3, \ldots, s_n)$$

Una segmentación es una lista de $k$ enteros, llamados fronteras:

$$B = (b_1, b_2, \ldots, b_k)$$

que cumplen tres restricciones:

1. $b_1 = 0$ — la primera frontera siempre es el inicio del documento.
2. $0 = b_1 < b_2 < \ldots < b_k \leq n - 1$ — están en orden estrictamente creciente.
3. $k \leq k_{\max}$ — el número de segmentos no excede un máximo configurado.

> Matiz de implementación: la restricción $k \leq k_{\max}$ es la del modelo formal. En el código, Brute Force enumera exactamente $k = k_{\max}$ segmentos (vía `combinations(range(1, n), k - 1)` en [src/algorithms/brute_force.py:79](src/algorithms/brute_force.py#L79)). Programación Dinámica llena la tabla `dp[i][j]` para todo $j \in [1, k_{\max}]$ y su backtracking ([src/algorithms/dynamic_programming.py:89-98](src/algorithms/dynamic_programming.py#L89-L98)) realiza un verdadero $\arg\max_j \text{dp}[n][j]$ sobre todos los $j \in [1, k_{\max}]$, seleccionando el número de segmentos que maximiza la cohesión total. En la práctica, la función objetivo ponderada por longitud crece (débilmente) con $j$, por lo que el argmax satura $k_{\max}$ en la mayoría de los documentos, especialmente cuando $n \geq k_{\max}$. Esto explica por qué DP y BF coinciden bit a bit en las 20 instancias del dataset `tiny`. Una mejora futura (§13) propone añadir una penalización por número de segmentos (tipo BIC/MDL) para que la selección de $k$ sea genuinamente adaptativa.

Cada frontera $b_j$ marca el inicio de un segmento. El segmento $S_j$ se define como las oraciones desde la posición $b_j$ hasta la posición $b_{j+1} - 1$ (y el último segmento llega hasta el final del documento).

Ejemplo: con $n = 11$ y $B = (0, 3, 7)$:

- $S_1 = (s_0, s_1, s_2)$ — oraciones 0 a 2 (inicio del documento hasta antes del 3)
- $S_2 = (s_3, s_4, s_5, s_6)$ — oraciones 3 a 6
- $S_3 = (s_7, s_8, s_9, s_{10})$ — oraciones 7 a 10 (hasta el final)

### 4.2 ¿Qué buscamos?

Buscamos la segmentación $B^*$ que maximice la cohesión total del documento, definida en la sección siguiente. Formalmente:

$$B^* = \arg\max_{B} \sum_{j=1}^{k} \text{cohesion}(b_j, b_{j+1} - 1)$$

Aquí $\text{cohesion}(i, j)$ es la cohesión del segmento que va desde la oración $i$ hasta la oración $j$.

### 4.3 Tamaño del espacio de búsqueda

¿Cuántas segmentaciones distintas existen? Si fijamos $k$ (número de segmentos), hay que escoger las $k - 1$ fronteras internas (la $b_1 = 0$ está fija) entre las $n - 1$ posiciones disponibles. El número de formas de hacerlo es el coeficiente binomial:

$$\binom{n - 1}{k - 1}$$

Tabla de valores ilustrativos:

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

- $\overline{\cos}(S)$ es la similitud coseno promedio entre todas las parejas de oraciones del segmento $S$.
- $\frac{j - i + 1}{n}$ es la fracción del documento que ocupa este segmento (longitud del segmento dividido entre el tamaño total).

#### Cálculo paso a paso de la parte $\overline{\cos}(S)$

Si el segmento tiene oraciones $s_i, s_{i+1}, \ldots, s_j$:

1. Calculamos el vector TF-IDF de cada oración.
2. Para cada par distinto $(s_a, s_b)$ con $a < b$ dentro del segmento, calculamos $\cos(s_a, s_b)$.
3. Promediamos todos esos cosenos.

> Ejemplo numérico: un segmento con 3 oraciones. Hay $\binom{3}{2} = 3$ pares: $(s_1, s_2)$, $(s_1, s_3)$, $(s_2, s_3)$. Si los cosenos respectivos son 0,8, 0,7 y 0,6, entonces $\overline{\cos}(S) = (0{,}8 + 0{,}7 + 0{,}6)/3 = 0{,}7$.

Caso especial: si el segmento tiene una sola oración, no hay pares, así que se define $\overline{\cos}(S) = 0$.

### 5.2 ¿Por qué multiplicamos por la longitud?

Para entender el factor de longitud, consideremos primero qué pasaría si no estuviera y la función objetivo fuera simplemente $\sum_j \overline{\cos}(S_j)$ (la suma de cosenos promedio por segmento).

Supongamos un documento de 10 oraciones y comparemos tres segmentaciones, todas evaluadas con la versión sin factor de longitud (solo $\overline{\cos}$):

- Segmentación A — 10 segmentos de 1 oración: cada segmento tiene 0 pares, así que $\overline{\cos} = 0$ por convención. Total: $0$.
- Segmentación B — 2 segmentos de 5 oraciones, cada uno con $\overline{\cos} = 0{,}8$. Total: $0{,}8 + 0{,}8 = 1{,}6$.
- Segmentación C — 1 segmento "casi todo" de 9 oraciones con $\overline{\cos} = 0{,}75$ + 1 oración aislada con $\overline{\cos} = 0$. Total: $0{,}75$.

Hasta aquí B parece la mejor. Pero hay un escenario patológico: si en el documento existen 2 oraciones casi idénticas (por ejemplo dos copias del mismo título), un segmento de tamaño 2 que las agrupe da $\overline{\cos} \approx 1$. Si el algoritmo es libre de elegir cuántos segmentos hacer y la función objetivo es solo $\sum \overline{\cos}$, le sale "gratis" insertar fronteras alrededor de ese par para conseguir una contribución de $1{,}0$ sin esfuerzo. El algoritmo se ve incentivado a buscar pares triviales en lugar de buena cobertura del documento.

La solución del factor $\frac{j - i + 1}{n}$: ahora la cohesión definida del segmento, $\text{cohesion}(i,j) = \overline{\cos}(S) \cdot \frac{j-i+1}{n}$, descuenta cualquier segmento corto. Con la fórmula completa:

- Segmentación B (2 segmentos × 5 oraciones, $\overline{\cos}=0{,}8$): $0{,}8 \cdot \tfrac{5}{10} + 0{,}8 \cdot \tfrac{5}{10} = 0{,}8$.
- Pareja trivial (segmento de 2 oraciones idénticas, $\overline{\cos}=1{,}0$): $1{,}0 \cdot \tfrac{2}{10} = 0{,}2$, no $1{,}0$.

Es decir, un segmento largo con cohesión moderada pesa más que uno corto trivial, que es exactamente el comportamiento que queremos.

### 5.3 Cohesión total de una segmentación

La cohesión total es simplemente la suma:

$$F(B) = \sum_{j=1}^{k} \text{cohesion}(b_j, b_{j+1} - 1)$$

Esto es lo que cada algoritmo intenta maximizar.

### 5.4 Limitación de esta función objetivo

Es importante reconocer una debilidad: esta función se basa en TF-IDF, que solo "ve" coincidencias de palabras exactas. Si dos oraciones hablan del mismo tema pero usando sinónimos (por ejemplo "automóvil" y "coche"), TF-IDF las considera distintas. Para textos reales con vocabulario rico esto puede subestimar la cohesión.

La sección 13 propone reemplazar TF-IDF por embeddings densos (vectores producidos por modelos como Sentence-BERT) como mejora futura.

---

## 6. Los cuatro algoritmos

Implementamos cuatro algoritmos que comparten la misma función objetivo y los mismos vectores TF-IDF. La diferencia entre ellos es cómo exploran el espacio de posibles segmentaciones.

### 6.1 Algoritmo 1 — Fuerza Bruta

#### Idea intuitiva

Si quiero estar 100 % seguro de encontrar la mejor segmentación, basta con probar todas las segmentaciones posibles y quedarme con la de mayor cohesión total. Es lento, pero correcto por definición.

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

Casi nunca en producción. Su valor es académico y de validación: sirve para verificar que algoritmos más rápidos (como Programación Dinámica) den la misma respuesta en instancias pequeñas. En este proyecto está limitado a $n \leq 15$ por código (más allá, el tiempo se dispara).

---

### 6.2 Algoritmo 2 — Programación Dinámica (DP)

#### Idea intuitiva

La fuerza bruta repite trabajo: muchas combinaciones distintas comparten subsegmentos. Por ejemplo, las segmentaciones $(0, 3, 7)$ y $(0, 3, 8)$ ambas necesitan calcular la cohesión del segmento $(0, 1, 2)$. La Programación Dinámica calcula cada subproblema una sola vez y reutiliza el resultado.

#### El truco: subestructura óptima

Observación clave: si la mejor segmentación del documento completo en $k$ segmentos termina con un último segmento que cubre las oraciones $i$ a $n-1$, entonces lo que viene antes (la primera parte del documento, oraciones $0$ a $i-1$) también tiene que ser la mejor segmentación posible de esa primera parte en $k-1$ segmentos.

(Si no lo fuera, podríamos sustituirla por una mejor y mejoraríamos el total, lo cual contradice que la solución global ya era óptima.)

Esto se llama principio de subestructura óptima, y es exactamente la condición que hace que la programación dinámica funcione.

#### Definición de la tabla

Sea $\text{dp}[i][j]$ = la cohesión total máxima alcanzable al segmentar las primeras $i$ oraciones en exactamente $j$ segmentos.

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
5. # Selección del mejor k: argmax_j dp[n][j] para j ∈ [1, k].
   best_k ← 1; best_val ← dp[n][1]
   PARA j = 2 HASTA k:
     SI dp[n][j] > best_val:
       best_val ← dp[n][j]
       best_k ← j
6. # Reconstrucción: seguir los punteros desde split[n][best_k].
   i ← n; j ← best_k
   B ← []
   MIENTRAS j > 0:
     B.insert(0, split[i][j])
     i ← split[i][j]
     j ← j - 1
7. RETORNAR B
```

#### Complejidad

- Tiempo: $O(n^2 \cdot k)$
- Espacio: $O(n^2)$

Para $n = 100$ y $k = 10$ son 100.000 operaciones: completamente manejable.

#### Garantía

DP devuelve la misma respuesta que la fuerza bruta (es exacto), pero en tiempo polinómico. Esto se valida empíricamente en el Experimento 2 (sección 11.1): en las 20 instancias del dataset `tiny`, los resultados de BF y DP son bit a bit idénticos.

---

### 6.3 Algoritmo 3 — Greedy (TextTiling)

#### Idea intuitiva

En vez de optimizar globalmente, hacemos una observación local: una frontera natural ocurre donde el "antes" y el "después" se parecen poco entre sí. Si la oración 7 cierra un bloque de cocina y la oración 8 abre un bloque de deportes, en el hueco entre 7 y 8 esperamos un "valle" de similitud: lo que está a la izquierda no se parece a lo que está a la derecha.

Este algoritmo está inspirado en TextTiling (Hearst, 1997), uno de los métodos clásicos de segmentación lingüística.

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

La profundidad de un valle mide cuánto "baja" la curva de similitud respecto a los picos circundantes. Si la curva baja mucho, probablemente es porque ahí cambia el tema.

#### Ejemplo intuitivo

Imagina que dibujamos la curva de similitud $\text{sim}[g]$ para todos los huecos. Si el documento tiene tres temas claros, la curva tendrá una forma parecida a una sierra con dos valles profundos: uno entre el tema 1 y el tema 2, otro entre el tema 2 y el tema 3. El algoritmo identifica esos dos valles y coloca fronteras allí.

#### Complejidad

- Tiempo: $O(n \cdot w)$ — para cada uno de los $n$ huecos, calculamos un promedio de $w$ vectores.
- Espacio: $O(n)$.

#### Ventajas y limitaciones

- Ventaja: extremadamente rápido. En nuestros experimentos, ~9× más rápido que DP.
- Limitación: no optimiza la función objetivo globalmente. Puede equivocarse si los cambios de tema son sutiles o si hay valles "falsos" por ruido léxico local.

---

### 6.4 Algoritmo 4 — Recocido Simulado (Simulated Annealing, SA)

#### Encuadre como técnica de simulación

A diferencia de BF, DP y Greedy, SA no es un algoritmo determinista: es una **simulación estocástica** de un sistema físico (el recocido metalúrgico) que se reinterpreta como optimización combinatoria. En la terminología del Cap. 1 de *Temas de Simulación* (García, Martí, Pérez), SA es un sistema dinámico no-estacionario cuya función de transición de estado es no-determinista y depende de variables aleatorias generadas en cada iteración. Concretamente, en cada paso se generan dos variables uniformes discretas (la frontera a perturbar y la dirección $\pm 1$) y, cuando el movimiento empeora la cohesión, una variable uniforme continua $U \sim \mathcal{U}(0,1)$ para decidir su aceptación según el criterio de Metropolis. La generación de estas variables corresponde a los métodos descritos en el Cap. 2 del libro (algoritmo de la transformada inversa y distribución uniforme), y la corrida del algoritmo es una realización (en el sentido del Cap. 4) del proceso estocástico subyacente. Por eso el Experimento 5 (§10.5, §11.5) trata a SA como un sistema que se analiza con réplicas independientes y herramientas estadísticas, no como un algoritmo cuyo resultado se reporta una sola vez.

#### Idea intuitiva

El recocido simulado está inspirado en la metalurgia: cuando se calienta un metal y se enfría lentamente, los átomos se acomodan en una configuración de baja energía (estable). Análogamente:

- Estado: una segmentación.
- Energía: el negativo de la cohesión (queremos minimizar energía, equivalente a maximizar cohesión).
- Temperatura: parámetro que controla qué tan dispuesto está el algoritmo a aceptar empeoramientos. Empieza alta y baja gradualmente.

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

El criterio de aceptación de Metropolis ($p = e^{\Delta / T}$) es lo que distingue a SA de un simple ascenso de colina (*hill climbing*): permite, con probabilidad decreciente, moverse a peores soluciones temporalmente. Esto es lo que permite escapar de óptimos locales.

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

- Ventaja: aplicable a problemas donde no hay subestructura óptima (DP no aplica).
- Limitación: sin ajuste cuidadoso de hiperparámetros, los resultados pueden ser inferiores a DP, como vemos en los experimentos.

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

### 7.1 Integración del LLM como evaluador

El sistema implementa un LLM como evaluador de coherencia semántica. El rol del LLM es claro: no decide dónde van las fronteras (eso lo hacen los algoritmos), sino que califica la coherencia de cada segmento que los algoritmos producen. 

Esta integración se logra a través de un prompt estructurado que solicita al LLM evaluar cada segmento en una escala discreta de 1 a 5, permitiendo medir qué tan bien el algoritmo agrupa oraciones semánticamente relacionadas más allá de lo que capturaría una métrica como TF-IDF.

### 7.2 El prompt

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

1. Escala 1–5 (no 0–10 ni 1–100): reduce la varianza entre llamadas y hace los puntajes interpretables.
2. Salida en JSON estricto: parseable por el programa sin riesgo de alucinaciones de formato.
3. Razonamiento incluido (`rationale`): para auditoría — podemos ver por qué el LLM dio un puntaje y detectar evaluaciones disparatadas.

### 7.3 Proveedores LLM utilizados

Este proyecto utiliza APIs de código abierto con acceso gratuito para la evaluación. Se implementó un patrón de fallback de dos capas:

#### Proveedor principal: Groq con Llama 3.3 70B

[Groq](https://groq.com) proporciona acceso gratuito a `llama-3.3-70b-versatile` con un tier que permite miles de peticiones diarias sin costo. Este es el proveedor principal del sistema.

#### Proveedor de respaldo: Mistral

Si Groq no responde (timeout, rate limit, error de conexión), el sistema reintenta automáticamente con Mistral AI (`mistral-large-latest`), que también ofrece API gratuita. Esto es el patrón de fallback que garantiza continuidad en los experimentos.

#### Degradación elegante

Si ambos proveedores fallan, el sistema registra un puntaje neutro (3) con `used_fallback=True` en los metadatos, permitiendo que los experimentos continúen y que luego se pueda auditar cuántas evaluaciones fueron forzadas a este modo.

### 7.4 Diagrama del flujo

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

### 7.5 Rate limiting

Las APIs gratuitas imponen límites estrictos. Implementamos:

- Throttle: intervalo mínimo entre peticiones (`LLM_MIN_REQUEST_INTERVAL_SECONDS=1.5`, configurable).
- Backoff exponencial: ante un error 429 (rate limit), esperamos 3 s y reintentamos; si vuelve a fallar, 6 s; tras dos reintentos infructuosos cae al proveedor de fallback. Estos valores son los de [src/llm/rate_limit.py:27](src/llm/rate_limit.py#L27) (`base_delay=3.0`, `max_retries=2`). El loader de Wikipedia usa una política distinta (2 s → 4 s → 8 s, definida en [src/dataset/wikipedia_loader.py:205](src/dataset/wikipedia_loader.py#L205)) porque la MediaWiki API tolera bursts más cortos.

Esto añade tiempo a los experimentos con LLM pero evita ser bloqueados.

---

## 8. El dataset

### 8.1 Dos fuentes de datos: sintética y real

Para evaluar objetivamente un algoritmo de segmentación necesitamos un ground truth: documentos donde sabemos cuáles son las fronteras "verdaderas". Este proyecto usa dos fuentes complementarias:

1. Datasets sintéticos (`small`, `tiny`): documentos generados concatenando oraciones de plantillas por tópico. Permiten control absoluto del experimento, reproducibilidad bit a bit con semilla fija y validación de los algoritmos en condiciones controladas.
2. Dataset Wikipedia (`wikipedia`): artículos reales en español descargados vía la MediaWiki API. Las secciones del artículo (encabezados `== ... ==` editados por humanos) son las fronteras de referencia, lo que aporta un ground truth natural sin anotación adicional.

Usar ambas fuentes permite contrastar comportamiento en condiciones ideales (sintético, vocabulario disjunto entre tópicos) contra comportamiento en texto natural (Wikipedia, vocabulario rico con sinónimos, transiciones graduales, ruido léxico). El experimento sobre Wikipedia (§10.4 y §11.4) mide cuánto se degrada el rendimiento absoluto y si el ordenamiento entre algoritmos se mantiene.

### 8.2 ¿Por qué estos datasets son válidos como benchmark del problema?

1. Las fronteras son no ambiguas y conocidas: en el sintético, cada documento se construye concatenando bloques de oraciones de tópicos distintos (deportes, tecnología, ciencia, política, arte, economía, salud, historia). En Wikipedia, las secciones editadas por humanos cumplen el mismo rol: son cambios de tema explícitos. En ambos casos el ground truth tiene baja varianza inter-anotador esperada.
2. Las longitudes son representativas y comparables entre ambos datasets: 15–36 oraciones por documento (sintético `small`, configurado para 12–40) frente a 17–40 (Wikipedia, tras truncación). Ambos reproducen el tamaño típico de una sección de artículo o entrada de blog corta. El dataset `tiny` (5–12 oraciones, configurado para 4–12) está calibrado específicamente para que Brute Force sea viable, habilitando la validación empírica de DP.
3. `overlap_level=low` en el sintético es una elección deliberada: las plantillas comparten poco vocabulario entre tópicos, lo que favorece a TF-IDF. Esto significa que los resultados absolutos del sintético son una cota superior del rendimiento esperado en texto natural. El dataset Wikipedia mide exactamente esta diferencia: vocabulario natural compartido entre temas, transiciones graduales, frases largas con cláusulas subordinadas.
4. El número de segmentos varía en el sintético: la distribución observada en `small` (3 segmentos en 8 docs, 4 en 8, 5 en 4) garantiza que los algoritmos no pueden "ganar" eligiendo siempre el mismo $k$ fijo. En Wikipedia la situación es distinta: tras la truncación a `max_segments_per_doc=5`, la distribución real es 5 segmentos en 24 docs y 3 segmentos en 1 doc, lo que significa que con `max_segments=5` el k predicho coincide con el k real en 24/25 instancias. Esto es una limitación del Experimento 4: las diferencias entre algoritmos en Wikipedia reflejan solo la calidad de las *posiciones* de las fronteras, no la elección del número de segmentos.
5. Reproducibilidad: el sintético se regenera bit a bit con `random_seed=42`. El dataset Wikipedia es reproducible mientras los artículos referenciados existan; un `metadata.json` registra cada título descargado.

### 8.3 Cómo se construye cada dataset

Datasets sintéticos (`src/dataset/generator.py`):

1. Define una biblioteca de tópicos (deportes, tecnología, política, ciencia, arte, economía, salud, historia). Cada tópico tiene un vocabulario característico y plantillas de oraciones.
2. Para cada documento:
   - Sortea cuántos segmentos tendrá (entre `min` y `max` del config).
   - Para cada segmento: sortea un tópico distinto y cuántas oraciones tendrá; genera esas oraciones desde las plantillas del tópico.
   - Concatena los segmentos secuencialmente.
   - Registra las posiciones donde empieza cada segmento (las fronteras de referencia).

Dataset Wikipedia (`src/dataset/wikipedia_loader.py`):

1. Lee una lista de títulos de artículos desde el YAML de configuración (28 artículos cubriendo los mismos 8 dominios temáticos que el sintético).
2. Para cada título, llama a la MediaWiki API (`action=query&prop=extracts&explaintext=1&exsectionformat=wiki`) que devuelve el texto plano con marcadores de sección `== Título ==`.
3. Parsea solo los encabezados de nivel 2 (`==`) como fronteras; las sub-secciones (`===`, `====`...) se pliegan dentro de su sección padre para que la granularidad coincida con el sintético.
4. Filtra secciones triviales (`Referencias`, `Véase también`, `Bibliografía`, `Notas`, `Enlaces externos`).
5. Trunca cada artículo a 5 secciones × 8 oraciones máximo por sección para emparejar el tamaño del dataset sintético `small`.
6. Reintenta con backoff exponencial ante errores HTTP 429 (rate limit). Pausa 1.5 s entre llamadas para no saturar la API pública.

### 8.4 Datasets usados

Hay tres datasets, todos en español.

#### Dataset `small` (sintético, uso principal)

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

#### Dataset `tiny` (sintético, validación BF vs DP)

Documentos más cortos (5–12 oraciones observadas; configurado para 4–12) para que la fuerza bruta sea viable. Usado únicamente para validar empíricamente que DP coincide con BF.

#### Dataset `wikipedia` (real, validación en texto natural)

Configuración (`config/datasets/wikipedia.yaml`): 28 títulos de artículos cubriendo los 8 dominios (deportes, tecnología, ciencia, política, arte, economía, salud, historia). Tras descarga y filtrado:

| Parámetro | Valor |
|---|---|
| Artículos solicitados | 28 |
| Documentos aceptados | 25 |
| Documentos descartados (429 persistente / filtros) | 3 |
| Oraciones por documento (media) | 35,8 |
| Oraciones por documento (rango) | 17 – 40 |
| Segmentos por documento (media) | 4,92 |
| Distribución de segmentos | 5 segmentos en 24 docs · 3 segmentos en 1 doc (concentrada en 5 por la truncación a `max_segments_per_doc=5`) |
| Fuente | MediaWiki API, es.wikipedia.org |

Las longitudes son comparables al sintético `small` (17–40 vs 15–36 oraciones), lo que hace que la comparación entre ambos sea directa: cualquier diferencia de rendimiento se atribuye a la naturaleza del texto, no a su tamaño.

### 8.5 Estructura en disco

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

Necesitamos medir, con números, qué tan buena es la salida de un algoritmo comparada con el ground truth. Usamos cuatro métricas de calidad (F1 de fronteras, Pk, WindowDiff y LLM Score) más una métrica de coste (runtime), que capturan aspectos distintos.

### 9.1 F1 de fronteras (con tolerancia ±1)

Idea: ¿cuántas de las fronteras predichas coinciden con las reales (con un margen de error de 1 oración)?

Definamos:

- TP (verdaderos positivos): fronteras predichas que están a distancia ≤ 1 de alguna frontera real.
- FP (falsos positivos): fronteras predichas que no.
- FN (falsos negativos): fronteras reales que no fueron predichas.

> $$P = \frac{TP}{TP + FP}, \quad R = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

Rango: $[0, 1]$. Mayor es mejor.

Ejemplo: ground truth $B^* = (0, 8, 15)$, predicción $B = (0, 7, 12, 16)$.

La implementación excluye la frontera 0 del cómputo (aparece trivialmente en cualquier segmentación, contarla inflaría la métrica). Comparamos entonces solo las fronteras internas: referencia $\{8, 15\}$ frente a predicción $\{7, 12, 16\}$. El emparejamiento es voraz de izquierda a derecha, y cada frontera real solo puede emparejarse con una predicción.

- La frontera 7 está a distancia 1 de 8 → TP, consume el 8.
- La frontera 12 no está cerca de ninguna real disponible → FP.
- La frontera 16 está a distancia 1 de 15 → TP, consume el 15.
- Reales no detectadas: ninguna → FN = 0.

$P = 2/3 \approx 0{,}667$; $R = 2/2 = 1{,}0$; $F_1 = 2 \cdot 0{,}667 \cdot 1 / 1{,}667 \approx 0{,}800$.

### 9.2 Pk

Idea: Pk (Beeferman et al., 1999) usa una ventana deslizante. Para cada par de posiciones a distancia $k$, ¿están en el mismo segmento según la referencia? ¿Y según la predicción? Si las dos respuestas difieren, es un error.

> $$P_k = \frac{1}{n-k} \sum_{i=1}^{n-k} \mathbf{1}\!\left[\text{ref}(i, i+k) \neq \text{pred}(i, i+k)\right]$$

donde $\text{ref}(i, j) = 0$ si $i$ y $j$ están en el mismo segmento de referencia, $1$ si están en segmentos distintos (análogamente para $\text{pred}$). $k$ se fija típicamente a la mitad del tamaño promedio de segmento real.

Rango: $[0, 1]$. Menor es mejor (es una tasa de error).

### 9.3 WindowDiff

Idea: variante de Pk que penaliza más cuando el número de fronteras dentro de la ventana es muy distinto entre referencia y predicción (no solo si difieren, sino por cuánto).

> $$WD = \frac{1}{n-k} \sum_{i=1}^{n-k} \mathbf{1}\!\left[|\text{ref\_count}(i, i+k) - \text{pred\_count}(i, i+k)| > 0\right]$$

donde $\text{ref\_count}(i, j)$ es el número de fronteras de referencia dentro de la ventana $[i, j]$.

Rango: $[0, 1]$. Menor es mejor.

WindowDiff es más estricto que Pk y se prefiere como métrica principal en la literatura moderna de segmentación.

### 9.4 LLM Score

Idea: la calificación promedio que el LLM dio a los segmentos producidos por el algoritmo.

> $$\text{LLMScore}(B) = \frac{1}{k} \sum_{j=1}^{k} \text{LLM}(S_j)$$

Rango: $[1, 5]$. Mayor es mejor.

Esta es la única métrica que mide calidad semántica real, no solo coincidencia con el ground truth. Útil porque puede pasar que un algoritmo produzca segmentaciones distintas a la referencia pero igualmente coherentes (e.g., el documento sintético tenía una transición ambigua y el algoritmo encontró otra agrupación válida).

### 9.5 Runtime

Idea: el tiempo en segundos que tomó procesar un documento.

Importancia: en producción muchas veces preferimos un algoritmo 5 % menos preciso pero 10× más rápido.

### 9.6 Implementación

Todas estas métricas están implementadas en `src/evaluation/metrics.py` sin dependencias externas (no usamos `segeval` para mantener el proyecto ligero).

---

## 10. Diseño experimental

Ejecutamos cuatro experimentos diseñados para responder preguntas distintas.

### 10.1 Experimento 1 — Comparación estructural sin LLM

ID: `exp_compare_algorithms`
Pregunta: ¿Cuál de los tres algoritmos escalables (DP, Greedy, SA) da mejores fronteras?
Dataset: `small` (20 documentos)
LLM: ninguno (solo métricas estructurales)
Por qué no incluimos Brute Force aquí: los documentos tienen 15–36 oraciones, por encima del límite de 15 de BF.

### 10.2 Experimento 2 — Validación BF vs DP

ID: `exp_bf_vs_dp`
Pregunta: ¿DP encuentra realmente el óptimo global, igual que la fuerza bruta?
Dataset: `tiny` (20 documentos con 5–12 oraciones)
LLM: ninguno
`max_segments`: 3 (calibrado al rango real de segmentos en `tiny`, que tiene 2–3 segmentos por documento; ver `config/experiments/exp_bf_vs_dp.yaml`). El resto de experimentos usa `max_segments = 5`.
Importancia: este es el experimento de validación de correctitud. Si DP no coincidiera con BF, sería evidencia de un bug en DP.

### 10.3 Experimento 3 — Evaluación con LLM

ID: `exp_llm_groq`
Pregunta: ¿Cómo califica un LLM la calidad semántica de los segmentos que cada algoritmo produce?
Dataset: `small`
LLM: Groq `llama-3.3-70b-versatile`, temperatura 0
Métricas: estructurales + `llm_score`

### 10.4 Experimento 4 — Validación en texto natural (Wikipedia)

ID: `exp_wikipedia`
Pregunta: ¿Los resultados obtenidos con datos sintéticos se generalizan a texto natural? ¿Se mantiene el ordenamiento relativo entre algoritmos cuando el vocabulario es rico y las transiciones son graduales?
Dataset: `wikipedia` (25 artículos reales en español)
LLM: ninguno (foco en métricas estructurales)
Importancia: usa artículos editados por humanos con secciones como ground truth para contrastar el comportamiento medido en condiciones ideales contra el comportamiento en texto natural. Es el experimento que separa "lo que funciona en un laboratorio" de "lo que funciona en producción".

### 10.5 Experimento 5 — Análisis de sensibilidad de hiperparámetros de SA

ID: `exp_sa_sensitivity`
Pregunta: ¿Cómo influyen los hiperparámetros del Recocido Simulado (temperatura inicial $T_0$, tasa de enfriamiento $\alpha$, y número de iteraciones $N_{\text{iter}}$) en la calidad de la segmentación y en el tiempo de ejecución del algoritmo?
Dataset: `small` (20 documentos)
Réplicas estocásticas: 30 semillas distintas por cada configuración del grid, generando un total de 16,200 ejecuciones individuales (27 configuraciones × 30 semillas × 20 documentos).
Metodología: Se realiza una búsqueda por rejilla (grid search) cruzando los siguientes rangos de parámetros:
- $T_0 \in [0.5, 1.0, 2.0]$
- $\alpha \in [0.990, 0.995, 0.999]$
- $N_{\text{iter}} \in [500, 1000, 2000]$

Para cada configuración se computan el promedio de las métricas estructurales ($F_1$-Boundary, $P_k$, WindowDiff) y del tiempo de ejecución en milisegundos, acompañados de sus respectivos intervalos de confianza al 95 %. Esta metodología sigue el marco del Cap. 4 de *Temas de Simulación* (García, Martí, Pérez): se trata cada par (configuración, semilla, documento) como una observación independiente y se estima la media poblacional $\theta$ mediante la esperanza muestral $\bar{X} = \sum_i X_i / N$ (sección 4.1 del libro), con varianza muestral insesgada $S^2 = \sum_i (X_i - \bar{X})^2 / (N-1)$ (Proposición 4.1.2) y margen $t_{\alpha/2,\,N-1} \cdot S / \sqrt{N}$.

Detalle técnico: el IC se computa sobre el conjunto de $N = 30 \times 20 = 600$ observaciones por configuración (30 semillas × 20 documentos), no sobre 30 réplicas agregadas. Con $N = 600$, $t_{0{,}025,\,599} \approx 1{,}96$ (en el código se usa la cota conservadora $2{,}045$ válida para $N \geq 30$, lo que infla el margen ~4 %). Esta decisión maximiza la potencia estadística pero supone independencia entre observaciones doc×semilla; como los documentos comparten dificultad latente, el IC reportado debe leerse como una cota inferior de la incertidumbre verdadera. Una variante más conservadora —promediar primero por semilla sobre los 20 documentos y construir el IC sobre las 30 medias resultantes— daría márgenes aproximadamente $\sqrt{600/30} \approx 4{,}5$ veces más anchos; la dirección del ranking entre configuraciones se mantiene en ambos casos.

### 10.6 Reproducibilidad

| Parámetro | Valor |
|---|---|
| Semilla aleatoria (todos los experimentos) | 42 |
| Python | 3.12.3 |
| scikit-learn | ≥ 1.3 |
| numpy | ≥ 1.24 |
| Fecha de ejecución | 2026-06-10 (exp 1–3) · 2026-06-11 (exp 4 y 5, UTC) |

La semilla se registra en `run_metadata.json` junto a cada resultado.

---

## 11. Resultados y análisis

### 11.1 Experimento 2 — BF vs DP (dataset `tiny`)

| Algoritmo | Pk ↓ | WindowDiff ↓ | F1-Boundary ↑ | Runtime (ms) |
|---|---|---|---|---|
| Brute Force | 0,1633 | 0,1717 | 0,8167 | 1,79 |
| Dynamic Programming | 0,1633 | 0,1717 | 0,8167 | 1,76 |

Lectura: las dos filas son idénticas en métricas de calidad. Esto prueba experimentalmente (no demostración formal, pero evidencia muy fuerte) que DP encuentra el óptimo global en las 20 instancias. Las medias de runtime por documento difieren en ~0,03 ms (ambas redondean a 1,8 ms en el `summary.csv`, que guarda 4 decimales en segundos): la ventaja de DP no es apreciable en este tamaño tan pequeño, y solo se vuelve relevante a medida que $n$ crece, consistente con su complejidad $O(n^2 k)$ vs $O(\binom{n-1}{k-1} \cdot n)$.

### 11.2 Experimento 1 — DP, Greedy, SA en `small`

| Algoritmo | Pk ↓ | WindowDiff ↓ | F1-Boundary ↑ | Runtime (ms) |
|---|---|---|---|---|
| Dynamic Programming | 0,1728 ± 0,157 | 0,1945 ± 0,162 | 0,797 ± 0,124 | 10,8 |
| Greedy (TextTiling) | 0,1288 ± 0,076 | 0,2373 ± 0,125 | 0,743 ± 0,140 | 1,2 |
| Simulated Annealing | 0,2121 ± 0,151 | 0,2261 ± 0,155 | 0,760 ± 0,119 | 16,2 |

### 11.3 Experimento 3 — Con LLM

| Algoritmo | Pk ↓ | WindowDiff ↓ | F1-Boundary ↑ | LLM Score ↑ | Runtime (ms) |
|---|---|---|---|---|---|
| Dynamic Programming | 0,1728 | 0,1945 | 0,797 | 4,54 ± 0,36 | 25,6 |
| Greedy (TextTiling) | 0,1288 | 0,2373 | 0,743 | 4,41 ± 0,53 | 3,6 |
| Simulated Annealing | 0,2121 | 0,2261 | 0,760 | 4,42 ± 0,34 | 29,6 |

Las métricas estructurales de SA son idénticas entre Exp. 1 y Exp. 3, validando la reproducibilidad: los parámetros del YAML —incluido `random_seed`— llegan correctamente al constructor de SA, eliminando cualquier no-determinismo entre corridas.

Uso del fallback LLM: la unidad de llamada es un segmento (no un documento). Con 20 docs × 3 algoritmos × 5 segmentos predichos por doc-algoritmo, son 300 llamadas LLM distribuidas en 60 pares doc-algoritmo. A nivel de llamada, Groq respondió correctamente en el 90,3 % (271/300); el resto cayó al fallback de Mistral. A nivel doc-algoritmo, 6 pares (de 60) activaron al menos una vez el fallback: 5 con tasa 100 % (los 5 segmentos a Mistral) y 1 con tasa 80 % (4 de 5 segmentos). Matiz importante: los 6 pares con fallback corresponden todos al mismo algoritmo (`simulated_annealing`, documentos 15 a 20), porque SA se ejecutó al final del experimento y heredó el límite de tasa acumulado de Groq durante las corridas previas de DP y Greedy. Es decir, la tasa de éxito por algoritmo no fue uniforme: DP y Greedy alcanzaron 100 % de respuestas Groq, mientras SA recibió ~71 % (71/100). Que ningún segmento cayese en el "score neutro = 3" se infiere indirectamente: las medias de LLM score por algoritmo se mantienen en 4,41–4,54 (lejos del piso 3,0 que produciría un neutral). El runner ya persiste por documento el campo `llm_n_neutral` ([src/experiments/runner.py:165-169](src/experiments/runner.py#L165-L169)) como cuenta directa de segmentos que llegaron al camino neutral; los `results.json` de esta entrega son anteriores a ese cambio y solo guardan `llm_used_fallback`, por lo que la verificación bit a bit estará disponible en cualquier re-corrida posterior.

### 11.4 Experimento 4 — DP, Greedy, SA en `wikipedia` (texto real)

| Algoritmo | Pk ↓ | WindowDiff ↓ | F1-Boundary ↑ | Runtime (ms) |
|---|---|---|---|---|
| Dynamic Programming | 0,3741 | 0,3963 | 0,4533 | 24,3 |
| Greedy (TextTiling) | 0,3624 | 0,4129 | 0,4233 | 2,5 |
| Simulated Annealing | 0,3513 | 0,3675 | 0,4833 | 31,0 |

Lectura inmediata — caída del rendimiento absoluto: comparando con el dataset sintético `small`:

| Métrica | DP small | DP wikipedia | Caída relativa |
|---|---|---|---|
| Pk ↓ | 0,1728 | 0,3741 | +117 % de error |
| WindowDiff ↓ | 0,1945 | 0,3963 | +104 % de error |
| F1-Boundary ↑ | 0,797 | 0,453 | −43 % |

Los algoritmos son aproximadamente el doble de inexactos en texto natural. La razón: el vocabulario natural de Wikipedia comparte términos entre temas (un artículo de fútbol y otro de política pueden hablar de "elecciones", "presidente", "votación"), TF-IDF detecta menos fronteras claras, y el espacio de búsqueda se vuelve más ambiguo.

Cambio en el ordenamiento: a diferencia del sintético donde DP gana en WD/F1, en Wikipedia SA gana en las tres métricas estructurales (Pk, WindowDiff y F1). Cuando la función objetivo TF-IDF deja de tener un óptimo claro (porque el vocabulario es ruidoso), la exploración estocástica de SA puede salir de óptimos locales que atrapan a DP por construcción de la tabla. DP sigue siendo "óptimo respecto a su función objetivo", pero esa función objetivo deja de ser un buen proxy de la verdad cuando el texto es natural.

Greedy mantiene su perfil de "rápido y razonable": ~10× más rápido que DP y resultados muy cercanos a DP en métricas estructurales — heurística práctica robusta.

### 11.5 Experimento 5 — Análisis de sensibilidad de hiperparámetros de SA

Se evaluó la sensibilidad del algoritmo de Recocido Simulado ante cambios en su temperatura inicial ($T_0$), tasa de enfriamiento ($\alpha$) y número de iteraciones ($N_{\text{iter}}$). La tabla a continuación muestra las 10 mejores configuraciones ordenadas por la métrica $F_1$-Boundary promedio, junto con sus intervalos de confianza al 95 % (calculados sobre las $N = 600$ observaciones doc×semilla por configuración, según el método descrito en §10.5; cota inferior de la incertidumbre por la no-independencia entre documentos).

| $T_0$ | $\alpha$ | $N_{\text{iter}}$ | $F_1$-Boundary ↑ | $P_k$ ↓ | WindowDiff ↓ | Tiempo (ms) |
|---|---|---|---|---|---|---|
| 0,5 | 0,995 | 2000 | **0,7815 ± 0,0106** | 0,1988 ± 0,0124 | 0,2153 ± 0,0128 | 17,8 ± 0,5 |
| 1,0 | 0,995 | 2000 | **0,7815 ± 0,0105** | 0,1929 ± 0,0123 | 0,2117 ± 0,0128 | 17,9 ± 0,5 |
| 0,5 | 0,995 | 1000 | 0,7811 ± 0,0105 | 0,1999 ± 0,0123 | 0,2163 ± 0,0128 | 14,2 ± 0,4 |
| 2,0 | 0,995 | 2000 | 0,7807 ± 0,0107 | 0,1979 ± 0,0127 | 0,2141 ± 0,0131 | 17,8 ± 0,5 |
| 0,5 | 0,990 | 1000 | 0,7760 ± 0,0106 | 0,2056 ± 0,0122 | 0,2203 ± 0,0127 | 14,4 ± 0,5 |
| 0,5 | 0,990 | 2000 | 0,7760 ± 0,0106 | 0,2056 ± 0,0122 | 0,2203 ± 0,0127 | 18,2 ± 0,5 |
| 2,0 | 0,990 | 1000 | 0,7756 ± 0,0105 | 0,2066 ± 0,0126 | 0,2222 ± 0,0130 | 14,8 ± 0,5 |
| 2,0 | 0,990 | 2000 | 0,7756 ± 0,0105 | 0,2066 ± 0,0126 | 0,2222 ± 0,0130 | 18,1 ± 0,5 |
| 0,5 | 0,990 | 500  | 0,7716 ± 0,0106 | 0,2106 ± 0,0120 | 0,2251 ± 0,0125 | 12,6 ± 0,5 |
| 1,0 | 0,995 | 1000 | 0,7714 ± 0,0104 | 0,2078 ± 0,0120 | 0,2256 ± 0,0125 | 14,1 ± 0,5 |

#### Análisis de tendencias y efectos principales:

1. **Efecto de la tasa de enfriamiento ($\alpha$):** 
   La tasa de enfriamiento intermedia $\alpha = 0,995$ es claramente la que ofrece el mejor compromiso. Si la tasa de enfriamiento es demasiado rápida ($\alpha = 0,990$), el sistema se enfría de forma brusca ("templado rápido" o *quenching*), congelando las fronteras prematuramente y quedando atrapado en óptimos locales ($F_1$ máximo de ~0,7760). Por otro lado, si es demasiado lenta ($\alpha = 0,999$), el sistema no se enfría lo suficiente en el número de iteraciones fijado, de modo que al final del proceso sigue aceptando movimientos aleatorios perjudiciales con alta probabilidad, lo que degrada notablemente el rendimiento (la mejor config con $\alpha = 0,999$ alcanza apenas $F_1 = 0,7671$).

2. **Efecto del número de iteraciones ($N_{\text{iter}}$):**
   Como era de esperar, incrementar el número de iteraciones mejora la calidad de la segmentación. Las configuraciones con $N_{\text{iter}} = 2000$ dominan el top de la tabla. Un mayor número de iteraciones proporciona más tiempo para explorar el espacio de búsqueda y permite que la temperatura disminuya de manera más suave y gradual. Sin embargo, esto duplica el tiempo de ejecución (de ~12,6 ms a ~18 ms por documento), aunque en términos absolutos el coste computacional sigue siendo bajo.

3. **Efecto de la temperatura inicial ($T_0$):**
   El algoritmo demuestra ser relativamente robusto ante variaciones en la temperatura inicial en el rango $[0.5, 2.0]$. No obstante, temperaturas iniciales más bajas ($T_0 = 0,5$ y $T_0 = 1,0$) combinadas con una tasa de enfriamiento lenta de $0,995$ alcanzan el mejor rendimiento promedio, sugiriendo que calentar el sistema en exceso al principio puede provocar que se deshagan agrupaciones semánticas correctas iniciales sin aportar una exploración útil posterior.

### 11.6 Lectura por métrica (dataset sintético `small`)

#### Pk

Greedy es el mejor (0,1288), un 25 % por debajo de DP. Esto puede sorprender, pero tiene una explicación: Pk solo mira pares de oraciones a una distancia fija. Greedy, al detectar valles de similitud locales, tiende a hacer pocos errores en pares de oraciones cercanos al cambio de tema.

#### WindowDiff

DP es el mejor (0,1945). WindowDiff es más estricto: castiga no solo errores binarios sino también discrepancias en el número de fronteras dentro de cada ventana. DP, al optimizar globalmente, distribuye mejor las fronteras.

#### F1-Boundary

DP es el mejor (0,797). DP coloca fronteras más cerca de las posiciones exactas reales, lo cual confirma que su optimización global paga dividendos en precisión local.

#### LLM Score

DP gana con 4,54/5 (± 0,36), seguido de SA (4,42 ± 0,34) y Greedy (4,41 ± 0,53) prácticamente empatados. Los tres están por encima de 4,4 (rango alto), lo que indica que el LLM consideró las segmentaciones de todos los algoritmos como semánticamente coherentes. DP mantiene la primera posición en LLM score y en F1 — evidencia de que la calidad estructural y la calidad semántica están alineadas en cuanto a quién manda. Entre SA y Greedy el LLM no distingue (la diferencia 4,42 vs 4,41 es muy inferior a la desviación estándar de cualquiera de los dos), lo cual sugiere que en términos puramente semánticos las heurísticas son intercambiables para este dataset.

#### Runtime

Greedy es ~9× más rápido que DP en el experimento puro sin LLM (1,2 ms vs 10,8 ms; ver §11.2). En el experimento con LLM el ratio cae a ~7× (3,6 ms vs 25,6 ms). El `runtime_seconds` que reportamos se mide dentro de `algorithm.segment(...)` y no incluye la llamada al LLM (la evaluación ocurre después en el runner, ver [src/experiments/runner.py:159-169](src/experiments/runner.py#L159-L169)). El hecho de que los tres algoritmos sean ~2–3× más lentos en Exp. 3 que en Exp. 1 se explica entonces por contención de CPU: cada llamada LLM abre un `ThreadPoolExecutor` con timeout ([src/llm/fallback_provider.py:58](src/llm/fallback_provider.py#L58)), y aunque el thread principal hace `future.result()` y bloquea, el costo de levantar y destruir threads más el `time.sleep` del throttle introducen jitter en el reloj que pesa sobre tiempos del orden de milisegundos. En documentos cortos (los nuestros) las diferencias absolutas son despreciables; en documentos de cientos de oraciones la diferencia entre $O(n^2 \cdot k)$ y $O(n \cdot w)$ se vuelve significativa.

### 11.7 Resumen visual de quién gana en qué

| Métrica | Ganador en sintético `small` | Ganador en Wikipedia |
|---|---|---|
| Pk | Greedy | SA |
| WindowDiff | DP | SA |
| F1-Boundary | DP | SA |
| LLM Score | DP | (no medido) |
| Runtime | Greedy | Greedy |

El cambio de ganador entre el dataset sintético y el real es el hallazgo más importante de §11.4: la elección de algoritmo óptimo depende de la naturaleza del texto, no solo de la métrica.

---

## 12. Discusión: por qué pasó lo que pasó

### 12.1 ¿Por qué DP no gana en Pk?

DP optimiza una función objetivo TF-IDF que no es idéntica a la métrica de evaluación Pk. Hay un gap entre lo que optimizamos y lo que medimos. Greedy, casualmente, está más alineado con Pk porque ambos enfatizan transiciones locales bruscas.

Esto es una observación importante: el mejor algoritmo depende de la métrica que valoremos. Si Pk es lo que importa para tu aplicación, Greedy es preferible.

### 12.2 ¿Por qué SA queda último en sintético pero gana en Wikipedia?

En el dataset sintético, el objetivo TF-IDF tiene un óptimo nítido porque los tópicos comparten poco vocabulario. DP encuentra ese óptimo por construcción; SA no puede superarlo con el mismo objetivo. A esto se sumaba que la configuración por defecto inicial (`initial_temp=1.0`, `cooling_rate=0.995`, `n_iterations=2000`) se había seleccionado sin una optimización formal de parámetros. El análisis de sensibilidad sistemático (Experimento 5) demostró que, efectivamente, se podía exprimir más rendimiento de SA: ajustando los parámetros a $T_0 = 0.5$ (o $T_0 = 1.0$) y $\alpha = 0.995$ con $N_{\text{iter}} = 2000$ iteraciones, el rendimiento estocástico promedio de SA en el dataset sintético asciende a $F_1 = 0,7815 \pm 0,0105$ (superando el $F_1 = 0,760$ inicial y cerrando significativamente la brecha frente al óptimo global exacto de DP, que es $F_1 = 0,797$).

En el dataset Wikipedia, el vocabulario natural aplana el paisaje de cohesión: "óptimo del proxy TF-IDF" deja de coincidir con "óptimo del ground truth". DP se queda atrapado maximizando un objetivo que ya no es buen indicador, mientras que la aleatoriedad de SA permite explorar fronteras que el proxy considera ligeramente peores pero que el ground truth premia. Es un caso clásico donde la metaheurística supera al exacto no por encontrar mejor óptimo del objetivo, sino por encontrar uno menos malo respecto al objetivo verdadero no observable. En este entorno real, contar con una parametrización optimizada resulta crítico para que la exploración no degenere en caminos subóptimos ruidosos.

### 12.3 ¿Por qué los tres algoritmos predicen siempre 5 segmentos?

Configuramos `max_segments = 5`. Dos factores se combinan:

1. BF enumera particiones de exactamente $k_{\max}$ segmentos. DP realiza un verdadero $\arg\max_j \text{dp}[n][j]$ sobre $j \in [1, k_{\max}]$, pero como la función objetivo ponderada por longitud crece con $j$, el argmax satura $k_{\max}$ en la práctica. SA arranca con una segmentación uniformemente espaciada de $k_{\max}$ segmentos y solo perturba posiciones, sin añadir ni quitar fronteras.
2. La función objetivo crece (débilmente) con más segmentos: dividir un segmento largo en dos suele aumentar la cohesión total porque cada mitad concentra mejor su tópico, así que aunque la implementación permitiera elegir un $k$ menor, el óptimo seguiría tendiendo a $k_{\max}$.

Resultado: los tres algoritmos predicen 5 segmentos en todos los documentos, mientras que la referencia tiene 3,80 segmentos en promedio. Esto es sobresegmentación sistemática y se traduce en una caída de F1 cuando la referencia tiene 3 segmentos. La sección 13 propone como mejora la selección automática de $k$ (penalizando el número de segmentos en la función objetivo, o usando el `dp[n][j]` para distintos $j$ con un criterio tipo BIC/MDL).

### 12.4 ¿El LLM es confiable como evaluador?

Posibles fuentes de error:

- Sesgo del modelo: Llama 3.3 puede tener sesgos en cómo califica español vs inglés, o ciertos temas vs otros.
- Inconsistencia: temperatura 0 reduce variabilidad pero no la elimina por completo.
- Sin ground truth para la métrica: no sabemos si "4,54" objetivamente significa "muy coherente" — es una opinión del modelo.

A pesar de eso, el LLM score correlaciona bien con F1-Boundary (DP gana en ambas), lo cual es una señal de validez convergente: dos métricas independientes apuntan al mismo orden.

### 12.5 Sintético vs Wikipedia: qué nos dice el contraste

El contraste entre los dos datasets es uno de los resultados más informativos del proyecto. Tres observaciones:

1. Caída absoluta del rendimiento en texto natural: F1 baja de 0,797 a 0,453 (−43 %), Pk sube de 0,17 a 0,37. Esta degradación es atribuible a que TF-IDF no captura sinónimos ni paráfrasis; el vocabulario natural de Wikipedia comparte términos entre tópicos y aplana las señales que el proxy usa para detectar fronteras.

2. DP no es universalmente el mejor: en sintético DP gana en WD y F1; en Wikipedia, SA gana en las tres métricas estructurales. Cuando la función objetivo TF-IDF deja de tener un óptimo nítido (porque el ruido léxico aplana el paisaje de cohesión), "ser óptimo respecto al proxy" deja de garantizar "ser bueno respecto al ground truth". La aleatoriedad de SA permite escapar de soluciones que el proxy considera buenas pero no lo son.

3. Greedy mantiene su perfil de "rápido y razonable": en ambos datasets queda cerca del mejor en métricas estructurales y es ~9× más rápido que DP en sintético (10,8 ms vs 1,2 ms) y ~10× en Wikipedia (24,3 ms vs 2,5 ms). Su simplicidad lo hace robusto frente al cambio de dominio.

El diseño multi-dataset es lo que permite extraer esta conclusión: con un solo dataset, cualquiera de los dos, el cuadro queda incompleto.

---

## 13. Limitaciones y mejoras futuras

| Prioridad | Limitación | Mejora propuesta | Impacto estimado |
|---|---|---|---|
| Alta | TF-IDF degrada en texto natural (−43 % F1 entre sintético y Wikipedia) | Reemplazar por embeddings densos (Sentence-BERT) | +15–20 % F1 en textos reales |
| Alta | $k$ se elige manualmente | Selección automática vía BIC/MDL o salto marginal de cohesión | Elimina el parámetro más crítico |
| Media | Sobresegmentación sistemática | Penalizar número de segmentos en la función objetivo | Acerca al número real |
| Completada | SA con hiperparámetros por defecto | Búsqueda por rejilla (grid search) de 27 configs con 30 réplicas (Exp. 5) | Logró +2,8 % de F1 (0,760 → 0,7815) y menor error en SA |
| Media | Wikipedia sin evaluación LLM | Correr `exp_wikipedia` con `provider: groq` para medir LLM score sobre texto real | Validación semántica del cambio de ranking |
| Baja | Wikipedia limitado a 28 títulos | Ampliar a categorías completas (e.g. todo "Ciencia") vía API de categorías | Mayor tamaño muestral |
| Baja | BF limitado a $n \leq 15$ | (Inherente al algoritmo; ya cumple su rol) | — |
| Baja | LLM evalúa segmento por segmento | Evaluar pares de segmentos (cohesión inter-segmento) | Mayor riqueza diagnóstica |

---

## 14. Conclusiones

### 14.1 Hallazgos principales

1. DP es el mejor algoritmo en datos sintéticos (F1, WindowDiff y LLM Score). Es la elección por defecto cuando el vocabulario está bien separado entre temas.
2. SA gana en Wikipedia (texto real) en las tres métricas estructurales. Cuando TF-IDF deja de ser un proxy fiable, la exploración estocástica de SA encuentra mejores fronteras que la optimización exacta sobre un objetivo ruidoso.
3. Greedy es la opción práctica universal: ~10× más rápido que DP y se mantiene cerca del mejor en ambos datasets. Robusta al cambio de dominio.
4. El rendimiento absoluto cae ~43 % de sintético a Wikipedia (F1: 0,797 → 0,453). Esto cuantifica el coste de usar TF-IDF en texto natural.
5. BF y DP coinciden bit a bit en 20/20 instancias del dataset `tiny`, validando empíricamente la correctitud de DP.
6. El LLM Score y las métricas estructurales coinciden en el ordenamiento sobre el sintético: ambas dan a DP el primer lugar — señal de validez convergente entre dos formas independientes de evaluar.
7. La función objetivo basada en TF-IDF es razonable en condiciones controladas pero degrada con vocabulario natural: hay margen claro de mejora con embeddings densos.
8. El análisis de sensibilidad estocástico con 30 réplicas (Experimento 5) demostró la importancia de calibrar los hiperparámetros de SA. Se determinó que una tasa de enfriamiento equilibrada de $\alpha = 0,995$ y una temperatura inicial moderada ($T_0 \le 1,0$) son estadísticamente óptimas, logrando elevar el F1-Boundary promedio de SA de $0,760$ a $0,7815$ y disminuyendo significativamente la varianza de los resultados. Tazas muy rápidas ($\alpha = 0,990$) congelan la solución en óptimos locales, mientras que tasas extremadamente lentas ($\alpha = 0,999$) no permiten estabilizar el proceso.

### 14.2 Lecciones generales

- No siempre el algoritmo óptimo gana en cada métrica. Greedy gana en Pk sobre el sintético; SA gana en todo sobre Wikipedia. La elección depende de la métrica y del dominio del texto.
- Validar con texto real es crucial. El dataset sintético sugiere que DP es la mejor opción; el dataset Wikipedia muestra que en texto natural la estrategia cambia. Un solo dataset deja el cuadro incompleto.
- Funciones objetivo proxy importan. Optimizar cohesión TF-IDF es una aproximación gruesa de "buena segmentación humana". La distancia entre proxy y verdad explica por qué DP no alcanza F1 = 1,0 y, sobre todo, por qué cae a 0,45 en texto natural.
- Los LLMs como evaluadores funcionan, al menos cuando se diseña un prompt restrictivo (escala discreta, salida JSON). Son una alternativa viable a la anotación humana costosa.
- El patrón de fallback es esencial para sistemas que dependen de APIs externas, sea para LLMs o para descargas masivas como la del dataset Wikipedia (3 artículos saltados por rate-limit persistente sobre 28).

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

### 15.3 Configurar API keys (opcional para experimentos con LLM)

Este paso es obligatorio si quieres ejecutar experimentos con evaluación LLM (Experimento 3). Para experimentos 1 y 2 (solo métricas estructurales), puedes saltarlo.

#### Paso 1: Copiar el template

```bash
cp .env.example .env
```

Esto crea un archivo `.env` (que no debe comitearse, ya está en `.gitignore`).

#### Paso 2: Obtener las claves

##### Proveedor principal: Groq (Llama 3.3 70B)

1. Ir a [console.groq.com](https://console.groq.com/) y registrarse (gratis, sin tarjeta de crédito)
2. Navegar a "API Keys" en el menú lateral
3. Copiar la clave generada (empieza con `gsk_`)
4. Pegarla en `.env` en la línea: `GROQ_API_KEY=gsk_<tu_clave_aquí>`

##### Proveedor fallback: Mistral (mistral-large-latest) — OPCIONAL

1. Ir a [console.mistral.ai](https://console.mistral.ai/) y registrarse (gratis)
2. Ir a "API Keys"
3. Copiar la clave generada (Mistral no impone un prefijo fijo en la clave)
4. Pegarla en `.env` en la línea: `MISTRAL_API_KEY=<tu_clave_aquí>`

Si no configuras Mistral, los experimentos seguirán funcionando pero sin fallback automático.

#### Paso 3: Verificar la configuración

El archivo `.env` debe quedar así:

```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MISTRAL_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_FALLBACK_ENABLED=true
LLM_TIMEOUT_SECONDS=30
LLM_MIN_REQUEST_INTERVAL_SECONDS=1.5
```

##### Notas sobre las variables opcionales

- `LLM_FALLBACK_ENABLED=true`: Si Groq falla, reintentar con Mistral automáticamente. Cambiar a `false` para desactivar fallback.
- `LLM_TIMEOUT_SECONDS=30`: Segundos de espera antes de fallar. Los tier gratuitos pueden ser lentos, 30 segundos es conservador.
- `LLM_MIN_REQUEST_INTERVAL_SECONDS=1.5`: Pausa mínima entre llamadas para no exceder rate limits gratuitos. Cambiar a `0` solo si tienes tier de pago (mucho más rápido).

### 15.4 Verificar conectividad con el LLM

```bash
python -m src.llm.check --provider groq
python -m src.llm.check --provider mistral
```

Debe imprimir un puntaje y un mensaje "OK". Si falla, revisa la clave en `.env`.

### 15.5 Generar los datasets

#### Datasets sintéticos (offline, sin internet)

```bash
# Dataset principal (20 documentos sintéticos)
python -m src.dataset.generator \
    --config config/datasets/small.yaml \
    --output data/small/

# Dataset pequeño para validar BF vs DP
python -m src.dataset.generator \
    --config config/datasets/tiny.yaml \
    --output data/tiny/
```

#### Dataset Wikipedia (requiere internet)

```bash
# Descarga 28 artículos de Wikipedia en español, extrae secciones como
# fronteras y produce 25 documentos validos (~1–2 minutos con 1,5 s entre llamadas)
python -m src.dataset.wikipedia_loader \
    --config config/datasets/wikipedia.yaml \
    --output data/wikipedia/
```

Los tres datasets quedan con la misma estructura en disco: `documents/`, `boundaries/`, `metadata.json`. El loader de Wikipedia añade el campo `source_title` en cada `boundaries/doc_*.json` para trazabilidad.

### 15.6 Ejecutar los experimentos

```bash
# Experimento 1: comparación sobre dataset sintético sin LLM
python -m src.experiments.runner --config config/experiments/exp_compare_algorithms.yaml

# Experimento 2: validación BF vs DP
python -m src.experiments.runner --config config/experiments/exp_bf_vs_dp.yaml

# Experimento 3: dataset sintético con LLM (requiere GROQ_API_KEY)
python -m src.experiments.runner --config config/experiments/exp_llm_groq.yaml

# Experimento 4: validación en texto natural (Wikipedia)
python -m src.experiments.runner --config config/experiments/exp_wikipedia.yaml
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

- Algoritmo exacto: garantiza encontrar la mejor solución posible (e.g., BF, DP).
- Algoritmo heurístico: usa atajos para ir rápido pero no garantiza el óptimo (e.g., Greedy).
- Backoff exponencial: técnica de reintento donde el tiempo de espera se duplica en cada fallo.
- Bolsa de palabras (*bag of words*): representación de texto que cuenta frecuencias de palabras ignorando el orden.
- Cohesión: medida de qué tan parecidas son las oraciones de un segmento entre sí.
- Complejidad algorítmica: cómo crece el tiempo de ejecución con el tamaño de la entrada (notación $O(\cdot)$).
- Coseno (similitud): medida de parecido entre dos vectores; 1 = idénticos en dirección, 0 = sin relación.
- DP (Programación Dinámica): paradigma algorítmico que resuelve problemas grandes combinando subproblemas óptimos.
- Embeddings densos: vectores producidos por redes neuronales (BERT, etc.) que capturan semántica profunda.
- Fallback: sistema de respaldo que entra en juego cuando el componente principal falla.
- Frontera (de segmento): posición donde empieza un segmento nuevo. La primera frontera siempre es 0.
- F1: media armónica de precisión y recall, una métrica clásica en clasificación.
- Función objetivo: fórmula que el algoritmo intenta maximizar (o minimizar).
- Ground truth: la respuesta verdadera contra la que medimos las predicciones.
- Hiperparámetro: parámetro del algoritmo elegido antes de ejecutarlo (e.g., temperatura inicial de SA).
- Inferencia LLM: una llamada al modelo de lenguaje para obtener una respuesta.
- LLM (*Large Language Model*): modelo de lenguaje grande (GPT, Claude, Llama, etc.).
- Metaheurística: estrategia general para guiar la búsqueda en problemas duros (SA, algoritmos genéticos...).
- Metropolis (criterio): en SA, regla que acepta movimientos malos con probabilidad $e^{\Delta/T}$.
- OFAC: oficina de control de activos extranjeros de EE.UU.; restringe servicios desde Cuba.
- Óptimo global: la mejor solución absoluta del espacio de búsqueda.
- Óptimo local: una solución que parece buena en su vecindad inmediata pero no es la mejor absoluta.
- Pk: métrica clásica de error en segmentación, basada en ventanas deslizantes.
- Prompt: el texto que enviamos al LLM para pedirle una respuesta.
- RAG (*Retrieval-Augmented Generation*): técnica donde se buscan fragmentos relevantes antes de enviarlos al LLM.
- Rate limit: límite de cuántas peticiones puedes hacer a una API por minuto/día.
- Recall: proporción de respuestas correctas que el sistema logró recuperar.
- Segmentación: división de un texto en bloques contiguos.
- Semilla aleatoria: número inicial que hace reproducible cualquier proceso pseudoaleatorio.
- Subestructura óptima: propiedad por la que la solución óptima de un problema se construye con soluciones óptimas de subproblemas. Habilita DP.
- TF-IDF: representación vectorial de texto que pondera frecuencia de palabras por su rareza global.
- TextTiling: algoritmo clásico de segmentación basado en valles de similitud entre bloques adyacentes.
- Throttle: limitar la velocidad de envío de peticiones para no exceder rate limits.
- Trade-off: compromiso entre dos cosas que no se pueden maximizar simultáneamente (e.g., velocidad vs precisión).
- WindowDiff: variante más estricta de Pk que considera la cantidad de fronteras en cada ventana.

---

*Fin del informe.*
