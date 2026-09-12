# Clasificación de movimientos REHAB con KNN manual

## Objetivo

Implementar una técnica clásica de aprendizaje máquina sin importar un
algoritmo ya construido. El proyecto utiliza K-Nearest Neighbors (KNN) para
predecir la categoría `movimiento` a partir de características de señales.

El archivo `df_final.csv` fue proporcionado para esta actividad. No se afirma
que haya sido generado por el autor de esta implementación.

## Por qué es clasificación

`movimiento` contiene códigos de categoría como `000`, `001` y `015`. Aunque
estén escritos con dígitos, no son cantidades continuas y, por tanto, el
problema no es de regresión.

## Estructura de los datos

El archivo recibido contiene 34,056 ventanas, 75 columnas, 4,257 repeticiones
y 15 clases. Las columnas de identificación son:

- `movimiento`: clase objetivo.
- `repeticion_id`: repetición de origen.
- `ventana`: una de las ocho ventanas de la repetición.

Las otras 72 columnas describen 12 señales mediante seis características:

- `std`: desviación estándar.
- `median`: mediana.
- `min`: mínimo.
- `max`: máximo.
- `iqr`: rango intercuartílico.
- `mad_diff`: cambio absoluto promedio entre observaciones consecutivas.

Las características `mean`, `range`, `rms` y `energy` no se utilizan.

## Dataset utilizado por el modelo

Las ocho ventanas de una misma repetición están relacionadas. Para evitar que
una repetición aparezca simultáneamente en entrenamiento y prueba, el programa
las agrupa y calcula la mediana de cada característica. El archivo generado
`data/dataset_modelo.csv` tiene una fila por repetición.

La división es estratificada: aproximadamente 80 % de las repeticiones de cada
movimiento se utiliza para entrenamiento y 20 % para prueba. La semilla `2026`
hace reproducible la selección.

## Implementación manual

`src/knn_manual.py` implementa:

1. Lectura y validación del CSV.
2. Agrupación por repetición.
3. División estratificada.
4. Normalización z-score calculada solo con entrenamiento.
5. Distancia euclidiana.
6. Selección de los `k` vecinos más cercanos.
7. Votación mayoritaria y desempate determinista.
8. Exactitud y matriz de confusión.
9. Exportación de predicciones.

No se utiliza `scikit-learn`. NumPy solamente se emplea para almacenar arreglos
y realizar operaciones aritméticas.

## Modelos compatibles y selección

El problema es de clasificación multiclase y las 72 entradas son numéricas.
Por ello son compatibles, entre otros, KNN, regresión logística multiclase,
árboles de decisión y Naive Bayes. Se eligió KNN porque:

- admite naturalmente múltiples clases;
- puede capturar fronteras no lineales mediante proximidad;
- es compatible con características numéricas normalizadas;
- su distancia y votación pueden implementarse manualmente de forma clara;
- no requiere importar un algoritmo ni un framework de aprendizaje máquina.

No se eligió regresión lineal porque la variable objetivo no es continua. Para
esta etapa se compararon distintas configuraciones del mismo KNN, alternativa
permitida por las instrucciones.

## Parte 1: resultado inicial

Con `k=5`, 80 % de entrenamiento, 20 % de prueba y semilla `2026`:

- Entrenamiento: 3,406 repeticiones.
- Prueba: 851 repeticiones.
- Exactitud: 0.8731 (87.31 %).

Este resultado inicial se obtuvo con una división 80/20 y demostró que la
implementación funcionaba. No se utilizó para seleccionar la configuración de
la segunda etapa.

## Parte 2: comparación de configuraciones

Para seleccionar `k` sin utilizar indebidamente la prueba, se creó una división
estratificada por repetición:

- 60 % para entrenamiento: 2,555 repeticiones.
- 20 % para validación: 851 repeticiones.
- 20 % para prueba final: 851 repeticiones.

Se compararon `k = 1, 3, 5, 7, 9, 11 y 15`. El criterio definido antes de ver
la prueba fue: mayor F1 macro de validación; después mayor exactitud; y, si
persistía un empate, menor `k`.

| k | Exactitud validación | Precisión macro | Recall macro | F1 macro |
|---:|---:|---:|---:|---:|
| 1 | 0.9271 | 0.9298 | 0.9231 | **0.9234** |
| 3 | 0.8884 | 0.8876 | 0.8831 | 0.8831 |
| 5 | 0.8625 | 0.8678 | 0.8572 | 0.8602 |
| 7 | 0.8484 | 0.8555 | 0.8424 | 0.8457 |
| 9 | 0.8555 | 0.8600 | 0.8491 | 0.8519 |
| 11 | 0.8437 | 0.8484 | 0.8370 | 0.8399 |
| 15 | 0.8261 | 0.8310 | 0.8193 | 0.8222 |

### Decisión final

Se seleccionó `k=1` porque obtuvo el F1 macro y la exactitud más altos en
validación. El descenso al aumentar `k` indica que vecinos de otras clases
empiezan a influir en la votación y suavizan demasiado las fronteras entre los
movimientos.

Después de seleccionar `k=1`, se unieron entrenamiento y validación (3,406
repeticiones), se recalculó la normalización únicamente con ese conjunto de
desarrollo y se evaluó una sola vez sobre las 851 repeticiones reservadas para
prueba:

- Exactitud final: **0.9542 (95.42 %)**.
- Precisión macro: **0.9561**.
- Recall macro: **0.9538**.
- F1 macro: **0.9526**.

La clase `014` no aparece en el dataset proporcionado y, por ello, no puede ser
aprendida ni evaluada. Los códigos deberán reemplazarse por nombres de
movimientos si posteriormente se proporciona el diccionario de etiquetas.

## Ejecución

Desde la carpeta `entrega_andy`:

```bash
python3 -m pip install -r requirements.txt
python3 src/knn_manual.py
python3 src/comparar_configuraciones.py
```

Opciones disponibles:

```bash
python3 src/knn_manual.py \
  --data data/df_final.csv \
  --k 5 \
  --test-size 0.20 \
  --seed 2026
```

El programa produce:

- `data/dataset_modelo.csv`: una fila por repetición.
- `resultados/predicciones.csv`: clase real, predicha y acierto por repetición.

La comparación produce:

- `resultados/comparacion_configuraciones.csv`.
- `resultados/predicciones_finales.csv`.
- `resultados/matriz_confusion_final.csv`.
- `resultados/resumen_seleccion.json`.

## Notebook adicional

`KNN_REHAB_explicado.ipynb` desarrolla paso a paso la justificación,
preparación, implementación, predicciones, exactitud y matriz de confusión. El
notebook complementa el proyecto, pero el programa `.py` es la implementación
independiente requerida por la actividad.

## Limitaciones

- Falta la clase `014`.
- No se recibió un diccionario que traduzca códigos a nombres de movimientos.
- La exactitud corresponde a la partición reproducible incluida y no garantiza
  el mismo desempeño en nuevos pacientes o condiciones de captura.
