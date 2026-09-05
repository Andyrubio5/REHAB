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

## Resultado reproducible

Con `k=5`, 80 % de entrenamiento, 20 % de prueba y semilla `2026`:

- Entrenamiento: 3,406 repeticiones.
- Prueba: 851 repeticiones.
- Exactitud: 0.8731 (87.31 %).

La clase `014` no aparece en el dataset proporcionado y, por ello, no puede ser
aprendida ni evaluada. Los códigos deberán reemplazarse por nombres de
movimientos si posteriormente se proporciona el diccionario de etiquetas.

## Ejecución

Desde la carpeta `entrega_andy`:

```bash
python3 -m pip install -r requirements.txt
python3 src/knn_manual.py
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

