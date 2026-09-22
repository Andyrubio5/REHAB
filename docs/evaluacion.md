# Evaluación y rúbrica

## Selección y configuración

Es clasificación supervisada multiclase de características tabulares. Se comparan árboles y ensembles,
modelos lineales, vecinos, SVM y MLP, además de Dummy como referencia. La imputación y, cuando corresponde,
el escalamiento están dentro del Pipeline y se ajustan solo con los datos de entrenamiento de cada fold.
Se usa codificación explícita de etiquetas para evitar la incompatibilidad observada con class_weight
y etiquetas numéricas de texto. F1 macro es el criterio de comparación y búsqueda; accuracy, reporte por
clase y matriz de confusión permiten interpretar qué movimientos se confunden.

## Entrenamiento, validación y evaluación externa

En modo completo, GroupKFold reserva un fold externo (aproximadamente 20% de las repeticiones) para
la evaluación externa. Dentro del 80% restante, cuatro folds internos separan ajuste y validación de
hiperparámetros. El ganador interno se reajusta con todo el entrenamiento externo antes de predecir
el fold externo. Ninguna ventana de una misma repetición cruza entrenamiento y evaluación de un fold.
Se repite para obtener una predicción externa por fila (OOF). Las particiones se guardan en CSV.

La comparación inicial usa cinco folds sin búsqueda interna. La búsqueda anidada se aplica a las tres
familias de ensembles predefinidas. El conjunto completo ha sido explorado previamente: ni estas métricas
ni un nuevo reparto del mismo dataset deben presentarse como una prueba nunca observada.

**Límite de la selección de familia:** los folds externos no participan en la optimización interna de
hiperparámetros, pero sí se consultan al elegir la familia final. La cifra de esa familia elegida no es una
estimación independiente de todo el proceso de selección. Para una evaluación final independiente,
se necesitan datos reservados antes de las decisiones, idealmente de participantes nuevos con ID real.
Si la rúbrica exige un conjunto de prueba fijo adicional, ese punto no se puede certificar retroactivamente
con resultados ya observados. La validación anidada documenta claramente los tres papeles disponibles.

## Modelo final

Se selecciona la familia por F1 macro medio externo, se vuelve a buscar su configuración sobre los datos
de desarrollo por grupos y se reajusta el mejor pipeline. Se guardan parámetros, características, clases,
codificador, versiones y hash de datos. HistGradientBoosting usa early_stopping=False: no se introduce una
partición interna aleatoria por ventanas distinta de la metodología definida. Este refit sirve para
inferencia y no aporta un nuevo score de prueba.

## Interpretación

- Comparar F1 macro medio y desviación entre folds, además de métricas OOF. La desviación no es un IC.
- Examinar precision, recall y F1 por movimiento y las confusiones concretas.
- No inferir generalización a sujetos nuevos: los grupos son repeticiones, no personas.
- No comparar numéricamente métricas de modo rápido con resultados del modo completo.
- No atribuir diferencias históricas solo al modelo sin controlar versiones, folds y datos.
- Las métricas de clasificación no constituyen validación clínica ni evaluación de calidad de rehabilitación.
