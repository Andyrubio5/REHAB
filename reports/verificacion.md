# Verificación de reproducibilidad

Pruebas realizadas en una copia local basada en `5178e321434eb2170907f1197f0815a36a824d5b`,
con los cambios de reproducibilidad presentes antes del commit de publicación. Los metadatos indican
ese commit padre y `working_tree_dirty: true`; no se atribuye la ejecución a un árbol limpio anterior.
Python 3.12.3 en Linux, entorno virtual dedicado, sin el PYTHONPATH externo de ROS de la terminal.
Versiones completas en `requirements-lock.txt`. `pip check` no detectó dependencias rotas.

## Comprobaciones completadas

- Los seis notebooks pasan validación nbformat y ejecución completa desde kernels nuevos.
  Los dos notebooks de entrenamiento se ejecutaron en modo rápido, su valor por defecto.
- Se reconstruyó el CSV desde los 30 arrays válidos. Coinciden las 34,056 filas, 75 columnas,
  identificadores y valores con tolerancia 1e-10. Diferencia absoluta máxima de la implementación
  vectorizada: 4.547473508864641e-13. El CSV publicado no se sobrescribió.
- Pasaron seis pruebas: repetición determinista del subconjunto, separación de grupos externa e interna,
  rechazo de ventanas duplicadas, rechazo de sensores faltantes, entrenamiento con etiquetas codificadas
  y persistencia del estado fallido cuando un estimador falla.
- Se ejecutó la búsqueda anidada rápida para las tres familias dos veces. Folds externos e internos,
  parámetros ganadores, predicciones, confusiones y métricas fueron idénticos. Los tiempos no se comparan.
- El refit rápido exportó un modelo; se cargó con joblib, predijo y decodificó sus etiquetas originales.
- Los notebooks de EDA ya no requieren data_profiling ni dependen de la ruta inexistente
  d02_processed_data en la raíz. El EDA también tenía una variable columnas_pairplot no definida;
  ahora se construye explícitamente antes de usarla.

La búsqueda rápida verificada tardó 5.5 segundos en esta máquina, con un trabajo y dos hilos.
Es un tiempo orientativo (puede variar con CPU, carga y versiones), no una promesa de rendimiento.
La búsqueda rápida usa 960 ventanas; sus métricas, conservadas en `referencia/busqueda_rapida`,
son **pruebas de funcionamiento**, no cifras finales del proyecto.

## Alcance y límites

No se repitió la búsqueda completa de 1,800 ajustes ni se entrenó su modelo final optimizado.
Esas rutas están implementadas, documentadas y disponibles mediante comando o workflow manual.
Tampoco se reconstruyó el procesamiento de señales crudas ni se evaluaron participantes nuevos.

Las cifras originales se conservan en `historico/`. No se certifica que se reproduzcan exactamente:
las versiones originales no estaban fijadas y los folds históricos no estaban persistidos.

## Comparación completa ejecutada

Se completaron los nueve modelos y cinco folds por modelo (45 ajustes) sobre las 34,056 ventanas.
Duración total aproximada: 8.2 minutos en esta máquina; hubo otras verificaciones simultáneas.
Es comparación sin búsqueda de hiperparámetros, no la búsqueda anidada completa.

| Modelo | Accuracy OOF | F1 macro OOF | F1 macro medio por fold |
|---|---:|---:|---:|
| HistGradientBoosting | 0.9654 | 0.9639 | 0.9640 |
| Extra Trees | 0.9597 | 0.9584 | 0.9583 |
| Random Forest | 0.9546 | 0.9531 | 0.9531 |
| SVM RBF | 0.9437 | 0.9420 | 0.9420 |
| KNN | 0.9313 | 0.9295 | 0.9295 |
| MLP | 0.9236 | 0.9214 | 0.9217 |
| Regresión logística | 0.7185 | 0.7153 | 0.7152 |
| LDA regularizado | 0.6171 | 0.6214 | 0.6213 |
| Dummy | 0.0904 | 0.0111 | 0.0111 |

Las métricas, folds externos, confusiones y reportes por clase están en `referencia/comparacion_completa/`.
Las predicciones por fila se generan en cada ejecución local (`reports/runs/`); no se duplican en esta referencia.
Estas cifras corresponden a evaluación por repeticiones, no por personas ni a una prueba final independiente.
