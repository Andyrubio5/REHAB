# Resultados históricos: busqueda_hiperparametros

Fuente: commit 5178e321434eb2170907f1197f0815a36a824d5b. No se regeneraron con el entorno fijado.

La búsqueda original se interrumpió al iniciar Random Forest. Las particiones separan repeticiones, no personas.


Celda 2:
```text
Ventanas: 34,056
Características: 72
Movimientos: 15
Grupos: 4,257

```


Celda 4:
```text

======================================================================
HistGradientBoosting
======================================================================
Fold 1: accuracy=0.9668 | F1 macro=0.9647
Mejores parámetros: {'modelo__min_samples_leaf': 20, 'modelo__max_leaf_nodes': 31, 'modelo__max_iter': 400, 'modelo__max_depth': None, 'modelo__learning_rate': 0.1, 'modelo__l2_regularization': 0.5}
Fold 2: accuracy=0.9638 | F1 macro=0.9624
Mejores parámetros: {'modelo__min_samples_leaf': 20, 'modelo__max_leaf_nodes': 31, 'modelo__max_iter': 400, 'modelo__max_depth': None, 'modelo__learning_rate': 0.1, 'modelo__l2_regularization': 0.5}
Fold 3: accuracy=0.9734 | F1 macro=0.9731
Mejores parámetros: {'modelo__min_samples_leaf': 20, 'modelo__max_leaf_nodes': 31, 'modelo__max_iter': 400, 'modelo__max_depth': None, 'modelo__learning_rate': 0.1, 'modelo__l2_regularization': 0.5}
Fold 4: accuracy=0.9727 | F1 macro=0.9717
Mejores parámetros: {'modelo__min_samples_leaf': 20, 'modelo__max_leaf_nodes': 31, 'modelo__max_iter': 400, 'modelo__max_depth': None, 'modelo__learning_rate': 0.1, 'modelo__l2_regularization': 0.5}
Fold 5: accuracy=0.9693 | F1 macro=0.9681
Mejores parámetros: {'modelo__min_samples_leaf': 40, 'modelo__max_leaf_nodes': 15, 'modelo__max_iter': 200, 'modelo__max_depth': 5, 'modelo__learning_rate': 0.15, 'modelo__l2_regularization': 0.1}

Classification report OOF — HistGradientBoosting
              precision    recall  f1-score   support

           0     0.9935    0.9811    0.9873      1856
           1     0.9592    0.9292    0.9440      1696
          10     0.9777    0.9883    0.9830      2488
          11     0.8370    0.9585    0.8936      1880
          12     0.9240    0.9497    0.9367      2344
          13     0.9935    0.9760    0.9847      2504
          15     0.9899    0.9816    0.9858      2504
           2     0.9718    0.9691    0.9705      2136
           3     0.9833    0.9720    0.9776      2000
           4     0.9855    0.9469    0.9658      2296
           5     0.9912    0.9578    0.9742      2344
           6     0.9766    0.9649    0.9707      2080
           7     0.9906    0.9877    0.9891      3080
           8     0.9815    0.9753    0.9784      2392
           9     0.9706    0.9796    0.9751      2456

    accuracy                         0.9692     34056
   macro avg     0.9684    0.9679    0.9678     34056
weighted avg     0.9705    0.9692    0.9695     34056


======================================================================
Extra Trees
======================================================================
Fold 1: accuracy=0.9598 | F1 macro=0.9576
Mejores parámetros: {'modelo__n_estimators': 500, 'modelo__min_samples_split': 2, 'modelo__min_samples_leaf': 1, 'modelo__max_features': 0.5, 'modelo__max_depth': None, 'modelo__bootstrap': False}
Fold 2: accuracy=0.9558 | F1 macro=0.9542
Mejores parámetros: {'modelo__n_estimators': 500, 'modelo__min_samples_split': 2, 'modelo__min_samples_leaf': 1, 'modelo__max_features': 0.5, 'modelo__max_depth': None, 'modelo__bootstrap': False}

```


Celda 4:
```text
d:\ANACONDA\Lib\site-packages\joblib\externals\loky\process_executor.py:782: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
  warnings.warn(

```


Celda 4:
```text
Fold 3: accuracy=0.9703 | F1 macro=0.9696
Mejores parámetros: {'modelo__n_estimators': 500, 'modelo__min_samples_split': 2, 'modelo__min_samples_leaf': 1, 'modelo__max_features': 0.5, 'modelo__max_depth': None, 'modelo__bootstrap': False}
Fold 4: accuracy=0.9705 | F1 macro=0.9693
Mejores parámetros: {'modelo__n_estimators': 500, 'modelo__min_samples_split': 2, 'modelo__min_samples_leaf': 1, 'modelo__max_features': 0.5, 'modelo__max_depth': None, 'modelo__bootstrap': False}
Fold 5: accuracy=0.9650 | F1 macro=0.9642
Mejores parámetros: {'modelo__n_estimators': 500, 'modelo__min_samples_split': 2, 'modelo__min_samples_leaf': 1, 'modelo__max_features': 0.5, 'modelo__max_depth': None, 'modelo__bootstrap': False}

Classification report OOF — Extra Trees
              precision    recall  f1-score   support

           0     0.9918    0.9779    0.9848      1856
           1     0.9533    0.9145    0.9335      1696
          10     0.9457    0.9795    0.9623      2488
          11     0.8097    0.9979    0.8940      1880
          12     0.9764    0.9360    0.9558      2344
          13     0.9850    0.9732    0.9791      2504
          15     0.9871    0.9792    0.9832      2504
           2     0.9756    0.9565    0.9660      2136
           3     0.9716    0.9585    0.9650      2000
           4     0.9726    0.9434    0.9578      2296
           5     0.9920    0.9526    0.9719      2344
           6     0.9771    0.9654    0.9712      2080
           7     0.9861    0.9896    0.9878      3080
           8     0.9773    0.9557    0.9664      2392
           9     0.9623    0.9678    0.9651      2456

    accuracy                         0.9643     34056
   macro avg     0.9643    0.9632    0.9629     34056
weighted avg     0.9665    0.9643    0.9647     34056


======================================================================
Random Forest
======================================================================

```


Error: KeyboardInterrupt:


Celda 5:
```text
Modelo guardado correctamente en: modelos\histgradientboosting_rehab.joblib

```
