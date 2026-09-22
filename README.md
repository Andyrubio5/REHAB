# REHAB: clasificación de movimientos de rehabilitación

Clasificación multiclase de 15 movimientos a partir de señales IMU y guante. Cada repetición
se divide en ocho ventanas y cada ventana tiene 72 características estadísticas.
El CSV publicado contiene **34,056 ventanas y 4,257 repeticiones**. El movimiento 014 se excluye
porque `014_1.npy` tiene la cabecera dañada. [Descripción y límites de los datos](docs/datos.md).

El proyecto permite comparar nueve modelos y ajustar HistGradientBoosting, Extra Trees y Random Forest
con validación cruzada anidada. La métrica principal es **F1 macro**: da igual peso a cada movimiento.
También se guardan accuracy, reportes por clase, matrices de confusión y predicciones fuera de muestra (OOF).

**Los grupos son repeticiones, no participantes.** Estos experimentos no demuestran generalización a
personas nuevas. La elección de una familia mediante sus resultados externos tampoco convierte esos
resultados en una prueba independiente de esa elección. [Metodología y rúbrica](docs/evaluacion.md).

## 1. Obtener una copia ligera

Requiere Git y **Python 3.12.3**. Desde una terminal:

```bash
git clone --filter=blob:none --no-checkout https://github.com/Andyrubio5/REHAB.git
cd REHAB
git sparse-checkout set --no-cone '/*' '!/data/REHAB/'
git checkout main
python -m venv .venv
```

Activar el entorno en Linux/macOS:

```bash
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instalar y verificar:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m rehab validate
python -m unittest discover -s tests -v
```

`requirements.txt` fija las dependencias directas; `requirements-lock.txt` restringe las transitivas
al entorno comprobado. Si tu terminal hereda `PYTHONPATH` de otro software (por ejemplo ROS),
abre una terminal limpia o elimina esa variable antes de instalar y ejecutar. `.python-version` y GitHub Actions usan 3.12.3. La referencia se validó en Linux;
no se promete igualdad bit a bit entre sistemas, CPU o implementaciones numéricas.
Una descarga completa normal también funciona, pero incluye decenas de miles de archivos no necesarios
para entrenar desde el CSV. Ejecutar todos los comandos siguientes **desde la raíz de REHAB**.

## 2. Comprobar el flujo rápido

```bash
python -m rehab selection --mode rapido --output reports/runs/mi-comparacion-rapida
python -m rehab search --mode rapido --output reports/runs/mi-busqueda-rapida
python -m rehab fit-final --run reports/runs/mi-busqueda-rapida
```

Este modo usa ocho repeticiones completas por clase (960 ventanas), tres folds externos,
dos internos, dos candidatos y modelos reducidos. **Sus métricas solo sirven para comprobar el código**,
no para reportar desempeño final. Las salidas no se sobrescriben: usar nombres nuevos para repetir
un experimento. Si omites `--output`, se crea una carpeta con fecha y hora.

## 3. Experimentos completos

```bash
python -m rehab selection --mode completo --output reports/runs/mi-comparacion-completa
python -m rehab search --mode completo --output reports/runs/mi-busqueda-completa
python -m rehab fit-final --run reports/runs/mi-busqueda-completa
```

- `selection`: compara Dummy, Extra Trees, Random Forest, HistGradientBoosting, SVM RBF,
  KNN, regresión logística, LDA y MLP con los mismos cinco folds por grupos.
- `search`: tres familias, cinco folds externos, cuatro internos y 30 candidatos por búsqueda:
  **1,800 ajustes internos + 15 refits**. Puede tardar horas; no se ejecuta en cada push.
- `fit-final`: elige la familia con mayor F1 macro medio externo y realiza otra búsqueda por grupos
  sobre los datos de desarrollo del modo elegido. Exporta el mejor pipeline reajustado y su codificador.
  Esto no añade una evaluación independiente ni utiliza parámetros finales escritos a mano.

La semilla, modelos, folds, candidatos y paralelismo están en `configs/rapido.json` y
`configs/completo.json`. Por defecto se usan un trabajo y dos hilos para evitar saturar la memoria.
Los árboles internos de la búsqueda usan un solo trabajo. Para limitar también bibliotecas numéricas,
Linux/macOS permite anteponer `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2` al comando.

## 4. Reconstruir el CSV desde los arrays procesados

```bash
git sparse-checkout add --no-cone '/data/REHAB/Rehab_exercise/d02_processed_data/'
python -m rehab validate --arrays
python -m rehab rebuild --output reports/runs/reconstruccion.csv
```

Se verifican los SHA-256 de los 30 archivos válidos y se exige que existan todos los pares de sensores.
La reconstrucción debe conservar filas e identificadores y coincidir numéricamente con el CSV publicado
con tolerancia `rtol=atol=1e-10`. Se escribe una salida nueva y un archivo de procedencia; no se modifica
`data/df_final.csv`. Esto verifica desde arrays **ya procesados**, no reproduce su procesamiento desde
señales crudas. El origen, versión y derechos del dataset necesitan completar su referencia bibliográfica:
no se inventa una fuente ni se extiende automáticamente la licencia MIT del código a los datos.

## 5. Notebooks

Después de instalar el entorno, ejecutar `jupyter notebook` desde la raíz. Elegir el kernel del entorno.
Los notebooks de entrenamiento usan el mismo módulo `rehab` que los comandos anteriores.

| Notebook | Uso |
|---|---|
| `exploracion_inicial.ipynb` | Inspección complementaria de arrays; requiere descargarlos |
| `eda_rehab_entrenamiento.ipynb` | Calidad, distribuciones y duplicados de arrays |
| `eda.ipynb` | EDA complementario; exporta a `reports/runs/eda`, sin `data_profiling` |
| `redefinicion_de_los_datos.ipynb` | Reconstruye y verifica el CSV |
| `seleccion_configuracion_entrenamiento_modelos.ipynb` | Comparación; `MODO="rapido"` por defecto |
| `busqueda_hiperparametros.ipynb` | Búsqueda anidada y exportación; cambiar `MODO` para ejecución completa |

Los EDA no son pasos obligatorios para entrenar desde el CSV. Las interacciones exploratorias del EDA
no constituyen una evaluación final: algunas variables se seleccionan mirando ese mismo conjunto.
Las salidas antiguas de entrenamiento están en [`reports/historico`](reports/historico), identificadas
por commit. La búsqueda antigua se interrumpió al iniciar Random Forest; no es una comparación completa
reproducida con las nuevas versiones.

## 6. Resultados y modelo

Cada ejecución genera:

- `metadata.json`: estado (`running`, `complete`, `failed` o `interrupted`), configuración, versiones,
  commit, indicador de cambios locales y hash del CSV.
- `outer_folds.csv` y, en búsqueda, `inner_folds_*.csv`: asignación de filas y repeticiones a folds.
- `fold_metrics.csv`, `summary.csv` y subcarpetas por modelo con predicciones, reportes y confusiones.
- En búsqueda, parámetros y `cv_results` por fold. Se guardan avances al terminar cada fold; una
  interrupción no marca el experimento como completo. No se implementa reanudación automática.

`models/<nombre-ejecucion>/model.joblib` contiene `pipeline`, `label_encoder` y `features`.
Ejemplo de inferencia, usando únicamente artefactos propios o de una fuente de confianza:

```python
import joblib
from rehab.data import load_data
artifact = joblib.load("models/mi-busqueda-completa/model.joblib")
X = load_data().loc[:4, artifact["features"]]
labels = artifact["label_encoder"].inverse_transform(artifact["pipeline"].predict(X))
print(labels)
```

Las etiquetas se conservan como `000`, `001`, etc.; se codifican a enteros solo para entrenar.
El modelo final no se evalúa sobre sus datos de entrenamiento como si fueran prueba.

## 7. Qué se ha verificado

Consultar [`reports/verificacion.md`](reports/verificacion.md) para las pruebas y límites reales.
Los tiempos allí son orientativos, no requisitos universales de hardware. Los resultados rápidos
no reemplazan la búsqueda completa ni certifican las métricas históricas.

GitHub Actions valida datos, notebooks, separación de grupos y entrenamiento/búsqueda rápidos.
La ejecución completa se solicita manualmente en **Actions → Reproducibilidad → Run workflow → completo**.
Los resultados se adjuntan como artefactos incluso si un paso falla; una ejecución manual puede alcanzar
el límite de seis horas. No se afirma éxito hasta que el estado de ejecución sea `complete`.

## Organización

```text
rehab/             Código reutilizable y comandos
configs/           Configuraciones rápida y completa
tests/             Validaciones de integridad y separación de grupos
data/              CSV, manifiesto y dataset existente
docs/              Datos y metodología
reports/historico/ Resultados antiguos identificados
reports/runs/      Ejecuciones nuevas (ignoradas por Git)
models/            Modelos exportados (ignorados por Git)
*.ipynb            Notebooks conservados en la raíz
```
