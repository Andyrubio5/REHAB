# Contribuir

Usa Python 3.12.3 y el entorno de `requirements.txt`. Ejecuta desde la raíz:

```bash
python -m rehab validate
python -m unittest discover -s tests -v
python -m rehab search --mode rapido
```

Los notebooks usan código común en `rehab/`. Mantén sus salidas limpias y conserva resultados
verificables en una carpeta de referencia dentro de `reports/`, con configuración, versiones y
procedencia. `reports/runs/` y `models/` contienen artefactos locales y no se versionan automáticamente.

No cambies silenciosamente datos, exclusiones, folds o métricas. Una nueva versión del dataset necesita
su manifiesto actualizado de forma explícita, explicación del cambio y nuevos resultados. No reemplaces
las cifras históricas por cifras de una ejecución rápida. Al modificar dependencias, verifica el flujo
en un entorno limpio antes de actualizar el archivo de versiones fijadas.
