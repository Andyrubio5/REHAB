# Datos y alcance de la reconstrucción

`data/df_final.csv` tiene 34,056 filas y 75 columnas: movimiento, repeticion_id, ventana y 72
características. Los canales son pitch1, yaw1, roll1, pitch2, yaw2, roll2, f1–f5 y pitch3.
Por cada canal se calculan std (ddof=0), median, min, max, iqr y media de diferencias absolutas
consecutivas (mad_diff). Una repetición de 880 muestras se divide en ocho ventanas no solapadas
de 110 muestras. El orden de características se define en `rehab/data.py` y se exporta con el modelo.

Cada par `<movimiento>_1.npy` / `<movimiento>_2.npy` debe tener idéntica forma `(repeticiones, 880, 6)`.
La alineación semántica de los índices entre sensores procede del dataset existente; comprobar la forma
no demuestra por sí mismo esa alineación. `repeticion_id` concatena movimiento e índice de repetición.
No hay en el CSV un identificador verificado de participante.

Se excluye completamente 014 porque `014_1.npy` tiene una firma NPY alterada. No se habilita pickle
ni se trata de recuperar el contenido automáticamente. `014_2.npy` se conserva en el repositorio
pero no se utiliza solo: hacen falta ambos sensores. Los movimientos incluidos son 000–013 y 015.

`manifest.json` registra SHA-256, tamaño y procedencia por commit del CSV y de los 30 arrays utilizados.
Los archivos no se sustituyen por una reconstrucción sin más: `rebuild` produce otra salida y compara
identidades y valores con el CSV publicado. Una ausencia, cambio de hash, columna inesperada, clase
faltante, duplicación del identificador de ventana o valor no finito provoca un error.

La reconstrucción comienza en d02_processed_data. Falta documentar completamente el procedimiento
original que produjo esos arrays a partir de datos crudos. También queda pendiente la referencia
bibliográfica exacta, versión y condiciones de uso del dataset; el repositorio no aporta una fuente
verificada suficiente para completarlas sin consultar a su autor.

El EDA encuentra señales nulas y duplicadas. Se mantienen para conservar la definición del experimento
histórico. Analizar su efecto requiere otro experimento documentado; no se eliminan silenciosamente.
La independencia por repetición no garantiza independencia entre personas ni entre patrones duplicados.
