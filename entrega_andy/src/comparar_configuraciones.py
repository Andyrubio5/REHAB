#!/usr/bin/env python3
"""Compara configuraciones del KNN manual y selecciona la mejor.

La selección de k se realiza con validación. El conjunto de prueba se utiliza
una sola vez, después de seleccionar la configuración ganadora.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    # Cuando se importa como parte del paquete ``src`` desde el notebook.
    from .knn_manual import (
        EstandarizadorManual,
        KNNManual,
        cargar_y_agrupar_csv,
        exactitud,
        guardar_predicciones,
        matriz_confusion,
    )
except ImportError:
    # Cuando se ejecuta directamente: python3 src/comparar_configuraciones.py
    from knn_manual import (
        EstandarizadorManual,
        KNNManual,
        cargar_y_agrupar_csv,
        exactitud,
        guardar_predicciones,
        matriz_confusion,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
K_CANDIDATOS = (1, 3, 5, 7, 9, 11, 15)


def dividir_tres_conjuntos(
    y: np.ndarray,
    proporcion_validacion: float = 0.20,
    proporcion_prueba: float = 0.20,
    semilla: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crea entrenamiento, validación y prueba de forma estratificada."""

    if proporcion_validacion <= 0 or proporcion_prueba <= 0:
        raise ValueError("Validación y prueba deben ser mayores que cero.")
    if proporcion_validacion + proporcion_prueba >= 1:
        raise ValueError("Validación más prueba debe ser menor que uno.")

    rng = np.random.default_rng(semilla)
    por_clase: dict[str, list[int]] = defaultdict(list)
    for indice, etiqueta in enumerate(y):
        por_clase[str(etiqueta)].append(indice)

    entrenamiento: list[int] = []
    validacion: list[int] = []
    prueba: list[int] = []

    for etiqueta in sorted(por_clase):
        indices = np.asarray(por_clase[etiqueta], dtype=int)
        rng.shuffle(indices)
        n_prueba = max(1, int(round(len(indices) * proporcion_prueba)))
        n_validacion = max(1, int(round(len(indices) * proporcion_validacion)))
        prueba.extend(indices[:n_prueba].tolist())
        validacion.extend(indices[n_prueba : n_prueba + n_validacion].tolist())
        entrenamiento.extend(indices[n_prueba + n_validacion :].tolist())

    rng.shuffle(entrenamiento)
    rng.shuffle(validacion)
    rng.shuffle(prueba)
    return (
        np.asarray(entrenamiento, dtype=int),
        np.asarray(validacion, dtype=int),
        np.asarray(prueba, dtype=int),
    )


def metricas_clasificacion(
    y_real: np.ndarray, y_predicha: np.ndarray
) -> dict[str, float]:
    """Calcula exactitud, precisión, recall y F1 macro manualmente."""

    clases = sorted(set(y_real.tolist()) | set(y_predicha.tolist()))
    precisiones: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for clase in clases:
        verdaderos_positivos = int(np.sum((y_real == clase) & (y_predicha == clase)))
        falsos_positivos = int(np.sum((y_real != clase) & (y_predicha == clase)))
        falsos_negativos = int(np.sum((y_real == clase) & (y_predicha != clase)))

        precision = (
            verdaderos_positivos / (verdaderos_positivos + falsos_positivos)
            if verdaderos_positivos + falsos_positivos
            else 0.0
        )
        recall = (
            verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
            if verdaderos_positivos + falsos_negativos
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisiones.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "exactitud": exactitud(y_real, y_predicha),
        "precision_macro": float(np.mean(precisiones)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }


def guardar_comparacion(resultados: list[dict[str, float]], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    campos = ["k", "exactitud", "precision_macro", "recall_macro", "f1_macro", "segundos"]
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)


def guardar_matriz(clases: list[str], matriz: np.ndarray, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["real/predicha", *clases])
        for clase, fila in zip(clases, matriz):
            escritor.writerow([clase, *fila.tolist()])


def comparar(
    ruta_csv: str | Path,
    valores_k: tuple[int, ...] = K_CANDIDATOS,
    semilla: int = 2026,
    directorio_resultados: str | Path = PROJECT_ROOT / "resultados",
) -> dict[str, object]:
    """Compara valores de k, selecciona uno y evalúa una sola vez en prueba."""

    dataset = cargar_y_agrupar_csv(ruta_csv)
    train_idx, val_idx, test_idx = dividir_tres_conjuntos(dataset.y, semilla=semilla)

    escalador = EstandarizadorManual().ajustar(dataset.x[train_idx])
    x_train = escalador.transformar(dataset.x[train_idx])
    x_val = escalador.transformar(dataset.x[val_idx])

    comparacion: list[dict[str, float]] = []
    for k in valores_k:
        inicio = perf_counter()
        predicciones = KNNManual(k=k).ajustar(x_train, dataset.y[train_idx]).predecir(x_val)
        segundos = perf_counter() - inicio
        fila = {"k": k, **metricas_clasificacion(dataset.y[val_idx], predicciones)}
        fila["segundos"] = segundos
        comparacion.append(fila)

    # Prioridad: F1 macro, exactitud y, si persiste el empate, menor k.
    mejor = max(comparacion, key=lambda r: (r["f1_macro"], r["exactitud"], -r["k"]))
    mejor_k = int(mejor["k"])

    # Una vez elegido k, se aprovechan entrenamiento y validación para el modelo final.
    desarrollo_idx = np.concatenate([train_idx, val_idx])
    escalador_final = EstandarizadorManual().ajustar(dataset.x[desarrollo_idx])
    x_desarrollo = escalador_final.transformar(dataset.x[desarrollo_idx])
    x_test = escalador_final.transformar(dataset.x[test_idx])
    modelo_final = KNNManual(k=mejor_k).ajustar(x_desarrollo, dataset.y[desarrollo_idx])
    predicciones_test = modelo_final.predecir(x_test)
    metricas_test = metricas_clasificacion(dataset.y[test_idx], predicciones_test)
    clases, confusion = matriz_confusion(dataset.y[test_idx], predicciones_test)

    resultados_dir = Path(directorio_resultados)
    guardar_comparacion(comparacion, resultados_dir / "comparacion_configuraciones.csv")
    guardar_predicciones(
        resultados_dir / "predicciones_finales.csv",
        [dataset.ids[i] for i in test_idx],
        dataset.y[test_idx],
        predicciones_test,
    )
    guardar_matriz(clases, confusion, resultados_dir / "matriz_confusion_final.csv")

    resumen = {
        "semilla": semilla,
        "k_evaluados": list(valores_k),
        "mejor_k": mejor_k,
        "muestras_entrenamiento": len(train_idx),
        "muestras_validacion": len(val_idx),
        "muestras_prueba": len(test_idx),
        "muestras_modelo_final": len(desarrollo_idx),
        "criterio_seleccion": "Mayor F1 macro de validación; luego exactitud; luego menor k.",
        "metricas_prueba": metricas_test,
    }
    with (resultados_dir / "resumen_seleccion.json").open("w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)

    return {
        "dataset": dataset,
        "comparacion": comparacion,
        "mejor_k": mejor_k,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "metricas_test": metricas_test,
        "clases": clases,
        "matriz_confusion": confusion,
        "y_test": dataset.y[test_idx],
        "predicciones_test": predicciones_test,
        "ids_test": [dataset.ids[i] for i in test_idx],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparación de configuraciones KNN manual")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "df_final.csv"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--k", nargs="+", type=int, default=list(K_CANDIDATOS))
    parser.add_argument("--resultados", default=str(PROJECT_ROOT / "resultados"))
    args = parser.parse_args()

    resultado = comparar(
        ruta_csv=args.data,
        valores_k=tuple(args.k),
        semilla=args.seed,
        directorio_resultados=args.resultados,
    )

    print("Comparación de configuraciones KNN manual")
    print("k  exactitud  precisión  recall   F1 macro  segundos")
    for fila in resultado["comparacion"]:
        print(
            f"{int(fila['k']):2d}  {fila['exactitud']:.4f}     "
            f"{fila['precision_macro']:.4f}     {fila['recall_macro']:.4f}  "
            f"{fila['f1_macro']:.4f}    {fila['segundos']:.3f}"
        )

    print(f"\nConfiguración seleccionada: k={resultado['mejor_k']}")
    print("Evaluación final en prueba:")
    for nombre, valor in resultado["metricas_test"].items():
        print(f"  {nombre}: {valor:.4f}")


if __name__ == "__main__":
    main()
