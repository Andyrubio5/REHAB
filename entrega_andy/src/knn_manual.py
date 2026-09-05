#!/usr/bin/env python3
"""Clasificador KNN implementado manualmente para el dataset REHAB.

No utiliza scikit-learn ni importa ningún algoritmo de aprendizaje automático.
NumPy se usa únicamente para representar arreglos y efectuar operaciones
aritméticas; la división, normalización, distancia, votación y evaluación se
implementan en este archivo.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ID_COLUMNS = ("movimiento", "repeticion_id", "ventana")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Dataset:
    """Contenedor sencillo para las muestras del modelo."""

    x: np.ndarray
    y: np.ndarray
    ids: list[str]
    feature_names: list[str]


def cargar_y_agrupar_csv(ruta: str | Path) -> Dataset:
    """Carga el CSV y representa cada repetición mediante medianas por columna.

    Las ocho ventanas de una repetición permanecen juntas. Esto evita fuga de
    información entre entrenamiento y prueba y reduce el costo de KNN.
    """

    ruta = Path(ruta)
    grupos: dict[str, list[list[float]]] = defaultdict(list)
    etiquetas: dict[str, str] = {}

    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if not lector.fieldnames:
            raise ValueError("El CSV no contiene encabezados.")

        faltantes = [c for c in ID_COLUMNS if c not in lector.fieldnames]
        if faltantes:
            raise ValueError(f"Faltan columnas obligatorias: {faltantes}")

        feature_names = [c for c in lector.fieldnames if c not in ID_COLUMNS]
        if not feature_names:
            raise ValueError("El CSV no contiene características numéricas.")

        for numero_fila, fila in enumerate(lector, start=2):
            repeticion = fila["repeticion_id"].strip()
            movimiento = fila["movimiento"].strip()
            if not repeticion or not movimiento:
                raise ValueError(f"Identificador vacío en la fila {numero_fila}.")

            if repeticion in etiquetas and etiquetas[repeticion] != movimiento:
                raise ValueError(
                    f"La repetición {repeticion} tiene más de una etiqueta."
                )

            try:
                valores = [float(fila[nombre]) for nombre in feature_names]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Valor no numérico en la fila {numero_fila}."
                ) from exc

            if not np.isfinite(valores).all():
                raise ValueError(f"Valor no finito en la fila {numero_fila}.")

            grupos[repeticion].append(valores)
            etiquetas[repeticion] = movimiento

    ids = sorted(grupos)
    x = np.asarray(
        [np.median(np.asarray(grupos[rep], dtype=float), axis=0) for rep in ids],
        dtype=float,
    )
    y = np.asarray([etiquetas[rep] for rep in ids], dtype=str)
    return Dataset(x=x, y=y, ids=ids, feature_names=feature_names)


def guardar_dataset_agrupado(dataset: Dataset, ruta: str | Path) -> None:
    """Guarda la versión de una fila por repetición utilizada por el modelo."""

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["repeticion_id", *dataset.feature_names, "movimiento"])
        for rep, fila, etiqueta in zip(dataset.ids, dataset.x, dataset.y):
            escritor.writerow([rep, *fila.tolist(), etiqueta])


def dividir_estratificado(
    y: np.ndarray, proporcion_prueba: float = 0.2, semilla: int = 2026
) -> tuple[np.ndarray, np.ndarray]:
    """Divide índices conservando aproximadamente la proporción de cada clase."""

    if not 0.0 < proporcion_prueba < 1.0:
        raise ValueError("La proporción de prueba debe estar entre 0 y 1.")

    rng = np.random.default_rng(semilla)
    por_clase: dict[str, list[int]] = defaultdict(list)
    for indice, etiqueta in enumerate(y):
        por_clase[str(etiqueta)].append(indice)

    entrenamiento: list[int] = []
    prueba: list[int] = []
    for etiqueta in sorted(por_clase):
        indices = np.asarray(por_clase[etiqueta], dtype=int)
        rng.shuffle(indices)
        cantidad_prueba = max(1, int(round(len(indices) * proporcion_prueba)))
        prueba.extend(indices[:cantidad_prueba].tolist())
        entrenamiento.extend(indices[cantidad_prueba:].tolist())

    rng.shuffle(entrenamiento)
    rng.shuffle(prueba)
    return np.asarray(entrenamiento), np.asarray(prueba)


class EstandarizadorManual:
    """Normalización z-score calculada únicamente con entrenamiento."""

    def __init__(self) -> None:
        self.centro: np.ndarray | None = None
        self.escala: np.ndarray | None = None

    def ajustar(self, x: np.ndarray) -> "EstandarizadorManual":
        self.centro = np.sum(x, axis=0) / x.shape[0]
        diferencias = x - self.centro
        varianza = np.sum(diferencias * diferencias, axis=0) / x.shape[0]
        self.escala = np.sqrt(varianza)
        self.escala[self.escala == 0.0] = 1.0
        return self

    def transformar(self, x: np.ndarray) -> np.ndarray:
        if self.centro is None or self.escala is None:
            raise RuntimeError("El estandarizador todavía no ha sido ajustado.")
        return (x - self.centro) / self.escala


class KNNManual:
    """K vecinos más cercanos con distancia euclidiana y voto mayoritario."""

    def __init__(self, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k debe ser un entero positivo.")
        self.k = k
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def ajustar(self, x: np.ndarray, y: np.ndarray) -> "KNNManual":
        if x.shape[0] != y.shape[0]:
            raise ValueError("X e y deben contener la misma cantidad de filas.")
        if self.k > x.shape[0]:
            raise ValueError("k no puede superar las muestras de entrenamiento.")
        self.x_train = np.asarray(x, dtype=float)
        self.y_train = np.asarray(y, dtype=str)
        return self

    def _predecir_una(self, muestra: np.ndarray) -> str:
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("El modelo todavía no ha sido ajustado.")

        diferencias = self.x_train - muestra
        distancias_cuadradas = np.sum(diferencias * diferencias, axis=1)
        indices = np.argpartition(distancias_cuadradas, self.k - 1)[: self.k]

        conteos = Counter(self.y_train[indices].tolist())
        mayor_votacion = max(conteos.values())
        empatadas = [clase for clase, votos in conteos.items() if votos == mayor_votacion]
        if len(empatadas) == 1:
            return empatadas[0]

        # Desempate determinista: menor suma de distancias; después, la etiqueta.
        return min(
            empatadas,
            key=lambda clase: (
                float(np.sum(distancias_cuadradas[indices][self.y_train[indices] == clase])),
                clase,
            ),
        )

    def predecir(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([self._predecir_una(fila) for fila in x], dtype=str)


def exactitud(y_real: np.ndarray, y_predicha: np.ndarray) -> float:
    """Proporción de predicciones correctas."""

    return float(np.sum(y_real == y_predicha) / len(y_real))


def matriz_confusion(
    y_real: np.ndarray, y_predicha: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """Construye manualmente una matriz: filas reales, columnas predichas."""

    clases = sorted(set(y_real.tolist()) | set(y_predicha.tolist()))
    posiciones = {clase: indice for indice, clase in enumerate(clases)}
    matriz = np.zeros((len(clases), len(clases)), dtype=int)
    for real, predicha in zip(y_real, y_predicha):
        matriz[posiciones[str(real)], posiciones[str(predicha)]] += 1
    return clases, matriz


def guardar_predicciones(
    ruta: str | Path,
    ids: list[str],
    y_real: np.ndarray,
    y_predicha: np.ndarray,
) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["repeticion_id", "movimiento_real", "movimiento_predicho", "correcta"])
        for rep, real, predicha in zip(ids, y_real, y_predicha):
            escritor.writerow([rep, real, predicha, int(real == predicha)])


def ejecutar_experimento(
    ruta_csv: str | Path,
    k: int = 5,
    proporcion_prueba: float = 0.2,
    semilla: int = 2026,
    ruta_predicciones: str | Path | None = None,
    ruta_dataset_agrupado: str | Path | None = None,
) -> dict[str, object]:
    """Ejecuta el flujo completo y devuelve resultados reutilizables."""

    dataset = cargar_y_agrupar_csv(ruta_csv)
    train_idx, test_idx = dividir_estratificado(dataset.y, proporcion_prueba, semilla)

    estandarizador = EstandarizadorManual().ajustar(dataset.x[train_idx])
    x_train = estandarizador.transformar(dataset.x[train_idx])
    x_test = estandarizador.transformar(dataset.x[test_idx])

    modelo = KNNManual(k=k).ajustar(x_train, dataset.y[train_idx])
    predicciones = modelo.predecir(x_test)
    precision = exactitud(dataset.y[test_idx], predicciones)
    clases, confusion = matriz_confusion(dataset.y[test_idx], predicciones)

    ids_prueba = [dataset.ids[i] for i in test_idx]
    if ruta_predicciones is not None:
        guardar_predicciones(
            ruta_predicciones, ids_prueba, dataset.y[test_idx], predicciones
        )
    if ruta_dataset_agrupado is not None:
        guardar_dataset_agrupado(dataset, ruta_dataset_agrupado)

    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_test": dataset.y[test_idx],
        "predicciones": predicciones,
        "exactitud": precision,
        "clases": clases,
        "matriz_confusion": confusion,
        "ids_prueba": ids_prueba,
        "modelo": modelo,
        "estandarizador": estandarizador,
    }


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clasificador KNN manual para movimientos del dataset REHAB."
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "df_final.csv"),
        help="Ruta al CSV",
    )
    parser.add_argument("--k", type=int, default=5, help="Cantidad de vecinos")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción de prueba")
    parser.add_argument("--seed", type=int, default=2026, help="Semilla reproducible")
    parser.add_argument(
        "--predicciones",
        default=str(PROJECT_ROOT / "resultados" / "predicciones.csv"),
        help="Archivo de salida para las predicciones",
    )
    parser.add_argument(
        "--dataset-modelo",
        default=str(PROJECT_ROOT / "data" / "dataset_modelo.csv"),
        help="Archivo de salida con una fila por repetición",
    )
    return parser


def main() -> None:
    args = construir_parser().parse_args()
    resultado = ejecutar_experimento(
        ruta_csv=args.data,
        k=args.k,
        proporcion_prueba=args.test_size,
        semilla=args.seed,
        ruta_predicciones=args.predicciones,
        ruta_dataset_agrupado=args.dataset_modelo,
    )

    dataset = resultado["dataset"]
    print("KNN manual - clasificación de movimientos REHAB")
    print(f"Repeticiones: {len(dataset.ids)}")
    print(f"Características: {len(dataset.feature_names)}")
    print(f"Entrenamiento: {len(resultado['train_idx'])}")
    print(f"Prueba: {len(resultado['test_idx'])}")
    print(f"k: {args.k}")
    print(f"Exactitud: {resultado['exactitud']:.4f}")
    print("\nPrimeras 10 predicciones:")
    for rep, real, predicha in list(
        zip(resultado["ids_prueba"], resultado["y_test"], resultado["predicciones"])
    )[:10]:
        print(f"  {rep}: real={real}, predicción={predicha}")

    print("\nMatriz de confusión (filas=reales, columnas=predichas):")
    print("clases:", " ".join(resultado["clases"]))
    print(resultado["matriz_confusion"])


if __name__ == "__main__":
    main()
