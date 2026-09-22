"""Configuraciones originales, con paralelismo explícito."""
import time
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

def baseline_models(seed=42, jobs=1):
    SEED = seed
    # ============================================================
    # 2. Pipelines
    # ============================================================

    def crear_pipeline(modelo, escalar=False):
        pasos = [
            ("imputacion", SimpleImputer(strategy="median")),
        ]

        if escalar:
            pasos.append(("escalamiento", StandardScaler()))

        pasos.append(("modelo", modelo))
        return Pipeline(pasos)


    modelos = {
        "Dummy": crear_pipeline(
            DummyClassifier(strategy="most_frequent")
        ),

        "Extra Trees": crear_pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
                n_jobs=jobs,
            )
        ),

        "Random Forest": crear_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=jobs,
            )
        ),

        "HistGradientBoosting": crear_pipeline(
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                # Evita una partición interna aleatoria por ventanas.
                early_stopping=False,
                random_state=SEED,
            )
        ),

        "SVM RBF": crear_pipeline(
            SVC(
                C=10,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                cache_size=512,
            ),
            escalar=True,
        ),

        "KNN": crear_pipeline(
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="minkowski",
                p=2,
                n_jobs=jobs,
            ),
            escalar=True,
        ),

        "Regresión logística": crear_pipeline(
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=SEED,
            ),
            escalar=True,
        ),

        "LDA regularizado": crear_pipeline(
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            ),
            escalar=True,
        ),

        "MLP": crear_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.01,
                learning_rate_init=0.001,
                batch_size=128,
                max_iter=400,
                # Evita validación interna que ignore los grupos.
                early_stopping=False,
                random_state=SEED,
            ),
            escalar=True,
        ),
    }
    return modelos

def search_models(seed=42):
    SEED = seed
    modelos_busqueda = {
        "HistGradientBoosting": {
            "pipeline": Pipeline([
                ("imputacion", SimpleImputer(strategy="median")),
                ("modelo", HistGradientBoostingClassifier(
                    early_stopping=False,
                    random_state=SEED,
                )),
            ]),

            "parametros": {
                "modelo__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
                "modelo__max_iter": [100, 200, 300, 400],
                "modelo__max_leaf_nodes": [7, 15, 31, 63],
                "modelo__max_depth": [None, 5, 10, 15],
                "modelo__min_samples_leaf": [10, 20, 30, 40],
                "modelo__l2_regularization": [0, 0.1, 0.5, 1, 5],
            },
        },

        "Extra Trees": {
            "pipeline": Pipeline([
                ("imputacion", SimpleImputer(strategy="median")),
                ("modelo", ExtraTreesClassifier(
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=1,
                )),
            ]),

            "parametros": {
                "modelo__n_estimators": [200, 300, 500, 700],
                "modelo__max_depth": [None, 10, 20, 30, 50],
                "modelo__min_samples_split": [2, 5, 10, 20],
                "modelo__min_samples_leaf": [1, 2, 4, 8],
                "modelo__max_features": [
                    "sqrt",
                    "log2",
                    0.5,
                    0.75,
                    None,
                ],
                "modelo__bootstrap": [False, True],
            },
        },

        "Random Forest": {
            "pipeline": Pipeline([
                ("imputacion", SimpleImputer(strategy="median")),
                ("modelo", RandomForestClassifier(
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=1,
                )),
            ]),

            "parametros": {
                "modelo__n_estimators": [200, 300, 500, 700],
                "modelo__max_depth": [None, 10, 20, 30, 50],
                "modelo__min_samples_split": [2, 5, 10, 20],
                "modelo__min_samples_leaf": [1, 2, 4, 8],
                "modelo__max_features": [
                    "sqrt",
                    "log2",
                    0.5,
                    0.75,
                ],
                "modelo__bootstrap": [True, False],
            },
        },
    }
    return modelos_busqueda
