"""Comparación, búsqueda anidada y exportación con trazabilidad por ejecución."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import json
import platform
import re
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from .data import ROOT, CSV, FEATURES, load_data, sha256, verify_manifest
from .models import baseline_models, search_models


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    tmp.replace(path)


def slug(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def environment():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'commit': commit, 'working_tree_dirty': dirty,
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p)
                              for p in sorted((ROOT / 'rehab').glob('*.py'))},
            'versions': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'threadpoolctl']}}


def get_config(mode):
    config = json.loads((ROOT / 'configs' / f'{mode}.json').read_text())
    return config


def prepare_data(config):
    verify_manifest()
    df = load_data()
    count = config['groups_per_class']
    if count is not None:
        rng = np.random.default_rng(config['seed'])
        selected = []
        for _, part in df.groupby('movimiento', sort=True):
            selected.extend(rng.choice(sorted(part.repeticion_id.unique()), size=count, replace=False))
        df = df[df.repeticion_id.isin(selected)].copy()
    # Mantiene orden estable y guarda la fila original para trazar cada predicción.
    return df.reset_index(names='source_row')


def group_splits(df, n_splits):
    groups = df.repeticion_id.to_numpy()
    splits = list(GroupKFold(n_splits=n_splits).split(df[FEATURES], df.movimiento, groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Hay repeticiones compartidas entre particiones.')
        if set(df.movimiento.iloc[train]) != set(df.movimiento):
            raise ValueError('Falta una clase en entrenamiento.')
    return splits


def save_splits(path, df, splits):
    fold = np.zeros(len(df), dtype=int)
    for number, (_, test) in enumerate(splits, 1):
        fold[test] = number
    if (fold == 0).any():
        raise ValueError('Hay filas sin fold.')
    table = df[['source_row', 'movimiento', 'repeticion_id', 'ventana']].copy()
    table['fold'] = fold
    table.to_csv(path, index=False)


def configured_models(config, stage):
    if stage == 'selection':
        models = baseline_models(config['seed'], config['jobs'])
        selected = {name: models[name] for name in config['selection_models']}
        if config['smoke_test']:
            for pipeline in selected.values():
                params = pipeline.get_params()
                if 'modelo__n_estimators' in params:
                    pipeline.set_params(modelo__n_estimators=20)
                if 'modelo__max_iter' in params:
                    pipeline.set_params(modelo__max_iter=20)
        return selected
    models = search_models(config['seed'])
    selected = {name: models[name] for name in config['search_models']}
    if config['smoke_test']:
        for name, model in selected.items():
            if name == 'HistGradientBoosting':
                model['parametros'] = {'modelo__max_iter': [20], 'modelo__max_leaf_nodes': [7, 15],
                                       'modelo__learning_rate': [0.1]}
            else:
                model['parametros'] = {'modelo__n_estimators': [20], 'modelo__max_depth': [10, None]}
    return selected


def make_search(spec, config, cv):
    return RandomizedSearchCV(clone(spec['pipeline']), spec['parametros'],
                              n_iter=config['n_iter'], scoring='f1_macro', cv=cv,
                              random_state=config['seed'], n_jobs=config['jobs'],
                              pre_dispatch=config['jobs'], error_score='raise', refit=True)


def run(stage='selection', mode='rapido', output=None):
    if stage not in ('selection', 'search'):
        raise ValueError('Etapa desconocida.')
    config = get_config(mode)
    df = prepare_data(config)
    encoder = LabelEncoder().fit(df.movimiento)
    y = encoder.transform(df.movimiento)  # Evita el fallo de class_weight con etiquetas de texto.
    X, groups = df[FEATURES], df.repeticion_id.to_numpy()
    folds = group_splits(df, config['outer_splits'])
    models = configured_models(config, stage)
    output = Path(output) if output is not None else ROOT / 'reports/runs' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + f'_{stage}_{mode}')
    output.mkdir(parents=True, exist_ok=False)  # Nunca mezcla ni sobrescribe experimentos.
    metadata = {'status': 'running', 'stage': stage, 'mode': mode, 'config': config,
                'dataset_sha256': sha256(CSV), 'features': FEATURES, 'classes': encoder.classes_.tolist(),
                'rows': len(df), 'groups': df.repeticion_id.nunique(), 'environment': environment(),
                'evaluation_unit': 'repeticion_id (no participante)',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'interpretation': ('Prueba de funcionamiento; no estima desempeño final.' if config['smoke_test']
                                   else 'Evaluación por repeticiones; no es prueba independiente de selección de familia.')}
    save_json(output / 'metadata.json', metadata)
    save_splits(output / 'outer_folds.csv', df, folds)
    # Particiones internas idénticas entre familias y persistidas para auditoría.
    if stage == 'search':
        for number, (train, _) in enumerate(folds, 1):
            inner_df = df.iloc[train].reset_index(drop=True)
            save_splits(output / f'inner_folds_{number}.csv', inner_df,
                        group_splits(inner_df, config['inner_splits']))
    metrics, summary = [], []
    try:
        with threadpool_limits(limits=config['threads']):
            for name, spec in models.items():
                model_dir = output / slug(name)
                model_dir.mkdir()
                predictions = np.full(len(y), -1, dtype=int)
                for number, (train, test) in enumerate(folds, 1):
                    started = time.monotonic()
                    if stage == 'search':
                        inner_df = df.iloc[train].reset_index(drop=True)
                        fitted = make_search(spec, config, group_splits(inner_df, config['inner_splits']))
                        fitted.fit(X.iloc[train], y[train])
                        save_json(model_dir / f'parameters_{number}.json', fitted.best_params_)
                        pd.DataFrame(fitted.cv_results_).to_csv(model_dir / f'cv_results_{number}.csv', index=False)
                    else:
                        fitted = clone(spec).fit(X.iloc[train], y[train])
                    pred = fitted.predict(X.iloc[test])
                    predictions[test] = pred
                    metrics.append({'model': name, 'fold': number,
                                    'accuracy': accuracy_score(y[test], pred),
                                    'f1_macro': f1_score(y[test], pred, labels=np.arange(len(encoder.classes_)),
                                                        average='macro', zero_division=0),
                                    'seconds': time.monotonic() - started,
                                    'validation_rows': len(test)})
                    pd.DataFrame(metrics).to_csv(output / 'fold_metrics.csv', index=False)
                    partial = df.loc[predictions >= 0, ['source_row', 'movimiento', 'repeticion_id', 'ventana']].copy()
                    partial['prediction'] = encoder.inverse_transform(predictions[predictions >= 0])
                    partial.to_csv(model_dir / 'predictions.csv', index=False)
                    print(f'{name} fold {number}: F1 macro={metrics[-1]["f1_macro"]:.4f}', flush=True)
                if (predictions < 0).any():
                    raise ValueError('Faltan predicciones OOF.')
                pd.DataFrame(classification_report(y, predictions, labels=np.arange(len(encoder.classes_)),
                    target_names=encoder.classes_, output_dict=True, zero_division=0)).T.to_csv(model_dir / 'classification_report.csv')
                pd.DataFrame(confusion_matrix(y, predictions), index=encoder.classes_,
                             columns=encoder.classes_).to_csv(model_dir / 'confusion_matrix.csv')
                model_metrics = pd.DataFrame([m for m in metrics if m['model'] == name])
                summary.append({'model': name, 'accuracy_oof': accuracy_score(y, predictions),
                                'f1_macro_oof': f1_score(y, predictions, average='macro', zero_division=0),
                                'f1_macro_mean': model_metrics.f1_macro.mean(),
                                'f1_macro_std': model_metrics.f1_macro.std()})
                pd.DataFrame(summary).sort_values('f1_macro_mean', ascending=False).to_csv(output / 'summary.csv', index=False)
        metadata.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_json(output / 'metadata.json', metadata)
    return output


def fit_final(search_run):
    """Reajusta la familia elegida: nunca reutiliza parámetros escritos a mano."""
    search_run = Path(search_run)
    metadata = json.loads((search_run / 'metadata.json').read_text())
    if metadata['stage'] != 'search' or metadata['status'] != 'complete':
        raise ValueError('Se requiere una búsqueda anidada completada.')
    if metadata['dataset_sha256'] != sha256(CSV):
        raise ValueError('Los datos cambiaron desde la búsqueda.')
    config = metadata['config']
    df = prepare_data(config)
    encoder = LabelEncoder().fit(df.movimiento)
    y = encoder.transform(df.movimiento)
    selected = pd.read_csv(search_run / 'summary.csv').iloc[0]['model']
    spec = configured_models(config, 'search')[selected]
    dest = ROOT / 'models' / search_run.name
    dest.mkdir(parents=True, exist_ok=False)
    final_metadata = {'status': 'running', 'selected_model': selected, 'config': config,
                      'dataset_sha256': sha256(CSV), 'features': FEATURES,
                      'classes': encoder.classes_.tolist(), 'environment': environment(),
                      'source_run': search_run.name, 'smoke_test': config['smoke_test'],
                      'evaluation': 'Refit para inferencia. No se mide accuracy de entrenamiento ni prueba independiente.'}
    save_json(dest / 'metadata.json', final_metadata)
    try:
        cv = group_splits(df, config['inner_splits'])
        save_splits(dest / 'tuning_folds.csv', df, cv)
        with threadpool_limits(limits=config['threads']):
            fitted = make_search(spec, config, cv).fit(df[FEATURES], y)
        joblib.dump({'pipeline': fitted.best_estimator_, 'label_encoder': encoder,
                     'features': FEATURES}, dest / 'model.joblib')
        pd.DataFrame(fitted.cv_results_).to_csv(dest / 'cv_results.csv', index=False)
        final_metadata.update(status='complete', best_parameters=fitted.best_params_)
    except BaseException as exc:
        final_metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_json(dest / 'metadata.json', final_metadata)
    return dest
