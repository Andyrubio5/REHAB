"""Rutas, integridad y reconstrucción de la versión publicada del dataset."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data/REHAB/Rehab_exercise/d02_processed_data'
CHANNELS = ['pitch1', 'yaw1', 'roll1', 'pitch2', 'yaw2', 'roll2',
            'f1', 'f2', 'f3', 'f4', 'f5', 'pitch3']
STATS = ['std', 'median', 'min', 'max', 'iqr', 'mad_diff']
FEATURES = [f'{channel}_{stat}' for channel in CHANNELS for stat in STATS]
CLASSES = [f'{i:03d}' for i in range(16) if i != 14]
CSV = ROOT / 'data/df_final.csv'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_frame(df):
    expected = ['movimiento', 'repeticion_id', 'ventana'] + FEATURES
    if list(df.columns) != expected:
        raise ValueError('Columnas u orden de características distintos de la versión publicada.')
    if df.shape != (34056, 75) or sorted(df.movimiento.unique()) != CLASSES:
        raise ValueError('Se esperaban 34,056 filas, 75 columnas y las 15 clases publicadas.')
    if df.isna().any().any() or not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError('El dataset contiene valores faltantes o no finitos.')
    grouped = df.groupby('repeticion_id', sort=True)
    if grouped.ngroups != 4257 or not grouped.size().eq(8).all():
        raise ValueError('Se esperaban 4,257 repeticiones con ocho ventanas cada una.')
    if not grouped.movimiento.nunique().eq(1).all():
        raise ValueError('Una repetición tiene varias clases.')
    if df.duplicated(['repeticion_id', 'ventana']).any():
        raise ValueError('Hay identificadores de ventana duplicados.')
    if not set(df.ventana.unique()) == set(range(1, 9)):
        raise ValueError('Las ventanas deben estar numeradas del 1 al 8.')
    if not df.repeticion_id.str.split('_').str[0].eq(df.movimiento).all():
        raise ValueError('La clase no coincide con el identificador de repetición.')
    return {'rows': len(df), 'features': len(FEATURES), 'classes': len(CLASSES),
            'groups': grouped.ngroups, 'windows_per_group': 8}


def load_data(path=CSV):
    df = pd.read_csv(path, dtype={'movimiento': str, 'repeticion_id': str})
    validate_frame(df)
    return df


def verify_manifest(include_arrays=False):
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    for entry in manifest['files']:
        if entry['role'] == 'processed_input' and not include_arrays:
            continue
        path = ROOT / entry['path']
        if not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError(f'Archivo ausente o modificado: {entry["path"]}')
    return manifest


def validate_inputs(directory=DATA_DIR):
    """Exige todos los pares; el movimiento 014 es una exclusión explícita."""
    pairs = []
    for movement in CLASSES:
        paths = [Path(directory) / f'{movement}_{sensor}.npy' for sensor in (1, 2)]
        arrays = [np.load(path, mmap_mode='r', allow_pickle=False) for path in paths]
        if arrays[0].shape != arrays[1].shape or arrays[0].shape[1:] != (880, 6):
            raise ValueError(f'Par de sensores incompatible: {movement}')
        if any(a.ndim != 3 or not np.isfinite(a).all() for a in arrays):
            raise ValueError(f'Dimensiones o valores inválidos: {movement}')
        pairs.append((movement, arrays))
    return pairs


def rebuild(directory=DATA_DIR):
    pairs = validate_inputs(directory)  # No produce un CSV parcial si falta una entrada.
    frames = []
    for movement, arrays in pairs:
        # Orden: repetición, ventana, tiempo, canal; coincide con el notebook original.
        parts = [np.asarray(a).reshape(-1, 8, 110, 6).reshape(-1, 110, 6) for a in arrays]
        x = np.concatenate(parts, axis=2)
        values = {'std': x.std(axis=1), 'median': np.median(x, axis=1),
                  'min': x.min(axis=1), 'max': x.max(axis=1),
                  'iqr': np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1),
                  'mad_diff': np.abs(np.diff(x, axis=1)).mean(axis=1)}
        count = len(arrays[0])
        columns = {'movimiento': np.repeat(movement, count * 8),
                   'repeticion_id': np.repeat([f'{movement}_{i+1:04d}' for i in range(count)], 8),
                   'ventana': np.tile(np.arange(1, 9), count)}
        columns.update({f'{channel}_{stat}': values[stat][:, i]
                        for i, channel in enumerate(CHANNELS) for stat in STATS})
        frames.append(pd.DataFrame(columns))
    df = pd.concat(frames, ignore_index=True)
    validate_frame(df)
    return df


def compare_rebuilt(df, reference=None):
    reference = load_data() if reference is None else reference
    keys = ['movimiento', 'repeticion_id', 'ventana']
    if not df[keys].equals(reference[keys]):
        raise ValueError('La reconstrucción cambió las identidades u orden de ventanas.')
    np.testing.assert_allclose(df[FEATURES], reference[FEATURES], rtol=1e-10, atol=1e-10)
    return float(np.abs(df[FEATURES].to_numpy() - reference[FEATURES].to_numpy()).max())
