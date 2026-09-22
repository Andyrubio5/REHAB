import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path
import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from rehab.data import FEATURES, load_data, validate_frame, validate_inputs
from rehab.experiments import get_config, prepare_data, group_splits, configured_models, run


class BrokenEstimator(BaseEstimator):
    def fit(self, X, y):
        raise RuntimeError("Fallo de prueba")


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = get_config('rapido')
        cls.df = prepare_data(cls.config)

    def test_groups_never_cross_outer_or_inner_partitions(self):
        for train, test in group_splits(self.df, 3):
            self.assertFalse(set(self.df.iloc[train].repeticion_id) & set(self.df.iloc[test].repeticion_id))
            inner = self.df.iloc[train].reset_index(drop=True)
            for a, b in group_splits(inner, 2):
                self.assertFalse(set(inner.iloc[a].repeticion_id) & set(inner.iloc[b].repeticion_id))

    def test_duplicate_windows_rejected(self):
        df = load_data()
        df.loc[1, 'ventana'] = df.loc[0, 'ventana']
        with self.assertRaisesRegex(ValueError, 'duplicados'):
            validate_frame(df)

    def test_missing_sensor_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory) / '000_1.npy', np.zeros((1, 880, 6)))
            with self.assertRaises(FileNotFoundError):
                validate_inputs(directory)

    def test_encoded_labels_fit_and_roundtrip(self):
        encoder = LabelEncoder().fit(self.df.movimiento)
        encoded = encoder.transform(self.df.movimiento)
        pipeline = clone(configured_models(self.config, 'selection')['Extra Trees'])
        train, test = group_splits(self.df, 3)[0]
        pipeline.fit(self.df.iloc[train][FEATURES], encoded[train])
        prediction = encoder.inverse_transform(pipeline.predict(self.df.iloc[test][FEATURES]))
        self.assertTrue(set(prediction) <= set(self.df.movimiento))
        self.assertTrue(all(len(label) == 3 for label in prediction))

    def test_failed_experiment_is_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failed-run'
            with patch('rehab.experiments.configured_models', return_value={'Broken': BrokenEstimator()}):
                with self.assertRaisesRegex(RuntimeError, 'Fallo de prueba'):
                    run('selection', 'rapido', output)
            metadata = json.loads((output / 'metadata.json').read_text())
            self.assertEqual(metadata['status'], 'failed')
            self.assertTrue((output / 'outer_folds.csv').exists())

    def test_dataset_subset_deterministic(self):
        self.assertTrue(self.df.equals(prepare_data(self.config)))
        self.assertTrue(self.df.groupby('repeticion_id').size().eq(8).all())


if __name__ == '__main__':
    unittest.main()
