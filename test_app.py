import unittest
import numpy as np
import app

class ForecastTests(unittest.TestCase):
    def test_demo_data_runs_pipeline_without_api_key(self):
        prices = app.demo_market_data()
        self.assertGreaterEqual(len(prices), 100)
        self.assertTrue((prices["Close"] > 0).all())
        features = app.build_features(prices)
        _, test, predictions, mae, baseline_mae, _, _, next_prediction = app.train_and_test(features)
        self.assertEqual(len(test), len(predictions))
        self.assertTrue(np.isfinite([mae, baseline_mae, next_prediction]).all())

if __name__ == "__main__":
    unittest.main()
