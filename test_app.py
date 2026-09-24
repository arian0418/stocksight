import unittest
import numpy as np
import forecast

class ForecastTests(unittest.TestCase):
    def test_demo_data_runs_pipeline_without_api_key(self):
        prices = forecast.demo_market_data()
        self.assertGreaterEqual(len(prices), 100)
        self.assertTrue((prices["Close"] > 0).all())
        features = forecast.build_features(prices)
        _, test, predictions, mae, baseline_mae, _, _, next_prediction = forecast.train_and_test(features)
        self.assertEqual(len(test), len(predictions))
        self.assertTrue(np.isfinite([mae, baseline_mae, next_prediction]).all())

    def test_demo_button_renders_dashboard_without_key(self):
        from streamlit.testing.v1 import AppTest
        dashboard = AppTest.from_file("app.py", default_timeout=20).run()
        self.assertEqual(len(dashboard.error), 0)
        dashboard.button[0].click().run()
        self.assertEqual(len(dashboard.error), 0)
        self.assertGreaterEqual(len(dashboard.metric), 8)
        self.assertEqual(len(dashboard.get("plotly_chart")), 1)

if __name__ == "__main__":
    unittest.main()
