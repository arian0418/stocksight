"""Data and model contracts, using deterministic real data transformations."""
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

import forecast


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.prices = forecast.demo_market_data()

    def test_demo_data_runs_pipeline_without_api_key(self):
        self.assertGreaterEqual(len(self.prices), 100)
        features = forecast.build_features(self.prices)
        _, test, predictions, mae, baseline_mae, _, _, next_prediction = forecast.train_and_test(features)
        self.assertEqual(len(test), len(predictions))
        self.assertTrue(np.isfinite([mae, baseline_mae, next_prediction]).all())

    def test_validation_sorts_dates_without_mutating_caller(self):
        self.assertTrue(hasattr(forecast, "validate_market_data"), "Need a shared data contract")
        source = self.prices.iloc[::-1].copy()
        clean = forecast.validate_market_data(source)
        self.assertTrue(clean.index.is_monotonic_increasing)
        self.assertTrue(source.index.is_monotonic_decreasing)

    def test_features_reject_nonfinite_or_impossible_prices(self):
        cases = [("Close", np.nan), ("Close", np.inf), ("Close", 0),
                 ("Open", -1), ("High", 1), ("Low", 9999),
                 ("Volume", -1), ("Volume", np.inf), ("Close", "invalid")]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                prices = self.prices.copy()
                if isinstance(value, str):
                    prices[column] = prices[column].astype(object)
                prices.loc[prices.index[5], column] = value
                try:
                    forecast.build_features(prices)
                except Exception as error:
                    self.assertIsInstance(error, ValueError)
                    self.assertRegex(str(error), "(?i)(finite|positive|range|numeric|volume|price|missing)")
                else:
                    self.fail("Invalid OHLCV must fail before feature generation")

    def test_features_reject_duplicate_days(self):
        prices = pd.concat([self.prices, self.prices.iloc[[-1]]])
        with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
            forecast.build_features(prices)

    def test_features_reject_short_history_before_training(self):
        with self.assertRaisesRegex(ValueError, "(?i)(60|history|observations)"):
            forecast.build_features(self.prices.head(30))

    def test_features_reject_missing_ohlcv_columns(self):
        with self.assertRaisesRegex(ValueError, "(?i)(missing|open)"):
            forecast.build_features(self.prices.drop(columns="Open"))

    def test_features_reject_invalid_dates(self):
        prices = self.prices.copy()
        prices.index = ["invalid"] + list(prices.index[1:])
        with self.assertRaisesRegex(ValueError, "(?i)date"):
            forecast.build_features(prices)

    def test_zero_volume_history_still_has_finite_features(self):
        self.prices["Volume"] = 0
        features = forecast.build_features(self.prices)
        self.assertGreater(len(features), 30)
        self.assertTrue(np.isfinite(features["Volume_Ratio"]).all())
        self.assertTrue((features["Volume_Ratio"] == 0).all())
        self.assertTrue(np.isfinite(forecast.train_and_test(features)[-1]))

    def test_target_dates_are_next_observation_and_latest_stays_unknown(self):
        features = forecast.build_features(self.prices)
        self.assertIn("TargetDate", features.columns)
        self.assertEqual(features.iloc[0]["TargetDate"], self.prices.index[21])
        self.assertEqual(features.iloc[0]["Target"], self.prices.iloc[21]["Close"])
        self.assertEqual(features.index[-1], self.prices.index[-1])
        self.assertTrue(pd.isna(features.iloc[-1]["TargetDate"]))
        self.assertTrue(pd.isna(features.iloc[-1]["Target"]))

    def test_changing_later_observations_does_not_leak_into_earlier_predictions(self):
        original = forecast.train_and_test(forecast.build_features(self.prices))
        changed = self.prices.copy()
        changed.loc[changed.index[-5:], ["Open", "High", "Low", "Close"]] *= 1.2
        updated = forecast.train_and_test(forecast.build_features(changed))
        np.testing.assert_allclose(original[2][:-5], updated[2][:-5])
        self.assertNotAlmostEqual(original[-1], updated[-1])

    def test_backtest_rows_are_labeled_with_target_date_and_keep_baseline(self):
        self.assertTrue(hasattr(forecast, "analyze_prices"), "Need a reusable analysis snapshot")
        result = forecast.analyze_prices(self.prices)
        table = result["backtest"]
        self.assertEqual(table.index.name, "Forecast date")
        self.assertEqual(table.index[-1], self.prices.index[-1])
        self.assertEqual(table.iloc[-1]["Actual close"], self.prices.iloc[-1]["Close"])
        self.assertEqual(table.iloc[-1]["No-change baseline"], self.prices.iloc[-2]["Close"])
        self.assertEqual(table.iloc[-1]["Observed date"], self.prices.index[-2])
        self.assertLess(result["train_end"], table.index[0])
        self.assertEqual(result["next_observed_date"], self.prices.index[-1])

    def test_csv_export_round_trips_all_backtest_rows(self):
        self.assertTrue(hasattr(forecast, "analyze_prices"), "Need a reusable analysis snapshot")
        table = forecast.analyze_prices(self.prices)["backtest"]
        exported = pd.read_csv(StringIO(table.to_csv(date_format="%Y-%m-%d")))
        self.assertEqual(len(exported), len(table))
        self.assertEqual(exported.iloc[-1]["Forecast date"], "2025-09-09")
        self.assertAlmostEqual(exported.iloc[-1]["Actual close"], self.prices.iloc[-1]["Close"])
        self.assertIn("No-change baseline", exported.columns)
        self.assertIn("Error", exported.columns)


class CSVTests(unittest.TestCase):
    def read(self, contents):
        self.assertTrue(hasattr(forecast, "read_market_csv"), "CSV import must validate user data")
        return forecast.read_market_csv(contents)

    def test_valid_csv_accepts_case_and_whitespace_in_headers(self):
        prices = forecast.demo_market_data().iloc[::-1]
        csv = prices.to_csv(index_label=" date ").replace("Open,High,Low,Close,Volume", " open ,HIGH,low,close,volume")
        imported = self.read(csv.encode())
        self.assertEqual(imported.index[0], pd.Timestamp("2025-01-01"))
        self.assertEqual(imported.index[-1], pd.Timestamp("2025-09-09"))
        self.assertEqual(list(imported.columns), ["Open", "High", "Low", "Close", "Volume"])

    def test_missing_date_column_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "(?i)date"):
            self.read(forecast.demo_market_data().to_csv(index=False).encode())

    def test_duplicate_dates_are_not_silently_dropped(self):
        prices = forecast.demo_market_data()
        csv = pd.concat([prices, prices.iloc[[-1]]]).to_csv(index_label="Date")
        with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
            self.read(csv.encode())

    def test_exact_duplicate_csv_headers_are_rejected_before_pandas_renames_them(self):
        csv = forecast.demo_market_data().to_csv(index_label="Date")
        lines = csv.splitlines()
        duplicate = lines[0] + ",Close\n" + "\n".join(line + ",1" for line in lines[1:])
        with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
            self.read(duplicate.encode())

    def test_binary_file_is_rejected_without_raw_parser_trace(self):
        with self.assertRaisesRegex(ValueError, "(?i)(UTF-8|CSV|text)"):
            self.read(b"\x00\xff\x81")

    def test_large_file_is_rejected_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "(?i)(5 MB|size|large)"):
            self.read(b"x" * (5 * 1024 * 1024 + 1))


class ProviderTests(unittest.TestCase):
    def response(self, payload):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    def test_transport_error_does_not_expose_key_or_request_url(self):
        with patch("forecast.requests.get", side_effect=requests.HTTPError(
            "https://www.alphavantage.co/query?apikey=super-secret-key"
        )):
            try:
                forecast.load_market_data("IBM", "super-secret-key")
            except Exception as error:
                self.assertIsInstance(error, ValueError)
                self.assertNotIn("super-secret-key", str(error))
                self.assertNotIn("https://", str(error))
            else:
                self.fail("A failed request should return a recoverable error")

    def test_provider_notes_do_not_echo_raw_content(self):
        payload = {"Information": "Rate limit. api_key=super-secret-key"}
        with patch("forecast.requests.get", return_value=self.response(payload)):
            with self.assertRaises(ValueError) as context:
                forecast.load_market_data("IBM", "super-secret-key")
        self.assertNotIn("super-secret-key", str(context.exception))
        self.assertRegex(str(context.exception), "(?i)(limit|unavailable|provider)")

    def test_malformed_json_is_recoverable_and_safe(self):
        response = self.response({})
        response.json = lambda: (_ for _ in ()).throw(ValueError("raw secret response"))
        with patch("forecast.requests.get", return_value=response):
            with self.assertRaises(ValueError) as context:
                forecast.load_market_data("IBM", "super-secret-key")
        self.assertNotIn("raw secret", str(context.exception))

    def test_invalid_symbol_is_rejected_before_network(self):
        with patch("forecast.requests.get", side_effect=AssertionError("Unexpected network")):
            with self.assertRaisesRegex(ValueError, "(?i)(symbol|ticker)"):
                forecast.load_market_data("https://invalid.example", "secret")

    def test_empty_key_is_rejected_before_network(self):
        with patch("forecast.requests.get", side_effect=AssertionError("Unexpected network")):
            with self.assertRaisesRegex(ValueError, "(?i)key"):
                forecast.load_market_data("IBM", "  ")

    def test_provider_prices_use_shared_ohlcv_validation(self):
        prices = forecast.demo_market_data()
        prices.iloc[-1, prices.columns.get_loc("Volume")] = -1
        renamed = prices.rename(columns={"Open": "1. open", "High": "2. high", "Low": "3. low", "Close": "4. close", "Volume": "5. volume"})
        renamed.index = renamed.index.strftime("%Y-%m-%d")
        payload = {"Time Series (Daily)": renamed.to_dict(orient="index")}
        with patch("forecast.requests.get", return_value=self.response(payload)):
            with self.assertRaisesRegex(ValueError, "(?i)(volume|valid|data)"):
                forecast.load_market_data("IBM", "secret")


if __name__ == "__main__":
    unittest.main()
