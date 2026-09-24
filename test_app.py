"""User-visible regressions for the real Streamlit app (no web service needed)."""
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def dashboard(self):
        dashboard = AppTest.from_file("app.py", default_timeout=20).run()
        self.assertEqual(len(dashboard.exception), 0)
        return dashboard

    def test_first_visit_contains_demo_results_without_a_key(self):
        dashboard = self.dashboard()
        self.assertEqual(len(dashboard.error), 0)
        self.assertGreaterEqual(len(dashboard.metric), 8)
        self.assertEqual(len(dashboard.get("plotly_chart")), 2)
        self.assertEqual(len(dashboard.text_input), 0)
        self.assertTrue(any("synthetic" in item.value.lower() for item in dashboard.info))

    def test_results_survive_chart_table_and_download_reruns(self):
        dashboard = self.dashboard()
        self.assertGreater(len(dashboard.metric), 0, "Demo should be ready on first load")
        latest_close = dashboard.metric[0].value
        dashboard.radio(key="history_range").set_value("All history").run()
        dashboard.selectbox(key="table_rows").set_value(20).run()
        dashboard.toggle(key="show_averages").set_value(False).run()
        dashboard.run()  # A download is also a rerun; no analyze button is pressed.
        self.assertEqual(len(dashboard.exception), 0)
        self.assertEqual(dashboard.metric[0].value, latest_close)
        self.assertEqual(len(dashboard.get("plotly_chart")), 2)
        self.assertGreater(len(dashboard.get("download_button")), 0)
        self.assertEqual(len(dashboard.dataframe[0].value), 20)

    def test_switching_to_live_does_not_request_data_or_relabel_demo(self):
        dashboard = self.dashboard()
        self.assertGreater(len(dashboard.metric), 0, "Demo should be ready on first load")
        latest_close = dashboard.metric[0].value
        with patch("forecast.requests.get", side_effect=AssertionError("Unexpected request")):
            dashboard.radio(key="source").set_value("Alpha Vantage (daily)").run()
        self.assertEqual(len(dashboard.exception), 0)
        self.assertEqual(dashboard.metric[0].value, latest_close)
        self.assertTrue(any("synthetic" in item.value.lower() for item in dashboard.info))
        self.assertEqual(len(dashboard.text_input), 2)

    def test_missing_live_key_keeps_previous_results_and_explains_recovery(self):
        dashboard = self.dashboard()
        self.assertGreater(len(dashboard.metric), 0, "Demo should be ready on first load")
        latest_close = dashboard.metric[0].value
        dashboard.radio(key="source").set_value("Alpha Vantage (daily)").run()
        dashboard.text_input(key="api_key").set_value("")
        dashboard.button(key="analyze_live").click().run()
        self.assertEqual(len(dashboard.exception), 0)
        self.assertTrue(any("key" in item.value.lower() for item in dashboard.error))
        self.assertEqual(dashboard.metric[0].value, latest_close)

    def test_provider_failure_never_displays_credentials(self):
        dashboard = self.dashboard()
        self.assertGreater(len(dashboard.metric), 0, "Demo should be ready on first load")
        dashboard.radio(key="source").set_value("Alpha Vantage (daily)").run()
        dashboard.text_input(key="api_key").set_value("private-test-key")
        with patch("forecast.requests.get", side_effect=requests.HTTPError(
            "https://example.invalid?apikey=private-test-key"
        )):
            dashboard.button(key="analyze_live").click().run()
        self.assertEqual(len(dashboard.exception), 0)
        self.assertGreater(len(dashboard.error), 0)
        self.assertNotIn("private-test-key", " ".join(item.value for item in dashboard.error))
        self.assertGreater(len(dashboard.metric), 0)

    def test_missing_csv_explains_required_upload_and_preserves_demo(self):
        dashboard = self.dashboard()
        self.assertGreater(len(dashboard.metric), 0, "Demo should be ready on first load")
        dashboard.radio(key="source").set_value("Upload CSV").run()
        dashboard.button(key="analyze_csv").click().run()
        self.assertEqual(len(dashboard.exception), 0)
        self.assertTrue(any("upload" in item.value.lower() for item in dashboard.error))
        self.assertGreater(len(dashboard.metric), 0)

    def test_reload_demo_recovers_after_an_input_error(self):
        dashboard = self.dashboard()
        self.assertGreater(len(dashboard.metric), 0, "Demo should be ready on first load")
        dashboard.radio(key="source").set_value("Upload CSV").run()
        dashboard.button(key="analyze_csv").click().run()
        self.assertGreater(len(dashboard.error), 0)
        dashboard.radio(key="source").set_value("Demo (synthetic)").run()
        dashboard.button(key="reload_demo").click().run()
        self.assertEqual(len(dashboard.error), 0)
        self.assertEqual(len(dashboard.exception), 0)
        self.assertEqual(len(dashboard.get("plotly_chart")), 2)


if __name__ == "__main__":
    unittest.main()
