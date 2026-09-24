"""Small, testable data-loading and chronological forecasting functions."""
from io import StringIO
import csv
import re

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
FEATURE_COLUMNS = [
    "Close", "Return_1D", "Return_5D", "MA_5", "MA_20",
    "Volatility_20", "Volume_Ratio",
]
MIN_HISTORY = 60
MAX_HISTORY = 10_000
MAX_CSV_BYTES = 5 * 1024 * 1024


def demo_market_data():
    """Deterministic synthetic prices; these dates are deliberately fixed."""
    dates = pd.bdate_range("2025-01-01", periods=180, name="Date")
    time = np.arange(len(dates), dtype=float)
    close = 100 + 0.12 * time + 3 * np.sin(time / 9)
    return pd.DataFrame({
        "Open": close - 0.4, "High": close + 1.2,
        "Low": close - 1.3, "Close": close,
        "Volume": 900_000 + 120_000 * (1 + np.sin(time / 7)),
    }, index=dates)


def validate_market_data(data):
    """Return a sorted copy, rejecting bad rows instead of silently dropping them."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Price data must be a table with daily OHLCV columns.")
    frame = data.copy()
    frame.columns = [str(column).strip().title() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names found. Keep one of each OHLCV column.")
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing) + ".")
    if len(frame) < MIN_HISTORY:
        raise ValueError(f"Use at least {MIN_HISTORY} daily observations to train and evaluate the model.")
    if len(frame) > MAX_HISTORY:
        raise ValueError(f"Use no more than {MAX_HISTORY:,} daily observations.")
    if pd.api.types.is_numeric_dtype(frame.index.dtype):
        raise ValueError("Dates must be calendar dates, such as 2025-01-31.")
    try:
        dates = pd.to_datetime(frame.index, format="mixed", errors="coerce", utc=True)
    except (TypeError, ValueError):
        raise ValueError("Some dates could not be read. Use YYYY-MM-DD dates.") from None
    if dates.isna().any():
        raise ValueError("Some dates are missing or invalid. Use YYYY-MM-DD dates.")
    # A daily series must contain at most one observation per calendar day.
    frame.index = dates.tz_convert(None).normalize().rename("Date")
    if frame.index.duplicated().any():
        raise ValueError("Duplicate dates found. Keep exactly one observation per day.")
    frame = frame[PRICE_COLUMNS]
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite numeric values with no missing cells.")
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Open, high, low, and close prices must be positive.")
    if (frame["Volume"] < 0).any():
        raise ValueError("Volume must be zero or positive.")
    bad_high = frame["High"] < frame[["Open", "Low", "Close"]].max(axis=1)
    bad_low = frame["Low"] > frame[["Open", "High", "Close"]].min(axis=1)
    if (bad_high | bad_low).any():
        raise ValueError("Invalid price range: low must be at or below open/close, and high at or above them.")
    return frame.sort_index().astype(float)


def read_market_csv(contents):
    """Read a bounded UTF-8 daily CSV, using the same rules as provider data."""
    if len(contents) > MAX_CSV_BYTES:
        raise ValueError("The CSV is too large. Use a file of 5 MB or less.")
    try:
        text = contents.decode("utf-8-sig")
        # pandas renames repeated headers (Close -> Close.1), so inspect the raw
        # header first to avoid silently selecting an ambiguous price column.
        header = next(csv.reader(StringIO(text)), [])
    except (UnicodeError, csv.Error):
        raise ValueError("Could not read that CSV. Save it as a comma-separated UTF-8 text file.") from None
    normalized_header = [column.strip().title() for column in header]
    if len(normalized_header) != len(set(normalized_header)):
        raise ValueError("Duplicate column names found in the CSV.")
    try:
        frame = pd.read_csv(StringIO(text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        raise ValueError("Could not read that CSV. Save it as a comma-separated UTF-8 text file.") from None
    frame.columns = [str(column).strip().title() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names found in the CSV.")
    if "Date" not in frame.columns:
        raise ValueError("The CSV needs a Date column, plus Open, High, Low, Close, and Volume.")
    return validate_market_data(frame.set_index("Date"))


def load_market_data(symbol, api_key):
    """Fetch daily observations. Never expose provider text or credential URLs."""
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.:-]{0,19}", symbol):
        raise ValueError("Enter a valid stock symbol using letters, numbers, dots, colons, or hyphens.")
    if not api_key.strip():
        raise ValueError("Enter your Alpha Vantage API key to load daily data.")
    params = {
        "function": "TIME_SERIES_DAILY", "symbol": symbol,
        "outputsize": "compact", "apikey": api_key.strip(),
    }
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        raise ValueError("Could not reach the data provider. Check your connection and try again, or use the demo/CSV.") from None
    try:
        payload = response.json()
    except ValueError:
        raise ValueError("The data provider returned an unreadable response. Try again later or use a CSV.") from None
    if not isinstance(payload, dict):
        raise ValueError("The data provider returned an unexpected response. Try again later.")
    if "Error Message" in payload:
        raise ValueError("Ticker not found. Check the symbol and try again.")
    if "Note" in payload or "Information" in payload:
        raise ValueError("Daily data is unavailable from the provider. Check your key and plan limits, or try again later.")
    series = payload.get("Time Series (Daily)")
    if not isinstance(series, dict) or not series or not all(isinstance(row, dict) for row in series.values()):
        raise ValueError("The provider did not return daily price data. Check the symbol or try a CSV.")
    frame = pd.DataFrame.from_dict(series, orient="index").rename(columns={
        "1. open": "Open", "2. high": "High", "3. low": "Low",
        "4. close": "Close", "5. volume": "Volume",
    })
    return validate_market_data(frame)


def build_features(data):
    """Every feature uses only the current observation and earlier history."""
    frame = validate_market_data(data)
    frame["Return_1D"] = frame["Close"].pct_change(fill_method=None)
    frame["Return_5D"] = frame["Close"].pct_change(5, fill_method=None)
    frame["MA_5"] = frame["Close"].rolling(5).mean()
    frame["MA_20"] = frame["Close"].rolling(20).mean()
    frame["Volatility_20"] = frame["Return_1D"].rolling(20).std()
    average_volume = frame["Volume"].rolling(20).mean()
    # No volume in the whole window means no volume signal, represented as 0.
    frame["Volume_Ratio"] = frame["Volume"].div(average_volume.mask(average_volume == 0)).fillna(0)
    frame["Target"] = frame["Close"].shift(-1)
    frame["TargetDate"] = pd.Series(frame.index, index=frame.index).shift(-1)
    # The first 20 observations warm up the indicators; the newest row remains
    # available even though its future target is not yet known.
    frame = frame.iloc[20:].copy()
    if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Price/volume ranges are too extreme to calculate finite model features.")
    return frame


def train_and_test(feature_df):
    """Evaluate on the newest 20%, then refit for one genuinely unknown target."""
    model_df = feature_df.dropna(subset=["Target"]).copy()
    cut = int(len(model_df) * 0.80)
    if cut < 20 or len(model_df) - cut < 5:
        raise ValueError("Not enough usable history to train and test the model.")
    train, test = model_df.iloc[:cut], model_df.iloc[cut:]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[FEATURE_COLUMNS], train["Target"])
    predictions = model.predict(test[FEATURE_COLUMNS])
    mae = mean_absolute_error(test["Target"], predictions)
    baseline_mae = mean_absolute_error(test["Target"], test["Close"])
    rmse = np.sqrt(mean_squared_error(test["Target"], predictions))
    actual_direction = np.sign(test["Target"].to_numpy() - test["Close"].to_numpy())
    predicted_direction = np.sign(predictions - test["Close"].to_numpy())
    direction_accuracy = (actual_direction == predicted_direction).mean() * 100
    # This refit is used only for the next observation, never for test metrics.
    model.fit(model_df[FEATURE_COLUMNS], model_df["Target"])
    next_prediction = model.predict(feature_df[FEATURE_COLUMNS].iloc[[-1]])[0]
    return model, test, predictions, mae, baseline_mae, rmse, direction_accuracy, next_prediction


def analyze_prices(data):
    """Build one reusable snapshot for rendering and full-period CSV export."""
    prices = validate_market_data(data)
    features = build_features(prices)
    _, test, predictions, mae, baseline_mae, rmse, direction_accuracy, next_prediction = train_and_test(features)
    backtest = pd.DataFrame({
        "Observed date": test.index.to_numpy(),
        "Actual close": test["Target"].to_numpy(),
        "Model forecast": predictions,
        "No-change baseline": test["Close"].to_numpy(),
        "Error": predictions - test["Target"].to_numpy(),
        "Direction correct": np.sign(predictions - test["Close"].to_numpy())
        == np.sign(test["Target"].to_numpy() - test["Close"].to_numpy()),
    }, index=pd.DatetimeIndex(test["TargetDate"], name="Forecast date"))
    training_rows = len(features.dropna(subset=["Target"])) - len(test)
    return {
        "prices": prices, "features": features, "backtest": backtest,
        "mae": float(mae), "baseline_mae": float(baseline_mae),
        "rmse": float(rmse), "direction_accuracy": float(direction_accuracy),
        "next_prediction": float(next_prediction),
        "next_observed_date": features.index[-1],
        "train_start": features.index[0],
        "train_end": features.iloc[training_rows - 1]["TargetDate"],
        "training_rows": training_rows,
    }
