"""Data loading and chronological evaluation for StockSight."""
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def demo_market_data():
    """Deterministic synthetic prices for exploring the dashboard offline."""
    dates = pd.bdate_range("2025-01-01", periods=180)
    t = np.arange(len(dates), dtype=float)
    close = 100 + 0.12 * t + 3 * np.sin(t / 9)
    return pd.DataFrame({
        "Open": close - 0.4, "High": close + 1.2,
        "Low": close - 1.3, "Close": close,
        "Volume": 900_000 + 120_000 * (1 + np.sin(t / 7))
    }, index=dates)

def load_market_data(symbol, api_key):
    url = "https://www.alphavantage.co/query"
    params = {"function":"TIME_SERIES_DAILY","symbol":symbol,"outputsize":"compact","apikey":api_key}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if "Error Message" in payload:
        raise ValueError("Ticker not found. Check the symbol and try again.")
    if "Note" in payload or "Information" in payload:
        raise ValueError(payload.get("Note") or payload.get("Information"))
    series = payload.get("Time Series (Daily)")
    if not series:
        raise ValueError("The API did not return daily price data.")
    df = pd.DataFrame.from_dict(series, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. volume":"Volume"}).astype(float)
    return df.sort_index()

def build_features(df):
    x = df.copy()
    x["Return_1D"] = x["Close"].pct_change()
    x["Return_5D"] = x["Close"].pct_change(5)
    x["MA_5"] = x["Close"].rolling(5).mean()
    x["MA_20"] = x["Close"].rolling(20).mean()
    x["Volatility_20"] = x["Return_1D"].rolling(20).std()
    x["Volume_Ratio"] = x["Volume"] / x["Volume"].rolling(20).mean()
    x["Target"] = x["Close"].shift(-1)
    # Keep the latest row even though its future target is unknown. It is needed
    # for a genuine next-day forecast; only training/evaluation rows need Target.
    return x.dropna(subset=["Close","Return_1D","Return_5D","MA_5","MA_20","Volatility_20","Volume_Ratio"])

def train_and_test(feature_df):
    features = ["Close","Return_1D","Return_5D","MA_5","MA_20","Volatility_20","Volume_Ratio"]
    model_df = feature_df.dropna(subset=["Target"]).copy()
    cut = int(len(model_df) * .80)
    if cut < 20 or len(model_df) - cut < 5:
        raise ValueError("Not enough usable history to train and test the model.")
    train, test = model_df.iloc[:cut], model_df.iloc[cut:]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[features], train["Target"])
    predictions = model.predict(test[features])
    mae = mean_absolute_error(test["Target"], predictions)
    baseline_mae = mean_absolute_error(test["Target"], test["Close"])
    rmse = np.sqrt(mean_squared_error(test["Target"], predictions))
    actual_direction = np.sign(test["Target"].values - test["Close"].values)
    predicted_direction = np.sign(predictions - test["Close"].values)
    direction_accuracy = (actual_direction == predicted_direction).mean() * 100
    # Retrain on every row whose next-day close is already known, then apply
    # the model to the newest feature row. This avoids labeling an already-known
    # historical target as a "next-day" forecast.
    model.fit(model_df[features], model_df["Target"])
    next_prediction = model.predict(feature_df[features].iloc[[-1]])[0]
    return model, test, predictions, mae, baseline_mae, rmse, direction_accuracy, next_prediction

