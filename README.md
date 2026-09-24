# StockSight — Python Stock Forecast Lab

StockSight is an educational stock forecasting and backtesting dashboard built around **Python**. It loads real historical stock data, engineers time-series features, trains an understandable regression model, evaluates it on later unseen observations, and presents the results in a Streamlit dashboard.

## Why I Built This

I was interested in stocks and wanted to understand how programming, data analysis, and machine learning could be used to study historical market data instead of only looking at a normal stock chart.

I built StockSight to experiment with that idea myself. It gave me a way to explore market trends, volatility, model performance, and simple next-day forecasts while learning how a machine-learning model can be trained and evaluated on time-series data.

## Stack

- Python
- Streamlit
- pandas / NumPy
- scikit-learn
- Plotly
- Alpha Vantage REST API

The project intentionally does not require JavaScript, Node.js, Docker, or a database.

## What it does

StockSight loads daily OHLCV data for a ticker, calculates 1-day and 5-day returns, 5-day and 20-day moving averages, 20-day volatility, and relative volume. It scales features using the training period only, then trains a Ridge regression model chronologically: the older 80% of usable observations is training data and the newer 20% is the backtest.

The dashboard shows the latest close, experimental next-day forecast, MAE, a naive no-change baseline MAE, RMSE, direction accuracy, annualized recent volatility, an actual-vs-predicted chart, model inputs, and recent backtest predictions.

## Windows setup

Open Command Prompt inside the project folder and run:

```
python -m pip install -r requirements.txt
```

Then start the app:

```
python -m streamlit run app.py
```

Streamlit will open StockSight in your browser. The app starts in **Demo (synthetic)** mode: click **Analyze stock** to explore a deterministic example without an API key. For real data, choose **Alpha Vantage (live)**, paste your API key into the sidebar, enter a ticker such as `IBM`, `AAPL`, or `NVDA`, and click **Analyze stock**. Demo data is synthetic and must not be treated as a real backtest.

You can optionally set an environment variable named `ALPHA_VANTAGE_API_KEY` instead of pasting the key into the app.

## Model design

The model uses scikit-learn's `StandardScaler` and Ridge regression pipeline. The scaler is fitted on the training data only, avoiding test-period leakage. Ridge regression is the model used for predictions. Ridge is a linear regression model with regularization, which makes it an understandable baseline for this project.

The model is evaluated on later observations rather than a randomly shuffled test set. The baseline predicts tomorrow's close is today's close, so a visitor can see whether the model actually improves on that simple guess. This avoids mixing earlier and later stock observations in the evaluation.

## Limitations

Stock prices are noisy and are affected by information that historical OHLCV data does not contain. Backtest performance does not guarantee future performance. StockSight is an educational software and machine-learning project, not a trading system, and its output is not financial advice.

Alpha Vantage request limits and data availability depend on the user's API plan.

## Tests

Run `python -m unittest discover -v`. GitHub Actions runs the same test on each pull request.
