# StockSight — Stock Price Forecast Lab

StockSight is an educational stock forecasting and backtesting dashboard. It combines real historical market data, transparent feature engineering, chronological model training, out-of-sample evaluation, and a polished browser interface.

## Features

- Load recent daily OHLCV history from Alpha Vantage.
- Engineer prior-price, return, moving-average, volatility, and relative-volume features.
- Train multivariable linear regression on the older 80% of usable observations.
- Evaluate on the newer 20% rather than randomly shuffling time-series data.
- Report MAE, RMSE, direction accuracy, and 20-day annualized volatility.
- Visualize actual vs. predicted prices in the unseen test period.
- Inspect the latest features and last 10 backtest predictions.
- Produce an experimental next-day forecast.
- Keep the API key in browser Local Storage instead of source control.

## Run

Open `index.html` in a modern browser. Get an Alpha Vantage API key, open **API Settings**, save the key, enter a ticker such as `IBM`, and click **Analyze Stock**.

No Node.js, Python environment, database, or build step is required.

## Model

The model is deliberately understandable. StockSight implements multivariable linear regression directly in JavaScript using a normal-equation system solved with Gaussian elimination and a small diagonal regularization term for numerical stability.

Each training row uses information available before its target close: current close, 1-day return, 5-day return, 5-day moving average, 20-day moving average, recent volatility, and relative volume. The target is the following trading day's close.

The data remains chronological: older observations train the model and newer observations test it. This makes the evaluation closer to the forecasting scenario and avoids randomly mixing later observations into training.

## Metrics

**MAE** is average absolute dollar error. **RMSE** is root mean squared dollar error and penalizes larger misses more. **Direction accuracy** measures how often predicted and actual daily direction agree in the test period. **20D volatility** is an annualized estimate based on recent daily returns.

## Limitations

Stock prices are noisy and are affected by information that historical OHLCV data cannot capture. Historical backtest performance does not guarantee future performance. StockSight is an educational software/ML project, not a trading system, and its output is not financial advice.

Alpha Vantage data availability and request limits depend on the user's API plan.

## Tech

HTML • CSS • Vanilla JavaScript • Canvas API • Alpha Vantage REST API • Local Storage
