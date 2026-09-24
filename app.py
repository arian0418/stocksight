import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from forecast import demo_market_data, load_market_data, build_features, train_and_test

st.set_page_config(page_title="StockSight", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp {background: #070b10; color: #edf4f8;}
[data-testid="stSidebar"] {background:#090e14; border-right:1px solid #1b2733;}
.block-container {max-width: 1250px; padding-top: 2rem;}
div[data-testid="stMetric"] {background:#0d141c; border:1px solid #1b2733; padding:18px; border-radius:10px;}
h1,h2,h3 {letter-spacing:-.02em;}
.small-note {color:#718293;font-size:.78rem}
.badge {display:inline-block;border:1px solid #315778;background:#10283c;color:#9ed0ff;padding:5px 9px;border-radius:5px;font-size:.7rem;letter-spacing:.08em}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 📈 STOCKSIGHT")
st.sidebar.caption("FORECAST LAB")
st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.caption("A transparent educational stock forecasting project built with Python and machine learning.")
st.sidebar.markdown("---")
data_source = st.sidebar.radio("Data source", ["Demo (synthetic)", "Alpha Vantage (live)"])
api_key = st.sidebar.text_input("Alpha Vantage API key", type="password", value=os.getenv("ALPHA_VANTAGE_API_KEY", ""))
st.sidebar.caption("Your key is used for this session and is not written to the repository.")
st.sidebar.markdown("---")
st.sidebar.warning("Educational project only. Forecasts are experimental and are not financial advice.")

st.markdown('<span class="badge">PYTHON • MACHINE LEARNING • BACKTESTING</span>', unsafe_allow_html=True)
st.title("StockSight")
st.markdown('<p class="small-note">Explore a stock forecasting pipeline that shows its errors instead of hiding them.</p>', unsafe_allow_html=True)

left, right = st.columns([3,1])
with left:
    if data_source == "Alpha Vantage (live)":
        symbol = st.text_input("Stock symbol", "IBM", max_chars=10).strip().upper()
    else:
        symbol = "DEMO"
        st.caption("Synthetic example: no ticker or API key required.")
with right:
    st.write("")
    st.write("")
    analyze = st.button("Analyze stock →", type="primary", width="stretch")

if analyze:
    if data_source == "Alpha Vantage (live)" and not api_key:
        st.error("Enter your Alpha Vantage API key in the sidebar first.")
        st.stop()
    if not symbol:
        st.error("Enter a stock ticker.")
        st.stop()
    try:
        with st.spinner("Loading market data and training the model..."):
            prices = demo_market_data() if data_source == "Demo (synthetic)" else load_market_data(symbol, api_key)
            feature_df = build_features(prices)
            model, test, predictions, mae, baseline_mae, rmse, direction_accuracy, next_prediction = train_and_test(feature_df)
        latest = prices.iloc[-1]
        previous = prices.iloc[-2]
        daily_change = (latest["Close"] / previous["Close"] - 1) * 100
        forecast_change = (next_prediction / latest["Close"] - 1) * 100
        volatility = prices["Close"].pct_change().tail(20).std() * np.sqrt(252) * 100

        st.markdown(f"### {symbol if data_source == 'Alpha Vantage (live)' else 'DEMO · SYNTHETIC DATA'} <span class='small-note'>• latest observation {prices.index[-1].date()}</span>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Latest close", f"${latest['Close']:,.2f}", f"{daily_change:+.2f}%")
        c2.metric("Next-day forecast", f"${next_prediction:,.2f}", f"{forecast_change:+.2f}% model move")
        c3.metric("MAE", f"${mae:,.2f}")
        c4.metric("RMSE", f"${rmse:,.2f}")
        st.caption(f"Naive baseline MAE (predict the next close equals today's close): ${baseline_mae:,.2f}. "
                   + ("The model beat this baseline on the test period." if mae < baseline_mae else "The model did not beat this baseline on the test period."))
        c5,c6 = st.columns(2)
        c5.metric("Direction accuracy", f"{direction_accuracy:.1f}%")
        c6.metric("20-day annualized volatility", f"{volatility:.1f}%")

        st.subheader("Out-of-sample backtest")
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=test.index, y=test["Target"], name="Actual next close", mode="lines"))
        chart.add_trace(go.Scatter(x=test.index, y=predictions, name="Model prediction", mode="lines"))
        chart.add_trace(go.Scatter(x=test.index, y=test["Close"], name="Naive baseline", mode="lines", line=dict(dash="dot")))
        chart.update_layout(template="plotly_dark", height=430, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor="#0d141c", plot_bgcolor="#0d141c", xaxis_title="", yaxis_title="Price ($)")
        st.plotly_chart(chart, width="stretch")

        st.subheader("Latest model inputs")
        last = feature_df.iloc[-1]
        f1,f2,f3,f4 = st.columns(4)
        f1.metric("5-day MA", f"${last['MA_5']:,.2f}")
        f2.metric("20-day MA", f"${last['MA_20']:,.2f}")
        f3.metric("5-day return", f"{last['Return_5D']*100:+.2f}%")
        f4.metric("Relative volume", f"{last['Volume_Ratio']:.2f}×")

        st.subheader("Recent predictions")
        recent = pd.DataFrame({"Actual":test["Target"], "Predicted":predictions, "Previous close":test["Close"]}).tail(10)
        recent["Error"] = recent["Predicted"] - recent["Actual"]
        recent["Direction correct"] = np.sign(recent["Predicted"]-recent["Previous close"]) == np.sign(recent["Actual"]-recent["Previous close"])
        st.dataframe(recent.style.format({"Actual":"${:.2f}","Predicted":"${:.2f}","Previous close":"${:.2f}","Error":"${:+.2f}"}), width="stretch")

        with st.expander("How the model works"):
            st.write("StockSight creates features from historical OHLCV data, including recent returns, 5- and 20-day moving averages, 20-day volatility, and relative volume. It keeps the observations in time order, scales each feature using only the training observations, trains a Ridge regression model on the older 80%, and evaluates it on the newer 20%. The naive baseline predicts that the next close equals the current close. MAE, baseline MAE, RMSE, direction accuracy, and the prediction chart are calculated from the same unseen test period.")
            st.info("A historical backtest is not proof that a stock can be predicted reliably. News, earnings, macroeconomic events, and many other factors are not represented by this simple model.")
    except (requests.RequestException, ValueError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
else:
    st.info("Run the synthetic demo without an API key, or choose Alpha Vantage for real historical data.")
    st.markdown("#### Pipeline")
    a,b,c,d = st.columns(4)
    a.markdown("**01 — Market data**\n\nDaily OHLCV history")
    b.markdown("**02 — Features**\n\nReturns, averages, volatility")
    c.markdown("**03 — Model**\n\nRidge regression")
    d.markdown("**04 — Evaluate**\n\nChronological backtest")
