"""StockSight's interface. Model and input logic live in forecast.py."""
from datetime import datetime, timezone
from html import escape
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast import analyze_prices, demo_market_data, load_market_data, read_market_csv


st.set_page_config(page_title="StockSight · Forecast lab", page_icon="📈", layout="wide")
st.markdown(f"<style>{Path(__file__).with_name('styles.css').read_text()}</style>", unsafe_allow_html=True)

DEMO = "Demo (synthetic)"
CSV = "Upload CSV"
LIVE = "Alpha Vantage (daily)"


def save_analysis(prices, source, label):
    """Replace the current result only after a complete analysis succeeds."""
    result = analyze_prices(prices)
    result.update(source=source, label=label, loaded_at=datetime.now(timezone.utc))
    st.session_state.analysis = result
    st.session_state.load_error = None


def style_chart(chart, y_title="Price · source units", height=365):
    """Shared readable axes, hover labels, and keyboard-accessible Plotly tools."""
    chart.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=10, r=16, t=18, b=12),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Arial, sans-serif", color="#364b5c", size=12),
        hovermode="x unified", dragmode=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
        yaxis_title=y_title, xaxis_title=None,
    )
    chart.update_xaxes(showgrid=False, zeroline=False)
    chart.update_yaxes(gridcolor="#edf1f3", zeroline=False)
    return chart


def render_history(result):
    prices = result["prices"]
    controls, average_control = st.columns([3, 1])
    with controls:
        history_range = st.radio(
            "History range", ["1 month", "3 months", "6 months", "All history"],
            index=1, horizontal=True, key="history_range",
            help="A display filter only. The model always uses the full loaded history.",
        )
    with average_control:
        show_averages = st.toggle("Moving averages", value=True, key="show_averages")
    months = {"1 month": 1, "3 months": 3, "6 months": 6}
    start = prices.index[-1] - pd.DateOffset(months=months[history_range]) if history_range in months else prices.index[0]
    visible = prices.loc[start:]
    chart = go.Figure()
    chart.add_trace(go.Scatter(
        x=visible.index, y=visible["Close"], name="Daily close", mode="lines",
        line=dict(color="#087f78", width=2.8), fill="tozeroy", fillcolor="rgba(8,127,120,0.06)",
        hovertemplate="%{y:,.2f}<extra>Daily close</extra>",
    ))
    if show_averages:
        for window, color, dash in [(5, "#b68024", "dot"), (20, "#627c9e", "dash")]:
            chart.add_trace(go.Scatter(
                x=visible.index, y=prices["Close"].rolling(window).mean().loc[start:],
                name=f"{window}-observation average", mode="lines",
                line=dict(color=color, width=1.6, dash=dash),
                hovertemplate="%{y:,.2f}<extra>Moving average</extra>",
            ))
    test_start = result["backtest"].index[0]
    chart.add_vrect(x0=max(test_start, visible.index[0]), x1=visible.index[-1], fillcolor="#edf4f1", opacity=0.55, line_width=0, layer="below")
    chart.update_yaxes(range=[visible["Low"].min() * 0.98, visible["High"].max() * 1.02])
    st.plotly_chart(style_chart(chart), width="stretch", config={"displaylogo": False, "scrollZoom": False}, key="history_chart")
    period_change = (visible["Close"].iloc[-1] / visible["Close"].iloc[0] - 1) * 100
    st.caption(
        f"{visible.index[0]:%d %b %Y} – {visible.index[-1]:%d %b %Y} · "
        f"{len(visible):,} observations · Close-to-close change {period_change:+.2f}%. "
        "The shaded period contains held-out forecast dates."
    )


def render_backtest(result):
    table = result["backtest"]
    st.markdown("#### How did the model hold up?")
    st.caption(
        f"Trained on {result['training_rows']} earlier examples. Evaluated on {len(table)} later closes, "
        f"{table.index[0]:%d %b %Y} – {table.index[-1]:%d %b %Y}. Lower error is better."
    )
    first, second, third = st.columns(3)
    first.metric("RMSE", f"{result['rmse']:,.3f}", help="Root mean squared error gives larger misses more weight. Units match the input prices.")
    second.metric("Direction hit rate", f"{result['direction_accuracy']:.1f}%", help="Share of predictions with the correct up, down, or unchanged direction. This is not a profit measure.")
    third.metric("Held-out forecasts", str(len(table)), help="These target closes were excluded from model fitting for the backtest.")
    gap = result["baseline_mae"] - result["mae"]
    if gap > 0:
        st.success(f"The model's average error was {gap:,.3f} lower than the no-change baseline on this test period.")
    elif gap < 0:
        st.warning(f"The model's average error was {abs(gap):,.3f} higher than the no-change baseline. A more complex model did not help here.")
    else:
        st.info("The model and the no-change baseline had the same average error on this test period.")
    chart = go.Figure()
    for column, color, dash in [
        ("Actual close", "#17394b", "solid"),
        ("Model forecast", "#087f78", "dash"),
        ("No-change baseline", "#b68024", "dot"),
    ]:
        chart.add_trace(go.Scatter(
            x=table.index, y=table[column], name=column, mode="lines",
            line=dict(color=color, width=2.2, dash=dash),
            hovertemplate="%{y:,.3f}<extra>" + column + "</extra>",
        ))
    st.plotly_chart(style_chart(chart), width="stretch", config={"displaylogo": False, "scrollZoom": False}, key="backtest_chart")
    st.caption("Each point is dated when the target close was observed. Its prediction used the preceding observation. The baseline simply repeats that preceding close.")
    heading, download = st.columns([2, 1])
    with heading:
        st.markdown("#### Forecast ledger")
        st.caption("Inspect individual misses. Positive error means the model predicted too high.")
    with download:
        file_label = {DEMO: "synthetic-demo", CSV: "uploaded-data", LIVE: "daily-data"}[result["source"]]
        st.download_button(
            "Download full backtest CSV", table.to_csv(date_format="%Y-%m-%d").encode("utf-8"),
            file_name=f"stocksight-{file_label}-backtest.csv", mime="text/csv", width="stretch",
            help=f"Exports all {len(table)} forecasts, including both dates, the baseline, and errors.",
        )
    rows = st.selectbox("Rows to show", [10, 20, 50], key="table_rows")
    recent = table.tail(rows).sort_index(ascending=False).reset_index()
    st.dataframe(
        recent, hide_index=True, width="stretch",
        column_config={
            "Forecast date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Observed date": st.column_config.DateColumn(format="DD MMM YYYY"),
            **{name: st.column_config.NumberColumn(format="%.3f") for name in ["Actual close", "Model forecast", "No-change baseline", "Error"]},
            "Direction correct": st.column_config.CheckboxColumn(),
        },
    )
    st.caption(f"Showing the newest {min(rows, len(table))} of {len(table)} forecasts. The download always contains the full test period.")


def render_model(result):
    latest = result["features"].iloc[-1]
    st.markdown("#### A small model you can understand")
    st.write("Ridge regression learns a regularized linear relationship between seven historical inputs and the next observed close. No news, fundamentals, or future prices are inputs.")
    columns = st.columns(4)
    columns[0].metric("5-observation average", f"{latest['MA_5']:,.2f}")
    columns[1].metric("20-observation average", f"{latest['MA_20']:,.2f}")
    columns[2].metric("5-observation return", f"{latest['Return_5D'] * 100:+.2f}%")
    columns[3].metric("Relative volume", f"{latest['Volume_Ratio']:.2f}×")
    st.caption(f"20-observation return volatility: {latest['Volatility_20'] * 100:.2f}% per observation, or {latest['Volatility_20'] * (252 ** 0.5) * 100:.1f}% annualized using a 252-session convention.")
    with st.expander("Training, evaluation, and the next forecast", expanded=True):
        st.markdown(
            "1. **Build inputs from the past.** Current close, 1- and 5-observation returns, "
            "5- and 20-observation averages, 20-observation return volatility, and relative volume. "
            "The first 20 observations warm up these indicators.\n"
            "2. **Keep time in order.** Fit the scaler and Ridge model on the oldest 80% of labeled "
            "examples. Hold the model fixed while predicting the newer 20%.\n"
            "3. **Check against a simple baseline.** Compare each next-close prediction with the "
            "actual close and a prediction that the price stays unchanged.\n"
            "4. **Refit for the unknown.** After evaluation, fit a separate final model on all "
            "known targets. Use the newest input row for the next-close estimate. That refit does not change the backtest."
        )
    st.markdown("#### Reading the numbers")
    st.markdown(
        "**MAE** is the average absolute miss, in the same units as the prices. "
        "**RMSE** penalizes larger errors more strongly. **Direction hit rate** counts "
        "up, down, and unchanged moves exactly; it does not include fees or measure trading returns."
    )
    st.caption("Windows count observations, not calendar days. A zero-volume window uses a relative-volume input of 0. Uploaded gaps are retained; dates are never filled in.")
    with st.expander("Data quality and limitations"):
        st.write(
            f"{len(result['prices']):,} daily observations, {result['prices'].index[0]:%d %b %Y} – "
            f"{result['prices'].index[-1]:%d %b %Y}. Duplicate dates, missing/nonfinite values, "
            "negative volume, and impossible OHLC ranges are rejected."
        )
        st.write("Provider data uses daily unadjusted prices. Splits and dividends can distort returns. CSV currency, adjustments, and date coverage are the uploader's responsibility; prices are shown in source units.")
        st.write("This is one historical holdout, with no uncertainty interval or transaction-cost model. Synthetic prices are unusually smooth. Good demo metrics are not evidence of market predictability. No exchange calendar is used, so the next observation is not assigned a future date.")


# The initial demo is local. Only explicit form submissions call the provider.
if "analysis" not in st.session_state:
    save_analysis(demo_market_data(), DEMO, "Synthetic market")

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">S</span><div>StockSight<small>FORECAST LAB</small></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">YOUR WORKSPACE</div>', unsafe_allow_html=True)
    source = st.radio("Data source", [DEMO, CSV, LIVE], key="source")
    try:
        if source == DEMO:
            st.caption("A fixed, synthetic series. Explore the complete workflow without an API key.")
            if st.button("Reload synthetic demo", key="reload_demo", type="primary", width="stretch"):
                save_analysis(demo_market_data(), DEMO, "Synthetic market")
        elif source == CSV:
            with st.form("csv_form"):
                upload = st.file_uploader("Daily OHLCV CSV", type=["csv"], help="UTF-8 CSV, up to 5 MB and 10,000 rows. At least 60 observations required.")
                st.caption("Columns: Date, Open, High, Low, Close, Volume. Dates: YYYY-MM-DD.")
                if st.form_submit_button("Analyze uploaded CSV", key="analyze_csv", type="primary", width="stretch"):
                    if upload is None:
                        raise ValueError("Upload a daily OHLCV CSV first, then select Analyze uploaded CSV.")
                    with st.spinner("Validating data and evaluating the model…"):
                        save_analysis(read_market_csv(upload.getvalue()), CSV, "Uploaded dataset")
            st.download_button("Download example CSV", demo_market_data().to_csv(index_label="Date").encode("utf-8"), "stocksight-synthetic-example.csv", "text/csv", width="stretch")
            st.caption("The example CSV contains synthetic data.")
        else:
            with st.form("live_form"):
                symbol = st.text_input("Stock symbol", "IBM", max_chars=20, key="live_symbol")
                api_key = st.text_input("Alpha Vantage API key", type="password", value=os.getenv("ALPHA_VANTAGE_API_KEY", ""), key="api_key")
                st.caption("Your key is sent only to Alpha Vantage and kept in this session. Daily observations may be delayed.")
                if st.form_submit_button("Load daily data", key="analyze_live", type="primary", width="stretch"):
                    with st.spinner("Loading daily data and evaluating the model…"):
                        save_analysis(load_market_data(symbol, api_key), LIVE, symbol.strip().upper())
    except ValueError as error:
        st.session_state.load_error = (source, str(error))
    except Exception:
        # Raw exception strings can contain request URLs and API credentials.
        st.session_state.load_error = (source, "Could not finish this analysis. Check your input and try again, or reload the synthetic demo.")
    st.divider()
    st.markdown("**Built to be questioned.**")
    st.caption("Compare the forecast with a no-change baseline. Every miss stays visible.")
    st.markdown('<div class="sidebar-footnote">EDUCATIONAL USE ONLY<br>Experimental forecasts. Not financial advice.</div>', unsafe_allow_html=True)

result = st.session_state.analysis
prices = result["prices"]
latest = prices.iloc[-1]
previous = prices.iloc[-2]
st.markdown('<div class="workspace-heading"><span>RESEARCH WORKSPACE</span><span>DAILY · OHLCV</span></div>', unsafe_allow_html=True)
st.title("Market overview")
st.markdown('<p class="intro">Follow the price history. See where the forecast gets it right—and where it misses.</p>', unsafe_allow_html=True)
if st.session_state.get("load_error") and st.session_state.load_error[0] == source:
    st.error(st.session_state.load_error[1])
if source != result["source"]:
    st.warning(f"Still showing {result['label']} from {result['source']}. Submit the selected source to replace this analysis.")
if result["source"] == DEMO:
    st.info("Synthetic demo · Generated locally, with fixed dates. These are not real market prices or live predictions.")
elif result["source"] == CSV:
    st.info("Uploaded CSV · Source, currency, and adjustment status are supplied by the uploader. These are not verified live quotes.")
else:
    age = (datetime.now(timezone.utc).date() - prices.index[-1].date()).days
    st.info(f"Alpha Vantage daily data · Latest observation is {age} calendar days old. This is a historical snapshot, not a streaming quote.")
    if age > 7:
        st.warning("The latest observation is more than a week old. Check the symbol and provider coverage before interpreting the next-close estimate.")

st.markdown(
    f'<div class="dataset-heading"><h2>{escape(result["label"])}</h2>'
    f'<span>{len(prices):,} observations · {prices.index[0]:%d %b} – {prices.index[-1]:%d %b %Y}</span></div>',
    unsafe_allow_html=True,
)
metrics = st.columns(4)
metrics[0].metric("Latest close", f"{latest['Close']:,.2f}", f"{(latest['Close'] / previous['Close'] - 1) * 100:+.2f}% vs previous", help="The last close in the loaded data, shown in the source's price units.")
metrics[1].metric("Next-close estimate", f"{result['next_prediction']:,.2f}", f"{(result['next_prediction'] / latest['Close'] - 1) * 100:+.2f}% model move", delta_color="off", help="Experimental point estimate for the observation after the last supplied close. This is not a confidence interval.")
metrics[2].metric("Model MAE", f"{result['mae']:,.3f}", help="Mean absolute error on held-out observations. Lower is better; units match the prices.")
metrics[3].metric("Baseline MAE", f"{result['baseline_mae']:,.3f}", help="Average error when predicting that the next close will equal the preceding close.")
st.caption(f"Prices in source units · Last observation {prices.index[-1]:%d %b %Y} · Next-close estimate follows that observation, not today's date.")
history_tab, backtest_tab, model_tab = st.tabs(["Price history", "Backtest & errors", "Model & data"])
with history_tab:
    with st.container(border=True):
        render_history(result)
with backtest_tab:
    with st.container(border=True):
        render_backtest(result)
with model_tab:
    with st.container(border=True):
        render_model(result)
st.markdown('<div class="page-footer"><span>STOCKSIGHT · TRANSPARENT BY DESIGN</span><span>Explore. Compare. Learn.</span></div>', unsafe_allow_html=True)
st.caption(f"Analysis prepared {result['loaded_at']:%d %b %Y, %H:%M UTC}. Results stay available in this browser session until you load another dataset.")
