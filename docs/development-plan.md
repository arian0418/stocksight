# StockSight improvement plan

## Goal and design

Keep the existing Python/Streamlit app understandable while making the complete
data → forecast → evaluation workflow usable on the first visit. The default
view is a deterministic, clearly labeled synthetic demo. Use a restrained navy,
teal, and warm-white palette, readable metric cards, descriptive controls, and
separate historical-price and held-out-prediction views.

An analysis is a session snapshot. Changing chart ranges, table length, tabs,
or downloading a CSV must not remove it or call the market-data provider again.
Changing the selected source shows that source's form; the previously loaded
snapshot keeps its original source label until a new analysis succeeds.

## Tasks

1. Capture the original two-test baseline and add behavioral regression tests.
2. Validate daily OHLCV input, safely load provider data, and support a bounded
   CSV import with an example download. Reject malformed/duplicate dates,
   nonfinite numbers, impossible price ranges, and insufficient history.
3. Preserve chronological training and evaluate forecasts on the date of the
   target close. Include the no-change baseline in the chart and CSV. Handle
   zero-volume windows explicitly and keep the newest unknown target available.
4. Build a persistent, responsive dashboard with history range controls,
   backtest metrics and explanations, source/freshness details, recent rows,
   model inputs, and actionable empty/error states.
5. Document setup, data contracts, architecture, and model limitations. Run the
   complete unit/AppTest suite and review the diff before a local commit.

## Verification

- Baseline: `python -m unittest discover -v` — 2 tests passed (2026-09-24).
- RED/GREEN: invalid input, zero-volume history, secret-safe provider failures,
  target-date alignment, chronological evaluation, default demo, rerun
  persistence, source switching, refresh/error recovery, and CSV output.
- Final: full unittest suite, Python compilation, clean diff check, and browser
  review by the coordinating agent. No live provider result is fabricated.

## Scope and limitations

No JavaScript frontend, database, accounts, paid services, trading/execution,
or deployment changes. Daily provider observations may be delayed and are
unadjusted; CSV units and adjustment status come from its author. Future output
means the next supplied market observation, not a calendar promise. The simple
Ridge model and synthetic demo are educational, not investment guidance.
