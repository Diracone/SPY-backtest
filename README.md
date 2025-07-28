# SPY-backtest
A machine learning-based strategy selector for SPY (S&amp;P 500 ETF), using RSI, SMA crossover, and momentum signals. Trains a Random Forest on historical data to pick the best signal daily. Includes backtest vs buy-and-hold, Sharpe ratio, and drawdown analysis. 

# ML Strategy Selector for SPY (2020–2024)

This notebook uses a machine learning approach to dynamically choose between multiple technical trading signals for SPY (the S&P 500 ETF). It compares a model-driven strategy to a simple buy-and-hold baseline.

## What it does

- Downloads daily SPY data from 2020 to 2025 using `yfinance`
- Computes indicators:
  - RSI (14-day)
  - SMA20 and SMA50
  - 20-day momentum
  - Rolling volatility
  - Simulated VIX and Fed Funds Rate
- Labels each row based on which signal would have produced the best 10-day return
- Trains a `RandomForestClassifier` on 2020–2023 data to predict the best strategy
- Applies the model to 2024 data for out-of-sample backtesting
- Tracks capital growth based on whether a trade is entered each day or not

## Strategy logic

The model predicts which of these four options is optimal:

- `RSI`: if RSI < 40, buy
- `MA`: if SMA20 > SMA50, buy
- `Momentum`: if 20-day return > 1%, buy
- `Cash`: otherwise stay out

Only one signal is selected per day, and trades are executed accordingly.

## Evaluation

- Capital curve of ML strategy vs SPY buy & hold
- Sharpe ratio of model strategy
- Maximum drawdown
- Number of active trading days

## Final output

You get a side-by-side plot of strategy performance and printouts like:

- Final capital (normalized to 1.0 start)
- Total return in %
- Sharpe ratio (annualized)
- Max drawdown %
- Active trading days out of total

## Requirements

Install these packages (if not already available):

```bash
pip install yfinance scikit-learn pandas numpy matplotlib
