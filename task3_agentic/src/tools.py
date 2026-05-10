import numpy as np
import pandas as pd
import yfinance as yf

from schemas import (
    PriceDataSummary,
    VolatilityResult,
)

from tracing import traced_tool


def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(series: pd.Series):
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()

    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd, signal


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20
):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()

    upper = sma + (2 * std)
    lower = sma - (2 * std)

    return upper, lower


@traced_tool
def get_price_data(
    ticker: str,
    period: str = "2y"
):
    stock = yf.Ticker(ticker)

    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    df = df.dropna()

    close = df["Close"]

    # Indicators
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()

    df["RSI_14"] = compute_rsi(close)

    macd, macd_signal = compute_macd(close)

    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd_signal

    bb_upper, bb_lower = compute_bollinger_bands(close)

    df["BB_UPPER"] = bb_upper
    df["BB_LOWER"] = bb_lower

    latest = df.iloc[-1]

    # Momentum Logic
    if (
        latest["SMA_50"] > latest["SMA_200"]
        and latest["RSI_14"] < 70
        and latest["MACD"] > latest["MACD_SIGNAL"]
    ):
        momentum_signal = "Bullish"

    elif (
        latest["SMA_50"] < latest["SMA_200"]
        and latest["RSI_14"] > 30
        and latest["MACD"] < latest["MACD_SIGNAL"]
    ):
        momentum_signal = "Bearish"

    else:
        momentum_signal = "Neutral"

    summary = PriceDataSummary(
        ticker=ticker,
        current_price=round(latest["Close"], 2),
        sma_50=round(latest["SMA_50"], 2),
        sma_200=round(latest["SMA_200"], 2),
        rsi_14=round(latest["RSI_14"], 2),
        macd=round(latest["MACD"], 2),
        macd_signal=round(latest["MACD_SIGNAL"], 2),
        bollinger_upper=round(latest["BB_UPPER"], 2),
        bollinger_lower=round(latest["BB_LOWER"], 2),
        momentum_signal=momentum_signal,
    )

    return {
        "summary": summary.model_dump(),
        "dataframe": df.tail(30).reset_index().to_dict(
            orient="records"
        ),
    }


@traced_tool
def calculate_volatility(
    ticker: str,
    window: int = 30
):
    stock = yf.Ticker(ticker)

    df = stock.history(period="1y")

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    returns = df["Close"].pct_change().dropna()

    rolling_std = returns.rolling(window).std()

    annualized_volatility = (
        rolling_std.iloc[-1] * np.sqrt(252)
    )

    result = VolatilityResult(
        ticker=ticker,
        annualized_volatility=round(
            annualized_volatility,
            4
        ),
    )

    return result.model_dump()