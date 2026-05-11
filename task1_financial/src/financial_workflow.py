import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK1_DIR = PROJECT_ROOT / "task1_financial"
TASK3_DIR = PROJECT_ROOT / "task3_agentic"
OUTPUTS_DIR = TASK1_DIR / "outputs"

# Task 3 already contains production-style tools for technical indicators,
# news, sentiment, schemas, and tracing. Adding task3_agentic to sys.path lets
# this notebook reuse that tested logic instead of copying another version into
# Task 1.
if str(TASK3_DIR) not in sys.path:
    sys.path.insert(0, str(TASK3_DIR))

from task3_agentic.src.schemas import PriceDataSummary, VolatilityResult  # noqa: E402
from task3_agentic.src.tools import (  # noqa: E402
    calculate_volatility,
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
    get_news,
    get_price_data,
    llm_sentiment,
)


JSONDict = Dict[str, Any]


class RiskInterpretation(BaseModel):
    """Human-readable risk interpretation for the notebook summary."""

    volatility_level: str
    momentum_view: str
    sentiment_view: str
    key_risks: List[str]


class InvestmentOutlook(BaseModel):
    """
    Structured final output for Task 1.

    This is intentionally small. Task 1 is a notebook analysis, not the full
    multi-agent Task 3 workflow, so the schema captures the final answer a
    reviewer needs without creating another orchestration framework.
    """

    ticker: str
    overall_view: str
    risk_level: str
    technical_summary: str
    sentiment_summary: str
    risk_interpretation: RiskInterpretation
    practical_next_steps: List[str]
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def normalize_ticker(ticker: str) -> str:
    """Normalize user input once so file names and API calls are consistent."""

    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        raise ValueError("Ticker must not be empty.")
    return clean_ticker


def _safe_call(step_name: str, fallback: Any, func, *args, **kwargs) -> Any:
    """
    Run an external-data step without breaking the whole notebook.

    Notebook workflows are exploratory. A news endpoint or LLM call can fail
    because of free-tier limits, network issues, or missing API keys. Returning
    a visible fallback keeps the analysis explainable instead of stopping in
    the middle.
    """

    try:
        print(f"[task1] {step_name}...")
        return func(*args, **kwargs)
    except Exception as exc:
        print(f"[task1] {step_name} failed; using fallback. Error: {exc}")
        return fallback


def load_market_dataframe(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Retrieve price history and add technical indicators for visualization.

    We reuse Task 3's indicator functions so RSI, MACD, and Bollinger Band
    logic has one source of truth. This avoids a common assessment mistake:
    implementing the same financial formula differently in two tasks.
    """

    normalized_ticker = normalize_ticker(ticker)
    stock = yf.Ticker(normalized_ticker)
    dataframe = stock.history(period=period).dropna()

    if dataframe.empty:
        raise ValueError(f"No market data found for ticker: {normalized_ticker}")

    close = dataframe["Close"]
    dataframe["SMA_50"] = close.rolling(50).mean()
    dataframe["SMA_200"] = close.rolling(200).mean()
    dataframe["RSI_14"] = compute_rsi(close)

    macd, macd_signal = compute_macd(close)
    dataframe["MACD"] = macd
    dataframe["MACD_SIGNAL"] = macd_signal

    bb_upper, bb_lower = compute_bollinger_bands(close)
    dataframe["BB_UPPER"] = bb_upper
    dataframe["BB_LOWER"] = bb_lower

    return dataframe


def collect_financial_evidence(
    ticker: str,
    *,
    period: str = "2y",
) -> JSONDict:
    """
    Collect structured evidence for the Task 1 notebook.

    The returned dictionary is notebook-friendly: every intermediate result is
    visible and can be displayed directly. It also keeps tool outputs separate
    from interpretation, which is useful for explaining production workflows.
    """

    normalized_ticker = normalize_ticker(ticker)
    dataframe = _safe_call(
        "Retrieving market dataframe",
        pd.DataFrame(),
        load_market_dataframe,
        normalized_ticker,
        period,
    )

    price_data = _safe_call(
        "Retrieving structured price summary",
        {"summary": {}, "dataframe": []},
        get_price_data,
        normalized_ticker,
        period,
    )

    volatility = _safe_call(
        "Calculating annualized volatility",
        {"ticker": normalized_ticker, "annualized_volatility": None},
        calculate_volatility,
        normalized_ticker,
    )

    news_items = _safe_call(
        "Retrieving recent news",
        [],
        get_news,
        normalized_ticker,
        10,
    )
    headlines = [
        item["title"]
        for item in news_items
        if isinstance(item, Mapping) and item.get("title")
    ]

    sentiment = _safe_call(
        "Classifying headline sentiment",
        {"results": [], "overall_sentiment_score": 0.0},
        llm_sentiment,
        headlines,
    )

    return {
        "ticker": normalized_ticker,
        "dataframe": dataframe,
        "price_summary": price_data.get("summary", {}),
        "volatility": volatility,
        "news": news_items,
        "sentiment": sentiment,
    }


def interpret_risk(
    *,
    price_summary: Mapping[str, Any],
    volatility: Mapping[str, Any],
    sentiment: Mapping[str, Any],
) -> RiskInterpretation:
    """
    Convert numeric/tool outputs into plain-English risk interpretation.

    This is deliberately rule-based rather than another LLM call. For a Task 1
    notebook, deterministic interpretation is easier to audit and explain.
    """

    annualized_volatility = volatility.get("annualized_volatility")
    sentiment_score = sentiment.get("overall_sentiment_score", 0.0)
    momentum_signal = str(price_summary.get("momentum_signal", "Unknown"))

    if annualized_volatility is None:
        volatility_level = "unknown"
    elif annualized_volatility < 0.20:
        volatility_level = "low"
    elif annualized_volatility < 0.40:
        volatility_level = "moderate"
    else:
        volatility_level = "high"

    if sentiment_score > 0.15:
        sentiment_view = "positive"
    elif sentiment_score < -0.15:
        sentiment_view = "negative"
    else:
        sentiment_view = "neutral"

    key_risks = []
    if volatility_level in {"high", "unknown"}:
        key_risks.append(
            "Volatility is elevated or unavailable, so position sizing matters."
        )
    if momentum_signal.lower() == "bearish":
        key_risks.append("Technical momentum is bearish based on current indicators.")
    if sentiment_view == "negative":
        key_risks.append("Recent news sentiment is negative.")
    if not key_risks:
        key_risks.append(
            "No single severe risk was detected, but market and company-specific risk remain."
        )

    return RiskInterpretation(
        volatility_level=volatility_level,
        momentum_view=momentum_signal,
        sentiment_view=sentiment_view,
        key_risks=key_risks,
    )


def generate_investment_outlook(evidence: Mapping[str, Any]) -> InvestmentOutlook:
    """
    Generate a structured final investment/risk outlook.

    The final outlook synthesizes Task 3 tool outputs but remains Task
    1-specific: concise, deterministic, and designed for notebook explanation.
    """

    ticker = str(evidence["ticker"])
    price_summary = evidence.get("price_summary", {})
    volatility = evidence.get("volatility", {})
    sentiment = evidence.get("sentiment", {})
    risk = interpret_risk(
        price_summary=price_summary,
        volatility=volatility,
        sentiment=sentiment,
    )

    momentum_signal = risk.momentum_view
    risk_level = (
        "high"
        if risk.volatility_level == "high" or risk.sentiment_view == "negative"
        else "moderate"
        if risk.volatility_level in {"moderate", "unknown"}
        else "low-to-moderate"
    )

    if momentum_signal.lower() == "bullish" and risk.sentiment_view != "negative":
        overall_view = "constructive but risk-aware"
    elif momentum_signal.lower() == "bearish" or risk.sentiment_view == "negative":
        overall_view = "cautious"
    else:
        overall_view = "neutral"

    technical_summary = (
        f"Momentum signal is {momentum_signal}. "
        f"Current price is {price_summary.get('current_price')}, "
        f"with SMA-50 at {price_summary.get('sma_50')} and "
        f"SMA-200 at {price_summary.get('sma_200')}."
    )
    sentiment_summary = (
        f"Overall sentiment score is "
        f"{sentiment.get('overall_sentiment_score', 0.0)} "
        f"({risk.sentiment_view})."
    )

    return InvestmentOutlook(
        ticker=ticker,
        overall_view=overall_view,
        risk_level=risk_level,
        technical_summary=technical_summary,
        sentiment_summary=sentiment_summary,
        risk_interpretation=risk,
        practical_next_steps=[
            "Review the latest company filings and earnings context.",
            "Compare this signal against sector and market conditions.",
            "Use conservative sizing when volatility or sentiment risk is elevated.",
        ],
    )


def save_task1_output(
    ticker: str,
    evidence: Mapping[str, Any],
    outlook: InvestmentOutlook,
) -> Path:
    """Save a JSON report for reproducibility and interview discussion."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"{normalize_ticker(ticker)}_task1_report_{timestamp}.json"

    serializable_evidence = {
        key: value
        for key, value in evidence.items()
        if key != "dataframe"
    }
    payload = {
        "evidence": serializable_evidence,
        "outlook": outlook.model_dump(),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)

    return output_path


def run_task1_analysis(
    ticker: str,
    *,
    period: str = "2y",
    save_output: bool = True,
) -> JSONDict:
    """
    End-to-end Task 1 helper for notebooks.

    This gives the notebook a clean single-call path while still exposing every
    component function for step-by-step teaching and debugging.
    """

    evidence = collect_financial_evidence(ticker, period=period)
    outlook = generate_investment_outlook(evidence)
    output_path: Optional[Path] = None

    if save_output:
        output_path = save_task1_output(evidence["ticker"], evidence, outlook)
        print(f"[task1] Saved structured report to {output_path}")

    return {
        "evidence": evidence,
        "outlook": outlook.model_dump(),
        "output_path": str(output_path) if output_path else None,
    }


def plot_price_and_sma(dataframe: pd.DataFrame, ticker: str) -> plt.Figure:
    """Plot price with SMA-50 and SMA-200."""

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(dataframe.index, dataframe["Close"], label="Close", linewidth=1.5)
    axis.plot(dataframe.index, dataframe["SMA_50"], label="SMA 50", linewidth=1.2)
    axis.plot(dataframe.index, dataframe["SMA_200"], label="SMA 200", linewidth=1.2)
    axis.set_title(f"{normalize_ticker(ticker)} Price with Moving Averages")
    axis.set_ylabel("Price")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_rsi(dataframe: pd.DataFrame, ticker: str) -> plt.Figure:
    """Plot RSI with common overbought/oversold reference lines."""

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(dataframe.index, dataframe["RSI_14"], label="RSI 14", color="purple")
    axis.axhline(70, color="red", linestyle="--", linewidth=1, label="Overbought")
    axis.axhline(30, color="green", linestyle="--", linewidth=1, label="Oversold")
    axis.set_title(f"{normalize_ticker(ticker)} RSI")
    axis.set_ylabel("RSI")
    axis.set_ylim(0, 100)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_macd(dataframe: pd.DataFrame, ticker: str) -> plt.Figure:
    """Plot MACD and signal line."""

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(dataframe.index, dataframe["MACD"], label="MACD", linewidth=1.3)
    axis.plot(
        dataframe.index,
        dataframe["MACD_SIGNAL"],
        label="Signal",
        linewidth=1.3,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(f"{normalize_ticker(ticker)} MACD")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_bollinger_bands(dataframe: pd.DataFrame, ticker: str) -> plt.Figure:
    """Plot close price with Bollinger Bands."""

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(dataframe.index, dataframe["Close"], label="Close", linewidth=1.5)
    axis.plot(dataframe.index, dataframe["BB_UPPER"], label="Upper Band", linewidth=1)
    axis.plot(dataframe.index, dataframe["BB_LOWER"], label="Lower Band", linewidth=1)
    axis.fill_between(
        dataframe.index,
        dataframe["BB_LOWER"],
        dataframe["BB_UPPER"],
        alpha=0.12,
    )
    axis.set_title(f"{normalize_ticker(ticker)} Bollinger Bands")
    axis.set_ylabel("Price")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_all_indicators(dataframe: pd.DataFrame, ticker: str) -> List[plt.Figure]:
    """Create all required Task 1 visualizations."""

    if dataframe.empty:
        raise ValueError("Cannot plot indicators because dataframe is empty.")

    return [
        plot_price_and_sma(dataframe, ticker),
        plot_rsi(dataframe, ticker),
        plot_macd(dataframe, ticker),
        plot_bollinger_bands(dataframe, ticker),
    ]


def validate_reused_outputs(evidence: Mapping[str, Any]) -> JSONDict:
    """
    Demonstrate structured validation of reused Task 3 outputs.

    This helper is useful in the notebook to show that Task 1 is consuming the
    same contracts produced by Task 3 tools, not loosely shaped dictionaries.
    """

    price_summary = PriceDataSummary.model_validate(evidence["price_summary"])
    volatility = VolatilityResult.model_validate(evidence["volatility"])
    return {
        "price_summary": price_summary.model_dump(),
        "volatility": volatility.model_dump(),
    }
