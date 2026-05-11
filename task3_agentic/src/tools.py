import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from ddgs import DDGS
from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from src.schemas import (
    NewsItem,
    PriceDataSummary,
    SentimentResult,
    VolatilityResult,
)

from src.tracing import traced_tool


# Notebook users often store API keys in a local .env file. Loading it here
# keeps setup simple while still avoiding hardcoded secrets in source control.
load_dotenv()


# Keeping prompts as module-level constants makes the LLM behavior easy to
# inspect, reuse, and test. In production, prompts are part of the application
# contract, not throwaway strings hidden deep inside business logic.
SENTIMENT_SYSTEM_PROMPT = """
You are a careful financial news sentiment classifier.

Return ONLY valid JSON. Do not include markdown, explanations outside JSON, or
extra keys.

Allowed sentiment labels:
- positive
- negative
- neutral

For each headline, judge the likely short-term market sentiment for the
company, not whether the writing style sounds happy or sad.
""".strip()

SENTIMENT_USER_PROMPT_TEMPLATE = """
Classify the sentiment of these headlines.

Return this exact JSON shape:
{{
  "results": [
    {{
      "headline": "original headline text",
      "sentiment": "positive | negative | neutral",
      "confidence": 0.0,
      "reason": "brief reason"
    }}
  ]
}}

Headlines:
{headlines_json}
""".strip()

ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}
SENTIMENT_SCORE_MAP = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


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


def _format_unix_timestamp(timestamp: Optional[int]) -> Optional[str]:
    """Convert a Unix timestamp into an ISO string when Yahoo provides one."""

    if timestamp is None:
        return None

    try:
        # ISO-8601 is a good notebook/API format because it is sortable,
        # timezone-aware, and easy for humans to read during debugging.
        return datetime.fromtimestamp(
            int(timestamp),
            tz=timezone.utc,
        ).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _extract_yfinance_news_item(raw_item: Dict[str, Any]) -> Optional[NewsItem]:
    """Normalize the different news shapes returned by yfinance/Yahoo."""

    # yfinance has returned both a flat shape and a newer nested "content"
    # shape over time. Handling both keeps the notebook stable even when the
    # upstream free endpoint changes small response details.
    content = raw_item.get("content") or {}

    title = (
        raw_item.get("title")
        or content.get("title")
        or content.get("headline")
    )
    if not title:
        return None

    provider = content.get("provider") or {}
    source = (
        raw_item.get("publisher")
        or provider.get("displayName")
        or provider.get("name")
        or raw_item.get("source")
    )

    published_at = (
        raw_item.get("providerPublishTime")
        or raw_item.get("publishTime")
        or content.get("pubDate")
        or content.get("displayTime")
    )

    if isinstance(published_at, int):
        published = _format_unix_timestamp(published_at)
    else:
        # Some yfinance versions already return a displayable date string.
        published = str(published_at) if published_at else None

    return NewsItem(
        title=str(title),
        source=str(source) if source else None,
        published=published,
    )


@traced_tool
def get_news(ticker: str, n: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch recent company news from yfinance and return validated news records.

    The tool returns plain dictionaries instead of Pydantic objects because
    notebooks, JSON trace logs, and downstream agents can display/serialize
    dictionaries without extra conversion code.
    """

    requested_count = max(n, 10)

    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []
    except Exception:
        # News is useful context, but it should not crash the full financial
        # workflow. Returning an empty list lets the agent continue with price
        # and volatility evidence while the trace records the tool outcome.
        return []

    news_items: List[NewsItem] = []
    seen_titles = set()

    for raw_item in raw_news:
        if len(news_items) >= requested_count:
            break

        item = _extract_yfinance_news_item(raw_item)
        if item is None:
            continue

        normalized_title = item.title.strip().lower()
        if normalized_title in seen_titles:
            continue

        seen_titles.add(normalized_title)
        news_items.append(item)

    # Graceful fallback: no fabricated headlines. An empty structured list is
    # honest, easy to check in downstream code, and safer than pretending the
    # model has evidence that the data source did not provide.
    return [item.model_dump() for item in news_items]


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Parse an LLM JSON response, tolerating common formatting mistakes.

    Even with a strict prompt, LLMs can occasionally wrap JSON in markdown or
    add a sentence. This defensive parser keeps the failure mode controlled
    instead of letting malformed text break the whole notebook.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object.")

    return json.loads(match.group(0))


def _fallback_sentiment_result(
    headline: str,
    reason: str,
) -> SentimentResult:
    """Create a valid neutral result when the model output cannot be trusted."""

    return SentimentResult(
        headline=headline,
        sentiment="neutral",
        confidence=0.0,
        reason=reason,
    )


def _validate_sentiment_result(
    raw_result: Dict[str, Any],
    original_headline: str,
) -> SentimentResult:
    """
    Validate and normalize one sentiment record.

    Pydantic validates the shape/types, while the explicit sentiment check
    enforces the business rule that downstream scoring only supports three
    labels. Keeping both layers visible is useful in interviews: schema
    validation protects data contracts, and business validation protects
    application meaning.
    """

    result = SentimentResult.model_validate(raw_result)
    normalized_sentiment = result.sentiment.strip().lower()

    if normalized_sentiment not in ALLOWED_SENTIMENTS:
        raise ValueError(
            f"Invalid sentiment label: {result.sentiment}"
        )

    confidence = max(0.0, min(1.0, float(result.confidence)))

    return SentimentResult(
        headline=result.headline or original_headline,
        sentiment=normalized_sentiment,
        confidence=confidence,
        reason=result.reason,
    )


def _calculate_overall_sentiment_score(
    results: Iterable[SentimentResult],
) -> float:
    """
    Aggregate item-level sentiment into one score between -1 and 1.

    Confidence weighting gives stronger model judgments more influence while
    keeping uncertain or fallback results close to neutral.
    """

    weighted_scores = []
    weights = []

    for result in results:
        confidence = max(0.0, min(1.0, result.confidence))
        weighted_scores.append(
            SENTIMENT_SCORE_MAP[result.sentiment] * confidence
        )
        weights.append(confidence)

    if not weights or sum(weights) == 0:
        return 0.0

    return round(sum(weighted_scores) / sum(weights), 4)


@traced_tool
def llm_sentiment(headlines: List[str]) -> Dict[str, Any]:
    """
    Classify headline sentiment with Groq and return validated structured data.

    Structured output matters in production because agents and dashboards need
    dependable fields, not prose that changes shape on every run. The LLM can
    reason over language, while Pydantic acts as the boundary that decides what
    is safe for the rest of the workflow to consume.
    """

    clean_headlines = [
        str(headline).strip()
        for headline in headlines
        if str(headline).strip()
    ]

    if not clean_headlines:
        return {
            "results": [],
            "overall_sentiment_score": 0.0,
        }

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        fallback_results = [
            _fallback_sentiment_result(
                headline=headline,
                reason=(
                    "GROQ_API_KEY is not configured, so sentiment was "
                    "returned as neutral instead of calling the LLM."
                ),
            )
            for headline in clean_headlines
        ]
        return {
            "results": [
                result.model_dump() for result in fallback_results
            ],
            "overall_sentiment_score": 0.0,
        }

    user_prompt = SENTIMENT_USER_PROMPT_TEMPLATE.format(
        headlines_json=json.dumps(clean_headlines, indent=2),
    )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            # Model is configurable so the assessment can run even if a free
            # Groq model changes. The default keeps setup simple for notebooks.
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {
                    "role": "system",
                    "content": SENTIMENT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed_response = _extract_json_object(raw_content)
        raw_results = parsed_response.get("results", [])

        if not isinstance(raw_results, list):
            raise ValueError("LLM JSON field 'results' was not a list.")

        validated_results = []
        for index, headline in enumerate(clean_headlines):
            try:
                raw_result = raw_results[index]
                validated_results.append(
                    _validate_sentiment_result(
                        raw_result=raw_result,
                        original_headline=headline,
                    )
                )
            except (IndexError, TypeError, ValidationError, ValueError):
                validated_results.append(
                    _fallback_sentiment_result(
                        headline=headline,
                        reason=(
                            "The LLM response for this headline was missing "
                            "or failed validation, so the safe neutral "
                            "fallback was used."
                        ),
                    )
                )

    except Exception as exc:
        # A production workflow should degrade predictably when a network call,
        # rate limit, or model response fails. Neutral fallbacks make the issue
        # visible without poisoning the quantitative analysis with guessed data.
        validated_results = [
            _fallback_sentiment_result(
                headline=headline,
                reason=f"Groq sentiment call failed: {exc}",
            )
            for headline in clean_headlines
        ]

    return {
        "results": [
            result.model_dump() for result in validated_results
        ],
        "overall_sentiment_score": _calculate_overall_sentiment_score(
            validated_results
        ),
    }


@traced_tool
def web_search(query: str) -> List[Dict[str, Optional[str]]]:
    """
    Search the web with DuckDuckGo and return the top five simple evidence rows.

    This supports agent reasoning by giving the workflow external context it can
    cite or inspect before forming a final answer. Keeping the output small and
    structured reduces noise: the agent gets titles, snippets, and URLs rather
    than an unbounded page dump.
    """

    clean_query = query.strip()
    if not clean_query:
        return []

    try:
        with DDGS() as ddgs:
            raw_results = list(
                ddgs.text(
                    clean_query,
                    max_results=5,
                )
            )
    except Exception:
        # Search is an enrichment tool. If the free endpoint is unavailable or
        # rate-limited, the rest of the workflow can still continue and the
        # trace log will show that search returned no evidence.
        return []

    search_results = []
    for result in raw_results[:5]:
        search_results.append(
            {
                "title": result.get("title"),
                "snippet": result.get("body"),
                "url": result.get("href"),
            }
        )

    return search_results
