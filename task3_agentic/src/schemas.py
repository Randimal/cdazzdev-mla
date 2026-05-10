from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class NewsItem(BaseModel):
    title: str
    source: Optional[str] = None
    published: Optional[str] = None


class SentimentResult(BaseModel):
    headline: str
    sentiment: str = Field(
        description="positive, negative, or neutral"
    )
    confidence: float
    reason: str


class PriceDataSummary(BaseModel):
    ticker: str
    current_price: float
    sma_50: float
    sma_200: float
    rsi_14: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    momentum_signal: str


class VolatilityResult(BaseModel):
    ticker: str
    annualized_volatility: float


class QuantAnalysis(BaseModel):
    ticker: str
    momentum_signal: str
    annualized_volatility: float
    sentiment_score: float
    key_metrics: Dict
    risks: List[str]


class FinalReport(BaseModel):
    ticker: str
    financial_health_summary: str
    top_three_risks: List[str]
    hedge_strategy: str