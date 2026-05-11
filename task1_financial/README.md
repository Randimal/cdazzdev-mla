# Task 1 — Financial AI Analysis Workflow

## Overview

This task implements a notebook-oriented Financial AI analysis workflow focused on:
- market data retrieval
- technical indicator analysis
- volatility estimation
- financial news retrieval
- LLM-based sentiment analysis
- investment risk interpretation
- structured financial outlook generation

The implementation prioritizes:
- explainability
- modular engineering
- reusable infrastructure
- notebook readability
- production-oriented design

Several reusable components from Task 3 were intentionally reused to reduce duplication and maintain consistent observability and structured output handling across the repository.

---

## Features

### Financial Data Analysis
- Historical stock price retrieval using `yfinance`
- SMA 50 / SMA 200 trend analysis
- RSI momentum analysis
- MACD trend analysis
- Bollinger Bands volatility analysis
- Annualized volatility estimation

### News & Sentiment Analysis
- Financial news retrieval
- LLM-based sentiment classification using Groq
- Structured sentiment outputs
- Risk-aware investment interpretation

### Visualization
The notebook generates:
- Price + SMA chart
- RSI chart
- MACD chart
- Bollinger Bands chart

### Engineering Features
- Reusable modular tooling
- Shared observability utilities from Task 3
- Structured outputs using Pydantic schemas
- Graceful degradation for API failures
- Notebook-friendly workflow design

---

## Workflow Structure

```text
Ticker Input
    ↓
Market Data Retrieval
    ↓
Technical Indicator Computation
    ↓
Volatility Analysis
    ↓
News Retrieval
    ↓
LLM Sentiment Analysis
    ↓
Risk Interpretation
    ↓
Final Investment Outlook
```

---

## Reused Components from Task 3

The following reusable infrastructure from Task 3 is intentionally reused:

- Financial data tooling
- Technical indicator computation
- Volatility analysis
- News retrieval
- LLM sentiment analysis
- Structured schemas
- Observability / tracing utilities

This demonstrates modular engineering and avoids unnecessary duplication across tasks.

---

## Included Outputs

The `outputs/` directory contains:
- structured JSON analysis outputs
- generated charts
- final investment outlook summaries

Example output artifacts:
- `AAPL_analysis.json`
- `AAPL_price_sma.png`
- `AAPL_rsi.png`
- `AAPL_macd.png`
- `AAPL_bollinger.png`

---

## Running the Notebook

Example usage:

```python
result = run_task1_analysis(
    ticker="AAPL",
    period="2y"
)
```

Generate charts:

```python
plot_all_indicators(result)
```

Save outputs:

```python
save_task1_output(result)
```

---

## Engineering Notes

This workflow intentionally favors:
- explicit analysis pipelines
- explainable computations
- modular reusable components
- lightweight production-style engineering

over unnecessarily complex abstractions.

The notebook structure is designed to remain:
- easy to debug
- easy to explain during interviews
- easy to extend for additional indicators or risk models

---

## Limitations

- Free-tier APIs may occasionally rate limit requests
- News quality depends on external providers
- LLM sentiment outputs may vary slightly between runs
- This workflow is intended for educational/research purposes and not real financial advice