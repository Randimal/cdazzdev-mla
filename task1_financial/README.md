# task1_financial/README.md

# Task 1 — Financial AI Analysis Workflow

## Overview

This task implements a modular notebook-oriented financial analysis workflow focused on:
- market data retrieval
- technical indicator analysis
- volatility estimation
- financial news retrieval
- LLM-based sentiment analysis
- risk interpretation
- and structured investment outlook generation.

One of the main goals of this workflow was combining explainable financial analysis with reusable engineering infrastructure.

Instead of building a single monolithic notebook, the implementation separates:
- tooling
- schemas
- orchestration
- analysis logic
- and visualization generation

into reusable components.

Several reusable utilities from Task 3 were intentionally reused to maintain consistent:
- observability
- structured output handling
- and workflow design patterns

across the repository.

---

# Workflow Structure

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
Structured Final Investment Outlook
```

---

# Financial Analysis Features

## Market Data Retrieval

Historical stock data is retrieved using `yfinance`.

The workflow supports:
- configurable ticker symbols
- adjustable time periods
- and notebook-friendly execution.

---

## Technical Indicators

The workflow computes:
- SMA 50 / SMA 200 trend analysis
- RSI momentum analysis
- MACD trend analysis
- Bollinger Bands volatility analysis
- annualized volatility estimation.

These indicators are used to generate a more explainable investment outlook rather than relying entirely on LLM-generated reasoning.

---

## Financial News and Sentiment

The workflow retrieves financial news and performs:
- LLM-based sentiment classification
- confidence estimation
- and structured sentiment summarization.

Sentiment outputs are normalized into validated schemas for consistency.

---

## Visualization

The notebook generates:
- price and moving average charts
- RSI charts
- MACD charts
- Bollinger Bands charts.

These visualizations improve interpretability and make the workflow easier to inspect interactively.

---

# Engineering Features

## Modular Design

Instead of placing all logic directly inside notebook cells, reusable utilities are separated into Python modules.

This improves:
- readability
- reuse
- maintainability
- and debugging.

---

## Shared Infrastructure

The workflow intentionally reuses infrastructure from Task 3, including:
- financial tooling
- volatility analysis
- sentiment analysis
- structured schemas
- and tracing utilities.

This demonstrates modular engineering and avoids unnecessary duplication.

---

## Structured Outputs

The workflow produces structured JSON outputs rather than unstructured notebook-only results.

This improves:
- reproducibility
- downstream processing
- and workflow transparency.

---

## Graceful Degradation

The workflow includes fallback handling for:
- API failures
- missing financial news
- malformed LLM responses
- and unavailable external services.

---

# Example Outputs

The `outputs/` directory contains:
- structured JSON analysis outputs
- generated visualizations
- and investment outlook summaries.

Example artifacts include:
- `AAPL_analysis.json`
- `AAPL_price_sma.png`
- `AAPL_rsi.png`
- `AAPL_macd.png`
- `AAPL_bollinger.png`

---

# Running the Workflow

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

# Engineering Notes

This workflow intentionally favors:
- explainable computations
- deterministic analysis pipelines
- reusable modular components
- and lightweight production-oriented engineering.

The goal was building a workflow that remains:
- easy to debug
- easy to explain
- and easy to extend

without unnecessary abstraction layers.

---

# Limitations

- Free-tier APIs may occasionally rate limit requests.
- Financial news quality depends on external providers.
- LLM sentiment outputs may vary slightly between runs.
- Technical indicators alone should not be treated as complete investment advice.
- The workflow is educational and engineering-focused rather than a production trading system.

---

# Main Takeaways

This task reinforced the value of:
- modular engineering
- reusable infrastructure
- explainable workflows
- and structured outputs

when building practical AI analysis systems.