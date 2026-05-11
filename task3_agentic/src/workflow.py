import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

# Notebooks commonly use either:
#   from src.workflow import run_investment_workflow
# or add task3_agentic/src to sys.path and import workflow directly.
# Adding both paths keeps this module compatible with both styles without
# forcing the user to learn Python packaging details during the assessment.
for path in (CURRENT_DIR, BASE_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from src.agents import (
    PortfolioStrategistAgent,
    QuantitativeAnalystAgent,
    SentimentResearchAgent,
    run_critique_loop,
)
from task3_agentic.src.schemas import FinalReport
from src.tools import (
    calculate_volatility,
    get_news,
    get_price_data,
    llm_sentiment,
    web_search,
)
from task3_agentic.src.tracing import log_tool_call


JSONDict = Dict[str, Any]

MEMORY_DIR = BASE_DIR / "memory"
OUTPUTS_DIR = BASE_DIR / "outputs"


def _now_utc() -> str:
    """Return a stable timestamp for cache metadata and output files."""

    return datetime.now(timezone.utc).isoformat()


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker input once at the workflow boundary.

    Keeping this rule in one place avoids cache misses caused by users entering
    "aapl", " AAPL ", or other harmless variations in a notebook.
    """

    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        raise ValueError("Ticker must not be empty.")
    return clean_ticker


def get_cache_path(ticker: str) -> Path:
    """Return the memory-cache path for a ticker report."""

    return MEMORY_DIR / f"{normalize_ticker(ticker)}_report_cache.json"


def get_output_path(ticker: str) -> Path:
    """Return a timestamped output path for the latest workflow run."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return OUTPUTS_DIR / f"{normalize_ticker(ticker)}_report_{timestamp}.json"


def load_cached_report(ticker: str) -> Optional[JSONDict]:
    """
    Load a cached report when available.

    This is the Task 3C short-term memory layer: the workflow remembers the
    last complete report for a ticker and can reuse it without re-calling
    market data, news, search, or LLM APIs.
    """

    cache_path = get_cache_path(ticker)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        # A corrupted cache should not break the analysis. We simply ignore it
        # and let the workflow rebuild the report from tools and agents.
        return None


def save_report_cache(ticker: str, workflow_result: Mapping[str, Any]) -> Path:
    """
    Save the complete workflow result to memory.

    We cache the full workflow state, not only the final report, because
    follow-up questions often need intermediate evidence such as risks,
    sentiment narratives, or quantitative metrics.
    """

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(ticker)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(workflow_result, file, indent=2, default=str)
    return cache_path


def save_report_output(ticker: str, workflow_result: Mapping[str, Any]) -> Path:
    """
    Save a timestamped report copy under outputs/.

    The memory cache is for reuse. The outputs folder is for auditability: it
    preserves what the workflow produced at a specific point in time.
    """

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = get_output_path(ticker)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(workflow_result, file, indent=2, default=str)
    return output_path


def _log_workflow_step(step_name: str, details: Mapping[str, Any]) -> None:
    """
    Log high-level workflow events to the same JSONL trace file as tools.

    Tool and agent decorators already log their calls. This helper adds the
    orchestration-level milestones so a reviewer can see cache hits, cache
    saves, and follow-up memory usage in one trace.
    """

    log_tool_call(
        tool_name=f"workflow.{step_name}",
        inputs=dict(details),
        output=details,
        duration=0.0,
    )


def _print_step(message: str, verbose: bool) -> None:
    """Keep notebook logging easy to turn on or off."""

    if verbose:
        print(message)


def _safe_tool_call(
    step_name: str,
    fallback: Any,
    verbose: bool,
    func,
    *args,
    **kwargs,
) -> Any:
    """
    Call a tool with graceful degradation.

    Tools are observable and already log exceptions, but the workflow should
    still continue where possible. Returning a typed fallback keeps later agent
    steps from crashing because one enrichment source failed.
    """

    try:
        _print_step(f"[workflow] {step_name}...", verbose)
        return func(*args, **kwargs)
    except Exception as exc:
        _print_step(
            f"[workflow] {step_name} failed; using fallback. Error: {exc}",
            verbose,
        )
        _log_workflow_step(
            f"{step_name}_fallback",
            {"error": str(exc)},
        )
        return fallback


def _extract_headlines(news_items: List[Mapping[str, Any]]) -> List[str]:
    """Convert news records into the headline list expected by llm_sentiment."""

    return [
        str(item.get("title", "")).strip()
        for item in news_items
        if str(item.get("title", "")).strip()
    ]


def _fallback_price_summary(ticker: str) -> JSONDict:
    """
    Minimal price summary used when market data is unavailable.

    This makes the workflow robust for demos, but the low-confidence agent
    fallbacks will clearly show that the final result should not be trusted as
    a complete market analysis.
    """

    return {
        "ticker": ticker,
        "current_price": None,
        "sma_50": None,
        "sma_200": None,
        "rsi_14": None,
        "macd": None,
        "macd_signal": None,
        "bollinger_upper": None,
        "bollinger_lower": None,
        "momentum_signal": "Unknown",
    }


def run_investment_workflow(
    ticker: str,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    debug: bool = False,
    verbose: bool = True,
) -> JSONDict:
    """
    Run the complete Task 3 multi-agent workflow for a ticker.

    The workflow is deliberately explicit:
    tools -> specialist agents -> critique loop -> strategist -> cache/output.
    That shape is easier to debug than an autonomous planner because every
    dependency is visible and every intermediate result can be inspected in a
    notebook cell.
    """

    normalized_ticker = normalize_ticker(ticker)
    _print_step(
        f"[workflow] Starting analysis for {normalized_ticker}",
        verbose,
    )

    if use_cache and not refresh_cache:
        cached_report = load_cached_report(normalized_ticker)
        if cached_report:
            _print_step(
                f"[workflow] Cache hit for {normalized_ticker}; skipping tools and agents.",
                verbose,
            )
            cached_report["cache_hit"] = True
            _log_workflow_step(
                "cache_hit",
                {
                    "ticker": normalized_ticker,
                    "cache_path": str(get_cache_path(normalized_ticker)),
                },
            )
            return cached_report

    _log_workflow_step(
        "start",
        {"ticker": normalized_ticker, "use_cache": use_cache},
    )

    price_data = _safe_tool_call(
        "get_price_data",
        {"summary": _fallback_price_summary(normalized_ticker), "dataframe": []},
        verbose,
        get_price_data,
        normalized_ticker,
    )
    price_summary = price_data.get(
        "summary",
        _fallback_price_summary(normalized_ticker),
    )

    volatility_result = _safe_tool_call(
        "calculate_volatility",
        {
            "ticker": normalized_ticker,
            "annualized_volatility": None,
        },
        verbose,
        calculate_volatility,
        normalized_ticker,
    )

    news_items = _safe_tool_call(
        "get_news",
        [],
        verbose,
        get_news,
        normalized_ticker,
        5,
    )
    headlines = _extract_headlines(news_items)

    sentiment_results = _safe_tool_call(
        "llm_sentiment",
        {
            "results": [],
            "overall_sentiment_score": 0.0,
        },
        verbose,
        llm_sentiment,
        headlines,
    )

    search_results = _safe_tool_call(
        "web_search",
        [],
        verbose,
        web_search,
        f"{normalized_ticker} stock latest business risks news",
    )

    _print_step("[workflow] Running quantitative analyst agent...", verbose)
    quantitative_analysis = QuantitativeAnalystAgent().analyze(
        ticker=normalized_ticker,
        price_data_summary=price_summary,
        volatility_result=volatility_result,
        debug=debug,
    )

    _print_step("[workflow] Running sentiment research agent...", verbose)
    sentiment_analysis = SentimentResearchAgent().analyze(
        ticker=normalized_ticker,
        news_items=news_items,
        sentiment_results=sentiment_results,
        search_results=search_results,
        debug=debug,
    )

    _print_step("[workflow] Running critique loop...", verbose)
    risk_review = run_critique_loop(
        quantitative_analysis=quantitative_analysis,
        sentiment_analysis=sentiment_analysis,
        debug=debug,
    )

    _print_step("[workflow] Running portfolio strategist agent...", verbose)
    final_report = PortfolioStrategistAgent().generate_report(
        ticker=normalized_ticker,
        quantitative_analysis=quantitative_analysis,
        sentiment_analysis=sentiment_analysis,
        risk_review=risk_review,
        debug=debug,
    )

    try:
        FinalReport.model_validate(final_report)
    except Exception as exc:
        _print_step(
            f"[workflow] Final report validation warning: {exc}",
            verbose,
        )

    workflow_result = {
        "ticker": normalized_ticker,
        "created_at": _now_utc(),
        "cache_hit": False,
        "final_report": final_report,
        "analysis": {
            "quantitative": quantitative_analysis,
            "sentiment": sentiment_analysis,
            "risk_review": risk_review,
        },
        "tool_outputs": {
            "price_data_summary": price_summary,
            "price_data_tail": price_data.get("dataframe", []),
            "volatility": volatility_result,
            "news": news_items,
            "sentiment_results": sentiment_results,
            "search_results": search_results,
        },
        "memory": {
            "cache_path": str(get_cache_path(normalized_ticker)),
            "output_path": None,
        },
    }

    cache_path = save_report_cache(normalized_ticker, workflow_result)
    output_path = save_report_output(normalized_ticker, workflow_result)
    workflow_result["memory"]["cache_path"] = str(cache_path)
    workflow_result["memory"]["output_path"] = str(output_path)

    # Save once more so the cache knows the timestamped output location too.
    save_report_cache(normalized_ticker, workflow_result)

    _log_workflow_step(
        "complete",
        {
            "ticker": normalized_ticker,
            "cache_path": str(cache_path),
            "output_path": str(output_path),
        },
    )
    _print_step(
        f"[workflow] Complete. Cache: {cache_path} | Output: {output_path}",
        verbose,
    )

    return workflow_result


def run_workflow(
    ticker: str,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    debug: bool = False,
    verbose: bool = True,
) -> JSONDict:
    """
    Notebook-friendly alias for run_investment_workflow().

    The longer name is explicit for code review; the short alias is convenient
    when demonstrating the workflow interactively.
    """

    return run_investment_workflow(
        ticker,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        debug=debug,
        verbose=verbose,
    )


def answer_follow_up_question(
    ticker: str,
    question: str,
    *,
    cached_report: Optional[Mapping[str, Any]] = None,
    verbose: bool = True,
) -> JSONDict:
    """
    Answer a follow-up question from cached workflow memory.

    This intentionally does not call tools or agents. It demonstrates
    short-term memory by reusing the previous structured report and returning a
    grounded answer based only on cached evidence.
    """

    normalized_ticker = normalize_ticker(ticker)
    memory = dict(cached_report or load_cached_report(normalized_ticker) or {})

    if not memory:
        return {
            "ticker": normalized_ticker,
            "question": question,
            "answer": (
                "No cached report was found. Run run_investment_workflow() "
                "first so the workflow has memory to answer from."
            ),
            "used_memory": False,
            "source": None,
        }

    final_report = memory.get("final_report", {})
    analysis = memory.get("analysis", {})
    tool_outputs = memory.get("tool_outputs", {})
    question_lower = question.lower()

    if "risk" in question_lower:
        answer = (
            "The top cached risks are: "
            + "; ".join(final_report.get("top_three_risks", []))
        )
    elif "hedge" in question_lower or "mitigation" in question_lower:
        answer = final_report.get(
            "hedge_strategy",
            "No hedge strategy was available in memory.",
        )
    elif "sentiment" in question_lower or "news" in question_lower:
        sentiment = analysis.get("sentiment", {})
        answer = (
            f"Cached sentiment is {sentiment.get('overall_sentiment')} "
            f"with score {sentiment.get('sentiment_score')}. "
            f"Reasoning: {sentiment.get('reasoning')}"
        )
    elif "volatility" in question_lower:
        volatility = tool_outputs.get("volatility", {})
        answer = (
            "Cached annualized volatility is "
            f"{volatility.get('annualized_volatility')}."
        )
    elif "summary" in question_lower or "outlook" in question_lower:
        answer = final_report.get(
            "financial_health_summary",
            "No final summary was available in memory.",
        )
    else:
        # The fallback answer is intentionally conservative. A more advanced
        # system could call an LLM over memory, but this helper demonstrates
        # short-term memory without triggering new API calls.
        answer = (
            "I found cached analysis, but this helper only answers common "
            "follow-up types: risks, hedge strategy, sentiment, volatility, "
            "or overall summary."
        )

    _print_step(
        f"[workflow] Answered follow-up for {normalized_ticker} from cache only.",
        verbose,
    )
    _log_workflow_step(
        "follow_up_from_memory",
        {
            "ticker": normalized_ticker,
            "question": question,
            "cache_path": str(get_cache_path(normalized_ticker)),
        },
    )

    return {
        "ticker": normalized_ticker,
        "question": question,
        "answer": answer,
        "used_memory": True,
        "source": str(get_cache_path(normalized_ticker)),
    }
