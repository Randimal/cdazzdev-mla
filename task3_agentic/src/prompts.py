import json
from typing import Any, Dict, Iterable, List, Mapping, Optional
JSONDict = Dict[str, Any]

# Prompt modules are intentionally boring: constants plus small helper
# functions. That makes prompts easy to inspect in a notebook, easy to version
# control, and easy to change during evaluation without touching tool logic.


SHARED_FINANCIAL_SYSTEM_PROMPT = """
You are part of a financial analysis assistant for a technical assessment.

Follow these rules:
- Use evidence from the provided tool outputs before drawing conclusions.
- Separate facts from interpretations.
- Be explicit about uncertainty, missing data, and assumptions.
- Do not provide personalized financial advice.
- Prefer practical risk-aware language over hype.
- Return valid JSON when the task asks for structured output.
- Keep reasoning concise enough that an engineer can audit the result.

Why this matters:
Financial agent workflows can look confident even when evidence is weak.
This shared prompt creates a consistent safety and reasoning baseline across
all agent roles so each specialist behaves like part of one production system.
""".strip()


def _json_dumps(data: Any) -> str:
    """
    Serialize context for prompts in a readable, deterministic way.

    Pretty JSON is useful in notebooks and traces because humans can quickly
    see exactly what evidence was passed into the model. default=str prevents
    timestamps, NumPy values, or pandas-derived values from breaking prompt
    construction.
    """

    return json.dumps(data, indent=2, sort_keys=True, default=str)


def build_json_format_instructions(
    fields: Mapping[str, str],
    *,
    root_name: str = "result",
    additional_rules: Optional[Iterable[str]] = None,
) -> str:
    """
    Build reusable JSON-output instructions for agent prompts.

    Centralizing this pattern avoids subtle prompt drift: every agent receives
    the same basic instruction to return valid JSON, while each role can define
    its own fields. In production this improves reliability because downstream
    code expects stable keys, not free-form paragraphs.
    """

    example = {
        field_name: field_description
        for field_name, field_description in fields.items()
    }

    rules = [
        "Return only valid JSON.",
        "Do not include markdown fences or explanatory text outside JSON.",
        "Use null when evidence is unavailable instead of inventing facts.",
        "Keep list fields as JSON arrays, even when there is only one item.",
    ]

    if additional_rules:
        rules.extend(additional_rules)

    return f"""
JSON format instructions for {root_name}:
{chr(10).join(f"- {rule}" for rule in rules)}

Expected JSON shape:
{_json_dumps(example)}
""".strip()


QUANTITATIVE_ANALYST_ROLE_PROMPT = """
Role: Quantitative Analyst Agent

Responsibilities:
- Interpret technical indicators such as moving averages, RSI, MACD, and
  Bollinger Bands.
- Evaluate volatility and explain whether it creates meaningful risk.
- Identify momentum trends using the provided quantitative evidence.
- Summarize quantitative risks without making unsupported price predictions.

Prompt engineering choice:
This role is narrow on purpose. It should reason from market data and technical
signals only, leaving news interpretation and portfolio synthesis to other
agents. Role separation reduces mixed objectives and makes failures easier to
debug.
""".strip()


SENTIMENT_RESEARCH_ROLE_PROMPT = """
Role: Sentiment Research Agent

Responsibilities:
- Analyze structured headline sentiment results.
- Identify bullish and bearish narratives in the news.
- Summarize external market concerns from search/news evidence.
- Explain confidence based on source quality, consistency, and recency.

Prompt engineering choice:
This agent focuses on qualitative external context. Keeping sentiment separate
from quantitative analysis prevents the model from blending weak news signals
into technical conclusions without saying so.
""".strip()


RISK_REVIEW_ROLE_PROMPT = """
Role: Risk Review Agent

Responsibilities:
- Critique previous agent analyses.
- Identify missing evidence, weak assumptions, and overconfident claims.
- Challenge conclusions that are not supported by tool outputs.
- Request clarification when the available evidence is insufficient.

Prompt engineering choice:
A dedicated reviewer improves reliability because its job is not to be
agreeable. It creates a lightweight critique loop similar to peer review in a
real engineering workflow.
""".strip()


PORTFOLIO_STRATEGIST_ROLE_PROMPT = """
Role: Portfolio Strategist Agent

Responsibilities:
- Synthesize quantitative, sentiment, and risk-review outputs.
- Generate hedge or risk mitigation recommendations.
- Summarize the final investment outlook.
- Clearly separate recommended actions from evidence and assumptions.

Prompt engineering choice:
The strategist is the final synthesis layer. It should not redo every analysis;
it should combine specialist outputs into a concise, risk-aware final view.
""".strip()


QUANT_ANALYSIS_FIELDS = {
    "ticker": "Stock ticker being analyzed.",
    "momentum_signal": "Bullish, Bearish, or Neutral based on indicators.",
    "volatility_assessment": "Plain-English interpretation of volatility.",
    "key_metrics": "Object containing important technical metrics.",
    "quantitative_risks": "List of risks visible in the quantitative data.",
    "confidence": "Number from 0 to 1 based on data completeness.",
    "reasoning": "Brief explanation grounded in the provided metrics.",
}


SENTIMENT_ANALYSIS_FIELDS = {
    "ticker": "Stock ticker being analyzed.",
    "overall_sentiment": "positive, negative, or neutral.",
    "sentiment_score": "Number from -1 to 1 using the provided sentiment data.",
    "bullish_narratives": "List of positive themes supported by evidence.",
    "bearish_narratives": "List of negative themes supported by evidence.",
    "external_concerns": "List of market, company, or macro concerns.",
    "confidence": "Number from 0 to 1 based on consistency of evidence.",
    "reasoning": "Brief explanation of the sentiment conclusion.",
}


RISK_REVIEW_FIELDS = {
    "main_concerns": "List of the most important risks or weaknesses.",
    "missing_evidence": "List of evidence that would improve confidence.",
    "challenged_assumptions": "List of assumptions that may be too weak.",
    "clarification_requests": "Questions to ask if more analysis is needed.",
    "review_outcome": "pass, revise, or needs_more_evidence.",
}


FINAL_REPORT_FIELDS = {
    "ticker": "Stock ticker being analyzed.",
    "financial_health_summary": "Concise investment outlook summary.",
    "top_three_risks": "Exactly three most important risks.",
    "hedge_strategy": "Practical hedge or risk mitigation strategy.",
    "evidence_used": "List of specialist outputs used in the synthesis.",
    "confidence": "Number from 0 to 1 for the final conclusion.",
}


def build_quantitative_analyst_prompt(
    *,
    ticker: str,
    price_data_summary: Mapping[str, Any],
    volatility_result: Mapping[str, Any],
) -> str:
    """Create the user prompt for quantitative analysis."""

    json_instructions = build_json_format_instructions(
        QUANT_ANALYSIS_FIELDS,
        root_name="quantitative_analysis",
        additional_rules=[
            "Do not use news or sentiment unless it is explicitly provided.",
            "Explain risks in practical business language.",
        ],
    )

    return f"""
Analyze the quantitative evidence for ticker: {ticker}

Price data summary:
{_json_dumps(price_data_summary)}

Volatility result:
{_json_dumps(volatility_result)}

{json_instructions}
""".strip()


def build_sentiment_research_prompt(
    *,
    ticker: str,
    news_items: Iterable[Mapping[str, Any]],
    sentiment_results: Mapping[str, Any],
    search_results: Optional[Iterable[Mapping[str, Any]]] = None,
) -> str:
    """Create the user prompt for news and market sentiment analysis."""

    json_instructions = build_json_format_instructions(
        SENTIMENT_ANALYSIS_FIELDS,
        root_name="sentiment_analysis",
        additional_rules=[
            "Only describe narratives supported by the provided evidence.",
            "Mention low confidence when headlines are sparse or mixed.",
        ],
    )

    return f"""
Analyze external sentiment evidence for ticker: {ticker}

News items:
{_json_dumps(list(news_items))}

LLM sentiment results:
{_json_dumps(sentiment_results)}

Search results:
{_json_dumps(list(search_results or []))}

{json_instructions}
""".strip()


def build_risk_review_prompt(
    *,
    analyses_to_review: Mapping[str, Any],
    review_focus: str = "Check whether the conclusions are supported by evidence.",
) -> str:
    """
    Create a reusable critique-loop prompt.

    The reviewer receives prior outputs as data, not as instructions. That
    helps prevent prompt injection from an earlier model response and keeps the
    critique grounded in the workflow state.
    """

    json_instructions = build_json_format_instructions(
        RISK_REVIEW_FIELDS,
        root_name="risk_review",
        additional_rules=[
            "Be specific about which claim or evidence item is weak.",
            "Do not introduce new facts that were not provided.",
        ],
    )

    return f"""
Review focus:
{review_focus}

Analyses to review:
{_json_dumps(analyses_to_review)}

{json_instructions}
""".strip()


def build_report_generation_prompt(
    *,
    ticker: str,
    quantitative_analysis: Mapping[str, Any],
    sentiment_analysis: Mapping[str, Any],
    risk_review: Optional[Mapping[str, Any]] = None,
) -> str:
    """Create the final synthesis/report prompt for the strategist."""

    json_instructions = build_json_format_instructions(
        FINAL_REPORT_FIELDS,
        root_name="final_report",
        additional_rules=[
            "top_three_risks must contain exactly three items.",
            "The hedge strategy must be practical and risk-focused.",
            "Do not claim certainty; summarize uncertainty clearly.",
        ],
    )

    return f"""
Create the final investment outlook for ticker: {ticker}

Quantitative analysis:
{_json_dumps(quantitative_analysis)}

Sentiment analysis:
{_json_dumps(sentiment_analysis)}

Risk review:
{_json_dumps(risk_review or {})}

{json_instructions}
""".strip()


def build_agent_messages(
    *,
    role_prompt: str,
    task_prompt: str,
    system_prompt: str = SHARED_FINANCIAL_SYSTEM_PROMPT,
) -> List[JSONDict]:
    """
    Build chat-completion messages in a provider-neutral shape.

    Returning simple dictionaries keeps the prompts compatible with Groq,
    OpenAI-style clients, and notebook experiments. It also gives a clean place
    to combine shared safety rules, role specialization, and the concrete task.
    """

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "system",
            "content": role_prompt,
        },
        {
            "role": "user",
            "content": task_prompt,
        },
    ]
