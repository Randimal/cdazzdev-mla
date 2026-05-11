import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError, field_validator

from task3_agentic.src.prompts import (
    PORTFOLIO_STRATEGIST_ROLE_PROMPT,
    QUANTITATIVE_ANALYST_ROLE_PROMPT,
    RISK_REVIEW_ROLE_PROMPT,
    SENTIMENT_RESEARCH_ROLE_PROMPT,
    build_agent_messages,
    build_quantitative_analyst_prompt,
    build_report_generation_prompt,
    build_risk_review_prompt,
    build_sentiment_research_prompt,
)
from task3_agentic.src.schemas import FinalReport
from task3_agentic.src.tracing import log_tool_call


load_dotenv()

JSONDict = Dict[str, Any]


class QuantitativeAnalysisOutput(BaseModel):
    """Structured output expected from the Quantitative Analyst Agent."""

    ticker: str
    momentum_signal: str
    volatility_assessment: str
    key_metrics: Dict[str, Any]
    quantitative_risks: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("momentum_signal")
    @classmethod
    def normalize_momentum_signal(cls, value: str) -> str:
        normalized = value.strip().capitalize()
        allowed = {"Bullish", "Bearish", "Neutral", "Unknown"}
        if normalized not in allowed:
            raise ValueError("momentum_signal must be Bullish, Bearish, or Neutral.")
        return normalized


class SentimentAnalysisOutput(BaseModel):
    """Structured output expected from the Sentiment Research Agent."""

    ticker: str
    overall_sentiment: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    bullish_narratives: List[str]
    bearish_narratives: List[str]
    external_concerns: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("overall_sentiment")
    @classmethod
    def validate_sentiment_label(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"positive", "negative", "neutral"}:
            raise ValueError(
                "overall_sentiment must be positive, negative, or neutral."
            )
        return normalized


class RiskReviewOutput(BaseModel):
    """Structured output expected from the Risk Review Agent."""

    main_concerns: List[str]
    missing_evidence: List[str]
    challenged_assumptions: List[str]
    clarification_requests: List[str]
    review_outcome: str

    @field_validator("review_outcome")
    @classmethod
    def validate_review_outcome(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"pass", "revise", "needs_more_evidence"}
        if normalized not in allowed:
            raise ValueError(
                "review_outcome must be pass, revise, or needs_more_evidence."
            )
        return normalized


class PortfolioStrategyOutput(BaseModel):
    """Structured output expected from the Portfolio Strategist Agent."""

    ticker: str
    financial_health_summary: str
    top_three_risks: List[str]
    hedge_strategy: str
    evidence_used: List[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("top_three_risks")
    @classmethod
    def validate_top_three_risks(cls, value: List[str]) -> List[str]:
        if len(value) != 3:
            raise ValueError("top_three_risks must contain exactly three items.")
        return value


def safe_extract_json(text: str) -> JSONDict:
    """
    Extract a JSON object from an LLM response.

    LLMs sometimes return valid reasoning but wrap the JSON in markdown or a
    short sentence. This helper keeps that common failure mode contained. The
    agent still validates the parsed object afterward, so extraction is only the
    first gate, not blind trust.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group(0))


class BaseAgent:
    """
    Small reusable base class for role-based agents.

    The base class centralizes API calls, JSON parsing, validation, and tracing.
    The individual agents stay simple: they only build the right prompt and
    provide a safe fallback. This is easier to debug than a large autonomous
    planning framework and fits the assessment's production-engineering focus.
    """

    agent_name = "BaseAgent"
    role_prompt = ""
    output_model: Optional[Type[BaseModel]] = None

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        # Environment-based model selection makes notebooks easy to run while
        # still allowing experiments without code changes:
        # GROQ_AGENT_MODEL overrides agent behavior, then GROQ_MODEL, then a
        # practical default.
        self.model = (
            model
            or os.getenv("GROQ_AGENT_MODEL")
            or os.getenv("GROQ_MODEL")
            or "llama-3.1-8b-instant"
        )
        self.temperature = temperature
        self.api_key = os.getenv("GROQ_API_KEY")
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        """Create the Groq client lazily so missing keys fail gracefully."""

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        if self._client is None:
            self._client = Groq(api_key=self.api_key)

        return self._client

    def run(
        self,
        *,
        task_prompt: str,
        fallback_output: JSONDict,
        debug: bool = False,
    ) -> JSONDict:
        """
        Execute one LLM-backed agent step and return structured JSON.

        Failures degrade into the caller-provided fallback. This is important in
        financial workflows: one unavailable model call should not erase the
        evidence already collected by tools or block notebook exploration.
        """

        start = time.time()
        error_message: Optional[str] = None
        raw_response: Optional[str] = None
        used_fallback = False

        try:
            messages = build_agent_messages(
                role_prompt=self.role_prompt,
                task_prompt=task_prompt,
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            raw_response = response.choices[0].message.content or "{}"
            parsed_output = safe_extract_json(raw_response)
            result = self._validate_output(parsed_output)

        except Exception as exc:
            # Safe fallback is explicit, traceable, and conservative. The
            # returned JSON tells the notebook that the agent failed instead of
            # silently pretending a high-confidence analysis happened.
            error_message = str(exc)
            used_fallback = True
            result = self._validate_output(fallback_output)

        duration = time.time() - start
        output_for_trace = {
            "result": result,
            "used_fallback": used_fallback,
            "error": error_message,
        }
        log_tool_call(
            tool_name=self.agent_name,
            inputs={
                "model": self.model,
                "task_prompt_preview": task_prompt[:500],
            },
            output=output_for_trace,
            duration=duration,
        )

        if debug:
            result["_debug"] = {
                "agent": self.agent_name,
                "model": self.model,
                "used_fallback": used_fallback,
                "error": error_message,
                "raw_response_preview": (
                    raw_response[:500] if raw_response else None
                ),
            }

        return result

    def _validate_output(self, output: JSONDict) -> JSONDict:
        """
        Validate model output with Pydantic when an output model is configured.

        Structured validation matters because downstream agents should consume
        predictable fields. If validation fails, the exception is caught in
        run() and the safe fallback is returned.
        """

        if self.output_model is None:
            return output

        validated = self.output_model.model_validate(output)
        return validated.model_dump()


class QuantitativeAnalystAgent(BaseAgent):
    """
    Interprets technical indicators and volatility evidence.

    Role specialization helps here because this agent should not be influenced
    by news narratives. It focuses on the numeric tool outputs and produces a
    structured summary that later agents can challenge or synthesize.
    """

    agent_name = "QuantitativeAnalystAgent"
    role_prompt = QUANTITATIVE_ANALYST_ROLE_PROMPT
    output_model = QuantitativeAnalysisOutput

    def analyze(
        self,
        *,
        ticker: str,
        price_data_summary: Mapping[str, Any],
        volatility_result: Mapping[str, Any],
        debug: bool = False,
    ) -> JSONDict:
        task_prompt = build_quantitative_analyst_prompt(
            ticker=ticker,
            price_data_summary=price_data_summary,
            volatility_result=volatility_result,
        )

        fallback_output = {
            "ticker": ticker,
            "momentum_signal": str(
                price_data_summary.get("momentum_signal", "Unknown")
            ),
            "volatility_assessment": (
                "LLM analysis unavailable; review annualized volatility "
                "directly from the tool output."
            ),
            "key_metrics": dict(price_data_summary),
            "quantitative_risks": [
                "Quantitative analyst LLM step was unavailable.",
                "Interpret technical indicators manually before relying on the result.",
            ],
            "confidence": 0.0,
            "reasoning": (
                "Fallback used because the agent could not produce validated "
                "structured output."
            ),
        }

        if "annualized_volatility" in volatility_result:
            fallback_output["key_metrics"]["annualized_volatility"] = (
                volatility_result["annualized_volatility"]
            )

        return self.run(
            task_prompt=task_prompt,
            fallback_output=fallback_output,
            debug=debug,
        )


class SentimentResearchAgent(BaseAgent):
    """
    Synthesizes headline sentiment, news, and search evidence.

    Separating this from quant analysis makes the system easier to explain:
    one agent handles market data, another handles narrative evidence, and the
    final strategist decides how much weight to give each.
    """

    agent_name = "SentimentResearchAgent"
    role_prompt = SENTIMENT_RESEARCH_ROLE_PROMPT
    output_model = SentimentAnalysisOutput

    def analyze(
        self,
        *,
        ticker: str,
        news_items: Iterable[Mapping[str, Any]],
        sentiment_results: Mapping[str, Any],
        search_results: Optional[Iterable[Mapping[str, Any]]] = None,
        debug: bool = False,
    ) -> JSONDict:
        task_prompt = build_sentiment_research_prompt(
            ticker=ticker,
            news_items=news_items,
            sentiment_results=sentiment_results,
            search_results=search_results,
        )

        sentiment_score = float(
            sentiment_results.get("overall_sentiment_score", 0.0)
        )
        if sentiment_score > 0.15:
            overall_sentiment = "positive"
        elif sentiment_score < -0.15:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        fallback_output = {
            "ticker": ticker,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score,
            "bullish_narratives": [],
            "bearish_narratives": [],
            "external_concerns": [
                "Sentiment research LLM step was unavailable."
            ],
            "confidence": 0.0,
            "reasoning": (
                "Fallback used the precomputed sentiment score but did not "
                "infer additional narratives."
            ),
        }

        return self.run(
            task_prompt=task_prompt,
            fallback_output=fallback_output,
            debug=debug,
        )


class RiskReviewAgent(BaseAgent):
    """
    Reviews prior outputs and highlights weak assumptions.

    A critique loop improves reliability because it gives the workflow a role
    whose explicit job is to find unsupported claims before the final report is
    generated.
    """

    agent_name = "RiskReviewAgent"
    role_prompt = RISK_REVIEW_ROLE_PROMPT
    output_model = RiskReviewOutput

    def review(
        self,
        *,
        analyses_to_review: Mapping[str, Any],
        review_focus: str = "Check whether the conclusions are supported by evidence.",
        debug: bool = False,
    ) -> JSONDict:
        task_prompt = build_risk_review_prompt(
            analyses_to_review=analyses_to_review,
            review_focus=review_focus,
        )

        fallback_output = {
            "main_concerns": [
                "Risk review LLM step was unavailable."
            ],
            "missing_evidence": [
                "Manual review is needed before final synthesis."
            ],
            "challenged_assumptions": [],
            "clarification_requests": [
                "Confirm whether the quantitative and sentiment evidence is sufficient."
            ],
            "review_outcome": "needs_more_evidence",
        }

        return self.run(
            task_prompt=task_prompt,
            fallback_output=fallback_output,
            debug=debug,
        )


class PortfolioStrategistAgent(BaseAgent):
    """
    Produces the final risk-aware investment outlook.

    The strategist consumes specialist outputs instead of directly calling
    every tool. That keeps orchestration lightweight and makes the final answer
    auditable: you can inspect exactly which prior analyses influenced it.
    """

    agent_name = "PortfolioStrategistAgent"
    role_prompt = PORTFOLIO_STRATEGIST_ROLE_PROMPT
    output_model = PortfolioStrategyOutput

    def generate_report(
        self,
        *,
        ticker: str,
        quantitative_analysis: Mapping[str, Any],
        sentiment_analysis: Mapping[str, Any],
        risk_review: Optional[Mapping[str, Any]] = None,
        debug: bool = False,
    ) -> JSONDict:
        task_prompt = build_report_generation_prompt(
            ticker=ticker,
            quantitative_analysis=quantitative_analysis,
            sentiment_analysis=sentiment_analysis,
            risk_review=risk_review,
        )

        fallback_output = {
            "ticker": ticker,
            "financial_health_summary": (
                "Final strategist LLM step was unavailable; use the specialist "
                "outputs directly and avoid making a high-confidence conclusion."
            ),
            "top_three_risks": [
                "Final synthesis was unavailable.",
                "Quantitative and sentiment evidence may be incomplete.",
                "Manual review is needed before any investment decision.",
            ],
            "hedge_strategy": (
                "Use conservative position sizing or stay unallocated until "
                "the analysis can be reviewed."
            ),
            "evidence_used": [
                "quantitative_analysis",
                "sentiment_analysis",
                "risk_review",
            ],
            "confidence": 0.0,
        }

        result = self.run(
            task_prompt=task_prompt,
            fallback_output=fallback_output,
            debug=debug,
        )

        # Validate the core assessment-facing report fields against the
        # existing schema. Extra strategist fields remain useful for notebooks,
        # but FinalReport compatibility shows the final output can satisfy the
        # simpler project contract too.
        try:
            FinalReport.model_validate(result)
        except ValidationError:
            result = self._validate_output(fallback_output)
            if debug:
                result["_debug"] = {
                    "agent": self.agent_name,
                    "model": self.model,
                    "used_fallback": True,
                    "error": "FinalReport schema validation failed.",
                    "raw_response_preview": None,
                }

        return result


def run_critique_loop(
    *,
    quantitative_analysis: Mapping[str, Any],
    sentiment_analysis: Mapping[str, Any],
    reviewer: Optional[RiskReviewAgent] = None,
    debug: bool = False,
) -> JSONDict:
    """
    Convenience helper for notebook-friendly critique loops.

    The function is deliberately small: it packages prior outputs and asks the
    reviewer to critique them. No autonomous planning or hidden control flow is
    introduced, so the workflow remains explainable during an interview.
    """

    reviewer = reviewer or RiskReviewAgent()
    return reviewer.review(
        analyses_to_review={
            "quantitative_analysis": dict(quantitative_analysis),
            "sentiment_analysis": dict(sentiment_analysis),
        },
        debug=debug,
    )
