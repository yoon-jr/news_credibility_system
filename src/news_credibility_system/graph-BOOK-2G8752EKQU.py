"""LangGraph workflow assembly for the URL-based LLM news credibility system."""

from __future__ import annotations

from typing import Dict, List, Optional

from .agents import (
    ArticleParserAgent,
    LLMBiasFramingAgent,
    LLMClaimCredibilityAgent,
    LLMSourceTransparencyAgent,
    LLMSummaryAgent,
    LLMVerificationPlannerAgent,
    ReportWriterAgent,
    ScoreAggregatorAgent,
    URLFetchAgent,
    build_agent_registry,
)
from .llm import DEFAULT_HF_MODEL, DEFAULT_HF_PROVIDER, HuggingFaceLLMClient
from .state import GraphState

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - safe fallback when dependency is missing.
    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False


def route_after_fetch(state: GraphState) -> str:
    """Conditional router used after URL/local extraction."""
    if not (state.get("article_text") or "").strip():
        return "extraction_failed"
    if len((state.get("article_text") or "").strip()) < 300:
        return "limited_article"
    return "normal_article"


def extraction_failed_handler(state: GraphState) -> GraphState:
    warnings = list(state.get("warnings", []))
    warnings.append("Conditional route selected: extraction failed. A minimal report will still be generated.")
    return {**state, "article_text": state.get("article_text", ""), "warnings": warnings, "route": "extraction_failed"}


def limited_article_handler(state: GraphState) -> GraphState:
    warnings = list(state.get("warnings", []))
    warnings.append("Conditional route selected: limited-article path because extracted content is short.")
    return {**state, "warnings": warnings, "route": "limited_article"}


class NewsCredibilityWorkflow:
    """Build and run the multi-agent LangGraph workflow."""

    def __init__(
        self,
        *,
        llm_model: str = DEFAULT_HF_MODEL,
        provider: str = DEFAULT_HF_PROVIDER,
        hf_token: Optional[str] = None,
        llm_timeout: int = 60,
        max_retries: int = 1,
        allow_llm_fallback: bool = True,
    ) -> None:
        self.llm_client = HuggingFaceLLMClient(
            model=llm_model,
            provider=provider,
            api_key=hf_token,
            timeout=llm_timeout,
            max_retries=max_retries,
            allow_fallback=allow_llm_fallback,
        )
        self.url_fetcher = URLFetchAgent()
        self.parser = ArticleParserAgent()
        self.summary = LLMSummaryAgent(self.llm_client)
        self.claim_credibility = LLMClaimCredibilityAgent(self.llm_client)
        self.source_transparency = LLMSourceTransparencyAgent(self.llm_client)
        self.bias_framing = LLMBiasFramingAgent(self.llm_client)
        self.verification_planner = LLMVerificationPlannerAgent(self.llm_client)
        self.score_aggregator = ScoreAggregatorAgent()
        self.report_writer = ReportWriterAgent()
        self.agent_registry = build_agent_registry(self.llm_client)
        self._compiled_graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    @property
    def framework_status(self) -> str:
        if LANGGRAPH_AVAILABLE:
            return "LangGraph workflow is active."
        return "LangGraph is not installed; sequential fallback is active. Install requirements.txt for formal evaluation."

    def _build_graph(self):
        if StateGraph is None:
            return None
        graph = StateGraph(GraphState)
        graph.add_node("url_fetch_agent", self.url_fetcher)
        graph.add_node("extraction_failed_handler", extraction_failed_handler)
        graph.add_node("limited_article_handler", limited_article_handler)
        graph.add_node("article_parser_agent", self.parser)
        graph.add_node("llm_summary_agent", self.summary)
        graph.add_node("llm_claim_credibility_agent", self.claim_credibility)
        graph.add_node("llm_source_transparency_agent", self.source_transparency)
        graph.add_node("llm_bias_framing_agent", self.bias_framing)
        graph.add_node("llm_verification_planner_agent", self.verification_planner)
        graph.add_node("score_aggregator_agent", self.score_aggregator)
        graph.add_node("report_writer_agent", self.report_writer)

        graph.add_edge(START, "url_fetch_agent")
        graph.add_conditional_edges(
            "url_fetch_agent",
            route_after_fetch,
            {
                "normal_article": "article_parser_agent",
                "limited_article": "limited_article_handler",
                "extraction_failed": "extraction_failed_handler",
            },
        )
        graph.add_edge("limited_article_handler", "article_parser_agent")
        graph.add_edge("extraction_failed_handler", "article_parser_agent")
        graph.add_edge("article_parser_agent", "llm_summary_agent")
        graph.add_edge("llm_summary_agent", "llm_claim_credibility_agent")
        graph.add_edge("llm_claim_credibility_agent", "llm_source_transparency_agent")
        graph.add_edge("llm_source_transparency_agent", "llm_bias_framing_agent")
        graph.add_edge("llm_bias_framing_agent", "llm_verification_planner_agent")
        graph.add_edge("llm_verification_planner_agent", "score_aggregator_agent")
        graph.add_edge("score_aggregator_agent", "report_writer_agent")
        graph.add_edge("report_writer_agent", END)
        return graph.compile()

    def run(
        self,
        *,
        article_url: str = "",
        article_text: str = "",
        input_path: str = "",
        max_article_chars: int = 12000,
        fetch_timeout: int = 20,
        fetch_max_chars: int = 20000,
    ) -> GraphState:
        initial_state: GraphState = {
            "article_url": article_url,
            "article_text": article_text,
            "input_path": input_path,
            "warnings": [],
            "route": "normal_article",
            "llm_reviews": {},
            "llm_config": {
                **self.llm_client.status(),
                "max_article_chars": max_article_chars,
            },
            "fetch_config": {
                "timeout": fetch_timeout,
                "max_chars": fetch_max_chars,
            },
        }
        if self._compiled_graph is not None:
            result = self._compiled_graph.invoke(initial_state)
            return dict(result)  # type: ignore[return-value]
        return self._run_fallback(initial_state)

    def _run_fallback(self, state: GraphState) -> GraphState:
        """Sequential fallback keeps the demo runnable if LangGraph was not installed."""
        state = self.url_fetcher(state)
        route = route_after_fetch(state)
        if route == "extraction_failed":
            state = extraction_failed_handler(state)
        elif route == "limited_article":
            state = limited_article_handler(state)
        for node in [
            self.parser,
            self.summary,
            self.claim_credibility,
            self.source_transparency,
            self.bias_framing,
            self.verification_planner,
            self.score_aggregator,
            self.report_writer,
        ]:
            state = node(state)
        return state

    def workflow_edges(self) -> List[str]:
        return [
            "START -> URL Fetch Agent",
            "URL Fetch Agent -> Article Parser Agent OR Limited Article Handler OR Extraction Failed Handler",
            "Limited Article Handler -> Article Parser Agent",
            "Extraction Failed Handler -> Article Parser Agent",
            "Article Parser Agent -> LLM Summary Agent",
            "LLM Summary Agent -> LLM Claim Credibility Agent",
            "LLM Claim Credibility Agent -> LLM Source Transparency Agent",
            "LLM Source Transparency Agent -> LLM Bias and Framing Agent",
            "LLM Bias and Framing Agent -> LLM Verification Planner Agent",
            "LLM Verification Planner Agent -> Score Aggregator Agent",
            "Score Aggregator Agent -> Report Writer Agent",
            "Report Writer Agent -> END",
        ]
