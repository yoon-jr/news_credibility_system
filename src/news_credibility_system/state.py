"""State definitions for the LangGraph news credibility workflow."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class GraphState(TypedDict, total=False):
    """Shared state passed between agents in the LangGraph workflow."""

    # Input
    article_url: str
    article_text: str
    input_path: str

    # URL fetching and parsing
    article_fetch: Dict[str, Any]
    fetch_config: Dict[str, Any]
    title: str
    metadata: Dict[str, Any]
    sentences: List[str]

    # LLM runtime configuration and agent outputs
    llm_config: Dict[str, Any]
    llm_reviews: Dict[str, Any]
    summary: Dict[str, Any]
    claim_assessment: Dict[str, Any]
    source_assessment: Dict[str, Any]
    bias_assessment: Dict[str, Any]
    verification_assessment: Dict[str, Any]

    # Final outputs
    scorecard: Dict[str, Any]
    report_markdown: str
    report_json: Dict[str, Any]
    warnings: List[str]
    route: str
