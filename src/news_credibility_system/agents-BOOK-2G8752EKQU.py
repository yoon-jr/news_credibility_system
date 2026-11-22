"""Agent implementations for the URL-based LLM news credibility system."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .llm import HuggingFaceLLMClient
from .state import GraphState
from .tools import (
    ANONYMOUS_MARKERS,
    CERTAINTY_WORDS,
    CLAIM_MARKERS,
    HEDGE_WORDS,
    LOADED_WORDS,
    OPINION_MARKERS,
    SOURCE_MARKERS,
    clamp,
    contains_any,
    count_keyword_hits,
    ensure_list,
    ensure_score,
    extract_metadata,
    extract_quotes,
    extract_title_and_body,
    fetch_article_from_url,
    safe_ratio,
    score_to_label,
    sentence_has_number_or_date,
    sentence_has_source,
    split_sentences,
    top_items,
    truncate_for_llm,
)


class BaseAgent:
    """Small base class used for reportable role metadata."""

    name: str = "Base Agent"
    role: str = "Base role"
    goal: str = "Base goal"

    def run(self, state: GraphState) -> GraphState:
        raise NotImplementedError

    def __call__(self, state: GraphState) -> GraphState:
        return self.run(state)


class URLFetchAgent(BaseAgent):
    name = "URL Fetch Agent"
    role = "Downloads the article URL and extracts readable article text."
    goal = "Turn a user-provided news URL into clean text for downstream LLM agents."

    def run(self, state: GraphState) -> GraphState:
        warnings = list(state.get("warnings", []))
        article_url = (state.get("article_url") or "").strip()
        article_text = state.get("article_text", "") or ""
        fetch_config = state.get("fetch_config", {}) if isinstance(state.get("fetch_config"), dict) else {}

        if article_url:
            result = fetch_article_from_url(
                article_url,
                timeout=int(fetch_config.get("timeout", 20)),
                max_chars=int(fetch_config.get("max_chars", 20000)),
            )
            warnings.extend(result.get("warnings", []))
            if result.get("status") == "ok" and result.get("text"):
                article_text = str(result.get("text", ""))
            else:
                warnings.append(result.get("error", "URL extraction failed."))
            route = "url_article" if article_text else "extraction_failed"
            return {
                **state,
                "article_text": article_text,
                "article_fetch": result,
                "warnings": warnings,
                "route": route,
            }

        if article_text.strip():
            return {
                **state,
                "article_fetch": {"status": "local_input", "text_length": len(article_text)},
                "warnings": warnings,
                "route": "local_article",
            }

        warnings.append("No URL or local article text was provided.")
        return {**state, "article_fetch": {"status": "missing_input"}, "warnings": warnings, "route": "extraction_failed"}


class ArticleParserAgent(BaseAgent):
    name = "Article Parser Agent"
    role = "Preprocesses the extracted article and builds metadata."
    goal = "Create title, clean sentences, dates, author information, URLs, and runtime warnings."

    def run(self, state: GraphState) -> GraphState:
        article_text = state.get("article_text", "") or ""
        warnings = list(state.get("warnings", []))
        fetch_result = state.get("article_fetch", {}) or {}
        page_metadata = dict(fetch_result.get("metadata", {}) or {})

        title, body = extract_title_and_body(article_text)
        page_title = fetch_result.get("title") or page_metadata.get("title")
        if page_title and str(page_title).strip() and title == "Untitled Article":
            title = str(page_title).strip()
        elif page_title and len(str(page_title)) > len(title):
            title = str(page_title).strip()[:180]

        sentences = split_sentences(body if body.strip() else article_text)
        metadata = extract_metadata(article_text, title, page_metadata=page_metadata)
        metadata["sentence_count"] = len(sentences)
        metadata["word_like_count"] = len(re.findall(r"[A-Za-z가-힣0-9]+", article_text))
        metadata["quote_count"] = len(extract_quotes(article_text))
        metadata["article_text_length"] = len(article_text)

        if len(article_text.strip()) < 300:
            warnings.append("The extracted article is short; credibility analysis may be limited.")
        if len(sentences) < 4:
            warnings.append("Fewer than four substantial sentences were detected.")
        if not metadata["has_author"]:
            warnings.append("No explicit author/reporter was detected.")
        if not metadata["has_date"]:
            warnings.append("No explicit publication or event date was detected.")

        return {
            **state,
            "title": title,
            "metadata": metadata,
            "sentences": sentences,
            "warnings": warnings,
        }


class LLMSummaryAgent(BaseAgent):
    name = "LLM Summary Agent"
    role = "Summarizes the article before detailed credibility analysis."
    goal = "Identify the topic, article type, and main points using a Hugging Face LLM."

    def __init__(self, llm_client: HuggingFaceLLMClient) -> None:
        self.llm_client = llm_client

    def run(self, state: GraphState) -> GraphState:
        text = truncate_for_llm(state.get("article_text", ""), int(state.get("llm_config", {}).get("max_article_chars", 12000)))
        title = state.get("title", "Untitled Article")
        fallback = self._fallback_summary(state)
        system_prompt = _json_system_prompt()
        user_prompt = f"""
You are a news analysis assistant. Summarize the article objectively.
Return JSON only using this schema:
{{
  "main_topic": "short topic",
  "article_type": "straight news / opinion / press release / analysis / unclear",
  "summary": "3-5 sentence summary",
  "key_points": ["point 1", "point 2", "point 3"],
  "potentially_sensitive_topic": true,
  "analysis": "brief reasoning"
}}

Title: {title}
Article:
{text}
""".strip()
        result = self.llm_client.complete_json(
            agent_name=self.name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_data=fallback,
            max_tokens=900,
        )
        summary = {
            **fallback,
            **result.data,
            "key_points": ensure_list(result.data.get("key_points"), fallback.get("key_points", []))[:6],
            "used_huggingface_llm": result.used_llm,
        }
        return _store_review(state, "summary", summary)

    @staticmethod
    def _fallback_summary(state: GraphState) -> Dict[str, Any]:
        sentences = state.get("sentences", [])[:4]
        title = state.get("title", "Untitled Article")
        return {
            "main_topic": title,
            "article_type": "unclear",
            "summary": " ".join(sentences) if sentences else "No substantial article text was available.",
            "key_points": sentences[:3],
            "potentially_sensitive_topic": False,
            "analysis": "Local fallback summary generated from the first extracted sentences.",
        }


class LLMClaimCredibilityAgent(BaseAgent):
    name = "LLM Claim Credibility Agent"
    role = "Evaluates factual claims and how well they are supported within the article."
    goal = "Use an LLM to rate claim specificity, support, verifiability, and missing evidence."

    def __init__(self, llm_client: HuggingFaceLLMClient) -> None:
        self.llm_client = llm_client

    def run(self, state: GraphState) -> GraphState:
        text = truncate_for_llm(state.get("article_text", ""), int(state.get("llm_config", {}).get("max_article_chars", 12000)))
        title = state.get("title", "Untitled Article")
        fallback = self._fallback_claims(state)
        system_prompt = _json_system_prompt()
        user_prompt = f"""
You are a careful news credibility analyst. Evaluate only the article text below; do not invent outside facts.
Focus on whether the article makes concrete claims, cites evidence, gives numbers/dates/context, and avoids unsupported leaps.
Return JSON only using this schema:
{{
  "score": 0-100,
  "claims": [
    {{"claim": "claim text", "verifiability": "high/medium/low", "support_in_article": "strong/moderate/weak", "risk": "short risk note"}}
  ],
  "unsupported_claims": ["claim that needs evidence"],
  "strengths": ["specific strength"],
  "concerns": ["specific concern"],
  "analysis": "why this score was assigned"
}}
Rubric: 90+ = concrete claims with clear evidence and context; 70 = mostly supported; 50 = mixed; 30 or below = many unsupported or vague claims.

Title: {title}
Article:
{text}
""".strip()
        result = self.llm_client.complete_json(
            agent_name=self.name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_data=fallback,
            max_tokens=1400,
        )
        assessment = _normalize_assessment(result.data, fallback, result.used_llm)
        assessment["claims"] = ensure_list(result.data.get("claims"), fallback.get("claims", []))[:8]
        assessment["unsupported_claims"] = ensure_list(result.data.get("unsupported_claims"), fallback.get("unsupported_claims", []))[:8]
        return _store_review(state, "claim_assessment", assessment)

    @staticmethod
    def _fallback_claims(state: GraphState) -> Dict[str, Any]:
        claims: List[Dict[str, Any]] = []
        for sentence in state.get("sentences", []):
            indicators: List[str] = []
            if sentence_has_number_or_date(sentence):
                indicators.append("number_or_date")
            if contains_any(sentence, CLAIM_MARKERS):
                indicators.append("claim_marker")
            if contains_any(sentence, SOURCE_MARKERS):
                indicators.append("source_marker")
            if indicators:
                claims.append(
                    {
                        "claim": sentence,
                        "verifiability": "high" if "number_or_date" in indicators else "medium",
                        "support_in_article": "moderate" if sentence_has_source(sentence) else "weak",
                        "risk": "Local heuristic extraction; verify with LLM or external sources.",
                    }
                )
        unsupported = [claim["claim"] for claim in claims if claim.get("support_in_article") == "weak"]
        support_ratio = safe_ratio(len(claims) - len(unsupported), max(len(claims), 1))
        score = clamp(40 + 45 * support_ratio + min(10, len(claims))) if claims else 35
        return {
            "score": round(score, 2),
            "claims": claims[:6],
            "unsupported_claims": unsupported[:6],
            "strengths": ["Some checkable claims were detected."] if claims else [],
            "concerns": ["No clear checkable claims were detected."] if not claims else ["Some claims may need stronger attribution."],
            "analysis": "Fallback claim assessment based on source markers, numbers, and dates.",
        }


class LLMSourceTransparencyAgent(BaseAgent):
    name = "LLM Source Transparency Agent"
    role = "Evaluates source transparency, attribution, dates, bylines, and citation quality."
    goal = "Use an LLM to judge whether readers can trace where the information came from."

    def __init__(self, llm_client: HuggingFaceLLMClient) -> None:
        self.llm_client = llm_client

    def run(self, state: GraphState) -> GraphState:
        text = truncate_for_llm(state.get("article_text", ""), int(state.get("llm_config", {}).get("max_article_chars", 12000)))
        metadata = state.get("metadata", {})
        fallback = self._fallback_source(state)
        system_prompt = _json_system_prompt()
        user_prompt = f"""
You are a media literacy evaluator. Assess source transparency in the article.
Evaluate byline, publication date, named sources, direct quotes, links, documents/data, anonymous sources, and whether attribution is specific.
Return JSON only using this schema:
{{
  "score": 0-100,
  "named_sources": ["source 1"],
  "anonymous_or_vague_sources": ["vague source phrase"],
  "missing_transparency_elements": ["missing item"],
  "strengths": ["specific strength"],
  "concerns": ["specific concern"],
  "analysis": "why this score was assigned"
}}

Detected metadata: {metadata}
Article:
{text}
""".strip()
        result = self.llm_client.complete_json(
            agent_name=self.name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_data=fallback,
            max_tokens=1200,
        )
        assessment = _normalize_assessment(result.data, fallback, result.used_llm)
        for key in ["named_sources", "anonymous_or_vague_sources", "missing_transparency_elements"]:
            assessment[key] = ensure_list(result.data.get(key), fallback.get(key, []))[:8]
        return _store_review(state, "source_assessment", assessment)

    @staticmethod
    def _fallback_source(state: GraphState) -> Dict[str, Any]:
        article_text = state.get("article_text", "")
        metadata = state.get("metadata", {})
        source_hits = count_keyword_hits(article_text, SOURCE_MARKERS)
        anonymous_hits = count_keyword_hits(article_text, ANONYMOUS_MARKERS)
        quotes = extract_quotes(article_text)
        source_score = 50.0
        source_score += 12 if metadata.get("has_author") else -8
        source_score += 12 if metadata.get("has_date") else -8
        source_score += min(14, metadata.get("url_count", 0) * 4)
        source_score += min(14, len(quotes) * 3)
        source_score += min(16, sum(source_hits.values()) * 2)
        source_score -= min(16, sum(anonymous_hits.values()) * 5)
        missing = []
        if not metadata.get("has_author"):
            missing.append("author")
        if not metadata.get("has_date"):
            missing.append("publication date")
        if not quotes:
            missing.append("direct quotes")
        return {
            "score": round(clamp(source_score), 2),
            "named_sources": [item["term"] for item in top_items(source_hits, 5)],
            "anonymous_or_vague_sources": [item["term"] for item in top_items(anonymous_hits, 5)],
            "missing_transparency_elements": missing,
            "strengths": ["Attribution markers were detected."] if source_hits else [],
            "concerns": ["Some transparency elements are missing."] if missing else [],
            "analysis": "Fallback source assessment based on byline/date/link/quote/source-marker signals.",
        }


class LLMBiasFramingAgent(BaseAgent):
    name = "LLM Bias and Framing Agent"
    role = "Assesses loaded language, sensationalism, and one-sided framing."
    goal = "Use an LLM to rate language neutrality and identify framing risks."

    def __init__(self, llm_client: HuggingFaceLLMClient) -> None:
        self.llm_client = llm_client

    def run(self, state: GraphState) -> GraphState:
        text = truncate_for_llm(state.get("article_text", ""), int(state.get("llm_config", {}).get("max_article_chars", 12000)))
        fallback = self._fallback_bias(state)
        system_prompt = _json_system_prompt()
        user_prompt = f"""
You are a neutral language and framing analyst. Evaluate whether the article uses balanced journalistic language.
Look for loaded wording, sensational phrases, unsupported blame, one-sided presentation, opinion mixed with facts, and missing counter-perspectives.
Return JSON only using this schema:
{{
  "score": 0-100,
  "loaded_phrases": ["phrase"],
  "one_sided_framing": ["example"],
  "counterperspectives_present": true,
  "strengths": ["specific strength"],
  "concerns": ["specific concern"],
  "analysis": "why this score was assigned"
}}
Rubric: higher score means more neutral and balanced language.

Article:
{text}
""".strip()
        result = self.llm_client.complete_json(
            agent_name=self.name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_data=fallback,
            max_tokens=1200,
        )
        assessment = _normalize_assessment(result.data, fallback, result.used_llm)
        assessment["loaded_phrases"] = ensure_list(result.data.get("loaded_phrases"), fallback.get("loaded_phrases", []))[:8]
        assessment["one_sided_framing"] = ensure_list(result.data.get("one_sided_framing"), fallback.get("one_sided_framing", []))[:8]
        assessment["counterperspectives_present"] = bool(result.data.get("counterperspectives_present", fallback.get("counterperspectives_present", False)))
        return _store_review(state, "bias_assessment", assessment)

    @staticmethod
    def _fallback_bias(state: GraphState) -> Dict[str, Any]:
        article_text = state.get("article_text", "")
        sentences = state.get("sentences", [])
        loaded_hits = count_keyword_hits(article_text, LOADED_WORDS)
        opinion_hits = count_keyword_hits(article_text, OPINION_MARKERS)
        certainty_hits = count_keyword_hits(article_text, CERTAINTY_WORDS)
        balance_markers = ["however", "but", "반면", "다만", "하지만", "한편", "또 다른", "according to critics", "전문가는"]
        balance_hits = count_keyword_hits(article_text, balance_markers)
        sensational_density = (sum(loaded_hits.values()) + sum(opinion_hits.values()) + sum(certainty_hits.values())) / max(len(sentences), 1)
        score = 88.0 - min(40, sensational_density * 20) + min(8, sum(balance_hits.values()) * 2)
        loaded_terms = [item["term"] for item in top_items({**loaded_hits, **opinion_hits, **certainty_hits}, 8)]
        return {
            "score": round(clamp(score), 2),
            "loaded_phrases": loaded_terms,
            "one_sided_framing": [],
            "counterperspectives_present": bool(balance_hits),
            "strengths": ["Few loaded-language markers were detected."] if not loaded_terms else [],
            "concerns": ["Potentially loaded or opinion-like wording was detected."] if loaded_terms else [],
            "analysis": "Fallback bias assessment based on loaded-language and balance-marker signals.",
        }


class LLMVerificationPlannerAgent(BaseAgent):
    name = "LLM Verification Planner Agent"
    role = "Creates an adversarial verification plan for the article's most important claims."
    goal = "Use an LLM to identify what should be externally verified and how serious the verification risk is."

    def __init__(self, llm_client: HuggingFaceLLMClient) -> None:
        self.llm_client = llm_client

    def run(self, state: GraphState) -> GraphState:
        text = truncate_for_llm(state.get("article_text", ""), int(state.get("llm_config", {}).get("max_article_chars", 12000)))
        title = state.get("title", "Untitled Article")
        fallback = self._fallback_verification(state)
        system_prompt = _json_system_prompt()
        user_prompt = f"""
You are an adversarial fact-check planning agent. You do NOT browse the web.
Your job is to inspect the article and list which claims a human should verify externally.
Return JSON only using this schema:
{{
  "score": 0-100,
  "highest_priority_checks": [
    {{"claim_or_issue": "what to check", "suggested_source": "where to verify", "risk_level": "high/medium/low"}}
  ],
  "red_flags": ["red flag"],
  "what_would_change_the_score": ["evidence that would improve or reduce trust"],
  "strengths": ["specific strength"],
  "concerns": ["specific concern"],
  "analysis": "why this verification-readiness score was assigned"
}}
Rubric: higher score means the article gives enough clear evidence that external verification would be straightforward; lower score means vague, hard-to-check, or suspicious claims.

Title: {title}
Article:
{text}
""".strip()
        result = self.llm_client.complete_json(
            agent_name=self.name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_data=fallback,
            max_tokens=1400,
        )
        assessment = _normalize_assessment(result.data, fallback, result.used_llm)
        for key in ["highest_priority_checks", "red_flags", "what_would_change_the_score"]:
            assessment[key] = ensure_list(result.data.get(key), fallback.get(key, []))[:8]
        return _store_review(state, "verification_assessment", assessment)

    @staticmethod
    def _fallback_verification(state: GraphState) -> Dict[str, Any]:
        sentences = state.get("sentences", [])
        checks = []
        for sentence in sentences:
            if sentence_has_number_or_date(sentence) or contains_any(sentence, CLAIM_MARKERS):
                checks.append(
                    {
                        "claim_or_issue": sentence,
                        "suggested_source": "official announcement, original report, public dataset, or named organization",
                        "risk_level": "medium" if sentence_has_source(sentence) else "high",
                    }
                )
        high_risk = sum(1 for item in checks if item["risk_level"] == "high")
        score = clamp(75 - min(35, high_risk * 6) + min(10, len(checks))) if checks else 40
        return {
            "score": round(score, 2),
            "highest_priority_checks": checks[:6],
            "red_flags": ["Some claims do not include clear attribution."] if high_risk else [],
            "what_would_change_the_score": ["Named primary sources, direct links to data, and clearer dates would improve confidence."],
            "strengths": ["Several claims can be translated into verification tasks."] if checks else [],
            "concerns": ["Few concrete claims were available for verification."] if not checks else [],
            "analysis": "Fallback verification plan generated from claim and source markers.",
        }


class ScoreAggregatorAgent(BaseAgent):
    name = "Score Aggregator Agent"
    role = "Aggregates LLM specialist assessments into a final reliability score."
    goal = "Produce a transparent score, label, and explanation using weighted LLM criteria."

    def run(self, state: GraphState) -> GraphState:
        claim_score = ensure_score(state.get("claim_assessment", {}).get("score"), 50)
        source_score = ensure_score(state.get("source_assessment", {}).get("score"), 50)
        bias_score = ensure_score(state.get("bias_assessment", {}).get("score"), 50)
        verification_score = ensure_score(state.get("verification_assessment", {}).get("score"), 50)
        metadata = state.get("metadata", {})
        warnings = list(state.get("warnings", []))

        weights = {
            "claim_credibility": 0.35,
            "source_transparency": 0.25,
            "language_neutrality": 0.20,
            "verification_readiness": 0.20,
        }
        final_score = (
            weights["claim_credibility"] * claim_score
            + weights["source_transparency"] * source_score
            + weights["language_neutrality"] * bias_score
            + weights["verification_readiness"] * verification_score
        )

        if metadata.get("article_text_length", 0) < 500:
            final_score -= 7
        if not metadata.get("has_date"):
            final_score -= 3
        if not metadata.get("has_author"):
            final_score -= 2

        reviews = state.get("llm_reviews", {})
        llm_agents = [
            "summary",
            "claim_assessment",
            "source_assessment",
            "bias_assessment",
            "verification_assessment",
        ]
        used_llm_count = sum(1 for key in llm_agents if reviews.get(key, {}).get("used_huggingface_llm"))
        if used_llm_count == 0:
            warnings.append("Hugging Face LLM was not used. Set HF_TOKEN to enable the intended LLM-based assessment.")
        elif used_llm_count < 4:
            warnings.append("Some LLM agents used fallback results because one or more HF API calls failed.")

        final_score = clamp(final_score)
        label = score_to_label(final_score)

        strengths: List[str] = []
        concerns: List[str] = []
        for key in ["claim_assessment", "source_assessment", "bias_assessment", "verification_assessment"]:
            assessment = state.get(key, {})
            strengths.extend(str(item) for item in ensure_list(assessment.get("strengths"))[:3])
            concerns.extend(str(item) for item in ensure_list(assessment.get("concerns"))[:3])
        concerns.extend(warnings[:5])

        scorecard = {
            "final_score": round(final_score, 2),
            "label": label,
            "component_scores": {
                "claim_credibility": claim_score,
                "source_transparency": source_score,
                "language_neutrality": bias_score,
                "verification_readiness": verification_score,
            },
            "weights": weights,
            "used_huggingface_llm_agent_count": used_llm_count,
            "total_llm_agent_count": len(llm_agents),
            "strengths": _dedupe(strengths)[:8],
            "concerns": _dedupe(concerns)[:10],
            "important_note": "The system analyzes the article text and asks an LLM to evaluate credibility signals. It does not perform live independent fact-checking beyond fetching the given URL.",
        }
        return {**state, "scorecard": scorecard, "warnings": warnings}


class ReportWriterAgent(BaseAgent):
    name = "Report Writer Agent"
    role = "Creates human-readable Markdown and machine-readable JSON reports."
    goal = "Summarize all agent findings in a clear report suitable for assignment evaluation."

    def run(self, state: GraphState) -> GraphState:
        title = state.get("title", "Untitled Article")
        metadata = state.get("metadata", {})
        fetch = state.get("article_fetch", {})
        summary = state.get("summary", {})
        claim = state.get("claim_assessment", {})
        source = state.get("source_assessment", {})
        bias = state.get("bias_assessment", {})
        verification = state.get("verification_assessment", {})
        scorecard = state.get("scorecard", {})
        warnings = state.get("warnings", [])
        llm_config = state.get("llm_config", {})

        md_lines: List[str] = []
        md_lines.append(f"# News Credibility Analysis Report: {title}")
        md_lines.append("")
        md_lines.append("## 1. Final Result")
        md_lines.append(f"- **Reliability label:** {scorecard.get('label', 'N/A')}")
        md_lines.append(f"- **Final score:** {scorecard.get('final_score', 'N/A')} / 100")
        md_lines.append(f"- **Hugging Face LLM agents used:** {scorecard.get('used_huggingface_llm_agent_count', 0)} / {scorecard.get('total_llm_agent_count', 0)}")
        md_lines.append(f"- **Model:** {llm_config.get('model', 'N/A')}")
        md_lines.append(f"- **Provider:** {llm_config.get('provider', 'N/A')}")
        md_lines.append("- **Scope:** URL article extraction + LLM-based credibility signal analysis. This is not a full independent fact-check.")
        md_lines.append("")
        md_lines.append("## 2. Input and Extraction")
        md_lines.append(f"- Source URL: {metadata.get('source_url') or state.get('article_url') or 'Local file input'}")
        md_lines.append(f"- Resolved URL: {metadata.get('resolved_url') or fetch.get('resolved_url') or 'N/A'}")
        md_lines.append(f"- Extraction status: {fetch.get('status', 'N/A')}")
        md_lines.append(f"- Extraction method: {metadata.get('extraction_method') or fetch.get('extraction_method') or 'N/A'}")
        md_lines.append(f"- Article text length: {metadata.get('article_text_length', 0)} characters")
        md_lines.append("")
        md_lines.append("## 3. Article Metadata")
        md_lines.append(f"- Title: {metadata.get('title', title)}")
        md_lines.append(f"- Site name: {metadata.get('site_name') or 'Not detected'}")
        md_lines.append(f"- Author detected: {metadata.get('author') or 'Not detected'}")
        md_lines.append(f"- Dates detected: {', '.join(metadata.get('dates', [])) if metadata.get('dates') else 'Not detected'}")
        md_lines.append(f"- Sentence count: {metadata.get('sentence_count', 0)}")
        md_lines.append(f"- URL count in article text: {metadata.get('url_count', 0)}")
        md_lines.append("")
        md_lines.append("## 4. LLM Summary")
        md_lines.append(f"- Main topic: {summary.get('main_topic', 'N/A')}")
        md_lines.append(f"- Article type: {summary.get('article_type', 'N/A')}")
        md_lines.append(f"- Summary: {summary.get('summary', 'N/A')}")
        if summary.get("key_points"):
            md_lines.append("- Key points:")
            for item in ensure_list(summary.get("key_points"))[:6]:
                md_lines.append(f"  - {item}")
        md_lines.append("")
        md_lines.append("## 5. Component Scores")
        for key, value in scorecard.get("component_scores", {}).items():
            md_lines.append(f"- **{key}:** {value} / 100")
        md_lines.append("")
        md_lines.append("### 5.1 Claim Credibility Agent")
        md_lines.append(f"- Score: {claim.get('score', 'N/A')} / 100")
        md_lines.append(f"- Analysis: {claim.get('analysis', 'N/A')}")
        if claim.get("claims"):
            md_lines.append("- Important claims:")
            for item in ensure_list(claim.get("claims"))[:6]:
                if isinstance(item, dict):
                    md_lines.append(f"  - {item.get('claim', item)}")
                else:
                    md_lines.append(f"  - {item}")
        if claim.get("unsupported_claims"):
            md_lines.append("- Unsupported or weakly supported claims:")
            for item in ensure_list(claim.get("unsupported_claims"))[:6]:
                md_lines.append(f"  - {item}")
        md_lines.append("")
        md_lines.append("### 5.2 Source Transparency Agent")
        md_lines.append(f"- Score: {source.get('score', 'N/A')} / 100")
        md_lines.append(f"- Analysis: {source.get('analysis', 'N/A')}")
        if source.get("missing_transparency_elements"):
            md_lines.append("- Missing transparency elements: " + ", ".join(map(str, ensure_list(source.get("missing_transparency_elements")))))
        md_lines.append("")
        md_lines.append("### 5.3 Bias and Framing Agent")
        md_lines.append(f"- Score: {bias.get('score', 'N/A')} / 100")
        md_lines.append(f"- Analysis: {bias.get('analysis', 'N/A')}")
        if bias.get("loaded_phrases"):
            md_lines.append("- Loaded phrases: " + ", ".join(map(str, ensure_list(bias.get("loaded_phrases"))[:8])))
        md_lines.append("")
        md_lines.append("### 5.4 Verification Planner Agent")
        md_lines.append(f"- Score: {verification.get('score', 'N/A')} / 100")
        md_lines.append(f"- Analysis: {verification.get('analysis', 'N/A')}")
        if verification.get("highest_priority_checks"):
            md_lines.append("- Highest-priority checks:")
            for item in ensure_list(verification.get("highest_priority_checks"))[:6]:
                if isinstance(item, dict):
                    md_lines.append(f"  - {item.get('claim_or_issue', item)} | Suggested source: {item.get('suggested_source', 'N/A')} | Risk: {item.get('risk_level', 'N/A')}")
                else:
                    md_lines.append(f"  - {item}")
        md_lines.append("")
        md_lines.append("## 6. Strengths and Concerns")
        md_lines.append("### Strengths")
        if scorecard.get("strengths"):
            for item in scorecard.get("strengths", []):
                md_lines.append(f"- {item}")
        else:
            md_lines.append("- No major strengths detected.")
        md_lines.append("### Concerns")
        if scorecard.get("concerns"):
            for item in scorecard.get("concerns", []):
                md_lines.append(f"- {item}")
        else:
            md_lines.append("- No major concerns detected.")
        if warnings:
            md_lines.append("")
            md_lines.append("## 7. Runtime Warnings")
            for warning in _dedupe([str(warning) for warning in warnings]):
                md_lines.append(f"- {warning}")

        report_json: Dict[str, Any] = {
            "title": title,
            "input": {"article_url": state.get("article_url"), "input_path": state.get("input_path")},
            "article_fetch": fetch,
            "metadata": metadata,
            "llm_config": llm_config,
            "summary": summary,
            "claim_assessment": claim,
            "source_assessment": source,
            "bias_assessment": bias,
            "verification_assessment": verification,
            "scorecard": scorecard,
            "warnings": _dedupe([str(warning) for warning in warnings]),
        }
        return {**state, "report_markdown": "\n".join(md_lines) + "\n", "report_json": report_json}


def _json_system_prompt() -> str:
    return (
        "You are an expert news credibility analysis agent. "
        "You must return one valid JSON object only. Do not use Markdown fences. "
        "Do not claim that you verified facts on the live web. If information is missing, say it is missing. "
        "Use concise English or Korean based on the article language."
    )


def _normalize_assessment(data: Dict[str, Any], fallback: Dict[str, Any], used_llm: bool) -> Dict[str, Any]:
    merged = {**fallback, **data}
    merged["score"] = ensure_score(merged.get("score"), fallback.get("score", 50))
    merged["strengths"] = ensure_list(merged.get("strengths"), fallback.get("strengths", []))[:8]
    merged["concerns"] = ensure_list(merged.get("concerns"), fallback.get("concerns", []))[:8]
    merged["analysis"] = str(merged.get("analysis") or fallback.get("analysis") or "No analysis returned.")
    merged["used_huggingface_llm"] = used_llm
    return merged


def _store_review(state: GraphState, key: str, assessment: Dict[str, Any]) -> GraphState:
    reviews = dict(state.get("llm_reviews", {}) or {})
    reviews[key] = assessment
    return {**state, key: assessment, "llm_reviews": reviews}


def _dedupe(items: List[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def build_agent_registry(llm_client: HuggingFaceLLMClient) -> List[BaseAgent]:
    """Return instantiated agents for report generation."""
    return [
        URLFetchAgent(),
        ArticleParserAgent(),
        LLMSummaryAgent(llm_client),
        LLMClaimCredibilityAgent(llm_client),
        LLMSourceTransparencyAgent(llm_client),
        LLMBiasFramingAgent(llm_client),
        LLMVerificationPlannerAgent(llm_client),
        ScoreAggregatorAgent(),
        ReportWriterAgent(),
    ]
