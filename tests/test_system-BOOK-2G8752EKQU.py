from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from news_credibility_system.graph import NewsCredibilityWorkflow


SAMPLE_TEXT = """테스트 뉴스 제목
작성자: 홍길동 기자
2026년 6월 10일

서울시가 2026년 6월 10일 새로운 대중교통 정책을 발표했다고 밝혔다.
시에 따르면 첫 단계 예산은 120억 원이며, 세부 집행 계획은 다음 달 공개된다.
전문가들은 정책 효과를 판단하려면 실제 이용률 데이터가 필요하다고 말했다.
다만 일부 시민단체는 예산 추산 방식이 충분히 공개되지 않았다고 지적했다.
"""


def test_workflow_runs_without_hf_token_on_local_text(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    workflow = NewsCredibilityWorkflow(llm_model="test/mock-model", provider="auto", max_retries=0)
    result = workflow.run(article_text=SAMPLE_TEXT, input_path="inline-test")

    assert result["report_markdown"].startswith("# News Credibility Analysis Report")
    assert 0 <= result["scorecard"]["final_score"] <= 100
    assert result["scorecard"]["used_huggingface_llm_agent_count"] == 0
    assert "claim_assessment" in result
    assert "source_assessment" in result
    assert "bias_assessment" in result
    assert "verification_assessment" in result


def test_url_input_uses_fetch_agent(monkeypatch):
    def fake_fetch(url: str, timeout: int = 20, max_chars: int = 20000):
        return {
            "status": "ok",
            "url": url,
            "resolved_url": url,
            "title": "URL 테스트 기사",
            "text": SAMPLE_TEXT,
            "metadata": {
                "title": "URL 테스트 기사",
                "author": "홍길동 기자",
                "published_date": "2026년 6월 10일",
                "source_url": url,
                "resolved_url": url,
                "site_name": "Example News",
                "extraction_method": "test_fake_fetch",
            },
            "warnings": [],
            "extraction_method": "test_fake_fetch",
        }

    monkeypatch.setattr("news_credibility_system.agents.fetch_article_from_url", fake_fetch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    workflow = NewsCredibilityWorkflow(llm_model="test/mock-model", provider="auto", max_retries=0)
    result = workflow.run(article_url="https://example.com/news")

    assert result["metadata"]["source_url"] == "https://example.com/news"
    assert result["article_fetch"]["status"] == "ok"
    assert result["report_json"]["input"]["article_url"] == "https://example.com/news"
