"""Helper tools for URL extraction, lightweight text analysis, and file output."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

SOURCE_MARKERS = [
    "according to",
    "said",
    "stated",
    "reported",
    "announced",
    "confirmed",
    "cited",
    "source",
    "official",
    "study",
    "report",
    "data",
    "survey",
    "전했다",
    "밝혔다",
    "말했다",
    "발표했다",
    "확인했다",
    "자료에 따르면",
    "보고서에 따르면",
    "관계자",
    "전문가",
    "조사",
    "통계",
    "데이터",
    "연구",
]

ANONYMOUS_MARKERS = [
    "anonymous",
    "unnamed",
    "sources said",
    "a source said",
    "관계자에 따르면",
    "익명의",
    "한 관계자",
    "복수의 관계자",
]

LOADED_WORDS = [
    "shocking",
    "outrageous",
    "disaster",
    "miracle",
    "secret",
    "explosive",
    "scandal",
    "fraud",
    "destroyed",
    "terrifying",
    "unbelievable",
    "충격",
    "경악",
    "대박",
    "난리",
    "끔찍",
    "완전",
    "절대",
    "반드시",
    "몰락",
    "폭망",
    "비밀",
    "분노",
    "참사",
    "논란",
    "최악",
    "기적",
]

HEDGE_WORDS = [
    "may",
    "might",
    "could",
    "possibly",
    "allegedly",
    "reportedly",
    "appears",
    "suggests",
    "추정",
    "가능성",
    "보인다",
    "전망",
    "예상",
    "의혹",
    "주장",
]

CERTAINTY_WORDS = [
    "definitely",
    "certainly",
    "always",
    "never",
    "undeniable",
    "guaranteed",
    "확실히",
    "무조건",
    "절대",
    "반드시",
    "100%",
]

CLAIM_MARKERS = [
    "announced",
    "reported",
    "confirmed",
    "claimed",
    "increased",
    "decreased",
    "rose",
    "fell",
    "will",
    "is expected",
    "shows",
    "found",
    "발표",
    "확인",
    "주장",
    "증가",
    "감소",
    "상승",
    "하락",
    "예상",
    "전망",
    "밝혔다",
    "나타났다",
    "집계",
]

OPINION_MARKERS = [
    "i think",
    "in my opinion",
    "clearly",
    "obviously",
    "should",
    "must",
    "생각한다",
    "분명하다",
    "당연히",
    "해야 한다",
    "비판했다",
]

DATE_PATTERNS = [
    r"\b\d{4}[-./]\d{1,2}[-./]\d{1,2}\b",
    r"\b\d{1,2}[-./]\d{1,2}[-./]\d{2,4}\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일\b",
    r"\b\d{1,2}\s*월\s*\d{1,2}\s*일\b",
]

NUMBER_PATTERN = re.compile(
    r"(?<![\w가-힣])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:%|퍼센트|명|건|개|억원|원|달러|명중|배|년|월|일|km|킬로미터|million|billion|people|cases)?",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
QUOTE_PATTERN = re.compile(r"[\"“”'‘’](.+?)[\"“”'‘’]|‘(.+?)’|“(.+?)”")
AUTHOR_PATTERN = re.compile(r"(?:^|\n)\s*(?:By\s+|기자\s*[:：]?\s*|작성자\s*[:：]?\s*)([A-Za-z가-힣 ._-]{2,40})", re.IGNORECASE)
KOREAN_REPORTER_PATTERN = re.compile(r"([가-힣]{2,5})\s*기자")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 NewsCredibilitySystem/2.0"
)


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    compact_lines: List[str] = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen:
                compact_lines.append("")
            blank_seen = True
        else:
            compact_lines.append(line)
            blank_seen = False
    return "\n".join(compact_lines).strip()


def split_sentences(text: str) -> List[str]:
    """A conservative sentence splitter for Korean and English news text."""
    cleaned = normalize_whitespace(text)
    candidates = re.split(r"(?<=[.!?。！？])\s+|(?<=[다요죠음임함])\.\s+|\n+", cleaned)
    sentences: List[str] = []
    for candidate in candidates:
        sentence = candidate.strip(" \t-•")
        if len(sentence) >= 8:
            sentences.append(sentence)
    return sentences


def _looks_like_metadata_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if AUTHOR_PATTERN.search(stripped) or KOREAN_REPORTER_PATTERN.search(stripped):
        return True
    if len(stripped) <= 50:
        for pattern in DATE_PATTERNS:
            if re.fullmatch(pattern, stripped, flags=re.IGNORECASE):
                return True
    return False


def extract_title_and_body(text: str) -> Tuple[str, str]:
    """Return title and body using the first substantial line as title."""
    cleaned = normalize_whitespace(text)
    lines = [line.strip(" #\t") for line in cleaned.split("\n") if line.strip()]
    if not lines:
        return "Untitled Article", ""
    title = lines[0]
    body_lines = [line for line in lines[1:] if not _looks_like_metadata_line(line)]
    body = "\n".join(body_lines) if body_lines else cleaned
    return title[:180], body


def regex_find_all(patterns: Sequence[str], text: str, flags: int = re.IGNORECASE) -> List[str]:
    matches: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags):
            matches.append(match.group(0))
    return matches


def count_keyword_hits(text: str, keywords: Sequence[str]) -> Dict[str, int]:
    lower = text.lower()
    hits: Dict[str, int] = {}
    for keyword in keywords:
        count = lower.count(keyword.lower())
        if count:
            hits[keyword] = count
    return hits


def contains_any(text: str, keywords: Sequence[str]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def extract_metadata(text: str, title: str, page_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dates = regex_find_all(DATE_PATTERNS, text)
    urls = URL_PATTERN.findall(text)
    author_match = AUTHOR_PATTERN.search(text)
    korean_reporters = KOREAN_REPORTER_PATTERN.findall(text)
    page_metadata = page_metadata or {}

    author = page_metadata.get("author") or None
    if not author:
        if author_match:
            author = author_match.group(1).strip()
        elif korean_reporters:
            author = korean_reporters[0].strip() + " 기자"

    page_date = page_metadata.get("published_date") or page_metadata.get("date")
    if page_date and page_date not in dates:
        dates.append(str(page_date))

    return {
        "title": title,
        "author": author,
        "dates": sorted(set(str(date) for date in dates if str(date).strip())),
        "url_count": len(urls),
        "urls": urls[:10],
        "has_author": bool(author),
        "has_date": bool(dates),
        "source_url": page_metadata.get("source_url"),
        "resolved_url": page_metadata.get("resolved_url"),
        "site_name": page_metadata.get("site_name"),
        "extraction_method": page_metadata.get("extraction_method"),
        "published_date_from_page": page_metadata.get("published_date"),
    }


def extract_quotes(text: str) -> List[str]:
    quotes: List[str] = []
    for match in QUOTE_PATTERN.findall(text):
        candidate = next((part for part in match if part), "").strip()
        if candidate and len(candidate) > 2:
            quotes.append(candidate)
    return quotes


def sentence_has_number_or_date(sentence: str) -> bool:
    return bool(NUMBER_PATTERN.search(sentence) or regex_find_all(DATE_PATTERNS, sentence))


def sentence_has_source(sentence: str) -> bool:
    return contains_any(sentence, SOURCE_MARKERS) or bool(URL_PATTERN.search(sentence))


def score_to_label(score: float) -> str:
    if score >= 80:
        return "High reliability"
    if score >= 60:
        return "Moderate reliability"
    if score >= 40:
        return "Low reliability"
    return "Very low reliability"


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def top_items(counter_like: Dict[str, int], limit: int = 10) -> List[Dict[str, Any]]:
    return [{"term": term, "count": count} for term, count in Counter(counter_like).most_common(limit)]


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must start with http:// or https:// and include a valid host.")


def _meta_content(soup: BeautifulSoup, *names: str) -> Optional[str]:
    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return str(tag.get("content")).strip()
    return None


def _extract_with_trafilatura(html: str, url: str) -> Tuple[str, Dict[str, Any], str]:
    """Use trafilatura when available; return empty text if extraction fails."""
    try:
        import trafilatura  # type: ignore
    except Exception:
        return "", {}, "trafilatura_not_installed"

    metadata: Dict[str, Any] = {}
    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
            output_format="txt",
        )
        meta_obj = trafilatura.extract_metadata(html, default_url=url)
        if meta_obj is not None:
            metadata = {
                "title": getattr(meta_obj, "title", None),
                "author": getattr(meta_obj, "author", None),
                "published_date": getattr(meta_obj, "date", None),
                "site_name": getattr(meta_obj, "sitename", None),
            }
        return normalize_whitespace(extracted or ""), metadata, "trafilatura"
    except Exception as exc:
        return "", {"trafilatura_error": str(exc)}, "trafilatura_error"


def _extract_with_bs4(html: str) -> Tuple[str, Dict[str, Any], str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = _meta_content(soup, "og:title", "twitter:title")
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()
    author = _meta_content(soup, "author", "article:author", "byl", "dc.creator")
    published_date = _meta_content(
        soup,
        "article:published_time",
        "article:modified_time",
        "date",
        "datePublished",
        "pubdate",
        "dc.date",
    )
    site_name = _meta_content(soup, "og:site_name")

    article = soup.find("article")
    paragraph_tags = article.find_all("p") if article else soup.find_all("p")
    paragraphs: List[str] = []
    for tag in paragraph_tags:
        text = tag.get_text(" ", strip=True)
        if len(text) >= 25:
            paragraphs.append(text)

    # Some pages use div spans instead of p tags.
    if len("\n".join(paragraphs)) < 300:
        candidates = []
        for tag in soup.find_all(["div", "section"]):
            text = tag.get_text(" ", strip=True)
            if 80 <= len(text) <= 2000:
                candidates.append(text)
        paragraphs.extend(candidates[:12])

    text = normalize_whitespace("\n\n".join(dict.fromkeys(paragraphs)))
    metadata = {
        "title": normalize_whitespace(title or ""),
        "author": normalize_whitespace(author or ""),
        "published_date": normalize_whitespace(published_date or ""),
        "site_name": normalize_whitespace(site_name or ""),
    }
    return text, metadata, "beautifulsoup"


def fetch_article_from_url(url: str, timeout: int = 20, max_chars: int = 20000) -> Dict[str, Any]:
    """Download a news URL and extract the likely article text.

    The function never raises for ordinary network/extraction failures. Instead,
    it returns a structured result with status="error" and a human-readable error.
    """
    warnings: List[str] = []
    result: Dict[str, Any] = {
        "status": "error",
        "url": url,
        "resolved_url": url,
        "status_code": None,
        "content_type": None,
        "title": "",
        "text": "",
        "metadata": {},
        "warnings": warnings,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        validate_url(url)
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            },
            timeout=timeout,
        )
        result["status_code"] = response.status_code
        result["resolved_url"] = response.url
        result["content_type"] = response.headers.get("Content-Type", "")
        response.raise_for_status()
    except requests.RequestException as exc:
        result["error"] = f"Failed to fetch URL: {exc}"
        return result

    html = response.text
    if "html" not in str(result["content_type"]).lower():
        warnings.append("The response Content-Type does not look like HTML; extraction may be poor.")

    text, metadata, method = _extract_with_trafilatura(html, response.url)
    if len(text) < 250:
        fallback_text, fallback_meta, fallback_method = _extract_with_bs4(html)
        if len(fallback_text) > len(text):
            text = fallback_text
            metadata = {**metadata, **{k: v for k, v in fallback_meta.items() if v}}
            method = fallback_method
        else:
            method = method or fallback_method

    soup_title = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        soup_title = _meta_content(soup, "og:title", "twitter:title") or (soup.title.string.strip() if soup.title and soup.title.string else "")
        soup_site = _meta_content(soup, "og:site_name")
        if soup_title and not metadata.get("title"):
            metadata["title"] = soup_title
        if soup_site and not metadata.get("site_name"):
            metadata["site_name"] = soup_site
    except Exception:
        pass

    text = normalize_whitespace(text)[:max_chars]
    title = normalize_whitespace(str(metadata.get("title") or soup_title or "Untitled Article"))[:180]
    if text and title and not text.startswith(title):
        combined_text = normalize_whitespace(title + "\n\n" + text)
    else:
        combined_text = text

    if len(combined_text) < 500:
        warnings.append("Extracted article text is short; the LLM assessment may be less reliable.")
    if not metadata.get("author"):
        warnings.append("Article author was not found in page metadata.")
    if not metadata.get("published_date"):
        warnings.append("Publication date was not found in page metadata.")

    result.update(
        {
            "status": "ok" if combined_text else "error",
            "title": title,
            "text": combined_text,
            "metadata": {
                **metadata,
                "source_url": url,
                "resolved_url": response.url,
                "extraction_method": method,
            },
            "extraction_method": method,
        }
    )
    if not combined_text:
        result["error"] = "Article text could not be extracted from the URL. Try another news URL or use --input with copied text."
    return result


def truncate_for_llm(text: str, max_chars: int = 12000) -> str:
    """Keep prompts under a manageable size without cutting too aggressively."""
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= max_chars:
        return cleaned
    head = cleaned[: int(max_chars * 0.72)].rstrip()
    tail = cleaned[-int(max_chars * 0.20) :].lstrip()
    return head + "\n\n[... middle of article truncated for prompt length ...]\n\n" + tail


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from an LLM response."""
    if not text:
        return None
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.IGNORECASE | re.DOTALL)
    candidates = []
    if fence_match:
        candidates.append(fence_match.group(1))
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(cleaned[first : last + 1])
    candidates.append(cleaned)
    for candidate in candidates:
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate.strip())
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def ensure_list(value: Any, default: Optional[List[Any]] = None) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return list(default or [])
    return [value]


def ensure_score(value: Any, default: float = 50.0) -> float:
    try:
        return round(clamp(float(value)), 2)
    except (TypeError, ValueError):
        return round(clamp(default), 2)
