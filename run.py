"""Command-line entry point for the URL-based LLM news credibility system."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from news_credibility_system.graph import NewsCredibilityWorkflow  # noqa: E402
from news_credibility_system.llm import DEFAULT_HF_MODEL, DEFAULT_HF_PROVIDER  # noqa: E402
from news_credibility_system.tools import write_json, write_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze credibility of a news URL or local article text using a LangGraph + Hugging Face LLM multi-agent workflow."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--url",
        "-u",
        help="URL of the news article to fetch and analyze.",
    )
    input_group.add_argument(
        "--input",
        "-i",
        help="Path to a UTF-8 text file containing copied article text. Useful when the URL blocks extraction.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs",
        help="Directory where report.md and report.json will be saved. Default: outputs",
    )
    parser.add_argument(
        "--prefix",
        default="news_credibility",
        help="Output filename prefix. Default: news_credibility",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_HF_MODEL,
        help=f"Hugging Face chat model. Default: {DEFAULT_HF_MODEL}",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_HF_PROVIDER,
        help=f"Hugging Face Inference Provider. Default: {DEFAULT_HF_PROVIDER}",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each Hugging Face LLM call. Default: 60",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Number of retries for each Hugging Face LLM call. Default: 1",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail if HF_TOKEN is missing or if the Hugging Face LLM call fails. By default, the system uses safe fallback results instead of crashing.",
    )
    parser.add_argument(
        "--max-article-chars",
        type=int,
        default=12000,
        help="Maximum article characters sent to each LLM prompt. Default: 12000",
    )
    parser.add_argument(
        "--fetch-timeout",
        type=int,
        default=20,
        help="Timeout in seconds for URL downloading. Default: 20",
    )
    parser.add_argument(
        "--fetch-max-chars",
        type=int,
        default=20000,
        help="Maximum extracted article characters kept from the URL. Default: 20000",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a short summary to the terminal after writing files.",
    )
    return parser.parse_args()


def _read_local_article(input_path: str) -> str:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    if not path.is_file():
        raise ValueError(f"input path is not a file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(exc.encoding, exc.object, exc.start, exc.end, "input file must be UTF-8 encoded") from exc


def main() -> int:
    args = parse_args()
    article_text = ""
    input_path = ""
    if args.input:
        try:
            article_text = _read_local_article(args.input)
            input_path = str(Path(args.input))
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    try:
        workflow = NewsCredibilityWorkflow(
            llm_model=args.model,
            provider=args.provider,
            llm_timeout=args.llm_timeout,
            max_retries=args.max_retries,
            allow_llm_fallback=not args.require_llm,
        )
        result = workflow.run(
            article_url=args.url or "",
            article_text=article_text,
            input_path=input_path,
            max_article_chars=args.max_article_chars,
            fetch_timeout=args.fetch_timeout,
            fetch_max_chars=args.fetch_max_chars,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    md_path = output_dir / f"{args.prefix}_report.md"
    json_path = output_dir / f"{args.prefix}_report.json"
    graph_path = output_dir / f"{args.prefix}_workflow.txt"

    write_text(md_path, result.get("report_markdown", ""))
    write_json(json_path, result.get("report_json", {}))
    write_text(graph_path, "\n".join(workflow.workflow_edges()) + "\n")

    scorecard = result.get("scorecard", {})
    print("News Credibility Analysis completed successfully.")
    print(f"Framework status: {workflow.framework_status}")
    print(f"Report: {md_path}")
    print(f"JSON: {json_path}")
    print(f"Workflow: {graph_path}")
    if args.print_summary:
        print(f"Final score: {scorecard.get('final_score', 'N/A')} / 100")
        print(f"Reliability label: {scorecard.get('label', 'N/A')}")
        print(
            "Hugging Face LLM agents used: "
            f"{scorecard.get('used_huggingface_llm_agent_count', 'N/A')} / {scorecard.get('total_llm_agent_count', 'N/A')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
