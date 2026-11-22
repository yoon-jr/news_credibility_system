# HW1 News Credibility Analysis System

This project implements a **URL-based news credibility analysis agentic system** using **LangGraph** and **Hugging Face Inference Providers**.

A user can provide either:

1. a news article URL, or
2. a local UTF-8 text file containing copied article text.

The system fetches/extracts article text, runs multiple LLM-based specialist agents, and generates Markdown/JSON credibility reports.

---

## Why this version is more dynamic

The first version used deterministic keyword rules. This version is more realistic:

- It accepts a real article URL through `--url`.
- It extracts article text using `trafilatura` with a BeautifulSoup fallback.
- It evaluates credibility with a Hugging Face-hosted LLM when `HF_TOKEN` is set.
- It still runs without crashing when a token or network is unavailable, because fallback outputs are generated with clear warnings.

---

## Recommended Hugging Face API and model

Recommended API:

- **Hugging Face Inference Providers** through `huggingface_hub.InferenceClient`
- Authentication variable: `HF_TOKEN`
- Default provider: `auto`
- Default model: `Qwen/Qwen2.5-1.5B-Instruct`

The default model was chosen because it is small enough for class-project use, instruction-tuned, and not gated like some Gemma models. For higher-quality analysis or longer articles, try:

```bash
--model Qwen/Qwen2.5-7B-Instruct-1M
```

The model and provider are configurable from the command line.

---

## Project structure

```text
news_credibility_system/
  README.md
  REPORT.md
  requirements.txt
  .env.example
  run.py
  sample_articles/
    sample_news_korean.txt
    sample_news_unreliable.txt
    sample_news_english.txt
  src/news_credibility_system/
    __init__.py
    agents.py
    graph.py
    llm.py
    state.py
    tools.py
  tests/
    test_system.py
  outputs/
```

---

## Installation

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Hugging Face token setup

Create a free Hugging Face account, create an access token with inference permission, and set it as an environment variable.

macOS/Linux:

```bash
export HF_TOKEN="hf_your_token_here"
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

If `HF_TOKEN` is not set, the system still runs and clearly reports that fallback analysis was used. For the intended LLM-based run, set `HF_TOKEN`.

---

## Run with a news URL

```bash
python run.py --url "https://example.com/news/article" --output outputs --print-summary
```

With a specific Hugging Face model:

```bash
python run.py --url "https://example.com/news/article" --model Qwen/Qwen2.5-1.5B-Instruct --provider auto --output outputs --print-summary
```

If you want the program to fail instead of using fallback results when the LLM is unavailable:

```bash
python run.py --url "https://example.com/news/article" --require-llm --print-summary
```

---

## Run with a local article text file

This is useful when a website blocks automated article extraction.

```bash
python run.py --input sample_articles/sample_news_korean.txt --output outputs --print-summary
```

---

## Output files

The program writes three files:

```text
outputs/news_credibility_report.md
outputs/news_credibility_report.json
outputs/news_credibility_workflow.txt
```

The Markdown report is for humans. The JSON report is useful for debugging and checking the exact agent outputs.

---

## Agents

This system uses 9 agents:

1. URL Fetch Agent
2. Article Parser Agent
3. LLM Summary Agent
4. LLM Claim Credibility Agent
5. LLM Source Transparency Agent
6. LLM Bias and Framing Agent
7. LLM Verification Planner Agent
8. Score Aggregator Agent
9. Report Writer Agent

---

## Agentic design patterns

This system uses more than five design patterns:

1. Multi-agent collaboration
2. State graph workflow
3. Tool use
4. Retrieval/extraction from URL
5. LLM-as-judge
6. Structured JSON output
7. Reflection / adversarial verification planning
8. Conditional routing
9. Score aggregation
10. Graceful fallback / fault tolerance

---

## Test

```bash
python -m pytest tests
```

The tests do not require a Hugging Face token. They verify that the workflow runs safely and produces reports even in offline or no-token environments.
