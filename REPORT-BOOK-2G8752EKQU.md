# Homework #1 Brief Report

# 뉴스 기사 신뢰도 분석 Agentic System

## 1. 시스템 개요

본 프로젝트는 **뉴스 기사 신뢰도 분석 Agentic System**을 구현한 것이다.
사용자는 분석하고 싶은 뉴스 기사의 URL 또는 로컬 텍스트 파일을 입력할 수 있으며, 시스템은 해당 기사 내용을 바탕으로 신뢰도를 분석한다.

본 시스템은 다음과 같은 기능을 수행한다.

1. 사용자가 입력한 뉴스 기사 URL에서 기사 본문을 추출한다.
2. 기사 제목, 작성자, 날짜, 문장, 인용문 등의 기본 정보를 구조화한다.
3. Hugging Face LLM 기반 agent들을 사용하여 기사 내용을 여러 관점에서 분석한다.
4. 주요 주장, 출처 투명성, 편향적 표현, 선정적 표현, 검증 필요성을 평가한다.
5. 여러 agent의 평가 결과를 종합하여 최종 신뢰도 점수를 계산한다.
6. 최종 분석 결과를 Markdown report와 JSON 파일로 저장한다.

본 시스템의 목적은 기사의 모든 주장이 실제로 참인지 외부 데이터베이스를 통해 완전하게 검증하는 것이 아니다. 대신, 기사가 얼마나 신뢰성 있게 작성되었는지, 출처와 근거가 충분한지, 편향적이거나 선정적인 표현이 있는지, 추가 검증이 필요한 주장이 무엇인지 분석하는 것을 목표로 한다.

본 시스템은 과제에서 요구한 agentic framework 중 하나인 **LangGraph**를 사용하여 구현하였다. LangGraph는 여러 agent를 상태 기반 workflow로 연결할 수 있기 때문에, 본 시스템처럼 여러 agent가 순차적으로 분석 결과를 공유하고 갱신하는 구조에 적합하다.

---

## 2. 사용한 Agentic Framework

본 Homework #1에서 사용한 agentic framework는 **LangGraph**이다.

LangGraph를 선택한 이유는 다음과 같다.

- 여러 agent를 graph node 형태로 구성할 수 있다.
- 각 agent가 공통 state를 읽고 업데이트할 수 있다.
- URL 수집, 기사 파싱, LLM 평가, 점수 계산, 리포트 생성 과정을 명확한 workflow로 표현할 수 있다.
- 기사 본문 추출 실패, LLM 호출 실패, 입력 부족 등의 상황을 조건 분기 형태로 처리할 수 있다.
- 각 agent의 역할과 실행 순서를 보고서에서 설명하기 쉽다.

본 시스템에서는 LangGraph의 `StateGraph`를 사용하여 agent들을 연결하였다. 각 agent는 현재 state를 입력으로 받아 특정 작업을 수행한 뒤, 그 결과를 다시 state에 저장한다.

---

## 3. 시스템 기능 설명

본 시스템은 크게 두 가지 입력 방식을 지원한다.

### 3.1 뉴스 기사 URL 입력

사용자는 분석하고 싶은 뉴스 기사의 URL을 입력할 수 있다.

실행 예시는 다음과 같다.

```bash
python run.py --url "https://example.com/news/article" --output outputs --print-summary
```

이 경우 시스템은 다음 과정을 수행한다.

1. URL에 접속하여 HTML을 다운로드한다.
2. 기사 본문으로 보이는 텍스트를 추출한다.
3. 추출된 기사 본문을 분석 가능한 형태로 정리한다.
4. Hugging Face LLM agent들이 신뢰도를 평가한다.
5. 최종 report를 생성한다.

실제 뉴스 웹사이트에 따라 자동 본문 추출이 완벽하지 않을 수 있다. 웹사이트가 JavaScript 렌더링을 사용하거나 크롤링을 차단하는 경우에는 로컬 텍스트 파일 입력 방식을 사용할 수 있다.

### 3.2 로컬 텍스트 파일 입력

사용자는 뉴스 기사 내용을 `.txt` 파일로 저장한 뒤 입력할 수도 있다.

실행 예시는 다음과 같다.

```bash
python run.py --input sample_articles/sample_news_korean.txt --output outputs --print-summary
```

이 방식은 URL 추출이 실패하거나, 사용자가 직접 복사한 기사 내용을 분석하고 싶을 때 유용하다.

### 3.3 출력 파일

시스템 실행 후 다음과 같은 결과 파일이 생성된다.

```text
outputs/news_credibility_report.md
outputs/news_credibility_report.json
outputs/news_credibility_workflow.txt
```

각 파일의 역할은 다음과 같다.

| 파일명                          | 설명                                                       |
| ------------------------------- | ---------------------------------------------------------- |
| `news_credibility_report.md`    | 사람이 읽기 쉬운 Markdown 형식의 최종 신뢰도 분석 report   |
| `news_credibility_report.json`  | 각 agent의 분석 결과와 최종 점수를 담은 구조화된 JSON 파일 |
| `news_credibility_workflow.txt` | agentic workflow의 실행 순서를 요약한 파일                 |

---

## 4. 실행 방법

### 4.1 가상환경 생성

프로젝트 폴더에서 다음 명령을 실행한다.

```bash
python -m venv .venv
```

Windows PowerShell에서는 다음 명령으로 가상환경을 활성화한다.

```powershell
.venv\Scripts\Activate.ps1
```

macOS 또는 Linux에서는 다음 명령을 사용한다.

```bash
source .venv/bin/activate
```

### 4.2 필요한 패키지 설치

```bash
python -m pip install -r requirements.txt
```

### 4.3 Hugging Face API Token 설정

본 시스템은 Hugging Face의 무료 LLM API를 사용하여 기사 신뢰도를 평가한다.
따라서 LLM 기반 분석을 사용하려면 Hugging Face token을 환경변수로 설정해야 한다.

Windows PowerShell에서는 다음과 같이 설정한다.

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

macOS 또는 Linux에서는 다음과 같이 설정한다.

```bash
export HF_TOKEN="hf_your_token_here"
```

토큰은 코드에 직접 작성하지 않고, 반드시 환경변수로 설정한다.

### 4.4 URL 기반 실행

```bash
python run.py --url "https://example.com/news/article" --output outputs --print-summary
```

### 4.5 로컬 기사 파일 기반 실행

```bash
python run.py --input sample_articles/sample_news_korean.txt --output outputs --print-summary
```

### 4.6 사용할 LLM 모델 지정

기본 모델 외에 다른 Hugging Face 모델을 사용하고 싶다면 다음과 같이 실행할 수 있다.

```bash
python run.py --url "https://example.com/news/article" --model Qwen/Qwen2.5-1.5B-Instruct --output outputs --print-summary
```

### 4.7 LLM 사용을 필수로 설정

기본적으로 시스템은 Hugging Face API 호출이 실패하더라도 fallback을 통해 제한적인 report를 생성할 수 있다.
하지만 LLM 호출이 반드시 성공해야 하도록 설정하고 싶다면 `--require-llm` 옵션을 사용할 수 있다.

```bash
python run.py --url "https://example.com/news/article" --require-llm --output outputs --print-summary
```

이 옵션을 사용하면 Hugging Face token이 없거나 LLM API 호출이 실패했을 때 프로그램이 오류를 발생시킨다.

---

## 5. 사용한 Agent와 역할

본 시스템은 총 **9개의 agent**를 사용한다.

| 번호 | Agent 이름                     | 역할                                                                                  |
| ---: | ------------------------------ | ------------------------------------------------------------------------------------- |
|    1 | URL Fetch Agent                | 사용자가 입력한 URL에서 뉴스 기사 HTML을 다운로드하고, 주요 기사 본문을 추출한다.     |
|    2 | Article Parser Agent           | 기사 제목, 작성자, 날짜, 문장, 인용문, 기본 메타데이터를 분석 가능한 형태로 정리한다. |
|    3 | LLM Summary Agent              | Hugging Face LLM을 사용하여 기사 내용을 요약하고 핵심 주제를 파악한다.                |
|    4 | LLM Claim Credibility Agent    | 기사에 포함된 주요 주장이 명확하고 검증 가능한지, 근거가 충분한지 평가한다.           |
|    5 | LLM Source Transparency Agent  | 작성자, 날짜, 출처, 인용, 링크 등 출처 정보가 투명하게 제시되었는지 평가한다.         |
|    6 | LLM Bias and Framing Agent     | 기사에 편향적 표현, 감정적 표현, 선정적 문구, 일방적 프레이밍이 있는지 분석한다.      |
|    7 | LLM Verification Planner Agent | 추가 검증이 필요한 주장과 그 주장을 검증하기 위한 방법을 제안한다.                    |
|    8 | Score Aggregator Agent         | 여러 LLM agent의 분석 결과를 종합하여 최종 신뢰도 점수와 등급을 계산한다.             |
|    9 | Report Writer Agent            | 최종 분석 결과를 Markdown report, JSON output, workflow summary 파일로 저장한다.      |

각 agent는 독립적인 역할을 수행하며, 이전 agent의 결과를 다음 agent가 활용한다.
이 구조는 단일 agent가 모든 작업을 처리하는 방식보다 모듈화가 잘 되어 있고, 각 분석 단계의 목적을 명확하게 설명할 수 있다는 장점이 있다.

---

## 6. 사용한 Agentic Design Pattern과 목적

본 시스템은 총 **10개의 agentic design pattern**을 사용한다.

| 번호 | Design Pattern               | 사용 목적                                                                                                                           |
| ---: | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
|    1 | Multi-Agent Collaboration    | 신뢰도 분석 작업을 여러 전문 agent로 분리하여 협업하도록 구성하였다.                                                                |
|    2 | State Graph Workflow         | LangGraph의 `StateGraph`를 사용하여 agent들을 graph node로 연결하고, 공통 state를 공유하도록 하였다.                                |
|    3 | Sequential Pipeline          | URL 수집 → 기사 파싱 → 요약 → 주장 평가 → 출처 평가 → 편향 평가 → 검증 계획 → 점수 계산 → report 생성 순서로 workflow를 구성하였다. |
|    4 | Tool Use                     | URL 다운로드, HTML 파싱, 본문 추출, 문장 분리, JSON 파싱, 파일 저장 등의 기능을 tool로 분리하여 사용하였다.                         |
|    5 | LLM-as-Judge                 | 단순 하드코딩 규칙만 사용하지 않고, Hugging Face LLM이 기사 신뢰도를 평가하도록 하였다.                                             |
|    6 | Structured Output            | LLM agent의 결과를 JSON 형태로 구조화하여 후속 agent가 안정적으로 처리할 수 있게 하였다.                                            |
|    7 | Reflection / Critical Review | Verification Planner Agent가 기사의 주장을 비판적으로 검토하고 추가 검증이 필요한 부분을 식별한다.                                  |
|    8 | Conditional Routing          | URL 추출 실패, 기사 본문 부족, LLM 호출 실패 등 상황에 따라 warning 또는 fallback 경로로 처리할 수 있도록 설계하였다.               |
|    9 | Score Aggregation            | 여러 평가 기준의 점수를 가중합하여 최종 신뢰도 점수를 산출한다.                                                                     |
|   10 | Graceful Fallback            | Hugging Face token이 없거나 API 호출에 실패해도 전체 프로그램이 즉시 종료되지 않고 제한적인 report를 생성할 수 있도록 하였다.       |

이러한 design pattern을 통해 본 시스템은 단순한 스크립트가 아니라, 여러 agent가 협력하여 분석하고, 검토하고, 결과를 종합하는 agentic system으로 동작한다.

---

## 7. 신뢰도 평가 방식

본 시스템은 여러 관점에서 기사의 신뢰도를 분석한다.

주요 평가 기준은 다음과 같다.

1. 주요 주장의 명확성
2. 주장에 대한 근거의 충분성
3. 출처와 인용의 투명성
4. 작성자와 날짜 정보의 존재 여부
5. 편향적 표현 또는 선정적 표현의 사용 여부
6. 반대 관점 또는 균형 잡힌 설명의 포함 여부
7. 추가 검증이 필요한 주장 여부

각 LLM 기반 agent는 자신의 분석 영역에 대해 점수와 설명을 생성한다.
이후 Score Aggregator Agent가 각 agent의 결과를 종합하여 0점부터 100점 사이의 최종 신뢰도 점수를 계산한다.

최종 점수는 다음과 같은 등급으로 변환된다.

```text
High reliability
Moderate reliability
Low reliability
```

최종 report에는 단순 점수뿐만 아니라, 점수의 이유, 우려 사항, 긍정적 요소, 추가 검증이 필요한 부분도 함께 포함된다.

---

## 8. 실행 예시

로컬 샘플 기사 파일을 사용한 실행 예시는 다음과 같다.

```bash
python run.py --input sample_articles/sample_news_korean.txt --output outputs --print-summary
```

실행 결과 예시는 다음과 같다.

```text
News Credibility Analysis completed successfully.
Framework status: LangGraph workflow is active.
Report: outputs/news_credibility_report.md
JSON: outputs/news_credibility_report.json
Workflow: outputs/news_credibility_workflow.txt
Final score: 77.77 / 100
Reliability label: Moderate reliability
```

Hugging Face LLM은 모델 응답에 따라 결과가 달라질 수 있으므로, 실제 점수는 실행 환경과 입력 기사에 따라 조금씩 달라질 수 있다.

---

## 9. 시스템의 장점

본 시스템의 장점은 다음과 같다.

첫째, 사용자가 직접 뉴스 기사 URL을 입력하여 분석할 수 있으므로 실제 활용성이 높다.

둘째, 단순 키워드 기반 하드코딩 평가가 아니라 Hugging Face LLM을 사용하여 기사 내용을 더 유연하게 분석한다.

셋째, 여러 agent가 각각 다른 관점에서 분석을 수행하므로 결과가 더 설명 가능하다.

넷째, Markdown report와 JSON output을 모두 생성하므로 사람이 읽기에도 좋고, 후속 프로그램에서 활용하기에도 좋다.

다섯째, Hugging Face token이 없거나 URL 본문 추출이 실패하는 경우에도 fallback 또는 로컬 텍스트 입력 방식을 통해 실행 가능성을 높였다.

---

## 10. 결론

본 프로젝트는 LangGraph 기반의 뉴스 기사 신뢰도 분석 agentic system을 구현하였다.
사용자는 뉴스 기사 URL 또는 로컬 텍스트 파일을 입력할 수 있으며, 시스템은 여러 LLM 기반 agent를 통해 기사 요약, 주장 신뢰도 평가, 출처 투명성 평가, 편향성 분석, 검증 계획 수립, 최종 점수 계산을 수행한다.
