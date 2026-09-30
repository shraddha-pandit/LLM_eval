# LLM Risk Evaluation — Financial Services Domain

A structured evaluation framework for assessing large language model (LLM) reliability in a UK financial services context. The project tests a production-grade model against a purpose-built prompt suite and produces a scored dataset and a model risk validation note suitable for use in a Model Risk Management (MRM) workflow.

---

## Why LLM Evaluation Matters in Financial Services

Financial services firms are under increasing pressure to adopt AI for customer-facing and internal decision-support use cases — from regulatory Q&A and product guidance to credit and investment advice. However, deploying an LLM without structured risk assessment creates exposure across three dimensions:

- **Hallucination risk** — models can confidently state incorrect regulatory figures, thresholds, or legislation, creating conduct risk and potential mis-selling.
- **Consistency risk** — the same query submitted twice may receive substantively different responses, making audit trails and standardised disclosures unreliable.
- **Bias risk** — models may treat customers differently based on name, gender, religion, or income level, in breach of the FCA's Consumer Duty and the Equality Act 2010.

This project applies a systematic evaluation methodology to surface these risks before deployment — mirroring the kind of pre-deployment validation expected under SR 11-7 (Model Risk Management) and the FCA's expectations for AI governance.

---

## Methodology

### Prompt Categories

The evaluation suite (`data/test_prompts.csv`) contains 45 prompts across five categories:

| Category | Code | Description | Prompts |
|---|---|---|---|
| **Factual Accuracy** | A | Regulatory figures, product definitions, legislative references. Expected answers provided for automated scoring. | 9 |
| **Ambiguous Queries** | B | Questions with no single correct answer (e.g. "should I fix my mortgage?"). Tests for overconfident directional advice. | 9 |
| **Consistency** | C | Concept explanations submitted three times each. Tests for output stability across repeated identical queries. | 9 × 3 runs |
| **Bias / Demographic Fairness** | D | Demographically sensitive scenarios testing for differential treatment by gender, religion, name origin, income, and age. | 9 |
| **Financial Domain Expertise** | E | Advanced topics: CDOs, IFRS 9 ECL model, VaR limitations, QE unintended consequences. Tests technical depth. | 9 |

### Scoring Dimensions

Three dimensions are scored automatically and supplemented by human review:

| Dimension | Method | Categories assessed |
|---|---|---|
| **Hallucination** | Key-term extraction from reference answers (regulatory figures, acronyms, percentages) checked against model output | A |
| **Consistency** | Pairwise string similarity (difflib SequenceMatcher) across 3 runs per prompt | C |
| **Bias** | Pattern-matching against demographically calibrated rules (gender risk framing, religious product omission, name-origin referencing) | D |

Each dimension is assigned a **RAG rating** (🟢 Green / 🟡 Amber / 🔴 Red) per prompt.

> **Note on human review:** Automated scoring provides a repeatable baseline, but hallucination and bias judgements require human oversight. Factual prompts with no extractable numeric key terms (A006, A007, A009) were rated Amber pending manual review. Bias pattern-matching can produce false positives where a legitimate financial term coincidentally matches a demographic keyword. All findings in the risk assessment report should be validated by a qualified model risk or compliance professional before informing deployment decisions.

---

## Tech Stack

| Component | Detail |
|---|---|
| **Language** | Python 3.12 |
| **LLM API** | [Groq](https://groq.com) — `openai/gpt-oss-20b` |
| **Data handling** | pandas |
| **Environment management** | python-dotenv |
| **Consistency scoring** | difflib (stdlib) |
| **Report generation** | Pure Python → Markdown |

---

## Key Findings

The evaluation produced an **overall Amber risk rating**. The model is assessed as not suitable for unsupervised deployment in regulated financial services contexts without the controls described in the risk assessment report.

| Dimension | Result | Detail |
|---|---|---|
| 🟡 **Hallucination** | 22% flag rate | 2 of 9 factual prompts flagged — Basel III CET1 ratio (4.5%) and UK ISA allowance (£20,000) not reproduced correctly |
| 🔴 **Consistency** | 100% Red | All 9 consistency prompts rated Red; mean pairwise string similarity 0.059 across three runs |
| 🟡 **Bias** | 3 of 9 flagged | Islamic finance options omitted (D003); name-origin referenced in credit risk comparison (D006); unprompted conservative risk framing for female executive (D008) |

The full findings, per-prompt RAG tables, and recommended controls are in [`outputs/risk_assessment_report.md`](outputs/risk_assessment_report.md).

---

## Project Structure

```
LLM eval/
├── data/
│   └── test_prompts.csv          # 45 prompts with expected answers and scoring notes
├── src/
│   ├── run_evaluation.py         # Sends prompts to Groq API, saves raw responses
│   ├── score_responses.py        # Scores responses across hallucination, consistency, bias
│   └── generate_report.py        # Produces the markdown risk assessment report
├── outputs/
│   ├── raw_responses.csv         # Model responses (one row per prompt × run)
│   ├── scored_results.csv        # Scored responses with RAG ratings per dimension
│   └── risk_assessment_report.md # Model risk validation note
├── .env.example                  # Template — copy to .env and add your key
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd "LLM eval"
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add your Groq API key

```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env  # Windows
```

Open `.env` and set:

```
GROQ_API_KEY=your_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Run the evaluation pipeline

Each script is run from the project root:

```bash
# Step 1 — send all prompts to the model and collect responses
python src/run_evaluation.py

# Step 2 — score responses across hallucination, consistency, and bias dimensions
python src/score_responses.py

# Step 3 — generate the markdown risk assessment report
python src/generate_report.py
```

Outputs are written to the `outputs/` directory. Category C prompts are sent 3 times automatically. A 2-second delay is applied between API calls.

---

## Outputs

| File | Description |
|---|---|
| `outputs/raw_responses.csv` | Raw model responses: `prompt_id`, `category`, `prompt_text`, `response`, `run_number` |
| `outputs/scored_results.csv` | All columns above plus RAG ratings and reasons for each scoring dimension |
| `outputs/risk_assessment_report.md` | Model risk validation note — executive summary, per-dimension findings, key risks, recommended controls |

---

## Limitations

- **String-similarity consistency scoring** penalises structural variation (e.g. prose vs. bullet points) even when semantic content is equivalent. Low scores on Category C reflect high output variability, not necessarily factual inconsistency. Semantic similarity scoring (e.g. embedding cosine similarity) would be a more robust measure.
- **Bias pattern-matching** is rules-based and calibrated to the specific test scenarios. It will not generalise to bias patterns outside the defined keyword sets without extending the rules.
- **Category B and E prompts** (ambiguous queries and expert domain) are not automatically scored in this version. They are included in `scored_results.csv` with `n/a` ratings and are intended for manual expert review.
