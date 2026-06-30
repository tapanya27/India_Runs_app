# 🧭 Redrob AI — Intelligent Candidate Discovery & Ranking

> A two-stage **retrieve → rerank → score** engine that ranks the **top 100 candidates** out of a **100,000-candidate pool** against a single Job Description — CPU-only, in under 5 minutes, with human-readable reasoning for every pick.

Built for the **India Runs Data & AI Challenge** (Redrob Hackathon). The job description is *Senior AI Engineer — Founding Team, Redrob AI*.

---

## ✨ Highlights

- **100,000 → 100** ranking from a prebuilt FAISS index — no per-query rescans of the full pool.
- **Two-stage retrieval:** dense bi-encoder recall (FAISS) → **cross-encoder reranking** of the shortlist.
- **6-component hybrid score:** Semantic · Skills · Experience · Projects · Education · Redrob behavioral signals.
- **Beats the dataset's traps:** a role/title-fit defense demotes keyword-stuffers (e.g. an *“HR Manager”* who lists `FAISS, embeddings, pinecone`) — **0 unrelated-title candidates in the top 100**.
- **Honeypot-aware:** impossible profiles are penalized (only 2 mild flags in top 100, **0 in top 10** — far under the 10% disqualification limit).
- **CPU-only & fast:** ranking step ≈ **145 s** wall-clock (budget: 300 s), using compact local models — no GPU, no network.
- **Valid, explainable output:** passes the official validator; every row carries specific, non-templated reasoning.

---

## 🏆 Results (current `submission.csv`)

| Metric | Value |
|---|---|
| Official validator | ✅ `Submission is valid.` |
| Rows / unique ranks / unique IDs | 100 / 100 / 100 |
| Score range (monotonic) | 0.8127 → 0.4755 |
| Top-100 with AI/ML/NLP/Search/DS titles | **100 / 100** |
| Unrelated-title (keyword-stuffer) in top 100 | **0** |
| Honeypot-flagged in top 100 (0 in top 10) | 2 (limit: <10) |
| Experience in 5–9 yr band | 75 / 100 |
| India-based | 92 / 100 |
| Reasoning: distinct / hallucinated-skill rows | 100 / **0** |

Top picks are genuine Senior ML/NLP/AI/Search Engineers at product companies (Flipkart, Netflix, Salesforce, Meta, CRED…) with real RAG / retrieval / reranking career evidence.

---

## 🧠 How it works

```
                          Job Description (canonical, shared by UI + CLI)
                                        │
                                        ▼
                          Sentence-Transformer embedding
                                        │
                                        ▼
                 ┌──────────────────────────────────────────┐
   100,000  ───► │  FAISS IndexFlatIP (cosine)  → top 700    │   recall
   prebuilt      └──────────────────────────────────────────┘
   index                                 │
                                         ▼
                 ┌──────────────────────────────────────────┐
                 │  Cross-Encoder rerank (top 700)           │   precision
                 │  blended = 0.55·CE + 0.45·FAISS cosine     │
                 └──────────────────────────────────────────┘
                                         │
                                         ▼
                 6-component weighted score  (on the 700 only)
                 Semantic·35  Skill·30  Exp·15  Proj·10  Edu·3  Redrob·7
                                         │
                                         ▼
                 × Penalty / Bonus modifiers
                 honeypot · consulting-only · role-fit · India-boost
                                         │
                                         ▼
                 Sort (score ↓, candidate_id ↑)  →  Top 100
                                         │
                                         ▼
                 submission.csv   +   per-candidate reasoning
```

**Stage 1 — Build once (`build_index.py`):** every candidate in `candidates.jsonl` is turned into a signal-dense text (name, role, experience domain, skills, summary, career history, education, platform signals), embedded with `all-MiniLM-L6-v2`, and stored in a FAISS index + a candidate map. This is the only slow step (~90 min on CPU) and is done a single time.

**Stage 2 — Rank per JD (`rank.py` / `app.py`):** the JD is embedded, FAISS returns the top 700 by cosine similarity, a cross-encoder reranks that shortlist, the six components are scored, multiplicative modifiers are applied, and the top 100 are written to `submission.csv`.

### Scoring weights

| Component | Weight | What it measures |
|---|---:|---|
| Semantic similarity | 35% | JD↔candidate fit (cross-encoder-blended) |
| Skill match (JD-aware) | 30% | Exact/strong/probable/category matches; required skills weighted higher |
| Experience | 15% | Fit to the 5–9 yr band (under/over-qualified penalties) |
| Projects / career | 10% | Career descriptions embedded vs the JD |
| Education | 3% | Institution tier + degree + field of study |
| Redrob signals | 7% | All 23 behavioral signals (response rate, interviews, GitHub, recency…) |

### Penalty / bonus modifiers (multiplicative)

- **Honeypot** — impossible profiles (expert-with-0-months, skill duration > career, keyword stuffing).
- **Consulting-only** — near-total careers at TCS/Infosys/Wipro/Accenture/… (JD disqualifier).
- **Role-fit** — the anti keyword-stuffer trap: role is judged from `current_title` + career-history **titles** and real career-**description** evidence, **never** the gameable headline/skills list.
- **India boost** — small 1.04× for Pune/Noida/India candidates.

---

## 📂 Repository structure

```
.
├── app.py                  # Streamlit UI: paste JD → top 100 + download submission.csv
├── rank.py                 # CLI: produces submission.csv (the reproduce command)
├── build_index.py          # One-time: builds candidate_index.faiss + candidate_map.pkl
├── challenge_ranker.py     # ★ Core engine: retrieve → rerank → 6-component score → top 100
├── skill_matcher.py        # Multi-layer semantic skill matching (exact/strong/probable/category)
├── cross_encoder_reranker.py  # Cross-encoder loader + scoring helpers
├── embedder.py             # Shared sentence-transformer loader + embed helpers
├── resume_parser.py        # Skill normalization/aliasing + JD skill extraction (spaCy)
├── config.py               # Models, dimensions, skill aliases/taxonomy, thresholds
│
├── candidate_index.faiss   # Prebuilt FAISS index (100,000 × 384)         [build artifact]
├── candidate_map.pkl       # Candidate records aligned to the index order  [build artifact]
├── submission.csv          # ★ The deliverable: top-100 ranking
├── submission_metadata.yaml# Portal metadata (fill in team/repo/sandbox)
│
├── requirements.txt
├── README.md               # This file
├── technical_details.txt   # Deep technical reference (every file + the math)
├── done_changes.txt        # Status of every requested change
├── CODE_CHANGES.txt        # File-by-file changelog
├── changes.txt             # Original change request
│
├── resources/              # Original challenge bundle (data, docs, validator)
│   ├── candidates.jsonl            # The 100,000-candidate pool
│   ├── validate_submission.py      # Official format validator
│   ├── candidate_schema.json       # Candidate JSON schema
│   ├── job_description.docx         # The JD being ranked against
│   ├── submission_spec.docx         # Rules, scoring, compute constraints
│   ├── redrob_signals_doc.docx      # The 23 behavioral signals
│   ├── sample_submission.csv        # Format reference (NOT an answer key)
│   └── …
│
├── testing/                # Your manual test outputs
└── trash/                  # Unused legacy pipeline + scratch files (recoverable)
```

---

## 🚀 Quick start

> Python 3.13, CPU-only. All models are local sentence-transformers (no API keys, no network at rank time).

```bash
# 1) Install dependencies (once)
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2) Build the index (once, ~90 min on CPU) — already built in this repo
python build_index.py

# 3a) Run the UI (recommended)
streamlit run app.py        # → http://localhost:8501 → click "Rank Candidates"

# 3b) …or the CLI
python rank.py --out submission.csv

# 4) Validate
python resources/validate_submission.py submission.csv   # → "Submission is valid."
```

The UI and the CLI rank against the **same** canonical JD, so they produce identical top-100 output (leave the pre-filled JD unchanged in the UI to match).

### Useful flags

```bash
python rank.py --out submission.csv --retrieve-k 700   # recall pool size
python rank.py --no-cross-encoder                      # faster, lower quality
```

---

## ⚙️ Compute & constraints (per `submission_spec`)

| Constraint | Limit | This system |
|---|---|---|
| Runtime (ranking step) | ≤ 5 min | ≈ 145 s |
| Memory | ≤ 16 GB | well within |
| Compute | CPU only | ✅ no GPU |
| Network at rank time | off | ✅ local models only |
| Models | compact, local | `all-MiniLM-L6-v2` (384-d) + `ms-marco-MiniLM-L-6-v2` |

---

## 📤 Output format

`candidate_id,rank,score,reasoning` — exactly 100 rows, ranks 1–100 unique, scores non-increasing, ties broken by `candidate_id` ascending.

```csv
candidate_id,rank,score,reasoning
CAND_0055905,1,0.8127,"Senior Machine Learning Engineer, 8.1 yrs exp, London | core JD skills: embeddings, opensearch… | relevant work at Flipkart: rag, retrieval, reranking | response rate 87%; open to work"
```

---

## ✅ Before you submit (hackathon)

- [ ] Rename `submission.csv` to your registered participant/team ID (e.g. `team_xxx.csv`).
- [ ] Fill in `submission_metadata.yaml` (team name, contact, **GitHub repo**, **sandbox link**).
- [ ] Push the code to GitHub (Stage-3 reproduction).
- [ ] Deploy a sandbox demo (HuggingFace Spaces / Streamlit Cloud can run `app.py`).
- [ ] Remember: 3 submissions max; the last valid one counts.

---

## 🛠️ Tech stack

`Python 3.13` · `sentence-transformers` · `FAISS (CPU)` · `PyTorch (CPU)` · `spaCy` · `Streamlit` · `pandas` / `numpy`

See **`technical_details.txt`** for the full module-by-module reference and the exact scoring math.
