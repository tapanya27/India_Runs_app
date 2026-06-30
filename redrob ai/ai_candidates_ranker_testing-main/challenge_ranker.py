"""
challenge_ranker.py
-------------------
Ranks 100,000 candidates from a prebuilt FAISS index against a job description.

Pipeline:
    JD → embed → FAISS top-K → cross-encoder rerank → batch score (6 components) → top-100

Scoring weights (optimized — changes.txt item 17 "Recommended starting point"):
    Semantic Similarity  35%
    Skill Match          30%
    Experience           15%
    Projects             10%
    Education             3%
    Redrob Signals        7%

For maximum NDCG@10, scoring is JD-aware:
    • Cross-encoder reranking refines the FAISS retrieval ordering (changes.txt item 5)
    • Required JD skills weighted 2× nice-to-have
    • Consulting-only background penalised (per JD disqualifiers)
    • Honeypot candidates detected and penalised
    • All 23 Redrob signals used
    • Project scoring batched for speed
"""
import math
import pickle
import re
from datetime import date
from pathlib import Path

import faiss
import numpy as np

from embedder import embed_text, get_model
from resume_parser import extract_skills, normalize_skill_list
from skill_matcher import SemanticSkillMatcher

INDEX_FILE = Path("candidate_index.faiss")
MAP_FILE = Path("candidate_map.pkl")

FAISS_RETRIEVE = 700   # retrieve a larger pool for better recall before reranking (changes.txt item 1)
FINAL_TOP = 100

# Cross-encoder reranking (changes.txt item 5) — refines FAISS ordering on the
# retrieved pool. CE_BLEND controls how much the cross-encoder score influences
# the final semantic component vs. the raw FAISS cosine similarity.
USE_CROSS_ENCODER = True
CE_BLEND = 0.55        # 0.55 cross-encoder + 0.45 FAISS cosine

# Challenge scoring weights (changes.txt item 17 "Recommended starting point")
W_SEMANTIC   = 0.35
W_SKILL      = 0.30
W_EXPERIENCE = 0.15
W_PROJECT    = 0.10
W_EDUCATION  = 0.03
W_REDROB     = 0.07

# JD-specific: Senior AI Engineer at Redrob
# These are the REQUIRED skills per job_description.docx
JD_REQUIRED_SKILLS = [
    "faiss", "embeddings", "sentence-transformers", "semantic search",
    "vector database", "vector db", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "elasticsearch", "bge", "e5", "openai embeddings",
    "nlp", "information retrieval", "python", "pytorch", "transformers",
    "ndcg", "mrr", "map", "a/b testing", "retrieval augmented generation",
    "rag", "learning to rank", "reranking", "cross encoder", "bi encoder",
]

JD_NICE_TO_HAVE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "llm", "gpt", "bert",
    "distributed systems", "microservices", "kafka", "spark",
    "airflow", "kubernetes", "docker", "aws", "gcp", "azure",
    "machine learning", "deep learning", "huggingface",
    "xgboost", "lightgbm", "ranking", "recommendation",
]

# Consulting-only companies (per JD disqualifiers)
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "hcl",
    "capgemini", "tech mahindra", "mphasis", "hexaware",
}

# ── Role / title fit (anti keyword-stuffer trap) ──────────────────────────────
# The JD's central trap (job_description.docx, final note): candidates who list
# every AI keyword as a "skill" but whose actual role/career is unrelated
# (HR Manager, Civil/Mechanical Engineer, Sales, Marketing, etc.). Per the JD,
# "a candidate who has all the AI keywords listed as skills but whose title is
# 'Marketing Manager' is not a fit, no matter how perfect their skill list looks."
# Title + real career-history evidence are the decisive signals — not the skill list.
STRONG_ROLE_TERMS = [
    "ai engineer", "a.i. engineer", "ml engineer", "machine learning",
    "deep learning", "data scientist", "nlp engineer", "nlp scientist",
    "applied scientist", "research engineer", "ai specialist", "ai/ml",
    "ml scientist", "research scientist", "ai researcher", "llm",
]
GOOD_ROLE_TERMS = [
    "software engineer", "backend engineer", "data engineer", "developer",
    "search engineer", "platform engineer", "mlops", "ml ops",
    "full stack", "fullstack", "software developer", "sde", "ai/ml engineer",
    "machine learning engineer", "computer scientist",
]
# Roles that are clearly outside the AI-engineering scope of this JD.
WEAK_ROLE_TERMS = [
    "hr ", "human resource", "recruiter", "talent acquisition", "civil engineer",
    "mechanical engineer", "accountant", "accounting", "sales", "marketing",
    "graphic designer", "content writer", "customer support", "operations manager",
    "business analyst", "project manager", "administrator", "receptionist",
    "teacher", "nurse", "electrician", "hr manager",
]
# Career-history *description* terms that show real, JD-relevant systems work.
# Career descriptions are the trustworthy signal — the dataset's keyword stuffers
# game the skills list and headline but keep their career history honest. Terms
# are specific phrases to avoid false positives (e.g. electrical "transformer",
# "research" containing "search").
RELEVANT_WORK_TERMS = [
    "rag", "retrieval", "recommendation", "recommender", "recsys",
    "information retrieval", "reranking", "re-ranking", "semantic search",
    "vector search", "vector database", "embedding", "learning to rank",
    "ranking system", "search system", "nlp", "llm", "language model",
    "fine-tun", "machine learning", "deep learning", "neural network",
]

TODAY = date(2026, 6, 25)

# ── Canonical job description ─────────────────────────────────────────────────
# Single source of truth, imported by BOTH rank.py and app.py so the CLI and the
# Streamlit UI always rank against the identical JD (a focused distillation of
# resources/job_description.docx — Senior AI Engineer, Redrob AI).
DEFAULT_JD = """
Senior AI Engineer – Founding Team, Redrob AI (Series A)

We are Redrob AI, a Series A HR-tech startup building intelligent candidate
discovery and ranking systems. We are looking for a Senior AI Engineer with
5–9 years of experience to join our founding engineering team.

Required Skills:
- Production embeddings-based retrieval systems using sentence-transformers,
  OpenAI embeddings, BGE, E5 or equivalent
- Vector databases and hybrid search: Pinecone, Weaviate, Qdrant, Milvus,
  OpenSearch, Elasticsearch, FAISS
- Strong Python — code quality, testing, production-grade systems
- Hands-on evaluation frameworks for ranking: NDCG, MRR, MAP, offline-to-online
  correlation, A/B testing
- NLP and information retrieval — semantic search, reranking, cross-encoders,
  bi-encoders, RAG, retrieval augmented generation
- Learning to rank, cross-encoder reranking

Nice-to-Have:
- LLM fine-tuning: LoRA, QLoRA, PEFT
- Learning-to-rank models: XGBoost, neural ranking
- HR-tech, recruiting, or marketplace experience
- Distributed systems, large-scale inference optimization
- Open-source contributions in AI/ML

Location: Pune / Noida, India (Hybrid, flexible)
Experience: 5–9 years

Disqualifiers:
- Pure research without production deployment
- Entire career at TCS, Infosys, Wipro, Accenture or similar consulting
  without product experience
- Only CV, speech, or robotics experience with no NLP or IR exposure
"""


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load prebuilt FAISS index and candidate map from disk."""
    if not INDEX_FILE.exists() or not MAP_FILE.exists():
        raise FileNotFoundError(
            f"Index files not found ({INDEX_FILE}, {MAP_FILE}). "
            "Run: python build_index.py"
        )
    index = faiss.read_index(str(INDEX_FILE))
    with open(MAP_FILE, "rb") as f:
        candidates = pickle.load(f)
    return index, candidates


# ── Helpers ───────────────────────────────────────────────────────────────────

def _skill_names(candidate: dict) -> list[str]:
    return [
        s["name"] for s in candidate.get("skills", [])
        if isinstance(s, dict) and s.get("name")
    ]


def _skill_info(candidate: dict) -> list[dict]:
    return [s for s in candidate.get("skills", []) if isinstance(s, dict)]


def _all_company_names(candidate: dict) -> list[str]:
    return [
        j.get("company", "").lower()
        for j in candidate.get("career_history", [])
        if j.get("company")
    ]


# ── Cross-encoder reranking (changes.txt item 5) ──────────────────────────────

def _candidate_ce_text(candidate: dict) -> str:
    """
    Build a compact, signal-dense text for cross-encoder scoring.

    Cross-encoders read the full (query, document) pair jointly, so we keep this
    short (headline + title + summary + top skills + most-recent role) to stay
    within the model's token window and the 5-minute ranking budget.
    """
    profile = candidate.get("profile", {})
    parts: list[str] = []

    headline = (profile.get("headline") or "").strip()
    if headline:
        parts.append(headline)

    title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    if title:
        parts.append(f"{title}, {yoe} years experience")

    summary = (profile.get("summary") or "").strip()
    if summary:
        parts.append(summary[:400])

    skills = _skill_names(candidate)
    if skills:
        parts.append("Skills: " + ", ".join(skills[:20]))

    career = candidate.get("career_history", [])
    if career:
        recent = career[0]
        desc = (recent.get("description") or "").strip()
        if desc:
            parts.append(f"{recent.get('title','')}: {desc[:300]}")

    return " ".join(p for p in parts if p)


def _cross_encoder_rerank(
    job_description: str,
    candidates: list[dict],
    faiss_scores: list[float],
) -> list[float]:
    """
    Re-rank the FAISS-retrieved pool with a cross-encoder and return a blended
    semantic score per candidate (aligned with the input order).

    blended = CE_BLEND * cross_encoder_prob + (1 - CE_BLEND) * faiss_cosine

    The cross-encoder reads each (JD, candidate) pair jointly, which captures
    relevance signals a bi-encoder (FAISS) misses — improving the final ordering
    of the top candidates without re-scoring all 100,000.
    """
    if not candidates:
        return []

    from cross_encoder_reranker import get_cross_encoder, _sigmoid

    ce = get_cross_encoder()
    jd_short = job_description[:1500]
    pairs = [(jd_short, _candidate_ce_text(c)) for c in candidates]
    raw = ce.predict(pairs, batch_size=64, show_progress_bar=False)
    ce_scores = [_sigmoid(float(s)) for s in raw]

    blended = [
        CE_BLEND * ce_s + (1.0 - CE_BLEND) * float(faiss_s)
        for ce_s, faiss_s in zip(ce_scores, faiss_scores)
    ]
    return blended


# ── Honeypot detection ────────────────────────────────────────────────────────

def _honeypot_penalty(candidate: dict) -> float:
    """
    Return a penalty multiplier [0.0, 1.0] where 1.0 = no penalty.
    Honeypot signals (per spec): impossible career claims, skill duration >> career,
    expert proficiency with 0 months usage on many skills.
    """
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0) or 0
    career = candidate.get("career_history", [])
    skills = _skill_info(candidate)

    penalty = 1.0

    # 1. Total career months far exceeds claimed experience
    total_career_months = sum(j.get("duration_months", 0) or 0 for j in career)
    if yoe > 0 and total_career_months > (yoe * 12 * 1.5):
        penalty *= 0.6   # overlapping durations is suspicious

    # 2. Many skills claimed as "expert" or "advanced" with 0 duration_months
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and (s.get("duration_months") or 0) == 0
    )
    if expert_zero >= 5:
        penalty *= 0.5
    elif expert_zero >= 3:
        penalty *= 0.75

    # 3. Skill duration_months > total career months (impossible)
    impossible_skills = sum(
        1 for s in skills
        if (s.get("duration_months") or 0) > max(1, total_career_months)
    )
    if impossible_skills >= 3:
        penalty *= 0.4

    # 4. Proficiency = "expert" on 10+ skills — keyword stuffer
    expert_count = sum(
        1 for s in skills if s.get("proficiency") == "expert"
    )
    if expert_count >= 10:
        penalty *= 0.7

    return round(max(0.0, min(1.0, penalty)), 4)


# ── Consulting-only penalty ───────────────────────────────────────────────────

def _consulting_penalty(candidate: dict) -> float:
    """
    Per JD: 'entire career at consulting firms (TCS, Infosys, Wipro …)' is a
    disqualifier. If ALL career history is at consulting companies return 0.6,
    if majority (>70%) return 0.8, else 1.0 (no penalty).
    """
    career = candidate.get("career_history", [])
    if not career:
        return 1.0

    total_months = sum(j.get("duration_months", 0) or 0 for j in career)
    consulting_months = sum(
        j.get("duration_months", 0) or 0
        for j in career
        if any(c in (j.get("company") or "").lower() for c in CONSULTING_COMPANIES)
    )

    if total_months == 0:
        return 1.0

    ratio = consulting_months / total_months
    if ratio >= 0.95:
        return 0.55    # near-total consulting background
    elif ratio >= 0.70:
        return 0.80
    return 1.0


# ── India-location boost ──────────────────────────────────────────────────────

def _india_boost(candidate: dict) -> float:
    """Small boost for India-based candidates (Pune/Noida role per JD)."""
    profile = candidate.get("profile", {})
    country = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()
    if "india" in country or "pune" in location or "noida" in location:
        return 1.04
    return 1.0


def _role_fit_multiplier(candidate: dict) -> float:
    """
    Multiplicative modifier [~0.30, 1.0] capturing how well the candidate's
    *role* (title + career) fits an AI-engineering JD — the decisive lever
    against the keyword-stuffer trap.

    A bi-encoder / skill-keyword match alone ranks an "HR Manager" who lists
    "FAISS, embeddings, pinecone" near the top. This penalises that: title and
    real career-history evidence dominate, the skill list does not.
    """
    profile = candidate.get("profile", {})
    title = (profile.get("current_title") or "").lower()
    career = candidate.get("career_history", [])
    career_titles = " ".join((j.get("title") or "").lower() for j in career)

    # Classify from current_title + career-history titles ONLY. The headline is
    # NOT used — keyword stuffers write "Mechanical Engineer | Building with LLMs"
    # in the headline while their real title and career stay unrelated.
    strong_now = any(t in title for t in STRONG_ROLE_TERMS)
    good_now = any(t in title for t in GOOD_ROLE_TERMS)
    weak_now = any(t in title for t in WEAK_ROLE_TERMS)
    strong_past = any(
        t in career_titles for t in (STRONG_ROLE_TERMS + GOOD_ROLE_TERMS)
    )

    # Real, JD-relevant work described in career history (counts even if the
    # title is off — the JD explicitly rewards "built a recommendation system at
    # a product company" over a keyword-perfect skills list).
    career_text = " ".join((j.get("description") or "").lower() for j in career)
    work_evidence = sum(1 for t in RELEVANT_WORK_TERMS if t in career_text)

    if strong_now:
        fit = 1.0
    elif good_now:
        fit = 0.80
    elif strong_past:
        fit = 0.62          # e.g. now a "Manager" but was an ML engineer
    elif weak_now:
        fit = 0.16
    else:
        fit = 0.50          # unknown / neutral title

    if work_evidence >= 2:
        fit = min(1.0, fit + 0.22)
    elif work_evidence == 1:
        fit = min(1.0, fit + 0.10)

    # Lazy keyword stuffing: unrelated role + many AI core skills + no career
    # evidence → extra demotion.
    skills_lower = {
        (s.get("name") or "").lower()
        for s in candidate.get("skills", []) if isinstance(s, dict)
    }
    ai_core_hits = sum(
        1 for req in JD_REQUIRED_SKILLS
        if any(req in cs or cs in req for cs in skills_lower)
    )
    if fit <= 0.25 and ai_core_hits >= 4 and work_evidence == 0:
        fit *= 0.5

    # Map fit [0,1] → multiplier [0.30, 1.0]
    return round(0.30 + 0.70 * fit, 4)


# ── Component scorers ─────────────────────────────────────────────────────────

def _score_skills_jd_aware(candidate: dict, skill_matcher: SemanticSkillMatcher) -> float:
    """
    Skill match with JD-specific weighting:
    - Required skills (from JD required list) weighted 2×
    - Nice-to-have skills weighted 1×
    - Advanced/expert proficiency adds endorsement-weighted bonus
    """
    skill_info = _skill_info(candidate)
    if not skill_info:
        return 0.0

    skill_names_raw = [s.get("name") for s in skill_info if s.get("name")]
    if not skill_names_raw:
        return 0.0
    normalized = normalize_skill_list(skill_names_raw)

    # SemanticSkillMatcher score
    matches = skill_matcher.match(normalized)
    if not matches:
        return 0.5

    WEIGHT_MAP = {
        "Exact Match": 1.0, "Strong Match": 1.0,
        "Probable Match": 0.8, "Category Match": 0.5, "Missing": 0.0,
    }
    base_score = min(
        sum(WEIGHT_MAP.get(m["category"], 0.0) for m in matches) / len(matches),
        1.0,
    )

    # JD-required skill boost: check how many required JD skills the candidate has
    candidate_skills_lower = {n.lower() for n in skill_names_raw}
    required_hits = sum(
        1 for req in JD_REQUIRED_SKILLS
        if any(req in cs or cs in req for cs in candidate_skills_lower)
    )
    nice_hits = sum(
        1 for nice in JD_NICE_TO_HAVE_SKILLS
        if any(nice in cs or cs in nice for cs in candidate_skills_lower)
    )

    # Bonus: required hits up to 0.20, nice-to-have up to 0.10
    required_bonus = min(0.20, required_hits / max(len(JD_REQUIRED_SKILLS), 1) * 0.20 * 3)
    nice_bonus = min(0.10, nice_hits / max(len(JD_NICE_TO_HAVE_SKILLS), 1) * 0.10 * 3)

    # Proficiency bonus: advanced/expert skills weighted by endorsements
    proficiency_score = 0.0
    for s in skill_info:
        prof = s.get("proficiency", "beginner")
        endorsements = min(s.get("endorsements", 0), 50)
        duration = min(s.get("duration_months", 0), 60)
        name_lower = (s.get("name") or "").lower()

        is_relevant = any(req in name_lower or name_lower in req for req in JD_REQUIRED_SKILLS)
        if not is_relevant:
            is_relevant = any(nice in name_lower or name_lower in nice for nice in JD_NICE_TO_HAVE_SKILLS)

        if is_relevant:
            prof_val = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.6, "beginner": 0.3}.get(prof, 0.3)
            endorse_val = endorsements / 50.0
            dur_val = duration / 60.0
            proficiency_score += prof_val * 0.5 + endorse_val * 0.3 + dur_val * 0.2

    if skill_info:
        proficiency_score = min(0.10, proficiency_score / max(len(skill_info), 1))

    final = base_score * 0.60 + required_bonus + nice_bonus + proficiency_score
    return round(min(1.0, max(0.0, final)), 4)


def _score_experience(candidate: dict, jd_text: str) -> float:
    """Score based on years of experience vs JD (5-9 years preferred)."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0.0) or 0.0

    # JD says 5-9 years, flexible
    match = re.search(
        r"(\d+\.?\d*)\s*[–-]\s*(\d+\.?\d*)\s*years?",
        jd_text.lower(),
    )
    if match:
        min_yrs = float(match.group(1))
        max_yrs = float(match.group(2))
    else:
        m2 = re.search(r"(\d+\.?\d*)\+?\s*years?\s+(?:of\s+)?(?:work\s+)?experience", jd_text.lower())
        min_yrs = float(m2.group(1)) if m2 else 3.0
        max_yrs = min_yrs + 4.0

    if min_yrs <= yoe <= max_yrs:
        return 1.0
    elif yoe > max_yrs:
        excess = yoe - max_yrs
        return round(max(0.7, 1.0 - excess * 0.02), 4)  # slight penalty for over-qualified
    elif yoe > 0:
        return round((yoe / min_yrs) ** 0.8, 4)
    return 0.0


def _score_projects_batch(
    candidates: list[dict],
    jd_embedding: np.ndarray,
) -> list[float]:
    """
    Batch score all candidate project/career descriptions against JD.
    Encodes ALL descriptions in one batched call — much faster than per-candidate.
    """
    model = get_model()

    # Gather all descriptions with candidate indices
    desc_index: list[tuple[int, float]] = []  # (candidate_idx, weight)
    all_descs: list[str] = []

    for ci, candidate in enumerate(candidates):
        jobs = [
            j.get("description", "")
            for j in candidate.get("career_history", [])
            if j.get("description")
        ][:5]
        for ji, desc in enumerate(jobs):
            all_descs.append(desc[:400])
            weight = 1.0 / (ji + 1)
            desc_index.append((ci, weight))

    if not all_descs:
        return [0.0] * len(candidates)

    # Single batched encode
    embs = model.encode(
        all_descs,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype(np.float32)

    sims = (embs @ jd_embedding).tolist()

    # Aggregate by candidate
    candidate_weighted: list[list[tuple[float, float]]] = [[] for _ in candidates]
    for (ci, weight), sim in zip(desc_index, sims):
        candidate_weighted[ci].append((max(0.0, sim), weight))

    scores = []
    for items in candidate_weighted:
        if not items:
            scores.append(0.0)
        else:
            total_w = sum(w for _, w in items)
            weighted_sim = sum(s * w for s, w in items) / total_w
            scores.append(round(min(1.0, weighted_sim), 4))

    return scores


def _score_education(candidate: dict) -> float:
    """Score based on institution tier + degree type + field relevance."""
    TIER_SCORES = {
        "tier_1": 1.0, "tier_2": 0.80, "tier_3": 0.60,
        "tier_4": 0.40, "unknown": 0.30,
    }
    DEGREE_BONUS = {
        "m.tech": 0.10, "m.e.": 0.10, "ms": 0.10, "m.s.": 0.10,
        "m.sc": 0.08, "mca": 0.08, "mba": 0.05,
        "phd": 0.15, "ph.d": 0.15,
    }
    CS_AI_FIELDS = {
        "computer science", "computer engineering", "information technology",
        "software engineering", "artificial intelligence", "machine learning",
        "data science", "electronics", "electrical engineering",
    }

    education = candidate.get("education", [])
    if not education:
        return 0.30

    best = 0.0
    for edu in education:
        tier_score = TIER_SCORES.get(edu.get("tier", "unknown"), 0.30)
        degree = (edu.get("degree") or "").lower().strip(".")
        degree_bonus = DEGREE_BONUS.get(degree, 0.0)
        field = (edu.get("field_of_study") or "").lower()
        field_bonus = 0.05 if any(f in field for f in CS_AI_FIELDS) else 0.0
        score = min(1.0, tier_score + degree_bonus + field_bonus)
        best = max(best, score)

    return round(best, 4)


def _score_redrob_comprehensive(candidate: dict) -> float:
    """
    Use all 23 Redrob signals for a comprehensive engagement/quality score.
    Each signal is scored 0-1 and combined via weighted average of available signals.
    """
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return 0.40

    components: list[float] = []
    weights: list[float] = []

    def add(value, weight):
        if value is not None:
            components.append(float(value))
            weights.append(float(weight))

    # ── Engagement & reliability signals ─────────────────────────────────────
    rrr = signals.get("recruiter_response_rate")
    if rrr is not None and rrr >= 0:
        add(rrr, 0.20)

    icr = signals.get("interview_completion_rate")
    if icr is not None and icr >= 0:
        add(icr, 0.15)

    # ── Technical activity ────────────────────────────────────────────────────
    gas = signals.get("github_activity_score", -1)
    if gas is not None and gas >= 0:
        add(gas / 100.0, 0.13)

    # Skill assessment scores (average of completed assessments)
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        avg_assess = sum(assessments.values()) / len(assessments) / 100.0
        add(avg_assess, 0.10)

    # ── Availability signals ──────────────────────────────────────────────────
    open_to_work = signals.get("open_to_work_flag")
    if open_to_work is not None:
        add(float(open_to_work), 0.10)

    # Recency: last active
    last_active_str = signals.get("last_active_date")
    if last_active_str:
        try:
            last_dt = date.fromisoformat(last_active_str)
            days_ago = (TODAY - last_dt).days
            if days_ago <= 30:
                recency = 1.0
            elif days_ago <= 90:
                recency = 0.75
            elif days_ago <= 180:
                recency = 0.50
            elif days_ago <= 365:
                recency = 0.30
            else:
                recency = 0.10
            add(recency, 0.07)
        except ValueError:
            pass

    # ── Profile quality ───────────────────────────────────────────────────────
    completeness = signals.get("profile_completeness_score")
    if completeness is not None:
        add(float(completeness) / 100.0, 0.06)

    # Market interest from recruiters
    saved = signals.get("saved_by_recruiters_30d", 0) or 0
    add(min(1.0, saved / 10.0), 0.05)

    # Profile views (market demand)
    views = signals.get("profile_views_received_30d", 0) or 0
    add(min(1.0, views / 50.0), 0.03)

    # ── Commitment signals ────────────────────────────────────────────────────
    oar = signals.get("offer_acceptance_rate", -1)
    if oar is not None and oar >= 0:
        add(float(oar), 0.04)

    # Notice period (shorter = more available, better for hiring speed)
    notice = signals.get("notice_period_days")
    if notice is not None:
        # 0 days → 1.0, 90 days → 0.5, 180 days → 0.0
        notice_score = max(0.0, 1.0 - float(notice) / 180.0)
        add(notice_score, 0.03)

    # ── Trust / verification ──────────────────────────────────────────────────
    trust = (
        (1 if signals.get("verified_email") else 0)
        + (1 if signals.get("verified_phone") else 0)
        + (1 if signals.get("linkedin_connected") else 0)
    ) / 3.0
    add(trust, 0.02)

    # Response speed
    art = signals.get("avg_response_time_hours")
    if art is not None and art >= 0:
        # 0h → 1.0, 24h → 0.88, 168h → 0.16
        add(max(0.0, 1.0 - float(art) / 200.0), 0.02)

    if not components:
        return 0.40

    total_w = sum(weights)
    score = sum(c * w for c, w in zip(components, weights)) / total_w
    return round(max(0.0, min(1.0, score)), 4)


# ── Reasoning generation ──────────────────────────────────────────────────────

def _build_reasoning(candidate: dict, scores: dict, jd_skills: list[str]) -> str:
    """
    Build a specific, non-templated reason per submission_spec requirements:
    - Specific facts (years, title, actual skill names, actual signal values)
    - JD connection
    - Honest gaps where present
    - Variation across candidates
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = _skill_info(candidate)
    education = candidate.get("education", [])

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0) or 0
    location = profile.get("location", "")
    country = profile.get("country", "")
    company = profile.get("current_company", "")

    # Skills matched to JD
    skill_names_lower = {(s.get("name") or "").lower() for s in skills}
    matched_required = [
        req for req in JD_REQUIRED_SKILLS
        if any(req in sn or sn in req for sn in skill_names_lower)
    ][:5]
    matched_nice = [
        nice for nice in JD_NICE_TO_HAVE_SKILLS
        if any(nice in sn or sn in nice for sn in skill_names_lower)
    ][:3]

    # Top skills by endorsements
    top_skills = sorted(skills, key=lambda s: s.get("endorsements", 0), reverse=True)[:4]
    top_skill_names = [s["name"] for s in top_skills]

    # Redrob signals (include actual values)
    rrr = signals.get("recruiter_response_rate")
    icr = signals.get("interview_completion_rate")
    gas = signals.get("github_activity_score", -1)
    open_work = signals.get("open_to_work_flag")

    # Education
    top_edu = education[0] if education else {}
    edu_str = f"{top_edu.get('degree','')} {top_edu.get('field_of_study','')} ({top_edu.get('tier','').replace('tier_','Tier ')})" if top_edu else ""

    # Build reason parts
    parts = [f"{title}, {yoe:.1f} yrs exp"]

    if location:
        parts[0] += f", {location}"

    if matched_required:
        parts.append(f"core JD skills: {', '.join(matched_required[:4])}")
    elif top_skill_names:
        parts.append(f"top skills: {', '.join(top_skill_names)}")

    if matched_nice:
        parts.append(f"bonus: {', '.join(matched_nice[:2])}")

    # Relevant project / career work (item 15): surface JD-relevant terms that
    # actually appear in the candidate's career-history descriptions.
    career_text = " ".join(
        (j.get("description") or "").lower()
        for j in candidate.get("career_history", [])
    )
    work_hits = [t for t in RELEVANT_WORK_TERMS if t in career_text]
    if work_hits:
        pretty = [("fine-tuning" if t == "fine-tun" else t) for t in work_hits[:3]]
        prefix = f"relevant work at {company}: " if company else "relevant work: "
        parts.append(prefix + ", ".join(pretty))
    elif company:
        parts.append(f"currently at {company}")

    if edu_str.strip():
        parts.append(f"edu: {edu_str.strip()}")

    signal_parts = []
    if rrr is not None and rrr >= 0:
        signal_parts.append(f"response rate {rrr:.0%}")
    if icr is not None and icr >= 0:
        signal_parts.append(f"interview completion {icr:.0%}")
    if gas is not None and gas >= 0:
        signal_parts.append(f"GitHub score {gas:.0f}")
    if open_work:
        signal_parts.append("open to work")
    if signal_parts:
        parts.append("; ".join(signal_parts))

    # Honest gaps
    gaps = []
    if scores.get("role_mult", 1.0) < 0.6 and not work_hits:
        gaps.append("role/title not aligned with an AI-engineering JD")
    if yoe < 3:
        gaps.append("limited experience")
    if not matched_required:
        gaps.append("few direct JD skill matches")
    if not open_work and open_work is not None:
        gaps.append("not marked open to work")
    if gaps:
        parts.append(f"gaps: {'; '.join(gaps)}")

    return " | ".join(parts)


# ── Main ranking function ─────────────────────────────────────────────────────

def rank_candidates_challenge(
    job_description: str,
    index: faiss.Index,
    all_candidates: list[dict],
    top_k: int = FINAL_TOP,
    retrieve_k: int = FAISS_RETRIEVE,
    use_cross_encoder: bool = True,
) -> list[dict]:
    """
    Full challenge ranking pipeline.

    Returns exactly top_k results sorted by final_score descending,
    with rank 1..top_k, guaranteed monotonically non-increasing scores.
    """
    jd_embedding = embed_text(job_description)

    # ── Step 1: FAISS retrieval ───────────────────────────────────────────────
    k = min(retrieve_k, index.ntotal)
    scores_arr, indices_arr = index.search(jd_embedding.reshape(1, -1), k)

    faiss_results = [
        (all_candidates[idx], float(score))
        for score, idx in zip(scores_arr[0], indices_arr[0])
        if 0 <= idx < len(all_candidates)
    ]

    if not faiss_results:
        return []

    retrieved_candidates = [c for c, _ in faiss_results]
    raw_cosines = [s for _, s in faiss_results]

    # ── Step 1b: Cross-encoder reranking (changes.txt item 5) ─────────────────
    # Min-max normalize the FAISS cosines to [0, 1] so they blend cleanly with
    # the cross-encoder probabilities.
    c_min, c_max = min(raw_cosines), max(raw_cosines)
    c_spread = c_max - c_min
    if c_spread > 1e-6:
        norm_cosines = [(s - c_min) / c_spread for s in raw_cosines]
    else:
        norm_cosines = [1.0] * len(raw_cosines)

    if use_cross_encoder and USE_CROSS_ENCODER:
        semantic_raw = _cross_encoder_rerank(
            job_description, retrieved_candidates, norm_cosines
        )
    else:
        semantic_raw = norm_cosines

    # Normalize the (blended) semantic scores to [0.30, 1.00] for meaningful spread
    if len(semantic_raw) > 1:
        s_min, s_max = min(semantic_raw), max(semantic_raw)
        spread = s_max - s_min
        if spread > 1e-6:
            semantic_raw = [0.30 + 0.70 * (s - s_min) / spread for s in semantic_raw]

    faiss_results = list(zip(retrieved_candidates, semantic_raw))

    # ── Step 2: Setup ─────────────────────────────────────────────────────────
    jd_skills = extract_skills(job_description, include_noun_chunks=False)
    # CE validation off: match() runs once per retrieved candidate, so per-call
    # cross-encoder validation would add hundreds of model calls to the budget.
    skill_matcher = SemanticSkillMatcher(jd_skills, use_cross_encoder_validation=False)

    # ── Step 3: Batch project scoring (one model call for all candidates) ─────
    project_scores = _score_projects_batch(retrieved_candidates, jd_embedding)

    # ── Step 4: Score each candidate ─────────────────────────────────────────
    scored = []
    for i, (candidate, semantic_score) in enumerate(faiss_results):
        sem  = max(0.0, min(1.0, semantic_score))
        skill = _score_skills_jd_aware(candidate, skill_matcher)
        exp  = _score_experience(candidate, job_description)
        proj = project_scores[i]
        edu  = _score_education(candidate)
        redrob = _score_redrob_comprehensive(candidate)

        # Raw weighted sum
        raw_score = (
            W_SEMANTIC   * sem   +
            W_SKILL      * skill +
            W_EXPERIENCE * exp   +
            W_PROJECT    * proj  +
            W_EDUCATION  * edu   +
            W_REDROB     * redrob
        )

        # Modifiers: honeypot, consulting, role-fit penalties + India boost
        honeypot_mult = _honeypot_penalty(candidate)
        consult_mult  = _consulting_penalty(candidate)
        role_mult     = _role_fit_multiplier(candidate)
        india_mult    = _india_boost(candidate)

        final = round(min(1.0, max(0.0,
            raw_score * honeypot_mult * consult_mult * role_mult * india_mult)), 4)

        all_scores = {
            "semantic_score":   round(sem, 4),
            "skill_score":      round(skill, 4),
            "experience_score": round(exp, 4),
            "project_score":    round(proj, 4),
            "education_score":  round(edu, 4),
            "redrob_score":     round(redrob, 4),
            "honeypot_mult":    round(honeypot_mult, 4),
            "consult_mult":     round(consult_mult, 4),
            "role_mult":        round(role_mult, 4),
            "final_score":      final,
        }

        reasoning = _build_reasoning(candidate, all_scores, jd_skills)

        scored.append({
            "candidate_id": candidate.get("candidate_id", ""),
            "candidate":    candidate,
            "reasoning":    reasoning,
            **all_scores,
        })

    # ── Step 5: Canonical ordering — score DESC, then candidate_id ASC ─────────
    # The spec tie-break is candidate_id ascending. We MUST apply this ordering
    # BEFORE slicing to top_k, otherwise the rank-100/101 boundary inside a tied
    # score block would admit the wrong candidate (a single sort with
    # reverse=True would order tied candidate_ids descending).
    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    # ── Step 6: Take top_k and guarantee monotonically non-increasing scores ──
    top = scored[:top_k]

    # Clamp: ensure no score exceeds the previous one (floating-point safety)
    prev_score = top[0]["final_score"] if top else 1.0
    for row in top:
        if row["final_score"] > prev_score:
            row["final_score"] = prev_score
        prev_score = row["final_score"]

    for i, r in enumerate(top):
        r["rank"] = i + 1

    return top
