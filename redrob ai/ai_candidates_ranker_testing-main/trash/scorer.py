import re
import math
import numpy as np
from config import (
    WEIGHT_SEMANTIC,
    WEIGHT_SKILL,
    WEIGHT_EXPERIENCE,
    WEIGHT_PROJECT,
    WEIGHT_HIRE_CONFIDENCE_TECH,
    WEIGHT_HIRE_CONFIDENCE_RECRUITER,
)
from embedder import embed_text, build_candidate_text
from resume_parser import extract_experience_domain


def _normalize_score(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1.0:
            numeric = numeric / 100.0
        return max(0.0, min(1.0, numeric))
    if isinstance(value, str):
        text = value.strip().rstrip("%")
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
        if numeric > 1.0:
            numeric = numeric / 100.0
        return max(0.0, min(1.0, numeric))
    return None


def _weighted_average(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if not values or total_weight <= 0:
        return 0.5
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def compute_skill_match(skill_matches: list[dict], candidate_skills: list[str]) -> float:
    """Weighted skill match score based on multi-layer matching results.

    Each match type contributes a different weight:
      Exact Match    = 1.0
      Strong Match   = 1.0
      Probable Match = 0.8
      Category Match = 0.5
      Missing        = 0.0

    Final score = sum(weights) / len(jd_skills)
    """
    if not skill_matches:
        return 0.5  # Unknown — neutral, not perfect

    from config import (
        SKILL_WEIGHT_EXACT, SKILL_WEIGHT_STRONG,
        SKILL_WEIGHT_PROBABLE, SKILL_WEIGHT_CATEGORY,
    )

    weight_map = {
        "Exact Match": SKILL_WEIGHT_EXACT,
        "Strong Match": SKILL_WEIGHT_STRONG,
        "Probable Match": SKILL_WEIGHT_PROBABLE,
        "Category Match": SKILL_WEIGHT_CATEGORY,
        "Missing": 0.0,
    }

    total_weight = sum(weight_map.get(m["category"], 0.0) for m in skill_matches)
    score = total_weight / len(skill_matches)

    return round(min(score, 1.0), 4)


def compute_experience_score(jd_text: str, candidate: dict) -> float:
    years = candidate.get("experience_years", 0.0)
    jd_lower = jd_text.lower()
    candidate_domain = candidate.get("experience_domain") or extract_experience_domain(candidate.get("raw_text", ""))
    jd_domain = extract_experience_domain(jd_text)

    # Extract required years from JD
    match = re.search(r"(\d+\.?\d*)\+?\s*years?\s+(?:of\s+)?(?:work\s+)?experience", jd_lower)
    required = float(match.group(1)) if match else 2.0

    domain_relevance = 0.5
    if jd_domain and candidate_domain:
        if jd_domain == candidate_domain:
            domain_relevance = 1.0
        elif jd_domain in candidate_domain or candidate_domain in jd_domain:
            domain_relevance = 0.8
        else:
            domain_relevance = 0.35

    # Compute semantic relevance between JD and candidate to weight years
    try:
        jd_emb = embed_text(jd_text)
        cand_emb = embed_text(build_candidate_text(candidate))
        relevance = float(np.dot(jd_emb, cand_emb))
    except Exception:
        relevance = 0.5

    relevance = max(0.0, min(1.0, relevance))
    relevance = max(0.0, min(1.0, 0.6 * relevance + 0.4 * domain_relevance))
    relevant_years = years * relevance

    if relevant_years >= required:
        excess_ratio = (relevant_years - required) / max(required, 1.0)
        bonus = 0.15 * math.log1p(excess_ratio)
        return round(min(1.0, 0.85 + bonus), 4)
    elif relevant_years > 0:
        return round((relevant_years / required) ** 0.8, 4)
    else:
        return 0.0


def compute_project_relevance(jd_text: str, candidate: dict) -> float:
    projects = candidate.get("projects", [])
    if not projects:
        return 0.0

    from embedder import embed_text
    jd_emb = embed_text(jd_text)

    scores = []
    for proj in projects[:6]:
        proj_emb = embed_text(proj)
        # Cosine similarity (embeddings already L2-normalized)
        sim = float(np.dot(jd_emb, proj_emb))
        scores.append(max(0.0, sim))

    if not scores:
        return 0.0

    # Weight top projects more: weighted average with decreasing weights
    weights = [1.0 / (i + 1) for i in range(len(scores))]
    weighted = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    return round(min(weighted, 1.0), 4)


def compute_education_bonus(jd_text: str, candidate: dict) -> float:
    """Additive bonus for degree level. Returns a direct score delta.

    PhD      → +5%  (0.05)
    M.Tech/MS/ME → +3%  (0.03)
    Bachelor's   → +0%  (0.00)

    The bonus is only applied when the JD explicitly values advanced
    education (mentions master/PhD/doctoral keywords).
    """
    education = candidate.get("education", [])
    jd_lower = jd_text.lower()

    # Only grant a bonus when the JD explicitly values advanced degrees
    jd_values_education = bool(
        re.search(r"\b(master|m\.?s\.?|m\.?tech|m\.?e\.?|m\.?sc|mca|mba|phd|doctoral|postgraduate|pg)\b", jd_lower)
    )

    if not jd_values_education or not education:
        return 0.0

    # Map each degree to its bonus tier
    phd_keywords = {"PhD"}
    masters_keywords = {"M.Tech", "M.E.", "M.Sc", "MBA", "MCA", "MS", "M.S."}

    best_bonus = 0.0
    for degree in education:
        if degree in phd_keywords:
            best_bonus = max(best_bonus, 0.05)  # +5%
        elif degree in masters_keywords:
            best_bonus = max(best_bonus, 0.03)  # +3%
        # Bachelor's and below: +0%

    return best_bonus


def compute_recruitability_score(candidate: dict) -> float:
    """Recruitability score derived from resume signals.

    When explicit recruiter signals (response_rate, etc.) are present they
    are used directly.  Otherwise the score is built entirely from
    heuristic signals extracted from the resume text so that different
    candidates actually receive different scores.
    """
    # ── Explicit recruiter signals (if present) ──────────────────────
    signal_fields = [
        ("response_rate", 0.20),
        ("profile_completeness", 0.15),
        ("interview_completion_rate", 0.15),
        ("activity_level", 0.10),
    ]

    explicit_values = []
    explicit_weights = []
    for field, weight in signal_fields:
        normalized = _normalize_score(candidate.get(field))
        if normalized is not None:
            explicit_values.append(normalized)
            explicit_weights.append(weight)

    # ── Heuristic signals derived from resume/profile text ───────────
    raw = (candidate.get("raw_text") or "").lower()
    projects = candidate.get("projects", [])
    education = candidate.get("education", [])

    # 1. Leadership experience
    leadership_keywords = r"\b(lead|leader|leadership|managed|manager|principal|director|head|vp|chief|cto|ceo|co-?founder|architect)\b"
    leadership = 1.0 if re.search(leadership_keywords, raw) else 0.0

    # 2. Seniority level
    seniority = 0.4  # default: mid-level assumption
    if re.search(r"\bprincipal\b|\bstaff\b|\bfellow\b", raw):
        seniority = 1.0
    elif re.search(r"\bsenior\b|\bsr\.?\b|\blead\b", raw):
        seniority = 0.85
    # Research scientists/engineers are typically senior-equivalent
    elif re.search(r"\bresearch\s+(scientist|engineer|lead)\b|\bstaff\s+scientist\b", raw):
        seniority = 0.80
    elif re.search(r"\bmid[- ]?level\b|\bintermediate\b", raw):
        seniority = 0.6
    elif re.search(r"\bjunior\b|\bjr\.?\b|\bintern\b|\bentry[- ]?level\b", raw):
        seniority = 0.25

    # 3. Promotions
    promotion_patterns = r"\b(promot(?:ed|ion)|advanced to|elevated to|grew from .* to)\b"
    promoted = 1.0 if re.search(promotion_patterns, raw) else 0.0

    # 4. Company reputation (presence of well-known companies)
    top_companies = re.compile(
        r"\b(google|meta|facebook|amazon|apple|microsoft|netflix|uber|airbnb|"
        r"linkedin|twitter|x\.com|stripe|openai|deepmind|nvidia|adobe|salesforce|"
        r"oracle|ibm|intel|samsung|tcs|infosys|wipro|flipkart|razorpay|"
        r"swiggy|zomato|paytm|phonepe|ola|myntra|zerodha)\b"
    )
    company_hits = len(set(top_companies.findall(raw)))
    company_reputation = min(1.0, company_hits / 2.0)  # 2+ top companies = 1.0

    # 5. Certifications
    cert_count = len(candidate.get("certifications", []))
    # Also look for certification keywords in raw text
    cert_keywords = re.findall(r"\b(certified|certification|certificate|aws certified|gcp certified|azure certified)\b", raw)
    cert_count = max(cert_count, len(cert_keywords))
    certs = min(1.0, cert_count / 3.0)  # 3+ certs = 1.0

    # 6. Open source contributions and research publications
    open_source_signals = [
        "github.com/", "gitlab.com/", "open-source", "open source",
        "contributor", "maintainer", "pull request", "merged pr",
        "npm package", "pypi", "crates.io",
        # Research publications are equivalent credibility signals
        "published", "paper", "arxiv", "conference", "journal", "acl", "neurips",
        "icml", "iclr", "emnlp", "aaai",
    ]
    os_hits = sum(1 for s in open_source_signals if s in raw)
    os_hits += sum(1 for p in projects if any(s in (p or "").lower() for s in ["github", "open-source", "open source", "contributor"]))
    open_source = min(1.0, os_hits / 2.0)  # 2+ signals = 1.0

    # 7. Education quality
    degree_tier = {"PhD": 1.0, "M.Tech": 0.85, "M.E.": 0.85, "M.Sc": 0.8,
                   "MBA": 0.75, "MCA": 0.7, "MS": 0.85, "M.S.": 0.85,
                   "B.Tech": 0.55, "B.E.": 0.55, "B.Sc": 0.5, "BCA": 0.45}
    edu_quality = max((degree_tier.get(d, 0.4) for d in education), default=0.3)

    # 8. Project complexity
    complex_keywords = re.compile(
        r"\b(distributed|large[- ]scale|production|scalab|high[- ]throughput|"
        r"low[- ]latency|million|billion|real[- ]time|microservice|orchestrat|"
        r"pipeline|infrastructure|deployed|serving)\b"
    )
    if projects:
        project_complexity = min(1.0, sum(1 for p in projects if complex_keywords.search(p.lower())) / max(1, len(projects)))
    else:
        # Fall back to checking raw text
        complexity_hits = len(complex_keywords.findall(raw))
        project_complexity = min(1.0, complexity_hits / 3.0)

    # ── Combine signals ──────────────────────────────────────────────
    if explicit_values:
        return round(_weighted_average(explicit_values, explicit_weights), 4)

    # Use a neutral baseline so candidates missing buzzwords aren't heavily penalized
    score = 0.40
    
    score += leadership * 0.10
    score += (seniority - 0.4) * 0.15      # 1.0 adds 0.09, 0.25 subtracts 0.02
    score += promoted * 0.10
    score += company_reputation * 0.10
    score += certs * 0.05
    score += open_source * 0.10
    score += (edu_quality - 0.3) * 0.10    # 1.0 adds 0.07, 0.3 adds 0.0
    score += project_complexity * 0.15

    return round(max(0.1, min(1.0, score)), 4)


def hybrid_score(
    semantic: float,
    skill: float,
    experience: float,
    project: float,
) -> float:
    return round(
        WEIGHT_SEMANTIC * semantic
        + WEIGHT_SKILL * skill
        + WEIGHT_EXPERIENCE * experience
        + WEIGHT_PROJECT * project,
        4,
    )


def score_candidate(jd_text: str, skill_matches: list[dict], candidate: dict, semantic_score: float) -> dict:
    skill = compute_skill_match(skill_matches, candidate.get("skills", []))
    experience = compute_experience_score(jd_text, candidate)
    project = compute_project_relevance(jd_text, candidate)
    education_bonus = compute_education_bonus(jd_text, candidate)
    recruitability = compute_recruitability_score(candidate)
    final = hybrid_score(semantic_score, skill, experience, project)

    # Education bonus is now a direct additive delta (0.05 for PhD, 0.03 for masters)
    final = round(min(1.0, max(0.0, final + education_bonus)), 4)
    hire_confidence = round(
        WEIGHT_HIRE_CONFIDENCE_TECH * final + WEIGHT_HIRE_CONFIDENCE_RECRUITER * recruitability,
        4,
    )

    return {
        "semantic_score": round(semantic_score, 4),
        "skill_score": round(skill, 4),
        "experience_score": round(experience, 4),
        "project_score": round(project, 4),
        "education_bonus": round(education_bonus, 4),
        "recruitability_score": round(recruitability, 4),
        "hire_confidence": hire_confidence,
        "final_score": final,
    }
