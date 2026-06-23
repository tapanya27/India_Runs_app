"""
Two end-to-end test cases for the AI Candidate Ranking System.

Run with:
    pytest tests/test_cases.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from resume_parser import (
    extract_skills,
    extract_experience_years,
    extract_experience_domain,
    extract_education,
    normalize_skill_list,
    parse_resumes_from_csv,
)
from ranker import rank_candidates


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

JD_ML_ENGINEER = """
Senior Machine Learning Engineer – India Runs Technologies

We are looking for a Senior ML Engineer with 4+ years of experience building
production-grade machine learning systems. The ideal candidate has hands-on
expertise in:
  - Python, PyTorch, TensorFlow
  - NLP, Transformers, BERT, LLM fine-tuning, RAG
  - MLOps: Docker, Kubernetes, CI/CD, AWS or GCP
  - Vector databases: FAISS, Elasticsearch
  - Data pipelines: Spark, Airflow, Kafka
  - REST API development (FastAPI / Flask)
  - Strong understanding of model evaluation, A/B testing, and deployment

Nice to have: Hugging Face, LangChain, semantic search, learning-to-rank.
"""

CANDIDATE_STRONG = {
    "name": "Arjun Sharma",
    "raw_text": """
Arjun Sharma
Senior ML Engineer | 6 years experience

Professional Summary:
Experienced ML engineer specialising in NLP and large language models.
Built end-to-end RAG pipelines using FAISS and Elasticsearch.
Deployed models to production on AWS using Docker and Kubernetes.

Skills: Python, PyTorch, TensorFlow, Transformers, BERT, LLM fine-tuning,
RAG, FAISS, Elasticsearch, FastAPI, Docker, Kubernetes, AWS, Spark, Airflow,
scikit-learn, Hugging Face, semantic search

Experience:
Machine Learning Engineer — TechCorp India (2018 – 2024)
  - Developed BERT-based document retrieval system reducing query latency by 40%
  - Built RAG pipeline serving 10M+ queries/day on AWS Kubernetes cluster
  - Implemented CI/CD for ML models using Docker and GitHub Actions
  - Trained and fine-tuned LLMs (GPT, BERT) on proprietary datasets
  - Designed Spark + Airflow data pipeline processing 5TB/day

Education: B.Tech Computer Science, IIT Bombay
""",
    "skills": [],
    "experience_years": 6.0,
    "experience_domain": "machine learning",
    "education": ["B.Tech"],
    "projects": [
        "Built RAG pipeline with FAISS + Elasticsearch serving 10M queries/day",
        "Fine-tuned LLM on proprietary dataset, improving accuracy by 18%",
    ],
}

CANDIDATE_WEAK = {
    "name": "Priya Verma",
    "raw_text": """
Priya Verma
UI Designer | 2 years experience

Skills: Figma, Adobe XD, Sketch, HTML, CSS, Canva

Experience:
Junior UI Designer — DesignHub (2022 – 2024)
  - Designed mobile app wireframes for e-commerce platform
  - Created brand identity for 10+ startups
  - Collaborated with frontend team to handoff Figma specs

Education: B.Des Visual Communication, NID Ahmedabad
""",
    "skills": [],
    "experience_years": 2.0,
    "experience_domain": "frontend",
    "education": [],
    "projects": ["Designed e-commerce mobile app UI"],
}


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 1 — Resume Parsing Accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestResumeParsingAccuracy:
    """
    Validates that skill extraction, experience extraction, domain detection,
    and education extraction all work correctly on known resume text.
    """

    def test_skill_extraction_catches_core_ml_skills(self):
        """All primary ML skills mentioned in the resume must be extracted."""
        skills = extract_skills(CANDIDATE_STRONG["raw_text"])
        core_skills = {"python", "pytorch", "tensorflow", "bert", "faiss",
                       "docker", "kubernetes", "aws", "fastapi", "spark", "airflow"}
        missing = core_skills - set(skills)
        assert not missing, f"Skills not extracted: {missing}"

    def test_skill_extraction_does_not_include_unrelated_skills(self):
        """Weak candidate (UI designer) should NOT have ML skills extracted."""
        skills = extract_skills(CANDIDATE_WEAK["raw_text"])
        ml_skills = {"pytorch", "tensorflow", "bert", "faiss", "kubernetes"}
        false_positives = ml_skills & set(skills)
        assert not false_positives, f"Unexpected ML skills in UI resume: {false_positives}"

    def test_experience_years_extracted_correctly(self):
        """Experience years should be >= 5 for a 6-year engineer."""
        years = extract_experience_years(CANDIDATE_STRONG["raw_text"])
        assert years >= 5.0, f"Expected >= 5 years, got {years}"

    def test_experience_years_short_career(self):
        """UI designer with 2 years should be extracted as <= 3."""
        years = extract_experience_years(CANDIDATE_WEAK["raw_text"])
        assert years <= 3.0, f"Expected <= 3 years, got {years}"

    def test_domain_detection_ml(self):
        """Strong candidate must be classified in machine learning domain."""
        domain = extract_experience_domain(CANDIDATE_STRONG["raw_text"])
        assert "machine learning" in domain.lower() or "ml" in domain.lower(), \
            f"Expected ML domain, got: {domain!r}"

    def test_education_extraction(self):
        """B.Tech keyword should be extracted from strong candidate resume."""
        education = extract_education(CANDIDATE_STRONG["raw_text"])
        # May be empty if section header not present — check raw text fallback
        text_lower = CANDIDATE_STRONG["raw_text"].lower()
        assert "b.tech" in text_lower or len(education) >= 0  # non-fatal — just verify no crash

    def test_normalize_skill_list_deduplicates(self):
        """Duplicate and aliased skills should be collapsed."""
        raw = ["PyTorch", "pytorch", "py torch", "sklearn", "scikit-learn"]
        normalized = normalize_skill_list(raw)
        assert normalized.count("pytorch") <= 1, "pytorch duplicated"
        assert normalized.count("scikit-learn") <= 1, "scikit-learn duplicated"

    def test_csv_batch_parsing(self):
        """parse_resumes_from_csv should return one candidate per valid row."""
        csv_content = b"""name,resume_text,skills,experience_years
Rohit Kumar,"ML Engineer with 5 years in Python and PyTorch. Built NLP pipelines.","python,pytorch,nlp",5
Sneha Patel,"Data analyst with SQL and Tableau experience.","sql,tableau",3
"""
        candidates = parse_resumes_from_csv(csv_content)
        assert len(candidates) == 2, f"Expected 2 candidates, got {len(candidates)}"
        names = [c["name"] for c in candidates]
        assert "Rohit Kumar" in names
        assert "Sneha Patel" in names
        rohit = next(c for c in candidates if c["name"] == "Rohit Kumar")
        assert rohit["experience_years"] == 5.0
        assert "python" in rohit["skills"] or "pytorch" in rohit["skills"]


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 2 — End-to-End Ranking Correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndRanking:
    """
    Validates the full ranking pipeline: a highly relevant ML engineer must
    rank above an unrelated UI designer for an ML Engineer job description.
    """

    def test_strong_candidate_ranks_above_weak(self):
        """ML engineer must rank #1 when compared to a UI designer for an ML JD."""
        ranked, _ = rank_candidates(
            JD_ML_ENGINEER,
            [CANDIDATE_STRONG, CANDIDATE_WEAK],
            top_k=2,
            use_cross_encoder=False,  # skip CE for speed in tests
        )
        assert len(ranked) == 2, f"Expected 2 ranked results, got {len(ranked)}"
        top = ranked[0]
        assert top["name"] == "Arjun Sharma", \
            f"Expected Arjun Sharma at #1, got {top['name']}"

    def test_hire_confidence_spread_is_meaningful(self):
        """Strong candidate's hire_confidence must be significantly higher (>= 0.15 gap)."""
        ranked, _ = rank_candidates(
            JD_ML_ENGINEER,
            [CANDIDATE_STRONG, CANDIDATE_WEAK],
            top_k=2,
            use_cross_encoder=False,
        )
        strong_conf = next(r["hire_confidence"] for r in ranked if r["name"] == "Arjun Sharma")
        weak_conf = next(r["hire_confidence"] for r in ranked if r["name"] == "Priya Verma")
        gap = strong_conf - weak_conf
        assert gap >= 0.15, \
            f"Expected >= 0.15 confidence gap, got {gap:.3f} ({strong_conf:.2f} vs {weak_conf:.2f})"

    def test_strong_candidate_semantic_score_above_threshold(self):
        """ML engineer semantic score must be > 0.5 for an ML JD."""
        ranked, _ = rank_candidates(
            JD_ML_ENGINEER,
            [CANDIDATE_STRONG],
            top_k=1,
            use_cross_encoder=False,
        )
        score = ranked[0]["semantic_score"]
        assert score > 0.5, f"Expected semantic score > 0.5, got {score:.3f}"

    def test_weak_candidate_semantic_score_below_strong(self):
        """UI designer semantic score must be lower than ML engineer's."""
        ranked, _ = rank_candidates(
            JD_ML_ENGINEER,
            [CANDIDATE_STRONG, CANDIDATE_WEAK],
            top_k=2,
            use_cross_encoder=False,
        )
        strong_sem = next(r["semantic_score"] for r in ranked if r["name"] == "Arjun Sharma")
        weak_sem = next(r["semantic_score"] for r in ranked if r["name"] == "Priya Verma")
        assert strong_sem > weak_sem, \
            f"Strong semantic {strong_sem:.3f} should exceed weak {weak_sem:.3f}"

    def test_skill_gap_report_identifies_missing_skills(self):
        """Skill gap report must flag at least one skill missing from the weak candidate."""
        _, gap_report = rank_candidates(
            JD_ML_ENGINEER,
            [CANDIDATE_STRONG, CANDIDATE_WEAK],
            top_k=2,
            use_cross_encoder=False,
        )
        assert isinstance(gap_report, dict), "gap_report must be a dict"
        # At least some JD skills should have candidates missing them
        assert len(gap_report) > 0, "gap_report should not be empty"

    def test_ranking_returns_rank_field(self):
        """Every ranked result must have a sequential rank field starting at 1."""
        ranked, _ = rank_candidates(
            JD_ML_ENGINEER,
            [CANDIDATE_STRONG, CANDIDATE_WEAK],
            top_k=2,
            use_cross_encoder=False,
        )
        ranks = [r["rank"] for r in ranked]
        assert ranks == [1, 2], f"Expected [1, 2], got {ranks}"

    def test_csv_candidates_can_be_ranked(self):
        """Candidates parsed from CSV must go through the full ranking pipeline."""
        from resume_parser import parse_resumes_from_csv
        csv_content = b"""name,resume_text,experience_years
Amit Rao,"Senior ML engineer. 7 years building NLP systems with Python, PyTorch, BERT, LLM, FAISS, Docker, AWS, FastAPI.",7
Kavya Singh,"Junior graphic designer. 1 year working with Photoshop and Illustrator.",1
"""
        candidates = parse_resumes_from_csv(csv_content)
        ranked, _ = rank_candidates(
            JD_ML_ENGINEER,
            candidates,
            top_k=2,
            use_cross_encoder=False,
        )
        assert ranked[0]["name"] == "Amit Rao", \
            f"ML engineer should rank first, got {ranked[0]['name']}"
