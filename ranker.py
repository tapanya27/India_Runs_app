from embedder import embed_text, embed_batch, build_candidate_text
from faiss_index import CandidateIndex
from cross_encoder_reranker import rerank
from scorer import score_candidate
from explainer import generate_explanation, generate_skill_gap_report
from resume_parser import extract_skills, normalize_skill_list
from skill_matcher import SemanticSkillMatcher
from config import FAISS_TOP_K, FINAL_TOP_K


def rank_candidates(
    job_description: str,
    candidates: list[dict],
    top_k: int = FINAL_TOP_K,
    use_cross_encoder: bool = True,
) -> tuple[list[dict], dict]:
    if not candidates:
        return []
    # Enrich each candidate's skills by re-extracting from raw_text
    enriched = []
    for candidate in candidates:
        existing = list(candidate.get("skills", []))
        raw_text = candidate.get("raw_text", "")
        if raw_text:
            extracted = extract_skills(raw_text)
            merged = list(set(existing) | set(extracted))
        else:
            merged = existing
        enriched.append({**candidate, "skills": normalize_skill_list(merged)})
    candidates = enriched

    # Extract JD skills without noun chunks — noun chunks on JD text add generic
    # phrases ("our team", "the ideal candidate") that inflate the denominator
    # of the skill match score and produce artificially low skill scores.
    jd_skills = extract_skills(job_description, include_noun_chunks=False)
    skill_matcher = SemanticSkillMatcher(jd_skills)

    # Build FAISS index over all candidates
    index = CandidateIndex()
    texts = [build_candidate_text(c) for c in candidates]
    embeddings = embed_batch(texts)
    index.add_candidates(candidates, embeddings)

    # FAISS: retrieve top candidates by semantic similarity
    jd_embedding = embed_text(job_description)
    faiss_results = index.search(jd_embedding, top_k=FAISS_TOP_K)

    # Cross-encoder re-ranking (returns normalized [0,1] scores via sigmoid)
    if use_cross_encoder and faiss_results:
        reranked = rerank(job_description, faiss_results)
    else:
        reranked = faiss_results  # faiss scores are cosine, already in [0, 1]

    # Normalize semantic scores across the pool so the best candidate anchors at
    # ~1.0 and the worst at ~0.3, creating meaningful spread instead of everyone
    # clustering at 85-96% cosine similarity.
    if len(reranked) > 1:
        raw_scores = [s for _, s in reranked]
        s_min, s_max = min(raw_scores), max(raw_scores)
        spread = s_max - s_min
        if spread > 0.005:
            reranked = [
                (c, 0.30 + 0.70 * (s - s_min) / spread)
                for c, s in reranked
            ]

    # Score + explain each candidate
    scored = []
    for candidate, semantic_score in reranked[:max(top_k, len(reranked))]:
        semantic_score = max(0.0, min(1.0, semantic_score))
        skill_matches = skill_matcher.match(candidate.get("skills", []))
        scores = score_candidate(job_description, skill_matches, candidate, semantic_score)
        explanation = generate_explanation(job_description, skill_matches, candidate, scores)
        scored.append({**candidate, **scores, "explanation": explanation})

    # Sort by hire confidence; break ties by technical fit, then semantic and skill.
    scored.sort(
        key=lambda x: (
            x["hire_confidence"],
            x["final_score"],
            x["semantic_score"],
            x["skill_score"],
        ),
        reverse=True,
    )

    for i, r in enumerate(scored[:top_k]):
        r["rank"] = i + 1

    result = scored[:top_k]

    # Attach skill gap report
    gap_report = generate_skill_gap_report(skill_matcher, candidates)

    return result, gap_report
