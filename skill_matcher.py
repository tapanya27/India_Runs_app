"""
Multi-layered skill matching engine.

Layers:
  1. Normalization   – lowercase, punctuation cleanup, alias resolution
  2. Exact Match     – string equality after normalization
  3. Semantic Match  – SBERT cosine similarity (greedy best-match)
  4. Cross-Encoder   – validation of borderline semantic matches
  5. Category Match  – taxonomy-level relevance for remaining unmatched skills
"""

import re
import math
import numpy as np
from embedder import embed_batch
from config import (
    SKILL_ALIASES,
    SKILL_STRONG_MATCH_THRESHOLD,
    SKILL_PROBABLE_MATCH_THRESHOLD,
    SKILL_CE_VALIDATION_THRESHOLD,
    SKILL_TAXONOMY_LABELS,
    SKILL_WEIGHT_EXACT,
    SKILL_WEIGHT_STRONG,
    SKILL_WEIGHT_PROBABLE,
    SKILL_WEIGHT_CATEGORY,
    CE_SIGMOID_SCALE,
)


# ── Layer 1: Normalization ───────────────────────────────────────────────────

def _normalize_skill(skill: str) -> str:
    """Normalize a skill string for comparison.

    - lowercase
    - collapse whitespace
    - strip trailing punctuation (but keep internal dots like Node.js)
    - basic plural → singular (trailing 's')
    - resolve known aliases as fallback
    """
    s = skill.lower().strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # strip leading/trailing punctuation except dots (for Node.js etc.)
    s = re.sub(r"^[^a-z0-9]+", "", s)
    s = re.sub(r"[^a-z0-9.+#/\-]+$", "", s)
    # basic plural normalization: "transformers" → "transformer"
    # protect: words ending in ss/js/is/us/as/es/ics (kubernetes, jenkins, pandas, analysis, etc.)
    protected_endings = ("ss", "js", "is", "us", "as", "es", "ics", "ns")
    if (len(s) > 5 and s.endswith("s")
            and not any(s.endswith(e) for e in protected_endings)):
        s_singular = s[:-1]
        # only de-pluralize if it doesn't break a known alias
        if s_singular not in SKILL_ALIASES:
            s = s_singular
    # resolve aliases (check both original and de-pluralized)
    s = SKILL_ALIASES.get(s, s)
    return s


# ── Layer 5 helper: Taxonomy classifier ──────────────────────────────────────

class _TaxonomyClassifier:
    """Classifies skills into taxonomy categories using SBERT similarity."""

    def __init__(self):
        self._label_embeddings: np.ndarray | None = None
        self._labels = SKILL_TAXONOMY_LABELS
        self._cache: dict[str, str] = {}

    def _ensure_embeddings(self):
        if self._label_embeddings is None:
            self._label_embeddings = embed_batch(self._labels)

    def classify(self, skill: str) -> str:
        """Return the best-matching taxonomy category for a skill."""
        if skill in self._cache:
            return self._cache[skill]

        self._ensure_embeddings()
        skill_emb = embed_batch([skill])
        sims = np.dot(self._label_embeddings, skill_emb.T).flatten()
        best_idx = int(np.argmax(sims))
        category = self._labels[best_idx]
        self._cache[skill] = category
        return category

    def classify_batch(self, skills: list[str]) -> list[str]:
        """Classify multiple skills at once (efficient batch embedding)."""
        self._ensure_embeddings()
        uncached = [s for s in skills if s not in self._cache]
        if uncached:
            embs = embed_batch(uncached)
            sims = np.dot(self._label_embeddings, embs.T)  # (n_labels, n_skills)
            for j, skill in enumerate(uncached):
                best_idx = int(np.argmax(sims[:, j]))
                self._cache[skill] = self._labels[best_idx]
        return [self._cache[s] for s in skills]


# Module-level singleton
_taxonomy = _TaxonomyClassifier()


# ── Cross-encoder helper ─────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x * CE_SIGMOID_SCALE))


def _cross_encoder_validate(pairs: list[tuple[str, str]]) -> list[float]:
    """Run cross-encoder on (jd_skill, resume_skill) pairs. Returns sigmoid scores."""
    if not pairs:
        return []
    from cross_encoder_reranker import get_cross_encoder
    ce = get_cross_encoder()
    # Format as natural language pairs for ms-marco model
    formatted = [(f"Technology skill: {a}", f"Technology skill: {b}") for a, b in pairs]
    raw_scores = ce.predict(formatted)
    return [_sigmoid(float(s)) for s in raw_scores]


# ── Main matcher class ───────────────────────────────────────────────────────

class SemanticSkillMatcher:
    """5-layer skill matching engine.

    Usage:
        matcher = SemanticSkillMatcher(jd_skills)
        results = matcher.match(candidate_skills)
    """

    def __init__(self, jd_skills: list[str]):
        self.jd_skills_raw = list(jd_skills)
        self.jd_skills = [_normalize_skill(s) for s in jd_skills]
        # Pre-embed JD skills for semantic matching
        if self.jd_skills:
            self.jd_embeddings = embed_batch(self.jd_skills)
        else:
            self.jd_embeddings = np.array([])
        self._emb_cache: dict[str, np.ndarray] = {}

    def _get_embeddings(self, skills: list[str]) -> np.ndarray:
        """Get embeddings for skills, using cache."""
        uncached = [s for s in skills if s not in self._emb_cache]
        if uncached:
            embs = embed_batch(uncached)
            for s, emb in zip(uncached, embs):
                self._emb_cache[s] = emb
        return np.array([self._emb_cache[s] for s in skills])

    def match(self, candidate_skills: list[str]) -> list[dict]:
        """Run all 5 matching layers and return results for each JD skill."""
        if not self.jd_skills:
            return []
        if not candidate_skills:
            return [
                self._missing(i) for i in range(len(self.jd_skills))
            ]

        # Normalize candidate skills
        c_skills_norm = [_normalize_skill(s) for s in candidate_skills]

        # Track which JD and candidate skills are still unmatched
        unmatched_jd = set(range(len(self.jd_skills)))
        unmatched_c = set(range(len(c_skills_norm)))
        results: dict[int, dict] = {}

        # ── Layer 2: Exact Match ─────────────────────────────────────
        for ji in list(unmatched_jd):
            for ci in list(unmatched_c):
                if self.jd_skills[ji] == c_skills_norm[ci]:
                    results[ji] = {
                        "jd_skill": self.jd_skills_raw[ji],
                        "best_match": candidate_skills[ci],
                        "similarity": 1.0,
                        "category": "Exact Match",
                        "match_layer": "exact",
                        "skill_category": None,
                    }
                    unmatched_jd.discard(ji)
                    unmatched_c.discard(ci)
                    break

        # ── Layer 3: Semantic Match (SBERT) ──────────────────────────
        if unmatched_jd and unmatched_c:
            jd_indices = sorted(unmatched_jd)
            c_indices = sorted(unmatched_c)

            jd_embs = self.jd_embeddings[jd_indices]
            c_skills_for_emb = [c_skills_norm[ci] for ci in c_indices]
            c_embs = self._get_embeddings(c_skills_for_emb)

            # Cosine similarity matrix (jd x candidate)
            sim_matrix = np.dot(jd_embs, c_embs.T)

            # Greedy assignment: highest similarity first, no duplicates
            flat_indices = np.argsort(sim_matrix.ravel())[::-1]
            assigned_jd = set()
            assigned_c = set()
            semantic_matches = []  # (ji_real, ci_real, sim, local_ji, local_ci)

            for flat_idx in flat_indices:
                local_ji = flat_idx // len(c_indices)
                local_ci = flat_idx % len(c_indices)
                if local_ji in assigned_jd or local_ci in assigned_c:
                    continue
                sim = float(sim_matrix[local_ji, local_ci])
                if sim < SKILL_PROBABLE_MATCH_THRESHOLD:
                    break  # remaining are all below threshold

                ji_real = jd_indices[local_ji]
                ci_real = c_indices[local_ci]

                if sim >= SKILL_STRONG_MATCH_THRESHOLD:
                    cat = "Strong Match"
                    layer = "semantic"
                else:
                    cat = "Probable Match"
                    layer = "semantic"

                semantic_matches.append((ji_real, ci_real, sim, cat, layer))
                assigned_jd.add(local_ji)
                assigned_c.add(local_ci)

            # ── Layer 4: Cross-Encoder Validation for Probable Matches ───
            probable_indices = [
                i for i, (_, _, _, cat, _) in enumerate(semantic_matches)
                if cat == "Probable Match"
            ]

            if probable_indices:
                ce_pairs = [
                    (self.jd_skills[semantic_matches[i][0]],
                     c_skills_norm[semantic_matches[i][1]])
                    for i in probable_indices
                ]
                ce_scores = _cross_encoder_validate(ce_pairs)

                for idx, ce_score in zip(probable_indices, ce_scores):
                    if ce_score < SKILL_CE_VALIDATION_THRESHOLD:
                        # Downgrade: CE says these aren't really equivalent
                        ji_real = semantic_matches[idx][0]
                        ci_real = semantic_matches[idx][1]
                        sim = semantic_matches[idx][2]
                        semantic_matches[idx] = (ji_real, ci_real, sim, "__REJECTED__", "cross_encoder")
                    else:
                        # Validated by CE
                        ji_real = semantic_matches[idx][0]
                        ci_real = semantic_matches[idx][1]
                        sim = semantic_matches[idx][2]
                        semantic_matches[idx] = (ji_real, ci_real, sim, "Probable Match", "cross_encoder")

            # Record results
            for ji_real, ci_real, sim, cat, layer in semantic_matches:
                if cat == "__REJECTED__":
                    continue  # will fall through to category match or missing
                results[ji_real] = {
                    "jd_skill": self.jd_skills_raw[ji_real],
                    "best_match": candidate_skills[ci_real],
                    "similarity": round(sim, 4),
                    "category": cat,
                    "match_layer": layer,
                    "skill_category": None,
                }
                unmatched_jd.discard(ji_real)
                unmatched_c.discard(ci_real)

        # ── Layer 5: Category Match (Taxonomy) ───────────────────────
        if unmatched_jd and unmatched_c:
            jd_remaining = sorted(unmatched_jd)
            c_remaining = sorted(unmatched_c)

            jd_cats = _taxonomy.classify_batch([self.jd_skills[i] for i in jd_remaining])
            c_cats = _taxonomy.classify_batch([c_skills_norm[i] for i in c_remaining])

            # For each unmatched JD skill, find a candidate in the same category
            for local_ji, ji_real in enumerate(jd_remaining):
                jd_cat = jd_cats[local_ji]
                best_ci = None
                best_sim = -1.0

                for local_ci, ci_real in enumerate(c_remaining):
                    if ci_real not in unmatched_c:
                        continue
                    if c_cats[local_ci] == jd_cat:
                        # Compute similarity to pick the best within category
                        jd_emb = self.jd_embeddings[ji_real]
                        c_emb = self._get_embeddings([c_skills_norm[ci_real]])[0]
                        sim = float(np.dot(jd_emb, c_emb))
                        if sim > best_sim:
                            best_sim = sim
                            best_ci = ci_real

                if best_ci is not None and best_sim > 0.3:
                    results[ji_real] = {
                        "jd_skill": self.jd_skills_raw[ji_real],
                        "best_match": candidate_skills[best_ci],
                        "similarity": round(best_sim, 4),
                        "category": "Category Match",
                        "match_layer": "category",
                        "skill_category": jd_cat,
                    }
                    unmatched_jd.discard(ji_real)
                    unmatched_c.discard(best_ci)

        # ── Fill remaining as Missing ────────────────────────────────
        for ji in unmatched_jd:
            results[ji] = self._missing(ji)

        # Return in JD skill order
        return [results[i] for i in range(len(self.jd_skills))]

    def _missing(self, ji: int) -> dict:
        return {
            "jd_skill": self.jd_skills_raw[ji],
            "best_match": None,
            "similarity": 0.0,
            "category": "Missing",
            "match_layer": None,
            "skill_category": None,
        }

    def match_debug(self, candidate_skills: list[str]) -> dict:
        """Return full debug info for all matching layers."""
        c_skills_norm = [_normalize_skill(s) for s in candidate_skills]
        matches = self.match(candidate_skills)

        exact = [m for m in matches if m["match_layer"] == "exact"]
        semantic = [m for m in matches if m["match_layer"] == "semantic"]
        cross_encoder = [m for m in matches if m["match_layer"] == "cross_encoder"]
        category = [m for m in matches if m["match_layer"] == "category"]
        missing = [m for m in matches if m["category"] == "Missing"]

        # Compute final score
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
        total_weight = sum(weight_map.get(m["category"], 0.0) for m in matches)
        final_score = total_weight / len(matches) if matches else 0.0

        return {
            "jd_skills_raw": self.jd_skills_raw,
            "resume_skills_raw": candidate_skills,
            "normalized_jd": self.jd_skills,
            "normalized_resume": c_skills_norm,
            "exact_matches": exact,
            "semantic_matches": semantic,
            "cross_encoder_validated": cross_encoder,
            "category_matches": category,
            "missing_skills": missing,
            "final_score": round(final_score, 4),
        }
