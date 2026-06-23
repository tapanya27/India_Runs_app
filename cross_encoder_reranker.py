import math
from sentence_transformers import CrossEncoder
from config import CROSS_ENCODER_MODEL, CE_SIGMOID_SCALE, CROSS_ENCODER_WEIGHT

_cross_encoder: CrossEncoder | None = None

# ms-marco CE outputs logits; sigmoid converts to (0, 1) probability
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x * CE_SIGMOID_SCALE))


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank(job_description: str, candidates: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    if not candidates:
        return []
    ce = get_cross_encoder()

    # Use up to 512 words of raw text for richer CE context (not 512 chars)
    def get_candidate_text(c: dict) -> str:
        raw = c.get("raw_text", "")
        words = raw.split()[:400]
        base = " ".join(words)
        skills = c.get("skills", [])
        if skills:
            base += " Skills: " + ", ".join(skills)
        return base

    pairs = [(job_description[:512], get_candidate_text(c)) for c, _ in candidates]
    raw_scores = ce.predict(pairs)

    # Convert logits → normalized [0, 1] via sigmoid
    ce_scores = [_sigmoid(float(s)) for s in raw_scores]

    # Combine CE score with original FAISS cosine score to leverage both signals
    combined = []
    for (cand, faiss_score), ce_score in zip(candidates, ce_scores):
        # faiss_score assumed in [0,1] (cosine similarity clamped earlier)
        combined_score = CROSS_ENCODER_WEIGHT * ce_score + (1.0 - CROSS_ENCODER_WEIGHT) * float(faiss_score)
        combined.append((cand, float(max(0.0, min(1.0, combined_score)))))

    # Sort by combined score desc
    ranked = sorted(combined, key=lambda x: x[1], reverse=True)
    return ranked
