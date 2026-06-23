import numpy as np
from sentence_transformers import SentenceTransformer
from config import SBERT_MODEL, EMBEDDING_DIM

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(SBERT_MODEL)
    return _model


def embed_text(text: str) -> np.ndarray:
    model = get_model()
    return model.encode(text, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32).astype(np.float32)


def build_candidate_text(candidate: dict) -> str:
    parts = []

    name = candidate.get("name") or candidate.get("auto_name", "")
    if name:
        parts.append(f"Candidate: {name}")

    raw = candidate.get("raw_text", "").strip()
    if raw:
        # Use up to 1500 chars of raw text for richer context
        parts.append(raw[:1500])

    skills = candidate.get("skills", [])
    if skills:
        parts.append("Technical Skills: " + ", ".join(skills))

    exp = candidate.get("experience_years", 0)
    parts.append(f"Years of Experience: {exp}")

    experience_domain = candidate.get("experience_domain", "")
    if not experience_domain:
        try:
            from resume_parser import extract_experience_domain

            experience_domain = extract_experience_domain(candidate.get("raw_text", ""))
        except Exception:
            experience_domain = ""
    if experience_domain:
        parts.append(f"Experience Domain: {experience_domain}")

    education = candidate.get("education", [])
    if education:
        parts.append("Education: " + ", ".join(education))

    projects = candidate.get("projects", [])
    if projects:
        parts.append("Projects: " + " | ".join(projects[:5]))

    return "\n".join(parts)
