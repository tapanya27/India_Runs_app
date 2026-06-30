"""
build_index.py
--------------
Run ONCE to build a searchable FAISS index from candidates.jsonl.

Usage:
    python build_index.py
    python build_index.py --candidates resources/candidates.jsonl

Outputs:
    candidate_index.faiss   — FAISS flat inner-product index (cosine via L2-norm)
    candidate_map.pkl       — list of candidate dicts in same order as index

The candidate text built here uses ALL available signal-bearing fields so the
FAISS retrieval step finds truly semantically relevant candidates, not just
those who happen to share surface keywords with the JD.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't encode the status emojis/arrows
# printed below; force UTF-8 so a direct `python build_index.py` run never crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


import numpy as np
from sentence_transformers import SentenceTransformer

# Model / dimension are sourced from config so the index, embedder, and ranker
# never drift out of sync.
from config import SBERT_MODEL, EMBEDDING_DIM

MAX_CANDIDATES = None

CANDIDATES_FILE = Path("resources/candidates.jsonl")
INDEX_FILE = Path("candidate_index.faiss")
MAP_FILE = Path("candidate_map.pkl")
BATCH_SIZE = 128

# Cap the embedded candidate text length: keeps the one-time build fast (item 19)
# while retaining the most discriminative fields. The cross-encoder rerank reads
# the fuller profile later, so deep career signal is recovered at ranking time.
MAX_TEXT_CHARS = 1100


def build_candidate_text(c: dict) -> str:
    """
    Build rich, signal-dense text from a JSONL candidate record.

    Includes every field that carries semantic meaning about the candidate's
    technical depth, domain expertise, and platform engagement. Fields are
    ordered by discriminative importance for the AI-engineer JD.
    """
    parts = []
    profile = c.get("profile", {})

    # ── Identity / positioning ────────────────────────────────────────────────
    name = (profile.get("anonymized_name") or "").strip()
    if name:
        parts.append(f"Name: {name}")

    headline = (profile.get("headline") or "").strip()
    if headline:
        parts.append(headline)

    title = profile.get("current_title", "")
    if title:
        parts.append(f"Role: {title}")

    yoe = profile.get("years_of_experience", 0)
    parts.append(f"Experience: {yoe} years")

    # ── Experience domain (current industry + recent role industries) ─────────
    domains = []
    cur_industry = (profile.get("current_industry") or "").strip()
    if cur_industry:
        domains.append(cur_industry)
    for job in c.get("career_history", [])[:3]:
        ind = (job.get("industry") or "").strip()
        if ind and ind not in domains:
            domains.append(ind)
    if domains:
        parts.append("Experience Domain: " + ", ".join(domains))

    # ── Skills (name + proficiency + duration) ────────────────────────────────
    skills = c.get("skills", [])
    if skills:
        # Build enriched skill string: "PyTorch (advanced, 36 months)"
        skill_parts = []
        for s in skills:
            if isinstance(s, dict) and s.get("name"):
                prof = s.get("proficiency", "")
                dur = s.get("duration_months", 0)
                endorsements = s.get("endorsements", 0)
                token = s["name"]
                if prof:
                    token += f" ({prof}"
                    if dur:
                        token += f", {dur}mo"
                    if endorsements:
                        token += f", {endorsements} endorsements"
                    token += ")"
                skill_parts.append(token)
        parts.append("Skills: " + ", ".join(skill_parts))

    # ── Profile summary (capped for embedding context) ────────────────────────
    summary = (profile.get("summary") or "").strip()
    if summary:
        parts.append(summary[:300])

    # ── Certifications ────────────────────────────────────────────────────────
    certs = c.get("certifications", [])
    if certs:
        cert_names = [cert.get("name", "") for cert in certs if cert.get("name")]
        if cert_names:
            parts.append("Certifications: " + ", ".join(cert_names))

    # ── Career history (each role's description) ──────────────────────────────
    # Kept (not dropped) because the JD's "right answers" are candidates whose
    # career descriptions show real systems work even when their skills/headline
    # lack AI keywords. Capped per-role to stay within MAX_TEXT_CHARS.
    for i, job in enumerate(c.get("career_history", [])[:3]):
        company = job.get("company", "")
        role_title = job.get("title", "")
        desc = (job.get("description") or "").strip()
        if role_title or desc:
            prefix = f"[{role_title} at {company}]" if company else f"[{role_title}]"
            snippet = desc[:250] if desc else ""
            parts.append(f"{prefix} {snippet}".strip())

    # ── Education ─────────────────────────────────────────────────────────────
    for edu in c.get("education", []):
        degree = edu.get("degree", "")
        field  = edu.get("field_of_study", "")
        inst   = edu.get("institution", "")
        tier   = edu.get("tier", "")
        if degree or field:
            edu_str = f"Education: {degree} in {field} at {inst}"
            if tier:
                edu_str += f" ({tier.replace('tier_', 'Tier ')})"
            parts.append(edu_str)

    # ── Redrob signals (include qualitative labels for semantic relevance) ─────
    signals = c.get("redrob_signals", {})
    if signals:
        sig_parts = []
        if signals.get("open_to_work_flag"):
            sig_parts.append("open to work")
        gas = signals.get("github_activity_score", -1)
        if gas is not None and gas > 20:
            sig_parts.append(f"GitHub active (score {gas:.0f})")
        assessments = signals.get("skill_assessment_scores", {})
        if assessments:
            assess_str = ", ".join(
                f"{k} {v:.0f}%" for k, v in list(assessments.items())[:5]
            )
            sig_parts.append(f"Assessed: {assess_str}")
        if sig_parts:
            parts.append("Platform: " + "; ".join(sig_parts))

    return "\n".join(filter(None, parts))[:MAX_TEXT_CHARS]


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from candidates.jsonl")
    parser.add_argument(
        "--candidates", default=str(CANDIDATES_FILE),
        help="Path to candidates.jsonl",
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found.")
        return

    print(f"Loading candidates from {candidates_path} …")
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Skipping malformed line: {e}")

    if MAX_CANDIDATES is not None:
        candidates = candidates[:MAX_CANDIDATES]
    print(f"Loaded {len(candidates):,} candidates.")

    print(f"Loading embedding model: {SBERT_MODEL} …")
    import torch

    # Use all CPU cores for encoding (item 19: performance optimization).
    try:
        torch.set_num_threads(os.cpu_count() or 4)
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}  |  torch threads: {torch.get_num_threads()}")

    model = SentenceTransformer(
        SBERT_MODEL,
        device=device,
    )

    print("Model device:", model.device)

    print("Building candidate texts …")
    texts = [build_candidate_text(c) for c in candidates]

    lengths = [len(t) for t in texts]
    print(f"Candidate texts: {len(texts)}  |  "
          f"shortest {min(lengths)}, avg {sum(lengths)/len(lengths):.0f}, "
          f"longest {max(lengths)} chars")

    print(f"Generating embeddings in batches of {BATCH_SIZE} (this is the one-time precompute) …")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    ).astype(np.float32)

    print("Building FAISS IndexFlatIP …")
    import faiss
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    print(f"Saving index  → {INDEX_FILE}")
    faiss.write_index(index, str(INDEX_FILE))

    print(f"Saving candidate map → {MAP_FILE}")
    with open(MAP_FILE, "wb") as f:
        pickle.dump(candidates, f)

    print(f"\n✅ Done!  Index has {index.ntotal:,} vectors.")
    print(f"   Files: {INDEX_FILE}  {MAP_FILE}")
    print(f"\nNext step:  streamlit run app.py")
    print(f"Or CLI:     python rank.py --out submission.csv")


if __name__ == "__main__":
    main()
