"""
rank.py — Standalone CLI for India Runs Challenge submission

Reproduce command (from submission_metadata.yaml):
    python rank.py --candidates ./resources/candidates.jsonl --out ./submission.csv

Options:
    --candidates  Path to candidates.jsonl  (default: resources/candidates.jsonl)
    --jd          Path to JD text file      (default: embedded JD from job_description.docx)
    --out         Output CSV path           (default: submission.csv)
    --top-k       Candidates to output      (default: 100)
    --retrieve-k  FAISS retrieve count      (default: 700)

The script uses the prebuilt FAISS index (candidate_index.faiss + candidate_map.pkl).
If the index does not exist, run: python build_index.py
"""
import argparse
import csv
import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't encode the status emojis/arrows
# printed below; force UTF-8 so the reproduce command never crashes on a print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from challenge_ranker import (
    load_index,
    rank_candidates_challenge,
    FINAL_TOP,
    FAISS_RETRIEVE,
    DEFAULT_JD,   # canonical JD — shared with app.py so CLI and UI rank identically
)


def validate_output(path: str) -> list[str]:
    """Run a subset of validate_submission.py checks inline."""
    import re
    errors = []
    CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["candidate_id", "rank", "score", "reasoning"]:
            errors.append(f"Bad header: {header}")
        rows = [r for r in reader if any(c.strip() for c in r)]

    if len(rows) != 100:
        errors.append(f"Expected 100 data rows, got {len(rows)}")

    seen_ids, seen_ranks = set(), set()
    by_rank = []
    for i, row in enumerate(rows):
        if len(row) != 4:
            errors.append(f"Row {i+2}: expected 4 columns, got {len(row)}")
            continue
        cid, rank_s, score_s, _ = row
        if not CANDIDATE_ID_PATTERN.match(cid.strip()):
            errors.append(f"Row {i+2}: invalid candidate_id '{cid}'")
        if cid.strip() in seen_ids:
            errors.append(f"Row {i+2}: duplicate candidate_id '{cid}'")
        seen_ids.add(cid.strip())
        try:
            rank = int(rank_s.strip())
            if rank in seen_ranks:
                errors.append(f"Row {i+2}: duplicate rank {rank}")
            seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {i+2}: invalid rank '{rank_s}'")
            rank = None
        try:
            score = float(score_s.strip())
        except ValueError:
            errors.append(f"Row {i+2}: invalid score '{score_s}'")
            score = None
        if rank is not None and score is not None:
            by_rank.append((rank, score, cid.strip()))

    by_rank.sort()
    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]
        r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(f"Non-monotonic scores: rank {r1} ({s1}) < rank {r2} ({s2})")

    return errors


def main():
    parser = argparse.ArgumentParser(description="India Runs Challenge — produce submission.csv")
    parser.add_argument("--candidates", default="resources/candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--jd", default=None,
                        help="Path to JD text file (uses embedded JD if not supplied)")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP,
                        help=f"Top-K results (default {FINAL_TOP})")
    parser.add_argument("--retrieve-k", type=int, default=FAISS_RETRIEVE,
                        help=f"FAISS retrieve K (default {FAISS_RETRIEVE})")
    parser.add_argument("--no-cross-encoder", action="store_true",
                        help="Disable cross-encoder reranking (faster, lower quality)")
    args = parser.parse_args()

    # Load JD
    if args.jd:
        jd_path = Path(args.jd)
        if not jd_path.exists():
            print(f"ERROR: JD file not found: {jd_path}", file=sys.stderr)
            sys.exit(1)
        job_description = jd_path.read_text(encoding="utf-8")
        print(f"JD loaded from {jd_path}")
    else:
        job_description = DEFAULT_JD
        print("Using embedded JD (Senior AI Engineer at Redrob AI).")

    # Load index
    print("Loading FAISS index …")
    try:
        index, all_candidates = load_index()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run: python build_index.py", file=sys.stderr)
        sys.exit(1)

    print(f"Index loaded: {index.ntotal:,} candidates.")

    # Rank
    use_ce = not args.no_cross_encoder
    print(f"Ranking (FAISS retrieve={args.retrieve_k}, final top-k={args.top_k}, "
          f"cross-encoder={'on' if use_ce else 'off'}) …")
    results = rank_candidates_challenge(
        job_description=job_description,
        index=index,
        all_candidates=all_candidates,
        top_k=args.top_k,
        retrieve_k=args.retrieve_k,
        use_cross_encoder=use_ce,
    )

    if len(results) < args.top_k:
        print(
            f"ERROR: only {len(results)} results produced (need {args.top_k}). "
            f"Increase --retrieve-k or check the index. Not writing an invalid CSV.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Write CSV
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                r["final_score"],
                r["reasoning"],
            ])

    print(f"Written: {out_path}  ({len(results)} rows)")

    # Validate
    print("Validating output …")
    errors = validate_output(str(out_path))
    if errors:
        print(f"\n⚠  Validation issues ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✅ Submission valid — all checks passed.")

    # Print top-5 preview
    print("\nTop 5:")
    for r in results[:5]:
        print(f"  #{r['rank']}  {r['candidate_id']}  {r['final_score']:.4f}  {r['reasoning'][:80]}")


if __name__ == "__main__":
    main()
