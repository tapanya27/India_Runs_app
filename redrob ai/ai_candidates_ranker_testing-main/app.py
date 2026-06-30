"""
app.py — India Runs Challenge: Intelligent Candidate Ranking (Streamlit UI)

One-time setup:  python build_index.py
Every run:       streamlit run app.py
"""
import io
import csv

import pandas as pd
import streamlit as st

from challenge_ranker import (
    load_index,
    rank_candidates_challenge,
    FINAL_TOP,
    FAISS_RETRIEVE,
    JD_REQUIRED_SKILLS,
    DEFAULT_JD,   # canonical JD — shared with rank.py so UI and CLI rank identically
)
from resume_parser import extract_skills

st.set_page_config(
    page_title="India Runs — AI Candidate Ranker",
    page_icon="🇮🇳",
    layout="wide",
)

st.markdown("""
<style>
.score-label { font-size:0.72rem; color:#888; text-transform:uppercase; letter-spacing:.05em; }
</style>
""", unsafe_allow_html=True)

st.title("🇮🇳 India Runs — AI Candidate Ranking System")
st.caption(
    f"FAISS retrieval (top {FAISS_RETRIEVE}) → cross-encoder rerank → 6-component scoring "
    f"→ top {FINAL_TOP} with honeypot & consulting-company filtering"
)

# ── Load FAISS index once ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading candidate index …")
def get_index():
    try:
        return load_index()
    except FileNotFoundError as e:
        return None, str(e)


index_obj, candidates_or_err = get_index()

if index_obj is None:
    st.error(
        f"❌ **Index not found.**\n\n{candidates_or_err}\n\n"
        "Run `python build_index.py` once, then refresh this page."
    )
    st.stop()

all_candidates: list[dict] = candidates_or_err
st.success(f"✅ Index loaded — **{index_obj.ntotal:,}** candidates ready.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top K results", 10, 100, FINAL_TOP, step=10)
    retrieve_k = st.slider(
        "FAISS retrieve K", 100, 1500, FAISS_RETRIEVE, step=100,
        help="Candidates retrieved from FAISS before detailed scoring. Higher = better recall but slower.",
    )
    use_ce = st.checkbox(
        "Cross-encoder rerank", value=True,
        help="Re-rank the retrieved pool with a cross-encoder for sharper ordering (slower).",
    )
    st.divider()
    st.markdown("**Scoring Weights**")
    st.markdown("""
| Component | Weight |
|---|---|
| Semantic Similarity | **35%** |
| Skill Match (JD-aware) | **30%** |
| Experience | **15%** |
| Projects / Career | **10%** |
| Education (tier) | **3%** |
| Redrob Signals (23) | **7%** |
""")
    st.divider()
    st.markdown("**Quality Filters**")
    st.markdown("✅ Honeypot detection  \n✅ Consulting-only penalty  \n✅ India location boost  \n✅ All 23 Redrob signals")

# ── JD Input ──────────────────────────────────────────────────────────────────
st.subheader("📋 Job Description")

# DEFAULT_JD is imported from challenge_ranker (shared with rank.py) so the UI
# and the CLI rank against the identical job description.

jd = st.text_area(
    "Paste the full job description (pre-filled with challenge JD)",
    value=DEFAULT_JD,
    height=250,
)

if jd.strip():
    jd_skills = extract_skills(jd)
    if jd_skills:
        st.caption(
            f"**Detected JD skills ({len(jd_skills)}):** "
            + " · ".join(f"`{s}`" for s in jd_skills[:25])
        )

st.divider()

# ── Rank button ───────────────────────────────────────────────────────────────
if st.button("🚀 Rank Candidates", type="primary", disabled=not jd.strip(), use_container_width=True):
    with st.spinner(f"Searching {index_obj.ntotal:,} candidates … scoring top {retrieve_k} …"):
        try:
            results = rank_candidates_challenge(
                job_description=jd,
                index=index_obj,
                all_candidates=all_candidates,
                top_k=top_k,
                retrieve_k=retrieve_k,
                use_cross_encoder=use_ce,
            )
        except Exception as exc:
            st.error(f"Ranking failed: {exc}")
            st.stop()

    if not results:
        st.warning("No candidates returned.")
        st.stop()

    st.success(f"Done — top **{len(results)}** candidates ranked.")

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Pool Size", f"{index_obj.ntotal:,}")
    m2.metric("FAISS Retrieved", str(retrieve_k))
    m3.metric("Final Ranked", str(len(results)))
    m4.metric("Top Score", f"{results[0]['final_score']:.1%}")
    avg_sem = sum(r["semantic_score"] for r in results) / len(results)
    m5.metric("Avg Semantic", f"{avg_sem:.1%}")

    # ── Score distribution ────────────────────────────────────────────────────
    st.subheader("📊 Score Breakdown — Top Candidates")
    chart_data = pd.DataFrame([
        {
            "ID": r["candidate_id"],
            "Final": r["final_score"],
            "Semantic": r["semantic_score"],
            "Skills": r["skill_score"],
            "Experience": r["experience_score"],
            "Projects": r["project_score"],
        }
        for r in results[:30]
    ]).set_index("ID")
    st.bar_chart(chart_data)

    # ── Results table ─────────────────────────────────────────────────────────
    st.subheader("🏆 Rankings")
    score_cols = ["Final Score", "Semantic", "Skills", "Exp", "Projects", "Education", "Redrob"]

    rows = []
    for r in results:
        profile = r["candidate"].get("profile", {})
        rows.append({
            "Rank":        r["rank"],
            "Candidate ID": r["candidate_id"],
            "Title":       profile.get("current_title", "—"),
            "YoE":         profile.get("years_of_experience", 0),
            "Location":    f"{profile.get('location','')} {profile.get('country','')}".strip(),
            "Final Score": r["final_score"],
            "Semantic":    r["semantic_score"],
            "Skills":      r["skill_score"],
            "Exp":         r["experience_score"],
            "Projects":    r["project_score"],
            "Education":   r["education_score"],
            "Redrob":      r["redrob_score"],
            "Honeypot":    r.get("honeypot_mult", 1.0),
            "Consult":     r.get("consult_mult", 1.0),
        })

    df = pd.DataFrame(rows)

    def color_score(val):
        if isinstance(val, float):
            if val >= 0.75:   return "background-color:#1a472a;color:#69db7c"
            elif val >= 0.55: return "background-color:#5c3d00;color:#ffd43b"
            elif val >= 0.35: return "background-color:#4a1942;color:#f783ac"
            else:             return "background-color:#3b1111;color:#ff6b6b"
        return ""

    styled = (
        df.style
        .map(color_score, subset=score_cols)
        .format({c: "{:.1%}" for c in score_cols})
        .format({"Honeypot": "{:.2f}", "Consult": "{:.2f}", "YoE": "{:.1f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── submission.csv download ───────────────────────────────────────────────
    st.subheader("⬇️ Download Submission")

    # Build the submission CSV in-memory with exact required format
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in results:
        writer.writerow([
            r["candidate_id"],
            r["rank"],
            r["final_score"],
            r["reasoning"],
        ])
    csv_bytes = buf.getvalue().encode("utf-8")

    col_dl, col_info = st.columns([1, 2])
    col_dl.download_button(
        label="⬇️ Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col_info.markdown("""
**Format:** `candidate_id, rank, score, reasoning`
**Rows:** exactly 100
**Scores:** monotonically non-increasing ✅
**Validate:** `python resources/validate_submission.py submission.csv`
""")

    # ── Candidate detail cards (top 20) ───────────────────────────────────────
    st.subheader("📝 Candidate Details (top 20)")
    for r in results[:20]:
        profile = r["candidate"].get("profile", {})
        signals = r["candidate"].get("redrob_signals", {})
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r["rank"], f"#{r['rank']}")
        label = (
            f"{medal} [{r['candidate_id']}] "
            f"{profile.get('current_title','?')} — "
            f"{r['final_score']:.1%}"
        )
        # Honeypot/consult flags
        flags = []
        if r.get("honeypot_mult", 1.0) < 0.9:
            flags.append(f"⚠️ Honeypot signal ({r['honeypot_mult']:.2f})")
        if r.get("consult_mult", 1.0) < 0.95:
            flags.append(f"⚠️ Consulting-heavy ({r['consult_mult']:.2f})")

        with st.expander(label):
            if flags:
                st.warning("  |  ".join(flags))
            c1, c2, c3 = st.columns(3)
            c1.metric("Semantic",   f"{r['semantic_score']:.1%}")
            c2.metric("Skills",     f"{r['skill_score']:.1%}")
            c3.metric("Experience", f"{r['experience_score']:.1%}")
            c4, c5, c6 = st.columns(3)
            c4.metric("Projects",   f"{r['project_score']:.1%}")
            c5.metric("Education",  f"{r['education_score']:.1%}")
            c6.metric("Redrob",     f"{r['redrob_score']:.1%}")
            st.divider()

            headline = profile.get("headline", "—")
            summary  = profile.get("summary",  "—")
            st.markdown(f"**Headline:** {headline}")
            st.markdown(f"**Summary:** {summary[:300]}{'…' if len(summary)>300 else ''}")
            loc = f"{profile.get('location','')}, {profile.get('country','')}".strip(", ")
            st.markdown(f"**Location:** {loc}  |  **YoE:** {profile.get('years_of_experience',0):.1f} yrs")

            skills = r["candidate"].get("skills", [])
            if skills:
                top_skills = sorted(skills, key=lambda s: s.get("endorsements", 0), reverse=True)[:10]
                st.markdown("**Top Skills:** " + " · ".join(f"`{s['name']}`" for s in top_skills))

            for edu in r["candidate"].get("education", [])[:2]:
                st.markdown(
                    f"**Education:** {edu.get('degree','')} {edu.get('field_of_study','')} "
                    f"@ {edu.get('institution','')} ({edu.get('tier','').replace('tier_','Tier ')})"
                )

            # Redrob key signals
            rrr = signals.get("recruiter_response_rate")
            icr = signals.get("interview_completion_rate")
            gas = signals.get("github_activity_score", -1)
            otw = signals.get("open_to_work_flag")
            sig_parts = []
            if rrr is not None: sig_parts.append(f"Response rate: {rrr:.0%}")
            if icr is not None: sig_parts.append(f"Interview completion: {icr:.0%}")
            if gas and gas >= 0: sig_parts.append(f"GitHub score: {gas}")
            if otw is not None: sig_parts.append("Open to work ✅" if otw else "Not open to work")
            if sig_parts:
                st.markdown("**Redrob Signals:** " + "  |  ".join(sig_parts))

            st.info(f"**Reasoning:** {r['reasoning']}")

elif not jd.strip():
    st.info("The JD is pre-filled above. Edit it if needed, then click **Rank Candidates**.")
