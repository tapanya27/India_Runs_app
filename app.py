import os
import hashlib
from pathlib import Path

import streamlit as st
import pandas as pd
import httpx

import resume_parser as rp

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.rank-badge   { font-size:1.5rem; font-weight:700; }
.score-label  { font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:.05em; }
.score-val    { font-size:1.1rem; font-weight:600; }
.card         { background:#1e1e2e; border-radius:12px; padding:16px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 AI-Powered Candidate Ranking System")
st.caption("Semantic matching · all-mpnet-base-v2 + Cross Encoder L-12 + FAISS · Explainable scoring")

# ── Session state init ────────────────────────────────────────────────────────
if "parsed_candidates" not in st.session_state:
    st.session_state.parsed_candidates = []   # [{...parsed...}]
if "file_hashes" not in st.session_state:
    st.session_state.file_hashes = set()
if "file_hash_to_index" not in st.session_state:
    st.session_state.file_hash_to_index = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top K candidates", 1, 20, 10)
    use_ce = st.toggle("Cross Encoder re-ranking", value=True,
                        help="More accurate but slower on first run")
    st.divider()
    st.markdown("**Hybrid Scoring Weights**")
    st.markdown("| Component | Weight |")
    st.markdown("|---|---|")
    st.markdown("| Semantic Match | **50%** |")
    st.markdown("| Skill Match | **20%** |")
    st.markdown("| Experience | **15%** |")
    st.markdown("| Project Relevance | **15%** |")
    st.markdown("| Recruitability | **25%** |")
    st.divider()
    if st.button("🗑️ Clear All Candidates", use_container_width=True):
        st.session_state.parsed_candidates = []
        st.session_state.file_hashes = set()
        st.session_state.file_hash_to_index = {}
        st.rerun()

# ── Job Description ───────────────────────────────────────────────────────────
st.subheader("📋 Job Description")
jd = st.text_area(
    "Paste the full job description",
    height=200,
    placeholder="e.g. We are looking for a Senior ML Engineer with 4+ years experience in PyTorch, transformers...",
)

if jd:
    jd_skills = rp.extract_skills(jd)
    if jd_skills:
        st.caption(f"**Detected JD skills ({len(jd_skills)}):** " + " · ".join(f"`{s}`" for s in jd_skills))

st.divider()

# ── Resume Upload ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📄 Upload Resumes")
    st.caption("Accepted formats: PDF, TXT (individual resumes) · CSV (batch — one candidate per row)")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or CSV resumes",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uf in uploaded_files:
            file_hash = hashlib.md5(uf.getvalue()).hexdigest()
            ext = Path(uf.name).suffix.lower()

            if ext == ".csv":
                with st.spinner(f"Parsing CSV batch: {uf.name}..."):
                    try:
                        resp = httpx.post(
                            f"{API_URL}/upload-csv",
                            files={"file": (uf.name, uf.getvalue(), "text/csv")},
                            timeout=120,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        added = 0
                        for candidate in data["candidates"]:
                            c_hash = hashlib.md5((candidate.get("name", "") + candidate.get("raw_text", "")[:200]).encode()).hexdigest()
                            if c_hash not in st.session_state.file_hashes:
                                st.session_state.parsed_candidates.append(candidate)
                                st.session_state.file_hashes.add(c_hash)
                                added += 1
                        st.success(f"✅ CSV: added {added} candidate(s) from {uf.name}")
                    except Exception as e:
                        st.error(f"Failed to parse CSV {uf.name}: {e}")
            else:
                with st.spinner(f"Parsing {uf.name}..."):
                    try:
                        resp = httpx.post(
                            f"{API_URL}/upload-resume",
                            files={"file": (uf.name, uf.getvalue(), uf.type or "application/octet-stream")},
                            data={"name": Path(uf.name).stem},
                            timeout=60,
                        )
                        resp.raise_for_status()
                        parsed = resp.json()
                        existing_index = st.session_state.file_hash_to_index.get(file_hash)
                        if existing_index is None:
                            st.session_state.parsed_candidates.append(parsed)
                            st.session_state.file_hash_to_index[file_hash] = len(st.session_state.parsed_candidates) - 1
                        else:
                            st.session_state.parsed_candidates[existing_index] = parsed
                        st.session_state.file_hashes.add(file_hash)
                        st.success(
                            f"✅ Parsed: {parsed.get('name', uf.name)} — {parsed.get('experience_domain') or 'domain not detected'} — {float(parsed.get('experience_years', 0) or 0):.1f} yrs detected"
                        )
                    except Exception as e:
                        st.error(f"Failed to parse {uf.name}: {e}")

with col2:
    st.subheader("✍️ Add Manually")
    with st.form("manual_form", clear_on_submit=True):
        c_name = st.text_input("Name *")
        c_text = st.text_area("Resume / Profile Text *", height=120)
        c_skills_raw = st.text_input("Skills (comma-separated, optional)")
        c_exp = st.number_input("Years of Experience", min_value=0.0, step=0.5)
        submitted = st.form_submit_button("➕ Add Candidate")

    if submitted:
        if not c_name or not c_text:
            st.warning("Name and resume text are required.")
        else:
            manual_skills = (
                rp.normalize_skill_list([s.strip() for s in c_skills_raw.split(",") if s.strip()])
                if c_skills_raw
                else rp.extract_skills(c_text)
            )
            candidate = {
                "name": c_name,
                "raw_text": c_text,
                "skills": manual_skills,
                "experience_years": c_exp or rp.extract_experience_years(c_text),
                "experience_domain": rp.extract_experience_domain(c_text),
                "projects": rp.extract_projects(c_text),
                "education": rp.extract_education(c_text),
            }
            st.session_state.parsed_candidates.append(candidate)
            st.success(f"Added {c_name}")

# ── Candidate List Preview ────────────────────────────────────────────────────
all_candidates = st.session_state.parsed_candidates

if all_candidates:
    st.divider()
    with st.expander(f"👥 {len(all_candidates)} candidate(s) loaded — click to preview"):
        for i, c in enumerate(all_candidates):
            name = c.get("name") or c.get("auto_name") or f"Candidate {i+1}"
            skills_preview = ", ".join(c.get("skills", [])[:6])
            exp = c.get("experience_years", 0)
            domain = c.get("experience_domain") or "domain not detected"
            st.markdown(f"**{i+1}. {name}** — {domain} — {exp:.1f} yrs — `{skills_preview or 'no skills detected'}`")

st.divider()

# ── Rank ──────────────────────────────────────────────────────────────────────
rank_disabled = not (jd.strip() and all_candidates)
if st.button("🚀 Rank Candidates", type="primary", disabled=rank_disabled, use_container_width=True):
    with st.spinner("Ranking... first run downloads models (~180MB)"):
        try:
            resp = httpx.post(
                f"{API_URL}/rank",
                json={
                    "job_description": jd,
                    "candidates": all_candidates,
                    "top_k": top_k,
                    "use_cross_encoder": use_ce,
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Ranking failed: {e}")
            st.stop()

    ranked = data["ranked_candidates"]
    gap_report = data["skill_gap_report"]

    if not ranked:
        st.warning("No candidates returned. Check your inputs.")
        st.stop()

    st.success(f"Ranked {len(ranked)} candidates successfully!")

    # ── Dashboard ────────────────────────────────────────────────────────────
    st.subheader("📌 Recruiter Dashboard")
    dashboard_cols = st.columns(4)
    dashboard_cols[0].metric("Candidates", len(all_candidates))
    dashboard_cols[1].metric("Top Hire Confidence", f"{ranked[0]['hire_confidence']:.1%}")
    dashboard_cols[2].metric(
        "Avg Recruitability",
        f"{pd.Series([c.get('recruitability_score', 0.5) for c in ranked]).mean():.1%}",
    )
    dashboard_cols[3].metric(
        "Avg Experience",
        f"{pd.Series([c.get('experience_years', 0) for c in all_candidates]).mean():.1f} yrs",
    )

    dashboard_left, dashboard_right = st.columns(2)

    skill_rows = []
    for candidate in all_candidates:
        for skill in candidate.get("skills", []):
            skill_rows.append(skill)

    if skill_rows:
        skill_counts = pd.Series(skill_rows).value_counts().head(12)
        dashboard_left.subheader("Top Skills")
        dashboard_left.bar_chart(skill_counts)
    else:
        dashboard_left.info("No skills detected yet.")

    experience_values = pd.Series([float(c.get("experience_years", 0) or 0) for c in all_candidates])
    if not experience_values.empty:
        bins = pd.cut(
            experience_values,
            bins=[-0.1, 1, 3, 5, 8, 20],
            labels=["0-1", "1-3", "3-5", "5-8", "8+"],
            include_lowest=True,
        )
        experience_distribution = bins.value_counts().reindex(["0-1", "1-3", "3-5", "5-8", "8+"], fill_value=0)
        dashboard_right.subheader("Experience Distribution")
        dashboard_right.bar_chart(experience_distribution)

    st.subheader("Candidate Distribution")
    hire_conf_series = pd.Series([c.get("hire_confidence", c.get("final_score", 0.0)) for c in ranked])
    if not hire_conf_series.empty:
        confidence_bins = pd.cut(
            hire_conf_series,
            bins=[0.0, 0.35, 0.55, 0.75, 0.9, 1.0],
            labels=["0-35", "35-55", "55-75", "75-90", "90-100"],
            include_lowest=True,
        )
        candidate_distribution = confidence_bins.value_counts().reindex(["0-35", "35-55", "55-75", "75-90", "90-100"], fill_value=0)
        st.bar_chart(candidate_distribution)

    st.subheader("Top Candidates")
    top_preview = pd.DataFrame(
        [
            {
                "Rank": f"#{r['rank']}",
                "Name": r.get("name") or r.get("auto_name") or "Unknown",
                "Hire Confidence": r["hire_confidence"],
                "Final Score": r["final_score"],
                "Recruitability": r.get("recruitability_score", 0.5),
            }
            for r in ranked[:5]
        ]
    )
    st.dataframe(top_preview, use_container_width=True, hide_index=True)

    # ── Score table ───────────────────────────────────────────────────────────
    st.subheader("🏆 Ranking Results")

    rows = []
    for r in ranked:
        rows.append({
            "Rank": f"#{r['rank']}",
            "Name": r.get("name") or r.get("auto_name") or "Unknown",
            "Hire Confidence": r["hire_confidence"],
            "Final Score": r["final_score"],
            "Semantic": r["semantic_score"],
            "Skills": r["skill_score"],
            "Experience": r["experience_score"],
            "Projects": r["project_score"],
            "Recruitability": r.get("recruitability_score", 0.5),
            "Exp. Years": r.get("experience_years", 0),
            "Education": ", ".join(r.get("education", [])) or "—",
        })

    df = pd.DataFrame(rows)

    def color_score(val):
        if isinstance(val, float):
            if val >= 0.75:
                return "background-color: #1a472a; color: #69db7c"
            elif val >= 0.55:
                return "background-color: #5c3d00; color: #ffd43b"
            elif val >= 0.35:
                return "background-color: #4a1942; color: #f783ac"
            else:
                return "background-color: #3b1111; color: #ff6b6b"
        return ""

    score_cols = ["Hire Confidence", "Final Score", "Semantic", "Skills", "Experience", "Projects", "Recruitability"]
    styled = (
        df.style
        .map(color_score, subset=score_cols)
        .format({c: "{:.1%}" for c in score_cols})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── CSV download ──────────────────────────────────────────────────────────
    csv = df.copy()
    for c in score_cols:
        csv[c] = csv[c].apply(lambda x: f"{x:.1%}" if isinstance(x, float) else x)
    st.download_button(
        "⬇️ Download Results CSV",
        csv.to_csv(index=False).encode(),
        file_name="candidate_rankings.csv",
        mime="text/csv",
    )

    # ── Score bar chart ───────────────────────────────────────────────────────
    st.subheader("📊 Score Comparison")
    chart_df = pd.DataFrame({
        "Candidate": [r.get("name") or r.get("auto_name") or f"#{r['rank']}" for r in ranked],
        "Hire Confidence": [r["hire_confidence"] for r in ranked],
        "Final Score": [r["final_score"] for r in ranked],
        "Semantic": [r["semantic_score"] for r in ranked],
        "Skills": [r["skill_score"] for r in ranked],
        "Experience": [r["experience_score"] for r in ranked],
    }).set_index("Candidate")
    st.bar_chart(chart_df)

    # ── Detailed Explanations ─────────────────────────────────────────────────
    st.subheader("📝 Detailed Explanations")
    for r in ranked:
        name = r.get("name") or r.get("auto_name") or "Unknown"
        final = r["final_score"]

        medal = "🥇" if r["rank"] == 1 else "🥈" if r["rank"] == 2 else "🥉" if r["rank"] == 3 else f"#{r['rank']}"
        with st.expander(f"{medal} {name} — {final:.1%}"):
            # Score pills
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Semantic", f"{r['semantic_score']:.1%}")
            c2.metric("Skills", f"{r['skill_score']:.1%}")
            c3.metric("Experience", f"{r['experience_score']:.1%}")
            c4.metric("Projects", f"{r['project_score']:.1%}")
            st.metric("Hire Confidence", f"{r['hire_confidence']:.1%}")
            st.divider()
            st.markdown(r["explanation"])

    # ── Skill Gap Report ──────────────────────────────────────────────────────
    if gap_report:
        st.subheader("🔍 Skill Gap Report")
        st.caption("How many candidates are missing each required JD skill")
        gap_df = pd.DataFrame(
            [{"Skill": k, "Candidates Missing": v} for k, v in gap_report.items()]
        )
        total = len(all_candidates)
        gap_df["Coverage"] = gap_df["Candidates Missing"].apply(
            lambda x: f"{(total - x) / total:.0%} have it"
        )
        st.dataframe(gap_df, use_container_width=True, hide_index=True)
        st.bar_chart(gap_df.set_index("Skill")["Candidates Missing"])

elif rank_disabled:
    if not jd.strip():
        st.info("Enter a job description to get started.")
    elif not all_candidates:
        st.info("Upload at least one resume or add a candidate manually.")
