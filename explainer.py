# Skill matching is handled by skill_matcher.SemanticSkillMatcher


def generate_explanation(jd_text: str, skill_matches: list[dict], candidate: dict, scores: dict) -> str:
    name = candidate.get("name") or candidate.get("auto_name") or "This candidate"
    c_skills = candidate.get("skills", [])

    lines = [f"**{name}** — Final Score: {scores['final_score']:.1%}"]
    lines.append("")

    lines.append(f"**Semantic Match: {scores['semantic_score']:.1%}**")
    lines.append("Overall profile-to-JD alignment based on deep language understanding.")
    lines.append("")

    lines.append(f"**Skill Match: {scores['skill_score']:.1%}**")

    exact_lines = []
    semantic_lines = []
    category_lines = []
    missing_lines = []

    for match in skill_matches:
        cat = match["category"]
        if cat == "Exact Match":
            exact_lines.append(f"✓ `{match['jd_skill']}`")
        elif cat in ["Strong Match", "Probable Match"]:
            semantic_lines.append(
                f"✓ `{match['jd_skill']}` ← `{match['best_match']}` ({match['similarity']:.2f})"
            )
        elif cat == "Category Match":
            short_cat = (match.get("skill_category") or "").split(" and ")[0]
            category_lines.append(
                f"~ `{match['jd_skill']}` ← `{match['best_match']}` [{short_cat}]"
            )
        else:
            missing_lines.append(f"✗ `{match['jd_skill']}`")

    if exact_lines:
        lines.append("Matched Skills:")
        lines.append("")
        lines.append(" ".join(exact_lines))
        lines.append("")

    if semantic_lines:
        lines.append("Semantic Matches:")
        lines.append("")
        lines.append(" ".join(semantic_lines))
        lines.append("")

    if category_lines:
        lines.append("Category Matches:")
        lines.append("")
        lines.append(" ".join(category_lines))
        lines.append("")

    if missing_lines:
        lines.append("Missing Skills:")
        lines.append("")
        lines.append(" ".join(missing_lines))
        lines.append("")

    matched_c_skills = {m["best_match"] for m in skill_matches if m["best_match"]}
    extra_skills = [s for s in c_skills if s not in matched_c_skills]
    if extra_skills:
        lines.append("Additional Skills:")
        lines.append("")
        lines.append(", ".join(f"`{s}`" for s in extra_skills[:8]))
    lines.append("")

    exp = candidate.get("experience_years", 0)
    experience_domain = candidate.get("experience_domain", "")
    if not experience_domain:
        try:
            from resume_parser import extract_experience_domain

            experience_domain = extract_experience_domain(candidate.get("raw_text", ""))
        except Exception:
            experience_domain = ""
    if experience_domain:
        lines.append(
            f"**Experience: {scores['experience_score']:.1%}** ({exp:.1f} years in {experience_domain})"
        )
    else:
        lines.append(f"**Experience: {scores['experience_score']:.1%}** ({exp:.1f} years detected)")
    lines.append("")

    recruitability = scores.get("recruitability_score", 0.5)
    hire_confidence = scores.get("hire_confidence", scores.get("final_score", 0.0))
    lines.append(f"**Recruitability: {recruitability:.1%}**")
    signal_parts = []
    for label, field in [
        ("response rate", "response_rate"),
        ("profile completeness", "profile_completeness"),
        ("interview completion", "interview_completion_rate"),
        ("activity level", "activity_level"),
    ]:
        if field in candidate:
            signal_parts.append(f"{label}={candidate[field]}")
    if signal_parts:
        lines.append("  • Signals: " + ", ".join(signal_parts))
    else:
        lines.append("  • No recruiter signals found; neutral fallback used.")
    lines.append("")

    lines.append(f"**Project Relevance: {scores['project_score']:.1%}**")
    projects = candidate.get("projects", [])
    if projects:
        for p in projects[:2]:
            preview = p[:130] + ("..." if len(p) > 130 else "")
            lines.append(f"  • {preview}")
    lines.append("")

    edu = candidate.get("education", [])
    if edu:
        lines.append(f"**Education**: {', '.join(edu)}")
        lines.append("")

    lines.append(f"**Hire Confidence: {hire_confidence:.1%}**")
    lines.append("A blended view of technical fit and recruiter signals.")
    lines.append("")

    lines.append(f"**Verdict**: {_verdict(scores['final_score'])}")
    return "\n".join(lines)


def _verdict(score: float) -> str:
    if score >= 0.75:
        return "🟢 Strong match — highly recommended for interview."
    elif score >= 0.55:
        return "🟡 Good match — worth a closer look."
    elif score >= 0.35:
        return "🟠 Partial match — notable skill gaps present."
    else:
        return "🔴 Weak match — significant gaps identified."


def generate_skill_gap_report(matcher, candidates: list[dict]) -> dict:
    gap_counts: dict[str, int] = {skill: 0 for skill in matcher.jd_skills_raw}
    for c in candidates:
        matches = matcher.match(c.get("skills", []))
        for m in matches:
            if m["category"] == "Missing":
                gap_counts[m["jd_skill"]] += 1
    return dict(sorted(gap_counts.items(), key=lambda x: -x[1]))
