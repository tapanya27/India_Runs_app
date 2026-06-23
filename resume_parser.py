import re
import csv
import io
import PyPDF2
import pdfplumber
import spacy
import nltk
from pathlib import Path

import numpy as np
from config import SKILL_KEYWORDS, SKILL_ALIASES

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

try:
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover - optional dependency
    pdfium = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover - optional dependency
    RapidOCR = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None

_ocr_engine = None

DEGREE_MAP = {
    r"(?<![a-z0-9])ph\.?d(?![a-z0-9])|doctor of philosophy": "PhD",
    r"(?<![a-z0-9])m\.?\s*tech(?![a-z0-9])|master of technology": "M.Tech",
    r"(?<![a-z0-9])m\.?\s*e\.?\b|master of engineering": "M.E.",
    r"(?<![a-z0-9])m\.?\s*s\.?\s*c\.?\b|master of science": "M.Sc",
    r"(?<![a-z0-9])m\.?\s*b\.?\s*a\.?\b|master of business administration|master of business": "MBA",
    r"(?<![a-z0-9])b\.?\s*tech(?![a-z0-9])|bachelor of technology": "B.Tech",
    r"(?<![a-z0-9])b\.?\s*e\.?\b|bachelor of engineering": "B.E.",
    r"(?<![a-z0-9])b\.?\s*s\.?\s*c\.?\b|bachelor of science": "B.Sc",
    r"(?<![a-z0-9])b\.?\s*c\.?\s*a\.?\b|bachelor of computer application": "BCA",
    r"(?<![a-z0-9])m\.?\s*c\.?\s*a\.?\b|master of computer application": "MCA",
}

EXPERIENCE_PATTERNS = [
    r"(\d+\.?\d*)\s*\+\s*years?",
    r"(\d+\.?\d*)\s*\+?\s*years?\s+(?:of\s+)?(?:work\s+)?experience",
    r"experience\s*[:\-–]?\s*(\d+\.?\d*)\s*\+?\s*years?",
    r"(\d+\.?\d*)\s*\+?\s*years?\s+(?:in\s+)?(?:the\s+)?(?:industry|field|domain)",
]

EXPERIENCE_DOMAIN_PATTERNS = [
    (r"\b(machine learning|ml|artificial intelligence|ai|deep learning|nlp|llm|transformers?|recommendation systems?|semantic search|retrieval(?:-augmented)? generation|rag|vector search|learning-to-rank|search relevance)\b", "machine learning"),
    (r"\b(frontend|front-end|ui|ux|react|angular|vue|web developer|web engineering)\b", "frontend"),
    (r"\b(backend|back-end|api|server-side|microservices|rest api|flask|fastapi|spring)\b", "backend"),
    (r"\b(data science|data scientist|analytics|statistics|statistician)\b", "data science"),
    (r"\b(data engineering|etl|data pipelines?|spark|airflow|dbt)\b", "data engineering"),
    (r"\b(devops|site reliability|sre|cloud engineer|platform engineer|kubernetes|docker|aws|azure|gcp)\b", "devops / cloud"),
    (r"\b(product management|product manager|pm)\b", "product management"),
    (r"\b(security|cybersecurity|information security|appsec)\b", "security"),
]

PROJECT_SECTION_RE = re.compile(
    r"(?:projects?|personal\s+projects?|key\s+projects?|notable\s+projects?)\s*[:\-–]?\s*\n",
    re.IGNORECASE,
)

EDUCATION_SECTION_RE = re.compile(
    r"(?:education|academic\s+qualifications?|qualifications?|educational\s+background)\s*[:\-–]?\s*\n",
    re.IGNORECASE,
)

SECTION_BOUNDARY_RE = re.compile(
    r"\n(?:experience|work\s+experience|professional\s+experience|projects?|skills|certifications?|awards|publications|summary|profile)\s*[:\-–]?\s*\n",
    re.IGNORECASE,
)


def _extract_section_text(text: str, section_re: re.Pattern[str]) -> str:
    match = section_re.search(text)
    if not match:
        return ""
    section_text = text[match.end(): match.end() + 4000]
    boundary = SECTION_BOUNDARY_RE.search(section_text)
    if boundary:
        section_text = section_text[: boundary.start()]
    return section_text


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            raise ValueError(f"Could not parse PDF: {e}")

    if len(text.strip()) >= 30:
        print(f"[PDF TEXT EXTRACT] Embedded text found ({len(text)} chars)")
        return text

    print(f"[PDF TEXT EXTRACT] No embedded text, trying OCR ({len(text)} chars from fallback)")
    ocr_text = extract_text_from_pdf_ocr(file_path)
    if ocr_text:
        print(f"[PDF OCR] Extracted {len(ocr_text)} chars via OCR")
    else:
        print(f"[PDF OCR] OCR returned empty")
    return ocr_text or text


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None and RapidOCR is not None:
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_image_with_pytesseract(image_array: np.ndarray) -> str:
    if pytesseract is None:
        return ""
    if Image is None:
        return ""
    image = Image.fromarray(image_array)
    try:
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def extract_text_from_pdf_ocr(file_path: str) -> str:
    if pdfium is None:
        print("[PDF OCR] pypdfium2 not available")
        return ""

    parts: list[str] = []
    try:
        doc = pdfium.PdfDocument(file_path)
    except Exception:
        print(f"[PDF OCR] Failed to open PDF with pypdfium2")
        return ""

    try:
        ocr_engine = _get_ocr_engine()
        print(f"[PDF OCR] Engine available: {ocr_engine is not None}")
        for page_index in range(len(doc)):
            page = doc[page_index]
            bitmap = page.render(scale=2)
            pil_image = bitmap.to_pil()
            image_array = np.array(pil_image)
            print(f"[PDF OCR] Page {page_index}: image shape {image_array.shape}")

            page_text = ""
            if ocr_engine is not None:
                try:
                    result = ocr_engine(image_array)
                    print(f"[PDF OCR] Page {page_index} OCR result: {type(result)}, items: {len(result) if result else 0}")
                    if result and len(result) > 0:
                        detections = result[0] if isinstance(result, (tuple, list)) and isinstance(result[0], list) else result
                        text_parts = []
                        if isinstance(detections, list):
                            for item in detections:
                                if not item or len(item) <= 1:
                                    continue
                                text_content = item[1]
                                if isinstance(text_content, str) and text_content.strip():
                                    text_parts.append(text_content)
                        page_text = "\n".join(text_parts)
                except Exception as e:
                    print(f"[PDF OCR] Page {page_index} OCR exception: {e}")
                    import traceback
                    traceback.print_exc()
                    page_text = ""

            if not page_text:
                page_text = _ocr_image_with_pytesseract(image_array)
                if page_text:
                    print(f"[PDF OCR] Page {page_index}: Fallback to pytesseract got {len(page_text)} chars")

            if page_text:
                parts.append(page_text)
                print(f"[PDF OCR] Page {page_index}: Added {len(page_text)} chars")
    finally:
        try:
            doc.close()
        except Exception:
            pass

    result_text = "\n".join(parts)
    print(f"[PDF OCR] Final result: {len(result_text)} chars from {len(parts)} pages")
    return result_text


def extract_text_from_txt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def normalize_skill(skill: str) -> str:
    s = skill.lower().strip()
    return SKILL_ALIASES.get(s, s)


def normalize_skill_list(skills: list[str]) -> list[str]:
    normalized = []
    seen = set()
    for skill in skills:
        canonical = normalize_skill(skill)
        if canonical and canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return sorted(normalized)


def extract_skills(text: str, include_noun_chunks: bool = True) -> list[str]:
    text_lower = text.lower()
    found = set()
    for skill in SKILL_KEYWORDS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.add(skill)
    # Also check aliases in text
    for alias, canonical in SKILL_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
            if canonical in SKILL_KEYWORDS:
                found.add(canonical)

    # Noun chunk extraction enriches resume skills but must NOT be used on JD text
    # because generic phrases ("our team", "the ideal candidate") inflate the
    # skill count and make the skill match denominator artificially large.
    if include_noun_chunks:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            if 1 <= len(chunk.text.split()) <= 4:
                clean_chunk = chunk.text.lower().strip("\n\t .,;:()[]{}")
                if clean_chunk and not clean_chunk.isnumeric():
                    found.add(clean_chunk)

    return normalize_skill_list(list(found))


def extract_experience_years(text: str) -> float:
    text_lower = text.lower()
    print(f"[EXPERIENCE EXTRACT] Input text: {len(text)} chars")

    # Try explicit patterns first
    for pattern in EXPERIENCE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            print(f"[EXPERIENCE EXTRACT] Found via pattern: {match.group(1)} years")
            return float(match.group(1))

    # Catch phrases like "5+ years" even when no other keyword follows.
    loose_year_match = re.search(r"(\d+\.?\d*)\s*\+\s*years?", text_lower)
    if loose_year_match:
        print(f"[EXPERIENCE EXTRACT] Found via loose match: {loose_year_match.group(1)} years")
        return float(loose_year_match.group(1))

    # Date range heuristic: count year spans like "2018 - 2022" or "Jan 2019 – Mar 2023"
    year_pairs = re.findall(r"\b(20\d{2}|19\d{2})\b.*?\b(20\d{2}|19\d{2}|present|current)\b", text_lower)
    total_years = 0.0
    for start, end in year_pairs:
        try:
            s = int(start)
            e = 2024 if end in ("present", "current") else int(end)
            diff = e - s
            if 0 < diff <= 20:
                total_years += diff
        except ValueError:
            continue
    if total_years > 0:
        print(f"[EXPERIENCE EXTRACT] Found via date range: {total_years} years")
        return min(round(total_years, 1), 30.0)

    print(f"[EXPERIENCE EXTRACT] No experience found, returning 0.0")
    return 0.0


def extract_experience_domain(text: str) -> str:
    text_lower = text.lower()
    for pattern, domain in EXPERIENCE_DOMAIN_PATTERNS:
        if re.search(pattern, text_lower):
            return domain

    top_lines = [line.strip() for line in text.splitlines()[:12] if line.strip()]
    for line in top_lines:
        if re.search(r"\b(engineer|developer|scientist|analyst|architect|manager|consultant)\b", line.lower()):
            return line.strip()

    return ""


def extract_education(text: str) -> list[str]:
    section_text = _extract_section_text(text, EDUCATION_SECTION_RE)
    if not section_text:
        return []

    text_lower = section_text.lower()
    found = []
    for pattern, label in DEGREE_MAP.items():
        if re.search(pattern, text_lower):
            found.append(label)
    # Deduplicate while preserving highest degree first
    degree_rank = ["PhD", "M.Tech", "M.E.", "M.Sc", "MBA", "MCA", "B.Tech", "B.E.", "B.Sc", "BCA"]
    found = sorted(set(found), key=lambda d: degree_rank.index(d) if d in degree_rank else 99)
    return found


def extract_projects(text: str) -> list[str]:
    projects = []

    # Try to find a Projects section
    match = PROJECT_SECTION_RE.search(text)
    if match:
        section_text = text[match.end(): match.end() + 3000]
        # Stop at next major section
        next_section = re.search(
            r"\n(?:education|experience|skills|certifications|awards|publications)\s*[:\-–]?\s*\n",
            section_text,
            re.IGNORECASE,
        )
        if next_section:
            section_text = section_text[: next_section.start()]
        lines = [l.strip() for l in section_text.split("\n") if len(l.strip()) > 25]
        projects = lines[:8]

    # Fallback: find lines with project-like signals
    if not projects:
        for line in text.split("\n"):
            stripped = line.strip()
            if (
                len(stripped) > 30
                and re.search(r"\b(built|developed|created|implemented|designed|trained|deployed)\b", stripped, re.IGNORECASE)
            ):
                projects.append(stripped)
            if len(projects) >= 6:
                break

    return projects


def extract_name(text: str) -> str:
    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    # Fallback: first non-empty line that looks like a name
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped.split()) <= 4 and stripped.replace(" ", "").isalpha():
            return stripped
    return ""


def parse_degree_from_text(text: str) -> list[str]:
    """Extract degree labels directly from a free-form text string.

    Works on both full resume text AND short strings like
    "PhD Natural Language Processing IISc Bangalore".
    Unlike extract_education() this does NOT require a section header.
    """
    text_lower = text.lower()
    found = []
    for pattern, label in DEGREE_MAP.items():
        if re.search(pattern, text_lower):
            found.append(label)
    degree_rank = ["PhD", "M.Tech", "M.E.", "M.Sc", "MBA", "MCA", "B.Tech", "B.E.", "B.Sc", "BCA"]
    return sorted(set(found), key=lambda d: degree_rank.index(d) if d in degree_rank else 99)


def parse_resumes_from_csv(file_bytes: bytes) -> list[dict]:
    """Parse a CSV file where each row represents one candidate.

    Recognised column names (case-insensitive):
      name, resume_text / raw_text / text, skills, experience_years / years,
      experience_domain / domain, education, projects
    Returns a list of candidate dicts ready for ranking.
    """
    text_io = io.StringIO(file_bytes.decode("utf-8", errors="ignore"))
    reader = csv.DictReader(text_io)

    def _col(row: dict, *candidates: str) -> str:
        for k in row:
            if k.strip().lower() in candidates:
                return row[k].strip()
        return ""

    results = []
    for row in reader:
        raw_text = _col(row, "resume_text", "raw_text", "text", "resume", "content")
        name = _col(row, "name", "candidate_name", "full_name") or "Unknown"

        if raw_text:
            skills_col = _col(row, "skills", "skill_list", "technologies")
            if skills_col:
                skills = normalize_skill_list([s.strip() for s in re.split(r"[;,|]", skills_col) if s.strip()])
            else:
                skills = normalize_skill_list(extract_skills(raw_text))

            exp_col = _col(row, "experience_years", "years", "exp_years", "experience")
            try:
                experience_years = float(exp_col) if exp_col else extract_experience_years(raw_text)
            except ValueError:
                experience_years = extract_experience_years(raw_text)

            domain_col = _col(row, "experience_domain", "domain", "field")
            experience_domain = domain_col or extract_experience_domain(raw_text)

            education_col = _col(row, "education", "degree", "qualification")
            if education_col:
                # parse_degree_from_text handles free-form strings like
                # "PhD Natural Language Processing IISc Bangalore"
                education = parse_degree_from_text(education_col) or [education_col]
            else:
                education = extract_education(raw_text) or parse_degree_from_text(raw_text)

            projects_col = _col(row, "projects", "project_list")
            projects = (
                [p.strip() for p in re.split(r"[;|]", projects_col) if p.strip()]
                if projects_col
                else extract_projects(raw_text)
            )
        else:
            # No resume text — build from individual columns if available
            skills_col = _col(row, "skills", "skill_list", "technologies")
            skills = normalize_skill_list([s.strip() for s in re.split(r"[;,|]", skills_col) if s.strip()])
            raw_text = f"Candidate: {name}. Skills: {skills_col}."
            exp_col = _col(row, "experience_years", "years", "exp_years", "experience")
            try:
                experience_years = float(exp_col) if exp_col else 0.0
            except ValueError:
                experience_years = 0.0
            experience_domain = _col(row, "experience_domain", "domain", "field")
            education_col = _col(row, "education", "degree", "qualification")
            education = parse_degree_from_text(education_col) if education_col else []
            projects = []

        results.append({
            "name": name,
            "raw_text": raw_text,
            "skills": skills,
            "experience_years": experience_years,
            "experience_domain": experience_domain,
            "education": education,
            "projects": projects,
        })

    return results


def parse_resume(file_path: str) -> dict:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in (".txt", ".md"):
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return {
        "raw_text": text,
        "skills": normalize_skill_list(extract_skills(text)),
        "experience_years": extract_experience_years(text),
        "experience_domain": extract_experience_domain(text),
        "education": extract_education(text),
        "projects": extract_projects(text),
        "auto_name": extract_name(text),
    }
