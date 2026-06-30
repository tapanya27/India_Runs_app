import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from resume_parser import parse_resume, parse_resumes_from_csv, normalize_skill_list
from ranker import rank_candidates

app = FastAPI(title="AI Candidate Ranker", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RankRequest(BaseModel):
    job_description: str
    candidates: list[dict]
    top_k: int = 10
    use_cross_encoder: bool = True


class RankResponse(BaseModel):
    ranked_candidates: list[dict]
    skill_gap_report: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Only .csv files are accepted here")
    content = await file.read()
    try:
        candidates = parse_resumes_from_csv(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {e}")
    if not candidates:
        raise HTTPException(status_code=422, detail="CSV contained no valid rows")
    return {"candidates": candidates, "count": len(candidates)}


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest):
    if not req.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description is required")
    if not req.candidates:
        raise HTTPException(status_code=400, detail="candidates list is empty")

    ranked, gap_report = rank_candidates(
        req.job_description,
        req.candidates,
        top_k=req.top_k,
        use_cross_encoder=req.use_cross_encoder,
    )
    return RankResponse(ranked_candidates=ranked, skill_gap_report=gap_report)


@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    name: str = Form(...),
):
    allowed = {".pdf", ".txt", ".md"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        parsed = parse_resume(tmp_path)
        parsed["skills"] = normalize_skill_list(parsed.get("skills", []))
        # Prefer user-provided name over auto-detected
        parsed["name"] = name or parsed.get("auto_name") or Path(file.filename).stem
        parsed["filename"] = file.filename
    finally:
        os.unlink(tmp_path)

    return parsed
