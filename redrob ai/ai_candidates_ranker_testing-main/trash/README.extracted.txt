Redrob Hackathon — Participant Bundle
Welcome to the Intelligent Candidate Discovery & Ranking Challenge.
What's in this bundle
Getting started
1. Read the docs (~30 minutes)
In this order:
1.       job_description.md — understand what role you're ranking candidates for
2.       submission_spec.md — understand the rules and evaluation pipeline
3.       redrob_signals_doc.md — understand the trap candidates and signal envelopes
4.       candidate_schema.json — understand the candidate data structure
5.       Open sample_candidates.json and skim a few candidates to see what real data looks like
2. Unpack the candidate pool
gunzip -k candidates.jsonl.gz   # -k keeps the .gz; you get both files
 wc -l candidates.jsonl      	# should print 100000
Or load the gzipped file directly in Python:
import gzip, json
 with gzip.open("candidates.jsonl.gz", "rt") as f:
 	candidates = [json.loads(line) for line in f if line.strip()]
 print(len(candidates))  # 100000
3. Build your ranker
Your job: produce a CSV with the top 100 candidates for the JD, ranked best-fit first, with a 1-2 sentence reasoning for each.
The format is described in submission_spec.md Section 2-3. The compute constraints are in Section 3 (5 min, 16 GB, CPU only, no network during ranking).
4. Validate before submitting
This catches format errors before you upload. The validator handles both .jsonl and gzipped .jsonl.gz files.
5. Submit
Submit via the portal. You'll be asked for:
·         The CSV file
·         All the metadata from submission_metadata_template.yaml (team name, GitHub repo, sandbox link, AI tools declaration, etc.)
·         See submission_spec.md Section 10 for the full list
Sandbox link is required — a working hosted environment (HuggingFace Spaces, Streamlit Cloud, Replit, Colab, Docker, or Binder) where your ranker can be run on a small sample. See Section 10.5 for what counts as a valid sandbox.
Key things to know
·         No live leaderboard. Scores are revealed only after submissions close. There is no feedback during the competition.
·         Three submissions max. Your last valid submission counts.
·         AI tools are allowed. Declare them honestly. The evaluation is designed so that AI-assisted work where you did real engineering succeeds, while AI-only submissions fail at Stages 3-5.
·         The dataset contains traps. Keyword stuffers, plain-language Tier 5s, behavioral twins, and ~80 honeypots with subtly impossible profiles.Submissions with honeypot rate > 10% in top 100 are disqualified you. See redrob_signals_doc.md.
·         You will be interviewed if you reach the top X. Be prepared to walk through your architecture and defend your design choices.
Asking for help
If you find a bug in the bundle (e.g., schema doesn't match data, validator rejects valid format) please report it via the official hackathon support channel.
Good luck.
File | What it is
candidates.jsonl.gz | The 100,000-candidate pool you'll rank. Gzipped JSONL (~52 MB compressed, ~465 MB uncompressed).
sample_candidates.json | First 50 candidates as pretty-printed JSON. Use this to inspect the schema quickly.
job_description.md | The job description you're ranking candidates against. Read it carefully — including the section at the end specifically for hackathon participants.
submission_spec.md | Read this in full before starting. Submission format, rules, compute constraints, evaluation stages.
submission_metadata_template.yaml | Template for the metadata you'll provide alongside your submission.
candidate_schema.json | JSON Schema describing every field in a candidate record.
redrob_signals_doc.md | Reference for the 23 behavioral signals in each candidate's redrob_signals object. 
sample_submission.csv | A format reference. Not a high-quality ranking — just an example of the CSV structure your submission should match.
validate_submission.py | Format validator. Run this on your submission before uploading.