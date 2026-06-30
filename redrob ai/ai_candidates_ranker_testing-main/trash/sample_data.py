"""
Quick CLI test — no server required.
  python sample_data.py
"""

from ranker import rank_candidates

JD = """
Senior Machine Learning Engineer

We are looking for an experienced ML Engineer to join our AI team.

Requirements:
- 4+ years of experience in machine learning or deep learning
- Strong proficiency in Python and PyTorch
- Experience with NLP models (BERT, transformers, LLMs)
- Familiarity with MLOps, Docker, and cloud platforms (AWS/GCP)
- Knowledge of REST API development with FastAPI or Flask
- Experience with vector databases (FAISS, Pinecone, Weaviate)
- Strong understanding of SQL and data pipelines

Nice to have:
- Experience with RAG systems or semantic search
- Contributions to open-source ML projects
"""

CANDIDATES = [
    {
        "name": "Alice Chen",
        "raw_text": """
        Alice Chen | Senior ML Engineer | 6 years experience

        Skills: Python, PyTorch, TensorFlow, BERT, transformers, FastAPI, Docker, AWS, FAISS,
                SQL, PostgreSQL, pandas, numpy, scikit-learn, LLM fine-tuning, RAG systems

        Experience:
        - Senior ML Engineer at TechCorp (2019 - 2022): Built semantic search using FAISS and SBERT.
          Developed RAG pipeline for enterprise Q&A. Deployed models on AWS SageMaker.
        - ML Engineer at DataInc (2022 - 2025): NLP model development, REST APIs with FastAPI.

        Projects:
        - Built open-source LLM fine-tuning toolkit (500+ GitHub stars)
        - Developed semantic resume ranker using sentence-transformers and FAISS
        - Implemented distributed training pipeline with PyTorch DDP

        Education: B.Tech Computer Science, IIT Bombay
        """,
        "skills": ["python", "pytorch", "transformers", "bert", "fastapi", "docker", "aws",
                   "faiss", "sql", "pandas", "numpy", "machine learning", "deep learning", "nlp",
                   "llm", "rag", "scikit-learn"],
        "experience_years": 6,
        "response_rate": 0.92,
        "profile_completeness": 0.96,
        "interview_completion_rate": 0.88,
        "activity_level": 0.81,
        "projects": [
            "Built open-source LLM fine-tuning toolkit with 500+ GitHub stars",
            "Developed semantic resume ranker using sentence-transformers and FAISS",
            "Implemented distributed training pipeline with PyTorch DDP",
        ],
        "education": ["B.Tech"],
    },
    {
        "name": "Bob Smith",
        "raw_text": """
        Bob Smith | Software Developer | 3 years experience

        Skills: Python, Flask, SQL, JavaScript, React, MySQL, Docker

        Experience:
        - Backend Developer at WebAgency (2022 - 2025): REST API development, database optimization.

        Projects:
        - Built e-commerce backend with Flask and PostgreSQL
        - Created React dashboard for analytics

        Education: B.Sc Computer Science
        """,
        "skills": ["python", "flask", "sql", "javascript", "react", "docker"],
        "experience_years": 3,
        "response_rate": 0.61,
        "profile_completeness": 0.74,
        "interview_completion_rate": 0.58,
        "activity_level": 0.46,
        "projects": [
            "Built e-commerce backend with Flask and PostgreSQL",
            "Created React dashboard for sales analytics",
        ],
        "education": ["B.Sc"],
    },
    {
        "name": "Priya Sharma",
        "raw_text": """
        Priya Sharma | Data Scientist | 5 years experience

        Skills: Python, PyTorch, scikit-learn, pandas, numpy, SQL, machine learning,
                deep learning, NLP, BERT, transformers, GCP, REST API

        Experience:
        - Data Scientist at AnalyticsCo (2020 - 2025): NLP classification models, recommendation systems,
          A/B testing, model deployment on GCP using Docker.

        Projects:
        - Trained sentiment analysis model at scale using BERT fine-tuning
        - Built recommendation engine for e-commerce using PyTorch
        - Developed text classification pipeline with transformers

        Education: M.Sc Data Science
        """,
        "skills": ["python", "pytorch", "scikit-learn", "pandas", "numpy", "sql", "machine learning",
                   "deep learning", "nlp", "bert", "transformers", "gcp", "rest api", "docker"],
        "experience_years": 5,
        "response_rate": 0.84,
        "profile_completeness": 0.9,
        "interview_completion_rate": 0.79,
        "activity_level": 0.73,
        "projects": [
            "Trained sentiment analysis model at scale using BERT fine-tuning",
            "Built recommendation engine for e-commerce using PyTorch",
            "Developed text classification pipeline with transformers",
        ],
        "education": ["M.Sc"],
    },
]


if __name__ == "__main__":
    print("Ranking candidates...\n")
    results, gap_report = rank_candidates(JD, CANDIDATES, top_k=3, use_cross_encoder=True)

    for r in results:
        print(f"Rank #{r['rank']}: {r['name']}")
        print(f"  Final Score : {r['final_score']:.1%}")
        print(f"  Recruitability: {r['recruitability_score']:.1%}")
        print(f"  Hire Confidence: {r['hire_confidence']:.1%}")
        print(f"  Semantic    : {r['semantic_score']:.1%}")
        print(f"  Skills      : {r['skill_score']:.1%}")
        print(f"  Experience  : {r['experience_score']:.1%}")
        print(f"  Projects    : {r['project_score']:.1%}")
        print()

    print("Skill Gap Report:")
    for skill, missing in gap_report.items():
        print(f"  {skill}: {missing}/{len(CANDIDATES)} candidates missing")
