from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

texts = [
    """Marketing Manager | Exploring AI & GenAI applications
Role: Marketing Manager
Experience: 13.9 years
Skills: Spark (intermediate, 21mo, 7 endorsements), Vue.js (intermediate, 21mo), Terraform (intermediate, 35mo, 11 endorsements), Accounting (beginner, 10mo, 15 endorsements), Databricks (beginner, 2mo, 15 endorsements), Illustrator (intermediate, 16mo, 6 endorsements), Snowflake (intermediate, 25mo, 6 endorsements)
""",
    """Software Engineer
Role: Backend Engineer
Experience: 5 years
Skills: Python, FastAPI, PostgreSQL, Redis, Docker
"""
]

print(model.encode(texts).shape)