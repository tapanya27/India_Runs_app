SBERT_MODEL = "all-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

FAISS_TOP_K = 20
FINAL_TOP_K = 10

# Hybrid scoring weights (sum to 1.0)
# Increased semantic and project importance per fixes.md recommendations
WEIGHT_SEMANTIC = 0.45
WEIGHT_SKILL = 0.20
WEIGHT_EXPERIENCE = 0.15
WEIGHT_PROJECT = 0.20

# Cross-encoder combination weight: how much CE should influence final semantic score
CROSS_ENCODER_WEIGHT = 0.75

WEIGHT_RECRUITABILITY = 0.25
WEIGHT_HIRE_CONFIDENCE_TECH = 0.75
WEIGHT_HIRE_CONFIDENCE_RECRUITER = 0.25

EMBEDDING_DIM = 768

# Cross-encoder ms-marco scores typically fall in [-10, 10]; sigmoid maps this to (0, 1)
CE_SIGMOID_SCALE = 1.1

# ── Skill Matching Thresholds ────────────────────────────────────────────────
SKILL_STRONG_MATCH_THRESHOLD = 0.75
SKILL_PROBABLE_MATCH_THRESHOLD = 0.62
SKILL_CE_VALIDATION_THRESHOLD = 0.45  # Cross-encoder sigmoid cutoff for borderline matches

# Scoring weights per match type (used in compute_skill_match)
SKILL_WEIGHT_EXACT = 1.0
SKILL_WEIGHT_STRONG = 1.0
SKILL_WEIGHT_PROBABLE = 0.8
SKILL_WEIGHT_CATEGORY = 0.5

# ── Skill Taxonomy (category labels for auto-classification) ─────────────────
# Each category is a short descriptive label. Skills are classified into
# categories via SBERT similarity against these labels — NOT via lookup.
SKILL_TAXONOMY_LABELS: list[str] = [
    "Programming Languages and Scripting",
    "Web Frameworks and Frontend Libraries",
    "Backend Frameworks and APIs",
    "Databases and Data Storage",
    "Cloud Platforms and Infrastructure",
    "DevOps and CI/CD and Containerization",
    "AI and Machine Learning and Deep Learning",
    "Data Engineering and ETL Pipelines",
    "Testing and Quality Assurance",
    "Security and Cybersecurity",
    "Analytics and Data Visualization",
    "Mobile Development",
    "Version Control and Collaboration",
]

# Skill aliases: all synonyms map to a canonical form for matching
SKILL_ALIASES: dict[str, str] = {
    "ml": "machine learning",
    "dl": "deep learning",
    "js": "javascript",
    "ts": "typescript",
    "tf": "tensorflow",
    "py torch": "pytorch",
    "pytorch": "pytorch",
    "sk-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "k8s": "kubernetes",
    "llms": "llm",
    "nlp": "nlp",
    "cv": "computer vision",
    "hugging face": "huggingface",
    "huggingface": "huggingface",
    "rest": "rest api",
    "restful": "rest api",
    "rdbms": "sql",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "es": "elasticsearch",
    "hf": "huggingface",
    "bert": "bert",
    "gpt": "gpt",
    "llm": "llm",
    "rag": "rag",
    "genai": "llm",
    "generative ai": "llm",
    "node js": "node.js",
    "node": "node.js",
    "nodejs": "node.js",
    "vue.js": "vue",
    "react js": "react",
    "reactjs": "react",
    "angularjs": "angular",
    "springboot": "spring",
    "spring boot": "spring",
    "xgboost": "machine learning",
    "lightgbm": "machine learning",
    "catboost": "machine learning",
    "devops": "ci/cd",
    "mlops": "ci/cd",
}

SKILL_KEYWORDS: list[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "scala", "r",
    "kotlin", "swift", "php", "ruby",
    # Web
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring",
    "html", "css", "graphql", "rest api",
    # ML/AI
    "pytorch", "tensorflow", "keras", "scikit-learn", "huggingface",
    "embeddings", "semantic search", "learning-to-rank",
    "machine learning", "deep learning", "nlp", "computer vision", "llm",
    "transformers", "bert", "gpt", "fine-tuning", "rag", "faiss",
    "reinforcement learning", "xgboost", "lightgbm",
    # Data
    "pandas", "numpy", "sql", "postgresql", "mysql", "mongodb", "redis",
    "elasticsearch", "spark", "kafka", "airflow", "dbt",
    # DevOps/Cloud
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ci/cd",
    "git", "linux", "microservices",
    # Other
    "opencv", "matplotlib", "seaborn", "plotly", "streamlit",
]
