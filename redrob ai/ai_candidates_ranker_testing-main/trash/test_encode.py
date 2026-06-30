from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

texts = [
    "Software Engineer with Python and Machine Learning experience.",
    "Backend developer with FastAPI and PostgreSQL."
]

print("Single:")
print(model.encode([texts[0]]).shape)

print("Two:")
print(model.encode(texts).shape)