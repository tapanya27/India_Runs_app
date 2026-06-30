from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

texts = []

for i in range(100):
    texts.append("Software engineer with Python FastAPI Docker PostgreSQL")

print(model.encode(texts).shape)