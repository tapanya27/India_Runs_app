from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("all-mpnet-base-v2")

print("Encoding...")

emb = model.encode(["hello world"])

print(emb.shape)