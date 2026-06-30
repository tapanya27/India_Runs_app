import faiss
from sentence_transformers import SentenceTransformer

print("FAISS imported")

model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

print(model.encode(["hello", "world"]).shape)