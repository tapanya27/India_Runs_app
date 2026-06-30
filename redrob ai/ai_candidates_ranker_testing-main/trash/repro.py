from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

with open("candidate0.txt", encoding="utf-8") as f:
    t0 = f.read()

with open("candidate1.txt", encoding="utf-8") as f:
    t1 = f.read()

print("One")
print(model.encode([t0]).shape)

print("Two")
print(model.encode([t0, t1]).shape)