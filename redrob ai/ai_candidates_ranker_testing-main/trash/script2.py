from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

text = open("candidate0.txt", encoding="utf-8").read()

print(len(text))
print(len(model.tokenizer.encode(text)))

print(model.encode([text]).shape)