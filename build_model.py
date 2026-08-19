from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("./embedding_model")
print("Embedding model downloaded and packaged.")
