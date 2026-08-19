from fastembed import TextEmbedding


model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="./fastembed_cache",
    threads=1,
)
next(model.embed(["Render build check"]))
print("Embedding model downloaded and packaged.")
