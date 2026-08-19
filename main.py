from chat import chat
from data import embed_data, qdrant


# only for laoding new data

# print("Loading and embedding data...")
# embeddings, text_chunks = embed_data()

# print("Saving to Qdrant...")
# qdrant("indian_constitution", embeddings, text_chunks)

# print("Done! all data loaded")

# for regular chat from qdrant
chat()

