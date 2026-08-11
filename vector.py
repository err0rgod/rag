from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rag import embed_data

chunk_embedings = embed_data()


client = QdrantClient(path="./qdrant_db")


vector_size = len(chunk_embedings[0])

print(vector_size)