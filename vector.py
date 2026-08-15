from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rag import embed_data

chunk_embedings,chunks = embed_data()

client = QdrantClient(path="./qdrant_db")

vector_size = len(chunk_embedings[0])

print(vector_size)








if not client.collection_exists(collection_name="indian_constitution"):
    client.create_collection(
        collection_name="indian_constitution",
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

points = []

for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embedings)):
    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": chunk
            }
        )
    )

client.upsert(
    collection_name="indian_constitution",
    points=points
)
client.close()