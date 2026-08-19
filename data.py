from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import glob
import os
import fitz
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def data_load_md() -> str:
    docs_path = snapshot_download(
        repo_id="err0rgod/indian_const",
        repo_type="dataset"
    )
    return docs_path
def data_load():
   return data_load_md()

def chunk_text(text, chunk_size=150,overlap=20):
  words = text.split() #break the string into words
  chunks = []
  step = chunk_size-overlap #how far we sslide the window each time

  for start in range(0, len(words), step):
    chunk_words = words[start:start+chunk_size]
    if not chunk_words:
      break
    chunks.append(" ".join(chunk_words))
    if start + chunk_size >= len(words):
      break
  return chunks


def embed_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    DATA_PATH = data_load()
    all_chunks =[]
    metadata = []
    for filepath in glob.glob(os.path.join(DATA_PATH,"*.pdf")):
        doc = fitz.open(filepath)
        text =""

        for page in doc:
           text += page.get_text()  
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({"source":filepath, "chunk_index":i})

    chunk_embeddings = model.encode(all_chunks, normalize_embeddings=True)
    print(chunk_embeddings.shape)
    return chunk_embeddings,all_chunks


def qdrant( data_collection_name, chunk_embeddings, chunks):
    client = QdrantClient(path="./qdrant_db")

    vector_size = len(chunk_embeddings[0])

    print(vector_size)

    if not client.collection_exists(collection_name=data_collection_name):
        client.create_collection(
            collection_name=data_collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
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
        collection_name=data_collection_name,
        points=points
    )
    client.close()

if __name__ == "__main__":
    print("Loading and embedding data...")
    embeddings, text_chunks = embed_data()
    
    print("Saving to Qdrant...")
    qdrant("indian_constitution", embeddings, text_chunks)
    
    print("Done!")
